"""
Trade Processor - Converts raw execution logs to enriched trade reviews.

Usage:
    python trade_processor.py <raw_csv_path> [--api-key YOUR_POLYGON_KEY] [--output output.csv] [--no-api]

Examples:
    # With Polygon API enrichment
    python trade_processor.py 02_24_26.csv --api-key xxxxxxxxxxx

    # Without API (basic metrics only, Polygon columns will be empty)
    python trade_processor.py 02_24_26.csv --no-api

    # Custom output path
    python trade_processor.py 02_24_26.csv --api-key xxxxxxxxxxx --output my_trades.csv

Input format:  Raw execution CSV from DAS/Maverick with columns:
               Exec Time, Symbol, Shares, Transaction, Price, Position, Running Net,
               Running Gross, STC, TAFee, orf, catfee, Comm, ECN, Contra, Liquidity

Output format: Trade-level CSV matching your review template with columns:
               Date, Symbol, Side, Net P&L, Gross P&L, Duration, Max % Up, Entry Time,
               Time Bucket, Avg Entry Price, Avg Exit Price, Total Shares, VWAP Status,
               Dist From VWAP %, Added To Position, Cum Vol at Entry, Relative Vol, Float,
               Total Day High/Low (with times), Pre/Regular/Post Market Highs (with times),
               Prev Close, Total Day Vol, Win, Strategy, Sub-Category 1, Sub-Category 2,
               Notes Link
"""

import pandas as pd
import numpy as np
import argparse
import time
import sys
import os
from datetime import datetime, timedelta

# ===================================================================
# USER CONFIG - Edit these before running, then just hit Run
# ===================================================================
INPUT_FILE   = "0.3_09_26.csv"          # <-- Change this to your raw CSV filename
OUTPUT_FILE  = ""                       # <-- Leave blank for auto-naming (adds _processed)
POLYGON_KEY  = os.environ.get("POLYGON_API_KEY", "")  # <-- Set via env var or --api-key flag
USE_API      = True                 # <-- Set to True once you add your API key
# ===================================================================

# ---------------------------------------------------------------------------
# Polygon API helper (extracted to data/polygon_client.py)
# ---------------------------------------------------------------------------
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from data.polygon_client import PolygonClient


# ---------------------------------------------------------------------------
# Core trade parsing
# ---------------------------------------------------------------------------

def parse_raw_executions(filepath):
    """Read the raw execution CSV and clean columns."""
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    df["Symbol"] = df["Symbol"].str.strip()
    df["Transaction"] = df["Transaction"].str.strip()
    df["Exec Time"] = df["Exec Time"].str.strip()
    df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce").fillna(0).astype(int)

    # Clean price and position (remove commas, spaces, parens for negatives)
    for col in ["Price", "Position", "Running Net", "Running Gross", "TAFee", "orf", "catfee", "Comm"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace("(", "-", regex=False)
            .str.replace(")", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["Exec Time"], format="%Y-%m-%d : %H:%M:%S")
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.time

    return df


def group_into_trades(df):
    """
    Group raw executions into individual trades.
    A trade starts when position goes from 0 to non-zero and ends when it returns to 0.
    """
    trades = []

    for symbol in df["Symbol"].unique():
        sym_df = df[df["Symbol"] == symbol].copy()
        # Sort by time, then by abs(Position) descending so Position=0 comes last
        # within fills that share the same timestamp (prevents early trade boundary)
        sym_df["_abs_pos"] = sym_df["Position"].abs()
        sym_df = sym_df.sort_values(["datetime", "_abs_pos"], ascending=[True, False]).reset_index(drop=True)
        sym_df = sym_df.drop(columns=["_abs_pos"])

        trade_fills = []
        for _, row in sym_df.iterrows():
            trade_fills.append(row)
            if row["Position"] == 0 and len(trade_fills) > 0:
                trades.append(trade_fills.copy())
                trade_fills = []

        # If there are leftover fills (position didn't close), still record them
        if len(trade_fills) > 0:
            trades.append(trade_fills)

    return trades


def classify_fill(txn, current_side):
    """Determine if a fill is an entry or exit based on transaction type and trade side."""
    txn = txn.upper()
    if current_side == "Long":
        return "entry" if "BOT" in txn else "exit"
    else:  # Short
        return "entry" if "SLD" in txn else "exit"


def get_time_bucket(hour, minute):
    """Map entry time to session bucket."""
    t = hour * 60 + minute
    if t < 570:       # Before 09:30
        return "00_Pre-Market"
    elif t < 600:     # 09:30 - 10:00
        return "01_Open (09:30-10:00)"
    elif t < 690:     # 10:00 - 11:30
        return "02_Mid-Morning (10:00-11:30)"
    elif t < 810:     # 11:30 - 13:30
        return "03_Lunch (11:30-13:30)"
    elif t < 960:     # 13:30 - 16:00
        return "04_Close (13:30-16:00)"
    else:             # 16:00+
        return "05_Post-Market"


def process_trade(fills):
    """Convert a list of fills into a single trade row dict."""
    first = fills[0]
    last = fills[-1]

    symbol = first["Symbol"]
    date_str = str(first["date"])

    # Determine side from first fill
    first_txn = first["Transaction"].upper()
    side = "Long" if "BOT" in first_txn else "Short"

    # Separate entry vs exit fills
    entry_fills = []
    exit_fills = []
    for f in fills:
        role = classify_fill(f["Transaction"], side)
        if role == "entry":
            entry_fills.append(f)
        else:
            exit_fills.append(f)

    # Calculate average entry and exit prices (share-weighted)
    entry_shares = sum(f["Shares"] for f in entry_fills)
    exit_shares = sum(f["Shares"] for f in exit_fills)
    total_shares = max(entry_shares, exit_shares)

    avg_entry = (sum(f["Price"] * f["Shares"] for f in entry_fills) / entry_shares) if entry_shares > 0 else 0
    avg_exit = (sum(f["Price"] * f["Shares"] for f in exit_fills) / exit_shares) if exit_shares > 0 else 0

    # Gross P&L
    if side == "Long":
        gross_pnl = sum(f["Price"] * f["Shares"] for f in exit_fills) - sum(f["Price"] * f["Shares"] for f in entry_fills)
    else:
        gross_pnl = sum(f["Price"] * f["Shares"] for f in entry_fills) - sum(f["Price"] * f["Shares"] for f in exit_fills)

    # Net P&L (gross minus all fees)
    total_fees = sum(abs(f["TAFee"]) + abs(f["orf"]) + abs(f["catfee"]) + abs(f["Comm"]) for f in fills)
    net_pnl = gross_pnl - total_fees

    # Duration
    entry_time = first["datetime"]
    exit_time = last["datetime"]
    duration = exit_time - entry_time
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Max % Up (best unrealized P&L as % of entry cost)
    # Track running position cost and mark-to-market at each fill
    max_pct_up = 0.0
    running_qty = 0
    running_cost = 0.0
    for f in fills:
        txn = f["Transaction"].upper()
        shares = f["Shares"]
        price = f["Price"]

        if classify_fill(f["Transaction"], side) == "entry":
            running_cost += price * shares
            running_qty += shares
        else:
            if running_qty > 0:
                cost_basis = running_cost / running_qty
                if side == "Long":
                    pct = ((price - cost_basis) / cost_basis) * 100
                else:
                    pct = ((cost_basis - price) / cost_basis) * 100
                max_pct_up = max(max_pct_up, pct)
                # Reduce cost proportionally
                running_cost -= (running_cost / running_qty) * shares
                running_qty -= shares

    # Added to position?
    added = len(entry_fills) > 1 and any(
        fills[i]["Position"] != 0 and classify_fill(fills[i]["Transaction"], side) == "entry"
        for i in range(1, len(fills))
    )

    # Entry time and time bucket
    entry_time_str = entry_time.strftime("%H:%M:%S")
    time_bucket = get_time_bucket(entry_time.hour, entry_time.minute)

    trade = {
        "Date": date_str,
        "Symbol": symbol,
        "Side": side,
        "Net P&L": round(net_pnl, 2),
        "Gross P&L": round(gross_pnl, 1),
        "Duration": duration_str,
        "Max % Up": f"{max_pct_up:.2f}%",
        "Entry Time": entry_time_str,
        "Time Bucket": time_bucket,
        "Avg Entry Price": round(avg_entry, 2),
        "Avg Exit Price": round(avg_exit, 2),
        "Total Shares": total_shares,
        "VWAP Status": "",
        "Dist From VWAP %": "",
        "Added To Position": "Yes" if added else "No",
        "Cum Vol at Entry": "",
        "Relative Vol": "",
        "Float": "",
        "Total Day High": "",
        "Total Day High Time": "",
        "Total Day Low": "",
        "Total Day Low Time": "",
        "Pre-Market High": "",
        "Pre-Market High Time": "",
        "Regular Market High": "",
        "Regular Market High Time": "",
        "Post-Market High": "",
        "Post-Market High Time": "",
        "Prev Close": "",
        "Total Day Vol": "",
        "Win": 1 if net_pnl > 0 else 0,
        "Strategy": "",
        "Sub 1": "",
        "Sub 2": "",
        "Sub 3": "",
        "Catalyst / Thesis": "",
        "AI Score (1-10)": "",
        "Notes Link": "",
    }

    return trade


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert raw execution CSV to trade review format.")
    parser.add_argument("input_csv", nargs="?", default=INPUT_FILE, help="Path to raw execution CSV file")
    parser.add_argument("--api-key", default=POLYGON_KEY or None, help="Polygon.io API key for market data enrichment")
    parser.add_argument("--output", "-o", default=OUTPUT_FILE or None, help="Output CSV path")
    parser.add_argument("--no-api", action="store_true", default=not USE_API, help="Skip Polygon API calls")
    args = parser.parse_args()

    input_file = args.input_csv or INPUT_FILE

    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    output_path = args.output
    if not output_path:
        base = os.path.splitext(input_file)[0]
        output_path = f"{base}_processed.csv"

    print(f"Reading: {input_file}")
    df = parse_raw_executions(input_file)
    print(f"  Found {len(df)} executions across {df['Symbol'].nunique()} tickers")

    print("Grouping into trades...")
    trade_groups = group_into_trades(df)
    print(f"  Found {len(trade_groups)} trades")

    print("Calculating trade metrics...")
    trades = [process_trade(fills) for fills in trade_groups]

    # Polygon enrichment
    use_api = args.api_key and not args.no_api
    if use_api:
        if not HAS_REQUESTS:
            print("Warning: 'requests' library not installed. Run: pip install requests")
            print("  Skipping API enrichment.")
        else:
            client = PolygonClient(args.api_key)
            tickers_done = set()
            for i, trade in enumerate(trades):
                ticker = trade["Symbol"]
                date = trade["Date"]
                status = f"  [{i + 1}/{len(trades)}] Enriching {ticker}..."
                if ticker not in tickers_done:
                    status += " (fetching bars + details)"
                    tickers_done.add(ticker)
                print(status)
                client.enrich_trade(trade, date)
    elif not args.no_api:
        print("  No API key provided. Use --api-key KEY for Polygon enrichment.")
        print("  Running with basic metrics only.")

    # Sort by entry time
    trades.sort(key=lambda t: (t["Date"], t["Entry Time"]))

    # Build output DataFrame
    columns = [
        "Date", "Symbol", "Side", "Net P&L", "Gross P&L", "Duration",
        "Max % Up", "Entry Time", "Time Bucket", "Avg Entry Price",
        "Avg Exit Price", "Total Shares", "VWAP Status", "Dist From VWAP %",
        "Added To Position", "Cum Vol at Entry", "Relative Vol", "Float",
        "Total Day High", "Total Day High Time", "Total Day Low",
        "Total Day Low Time", "Pre-Market High", "Pre-Market High Time",
        "Regular Market High", "Regular Market High Time", "Post-Market High",
        "Post-Market High Time", "Prev Close", "Total Day Vol", "Win",
        "Strategy", "Sub 1", "Sub 2", "Sub 3", "Catalyst / Thesis", "AI Score (1-10)", "Notes Link",
    ]

    out_df = pd.DataFrame(trades, columns=columns)
    out_df.to_csv(output_path, index=False)

    # Print summary
    total_net = sum(t["Net P&L"] for t in trades)
    winners = sum(1 for t in trades if t["Net P&L"] > 0)
    losers = sum(1 for t in trades if t["Net P&L"] <= 0)

    print(f"\n{'=' * 50}")
    print(f"Output: {output_path}")
    print(f"Trades: {len(trades)}  |  Winners: {winners}  |  Losers: {losers}")
    print(f"Win Rate: {winners / len(trades) * 100:.1f}%")
    print(f"Net P&L: ${total_net:,.2f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()