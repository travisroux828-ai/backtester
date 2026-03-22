"""
CSV export for backtest results - designed for TradingView chart review.
"""

from __future__ import annotations

import csv
import io

import pandas as pd

from engine.models import BacktestResult, Trade


COLUMNS = [
    "date",
    "ticker",
    "direction",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "shares",
    "gross_pnl",
    "signal_reason",
    "exit_reason",
    "stop_price",
    "target_price",
    "vwap_at_entry",
    "rsi_at_entry",
    "volume_ratio_at_entry",
    "gap_percent",
    "premarket_high",
    "orb_high",
    "orb_low",
]


def trade_to_row(trade: Trade) -> dict:
    """Convert a Trade to a CSV row dict."""
    meta = trade.metadata or {}
    return {
        "date": trade.entry_time.strftime("%Y-%m-%d"),
        "ticker": trade.ticker,
        "direction": trade.direction,
        "entry_time": trade.entry_time.strftime("%H:%M:%S"),
        "exit_time": trade.exit_time.strftime("%H:%M:%S"),
        "entry_price": round(trade.entry_price, 2),
        "exit_price": round(trade.exit_price, 2),
        "shares": trade.shares,
        "gross_pnl": round(trade.gross_pnl, 2),
        "signal_reason": trade.signal_reason,
        "exit_reason": trade.exit_reason,
        "stop_price": round(trade.stop_price, 2) if trade.stop_price else "",
        "target_price": round(trade.target_price, 2) if trade.target_price else "",
        "vwap_at_entry": meta.get("vwap_at_entry", ""),
        "rsi_at_entry": meta.get("rsi_at_entry", ""),
        "volume_ratio_at_entry": meta.get("volume_ratio_at_entry", ""),
        "gap_percent": round(meta["gap_percent"], 2) if meta.get("gap_percent") is not None else "",
        "premarket_high": round(meta["pm_high"], 2) if meta.get("pm_high") is not None else "",
        "orb_high": round(meta["orb_high"], 2) if meta.get("orb_high") is not None else "",
        "orb_low": round(meta["orb_low"], 2) if meta.get("orb_low") is not None else "",
    }


def export_to_csv(result: BacktestResult, filepath: str | None = None) -> str:
    """
    Export backtest results to CSV.

    If filepath is provided, writes to file. Always returns CSV string.
    """
    rows = [trade_to_row(t) for t in result.trades]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

    csv_string = output.getvalue()

    if filepath:
        with open(filepath, "w", newline="") as f:
            f.write(csv_string)

    return csv_string


def result_to_dataframe(result: BacktestResult) -> pd.DataFrame:
    """Convert backtest results to a pandas DataFrame for display."""
    rows = [trade_to_row(t) for t in result.trades]
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(rows, columns=COLUMNS)
