"""
Stock scanner - finds tickers matching criteria using Polygon API.

Uses the grouped daily bars endpoint to get all tickers in one call,
then filters by price/volume. For float/market cap, fetches ticker
details only for candidates that pass the initial filters.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta

from data.polygon_client import PolygonClient, CACHE_DIR

GROUPED_CACHE_DIR = os.path.join(CACHE_DIR, "grouped")
DETAILS_CACHE_DIR = os.path.join(CACHE_DIR, "details")


def _ensure_dirs():
    os.makedirs(GROUPED_CACHE_DIR, exist_ok=True)
    os.makedirs(DETAILS_CACHE_DIR, exist_ok=True)


def get_grouped_daily(client: PolygonClient, date_str: str) -> list[dict] | None:
    """
    Get grouped daily bars for ALL tickers on a given date.
    Returns list of dicts with keys: T (ticker), o, h, l, c, v, vw
    Uses disk cache to avoid repeat API calls.
    """
    _ensure_dirs()
    cache_path = os.path.join(GROUPED_CACHE_DIR, f"{date_str}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    data = client._get(
        f"/v2/aggs/grouped/locale/us/market/stocks/{date_str}",
        params={"adjusted": "true"},
    )

    if not data or "results" not in data:
        return None

    results = data["results"]

    # Cache to disk
    with open(cache_path, "w") as f:
        json.dump(results, f)

    time.sleep(0.15)
    return results


def get_cached_ticker_details(client: PolygonClient, ticker: str) -> dict | None:
    """
    Get ticker details with persistent disk cache.
    Details like float and market cap don't change often,
    so we cache them for 7 days.
    """
    _ensure_dirs()
    cache_path = os.path.join(DETAILS_CACHE_DIR, f"{ticker}.json")

    # Check disk cache (valid for 7 days)
    if os.path.exists(cache_path):
        mtime = os.path.getmtime(cache_path)
        age_days = (time.time() - mtime) / 86400
        if age_days < 7:
            with open(cache_path, "r") as f:
                return json.load(f)

    details = client.get_ticker_details(ticker)
    if details:
        with open(cache_path, "w") as f:
            json.dump(details, f)

    return details


def scan_tickers(
    client: PolygonClient,
    date_str: str,
    min_price: float = 0,
    max_price: float = float("inf"),
    min_volume: int = 0,
    min_dollar_volume: float = 0,
    min_float: float = 0,
    max_float: float = float("inf"),
    min_market_cap: float = 0,
    max_market_cap: float = float("inf"),
    min_gap_percent: float = 0,
    min_change_percent: float = 0,
    max_results: int = 50,
    progress_callback=None,
) -> list[dict]:
    """
    Scan for tickers matching the given criteria on a specific date.

    Phase 1: Filter by price/volume using grouped daily bars (1 API call).
    Phase 2: Filter by float/market cap using ticker details (only for phase 1 candidates).

    Returns list of dicts with ticker info, sorted by volume descending.
    """
    grouped = get_grouped_daily(client, date_str)
    if not grouped:
        return []

    # Phase 1: Price and volume filters (no extra API calls)
    candidates = []
    for bar in grouped:
        ticker = bar.get("T", "")

        # Skip non-stock tickers (ETFs starting with X:, crypto, etc.)
        if not ticker or ":" in ticker or len(ticker) > 5:
            continue
        # Skip common ETFs/indices
        if ticker in ("SPY", "QQQ", "IWM", "DIA", "VXX", "UVXY", "SQQQ", "TQQQ"):
            continue

        close = bar.get("c", 0)
        volume = bar.get("v", 0)
        open_price = bar.get("o", 0)
        vwap = bar.get("vw", 0)

        # Price filter
        if close < min_price or close > max_price:
            continue

        # Volume filter
        if volume < min_volume:
            continue

        # Dollar volume filter
        dollar_vol = close * volume
        if dollar_vol < min_dollar_volume:
            continue

        # Change % filter (intraday)
        change_pct = 0
        if open_price > 0:
            change_pct = ((close - open_price) / open_price) * 100

        if min_change_percent > 0 and abs(change_pct) < min_change_percent:
            continue

        candidates.append({
            "ticker": ticker,
            "open": round(open_price, 2),
            "high": round(bar.get("h", 0), 2),
            "low": round(bar.get("l", 0), 2),
            "close": round(close, 2),
            "volume": int(volume),
            "vwap": round(vwap, 2),
            "dollar_volume": round(dollar_vol, 0),
            "change_pct": round(change_pct, 2),
        })

    # Sort by volume descending for prioritization
    candidates.sort(key=lambda x: x["volume"], reverse=True)

    # Phase 2: Float and market cap filters (requires ticker details API calls)
    needs_details = (
        min_float > 0 or max_float < float("inf") or
        min_market_cap > 0 or max_market_cap < float("inf")
    )

    if not needs_details:
        return candidates[:max_results]

    # Only fetch details for top candidates by volume to limit API calls
    check_limit = min(len(candidates), max_results * 3)
    filtered = []

    for i, cand in enumerate(candidates[:check_limit]):
        if progress_callback:
            progress_callback(i + 1, check_limit, cand["ticker"])

        details = get_cached_ticker_details(client, cand["ticker"])
        if not details:
            continue

        # Float
        float_shares = (
            details.get("weighted_shares_outstanding")
            or details.get("share_class_shares_outstanding")
            or 0
        )
        if float_shares < min_float or float_shares > max_float:
            continue

        # Market cap
        market_cap = details.get("market_cap", 0) or 0
        if market_cap < min_market_cap or market_cap > max_market_cap:
            continue

        cand["float"] = float_shares
        cand["market_cap"] = market_cap
        cand["name"] = details.get("name", "")
        filtered.append(cand)

        if len(filtered) >= max_results:
            break

    return filtered


def scan_tickers_multi_day(
    client: PolygonClient,
    start_date: str,
    end_date: str,
    progress_callback=None,
    **filter_kwargs,
) -> dict[str, list[dict]]:
    """
    Scan across multiple days. Returns {date_str: [candidates]}.
    Useful for finding which tickers met criteria on each day.
    """
    from engine.backtest import get_trading_days

    days = get_trading_days(start_date, end_date)
    results = {}

    for i, date_str in enumerate(days):
        if progress_callback:
            progress_callback(i + 1, len(days), f"Scanning {date_str}...")

        candidates = scan_tickers(client, date_str, **filter_kwargs)
        results[date_str] = candidates

    return results


def format_number(n):
    """Format large numbers: 1500000 -> '1.5M'"""
    if n is None or n == 0:
        return "0"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(int(n))
