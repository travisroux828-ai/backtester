"""
Polygon.io API client with disk caching for trade enrichment and backtesting.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

import pandas as pd

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


class PolygonClient:
    """Lightweight Polygon.io client with disk caching."""

    BASE = "https://api.polygon.io"

    def __init__(self, api_key):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.params = {"apiKey": api_key}
        self._agg_cache = {}
        self._ref_cache = {}
        self._prev_cache = {}
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _get(self, url, params=None):
        """GET with basic rate-limit retry."""
        for attempt in range(3):
            r = self.session.get(f"{self.BASE}{url}", params=params or {})
            if r.status_code == 429:
                wait = 12 if attempt == 0 else 30
                print(f"  [Polygon] Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code == 200:
                return r.json()
            else:
                print(f"  [Polygon] Warning: {r.status_code} for {url}")
                return None
        return None

    def _disk_cache_path(self, ticker, date_str):
        return os.path.join(CACHE_DIR, f"{ticker}_{date_str}.json")

    def get_minute_bars(self, ticker, date_str):
        """Get minute-level aggregates for full day (extended hours included)."""
        key = (ticker, date_str)
        if key in self._agg_cache:
            return self._agg_cache[key]

        # Check disk cache
        cache_path = self._disk_cache_path(ticker, date_str)
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                results = json.load(f)
            bars = pd.DataFrame(results)
            bars["dt"] = pd.to_datetime(bars["t"], unit="ms", utc=True)
            bars["dt"] = bars["dt"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
            bars["time_str"] = bars["dt"].dt.strftime("%H:%M")
            self._agg_cache[key] = bars
            return bars

        # Fetch from API
        data = self._get(
            f"/v2/aggs/ticker/{ticker}/range/1/minute/{date_str}/{date_str}",
            params={"adjusted": "true", "sort": "asc", "limit": "50000"},
        )
        if not data or "results" not in data:
            self._agg_cache[key] = None
            return None

        # Save to disk cache
        with open(cache_path, "w") as f:
            json.dump(data["results"], f)

        bars = pd.DataFrame(data["results"])
        bars["dt"] = pd.to_datetime(bars["t"], unit="ms", utc=True)
        bars["dt"] = bars["dt"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
        bars["time_str"] = bars["dt"].dt.strftime("%H:%M")

        self._agg_cache[key] = bars
        time.sleep(0.15)
        return bars

    def get_ticker_details(self, ticker):
        """Get float / shares outstanding from reference endpoint."""
        if ticker in self._ref_cache:
            return self._ref_cache[ticker]

        data = self._get(f"/v3/reference/tickers/{ticker}")
        if data and "results" in data:
            self._ref_cache[ticker] = data["results"]
            return data["results"]
        self._ref_cache[ticker] = None
        return None

    def get_prev_close(self, ticker, date_str):
        """Get previous day close."""
        key = (ticker, date_str)
        if key in self._prev_cache:
            return self._prev_cache[key]

        data = self._get(f"/v2/aggs/ticker/{ticker}/prev")
        if data and "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            self._prev_cache[key] = result.get("c")
            return result.get("c")
        self._prev_cache[key] = None
        return None

    def enrich_trade(self, trade_row, date_str):
        """Add Polygon data columns to a trade dict. Modifies in place."""
        ticker = trade_row["Symbol"]
        entry_time_str = trade_row["Entry Time"]

        try:
            entry_dt = datetime.strptime(f"{date_str} {entry_time_str}", "%Y-%m-%d %H:%M:%S")
        except Exception:
            try:
                entry_dt = datetime.strptime(f"{date_str} {entry_time_str}", "%Y-%m-%d %H:%M")
            except Exception:
                entry_dt = None

        bars = self.get_minute_bars(ticker, date_str)

        if bars is not None and len(bars) > 0 and entry_dt is not None:
            bars_before = bars[bars["dt"] <= entry_dt]
            if len(bars_before) > 0:
                cum_vol = bars_before["v"].sum()
                cum_vwap_dollar = (bars_before["vw"] * bars_before["v"]).sum()
                vwap_at_entry = cum_vwap_dollar / cum_vol if cum_vol > 0 else None

                trade_row["Cum Vol at Entry"] = self._format_vol(cum_vol)

                if vwap_at_entry is not None:
                    entry_px = trade_row["Avg Entry Price"]
                    dist = ((entry_px - vwap_at_entry) / vwap_at_entry) * 100
                    trade_row["VWAP Status"] = "Above" if entry_px > vwap_at_entry else "Below"
                    trade_row["Dist From VWAP %"] = f"{dist:.2f}%"

            total_vol = bars["v"].sum()
            trade_row["Total Day Vol"] = self._format_vol(total_vol)

            if entry_dt is not None and len(bars_before) > 0:
                minutes_elapsed = max(1, len(bars_before))
                total_minutes = len(bars)
                if total_minutes > 0:
                    expected_vol_fraction = minutes_elapsed / total_minutes
                    expected_vol = total_vol * expected_vol_fraction
                    if expected_vol > 0:
                        trade_row["Relative Vol"] = round(cum_vol / expected_vol, 2)

            pre = bars[(bars["dt"].dt.hour >= 4) & (bars["dt"].dt.hour * 60 + bars["dt"].dt.minute < 570)]
            reg = bars[(bars["dt"].dt.hour * 60 + bars["dt"].dt.minute >= 570) &
                       (bars["dt"].dt.hour * 60 + bars["dt"].dt.minute < 960)]
            post = bars[bars["dt"].dt.hour >= 16]

            idx_h = bars["h"].idxmax()
            idx_l = bars["l"].idxmin()
            trade_row["Total Day High"] = round(bars.loc[idx_h, "h"], 2)
            trade_row["Total Day High Time"] = bars.loc[idx_h, "time_str"]
            trade_row["Total Day Low"] = round(bars.loc[idx_l, "l"], 2)
            trade_row["Total Day Low Time"] = bars.loc[idx_l, "time_str"]

            for session, label in [(pre, "Pre-Market"), (reg, "Regular Market"), (post, "Post-Market")]:
                if len(session) > 0:
                    ih = session["h"].idxmax()
                    trade_row[f"{label} High"] = round(session.loc[ih, "h"], 2)
                    trade_row[f"{label} High Time"] = session.loc[ih, "time_str"]

        details = self.get_ticker_details(ticker)
        if details:
            ws = details.get("weighted_shares_outstanding") or details.get("share_class_shares_outstanding")
            if ws:
                trade_row["Float"] = self._format_vol(ws)

        prev = self.get_prev_close(ticker, date_str)
        if prev is not None:
            trade_row["Prev Close"] = round(prev, 2)

        time.sleep(0.15)

    @staticmethod
    def _format_vol(v):
        """Format volume: 1234567 -> '1.23M', 12345 -> '12.35K'"""
        if v is None:
            return ""
        if v >= 1_000_000_000:
            return f"{v / 1_000_000_000:.2f}B"
        elif v >= 1_000_000:
            return f"{v / 1_000_000:.2f}M"
        elif v >= 1_000:
            return f"{v / 1_000:.2f}K"
        return str(int(v))
