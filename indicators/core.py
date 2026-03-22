"""
Technical indicators for backtesting. All operate on Polygon minute bar DataFrames.

Expected columns: o (open), h (high), l (low), c (close), v (volume), vw (vwap), dt (datetime)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def vwap(bars: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP, resets at 09:30 market open."""
    result = pd.Series(np.nan, index=bars.index)
    market_open_mask = bars["dt"].dt.hour * 60 + bars["dt"].dt.minute >= 570  # 09:30

    # Pre-market VWAP (from 04:00)
    pre_mask = ~market_open_mask
    if pre_mask.any():
        pre = bars.loc[pre_mask]
        cum_vol = pre["v"].cumsum()
        cum_dollar_vol = (pre["vw"] * pre["v"]).cumsum()
        result.loc[pre_mask] = cum_dollar_vol / cum_vol.replace(0, np.nan)

    # Regular hours VWAP (resets at 09:30)
    if market_open_mask.any():
        reg = bars.loc[market_open_mask]
        cum_vol = reg["v"].cumsum()
        cum_dollar_vol = (reg["vw"] * reg["v"]).cumsum()
        result.loc[market_open_mask] = cum_dollar_vol / cum_vol.replace(0, np.nan)

    return result


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=1).mean()


def rsi(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Relative Strength Index on close prices."""
    delta = bars["c"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = bars["h"]
    low = bars["l"]
    prev_close = bars["c"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=1).mean()


def volume_ratio(bars: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Current bar volume / average volume over lookback bars."""
    avg_vol = bars["v"].rolling(window=lookback, min_periods=1).mean()
    return bars["v"] / avg_vol.replace(0, np.nan)


def cum_volume(bars: pd.DataFrame) -> pd.Series:
    """Cumulative volume from start of data."""
    return bars["v"].cumsum()


def opening_range(bars: pd.DataFrame, minutes: int = 5) -> tuple[float | None, float | None]:
    """High/low of first N minutes after 09:30."""
    market_open = bars[bars["dt"].dt.hour * 60 + bars["dt"].dt.minute >= 570]
    if len(market_open) == 0:
        return None, None

    open_time = market_open.iloc[0]["dt"]
    from datetime import timedelta
    cutoff = open_time + timedelta(minutes=minutes)

    orb_bars = market_open[market_open["dt"] < cutoff]
    if len(orb_bars) == 0:
        return None, None

    return orb_bars["h"].max(), orb_bars["l"].min()


def premarket_levels(bars: pd.DataFrame) -> tuple[float | None, float | None]:
    """High/low from pre-market (04:00 - 09:29)."""
    pre = bars[(bars["dt"].dt.hour >= 4) &
               (bars["dt"].dt.hour * 60 + bars["dt"].dt.minute < 570)]
    if len(pre) == 0:
        return None, None
    return pre["h"].max(), pre["l"].min()


def gap_percent(bars: pd.DataFrame, prev_close: float | None) -> float | None:
    """Gap percentage: (open - prev_close) / prev_close * 100."""
    if prev_close is None or prev_close == 0:
        return None
    market_open = bars[bars["dt"].dt.hour * 60 + bars["dt"].dt.minute >= 570]
    if len(market_open) == 0:
        return None
    open_price = market_open.iloc[0]["o"]
    return ((open_price - prev_close) / prev_close) * 100


def dist_from_vwap(bars: pd.DataFrame) -> pd.Series:
    """Percentage distance from VWAP: (close - vwap) / vwap * 100."""
    v = vwap(bars)
    return ((bars["c"] - v) / v.replace(0, np.nan)) * 100


def compute_all_indicators(bars: pd.DataFrame, prev_close: float | None = None) -> dict:
    """Compute all indicators for a full day of bars. Returns a dict of Series/values."""
    orb_high, orb_low = opening_range(bars)
    pm_high, pm_low = premarket_levels(bars)

    return {
        "vwap": vwap(bars),
        "ema_9": ema(bars["c"], 9),
        "ema_20": ema(bars["c"], 20),
        "rsi_14": rsi(bars, 14),
        "atr_14": atr(bars, 14),
        "volume_ratio": volume_ratio(bars),
        "cum_volume": cum_volume(bars),
        "dist_from_vwap": dist_from_vwap(bars),
        "orb_high": orb_high,
        "orb_low": orb_low,
        "pm_high": pm_high,
        "pm_low": pm_low,
        "gap_percent": gap_percent(bars, prev_close),
        "prev_close": prev_close,
    }
