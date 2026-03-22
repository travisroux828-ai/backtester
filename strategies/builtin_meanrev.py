"""
VWAP Fade Strategy - Mean reversion when price overextends from VWAP.

Logic:
1. Wait for price to extend > 2 ATR above/below VWAP.
2. Look for a reversal candle (red candle for shorts, green for longs).
3. Enter on the reversal candle close.
4. Stop above/below the extension high/low + 0.5 ATR.
5. Target VWAP.
"""

import pandas as pd

from engine.models import Signal, Position
from strategies.base import Strategy


class VWAPFade(Strategy):
    name = "VWAP Fade"
    description = "Mean reversion fade when price overextends from VWAP. Targets VWAP as profit target."

    DEFAULT_CONFIG = {
        "entry_start": "09:45",
        "entry_end": "15:00",
        "atr_extension": 2.0,       # How many ATRs from VWAP to trigger
        "stop_atr_buffer": 0.5,     # Extra ATR added to stop beyond extremes
        "direction": "both",
        "min_rsi_for_short": 65,    # RSI must be above this to short
        "max_rsi_for_long": 35,     # RSI must be below this to go long
        "position_sizing": {
            "type": "fixed_shares",
            "shares": 100,
            "max_shares": 5000,
        },
    }

    def __init__(self, config: dict | None = None):
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)

    def on_bar(self, ticker: str, bar_idx: int, bars: pd.DataFrame,
               indicators: dict, position: Position | None) -> Signal | None:
        if position is not None:
            return None

        if bar_idx < 14:  # Need enough bars for ATR
            return None

        bar = bars.iloc[bar_idx]
        bar_time = bar["dt"]
        time_minutes = bar_time.hour * 60 + bar_time.minute

        start_h, start_m = map(int, self.config["entry_start"].split(":"))
        end_h, end_m = map(int, self.config["entry_end"].split(":"))
        if time_minutes < start_h * 60 + start_m or time_minutes > end_h * 60 + end_m:
            return None

        vwap_val = indicators.get("vwap_val")
        atr_val = indicators.get("atr_14_val")
        rsi_val = indicators.get("rsi_14_val")

        if vwap_val is None or atr_val is None or atr_val == 0:
            return None

        price = bar["c"]
        extension = self.config["atr_extension"]
        stop_buffer = self.config["stop_atr_buffer"]
        direction = self.config["direction"]

        dist_from_vwap = price - vwap_val
        atr_distance = abs(dist_from_vwap) / atr_val

        # Short fade: price extended above VWAP + red candle
        if direction in ("short", "both"):
            if atr_distance >= extension and dist_from_vwap > 0:
                is_red = bar["c"] < bar["o"]
                rsi_ok = rsi_val is None or rsi_val >= self.config["min_rsi_for_short"]
                if is_red and rsi_ok:
                    stop = bar["h"] + (atr_val * stop_buffer)
                    target = vwap_val
                    reason = (f"VWAP fade short: price {atr_distance:.1f} ATR above VWAP "
                              f"({vwap_val:.2f}), red candle")
                    if rsi_val is not None:
                        reason += f", RSI {rsi_val:.0f}"
                    return Signal(
                        timestamp=bar_time,
                        direction="short",
                        reason=reason,
                        stop_price=stop,
                        target_price=target,
                        metadata={"vwap": vwap_val, "atr": atr_val, "extension_atr": atr_distance},
                    )

        # Long fade: price extended below VWAP + green candle
        if direction in ("long", "both"):
            if atr_distance >= extension and dist_from_vwap < 0:
                is_green = bar["c"] > bar["o"]
                rsi_ok = rsi_val is None or rsi_val <= self.config["max_rsi_for_long"]
                if is_green and rsi_ok:
                    stop = bar["l"] - (atr_val * stop_buffer)
                    target = vwap_val
                    reason = (f"VWAP fade long: price {atr_distance:.1f} ATR below VWAP "
                              f"({vwap_val:.2f}), green candle")
                    if rsi_val is not None:
                        reason += f", RSI {rsi_val:.0f}"
                    return Signal(
                        timestamp=bar_time,
                        direction="long",
                        reason=reason,
                        stop_price=stop,
                        target_price=target,
                        metadata={"vwap": vwap_val, "atr": atr_val, "extension_atr": atr_distance},
                    )

        return None
