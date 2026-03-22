"""
ORB Breakout Strategy - Opening Range Breakout with volume confirmation.

Logic:
1. After the first 5 minutes (09:35), compute the opening range high/low.
2. Enter long when price breaks above ORB high with volume > 1.5x avg.
3. Enter short when price breaks below ORB low with volume > 1.5x avg.
4. Stop at opposite end of opening range.
5. Target at 2:1 risk/reward.
"""

from __future__ import annotations

import pandas as pd

from engine.models import Signal, Position
from strategies.base import Strategy


class ORBBreakout(Strategy):
    name = "ORB Breakout"
    description = "Opening range breakout with volume confirmation. Enters after 5-min ORB forms."

    DEFAULT_CONFIG = {
        "orb_minutes": 5,
        "entry_start": "09:35",
        "entry_end": "11:30",
        "volume_threshold": 1.5,
        "rsi_max_long": 75,
        "rsi_min_short": 25,
        "risk_reward": 2.0,
        "direction": "both",  # "long", "short", or "both"
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

        bar = bars.iloc[bar_idx]
        bar_time = bar["dt"]
        time_minutes = bar_time.hour * 60 + bar_time.minute

        # Parse entry window
        start_h, start_m = map(int, self.config["entry_start"].split(":"))
        end_h, end_m = map(int, self.config["entry_end"].split(":"))
        entry_start = start_h * 60 + start_m
        entry_end = end_h * 60 + end_m

        if time_minutes < entry_start or time_minutes > entry_end:
            return None

        orb_high = indicators.get("orb_high")
        orb_low = indicators.get("orb_low")
        if orb_high is None or orb_low is None:
            return None

        price = bar["c"]
        vol_ratio = indicators.get("volume_ratio_val")
        rsi_val = indicators.get("rsi_14_val")
        vol_threshold = self.config["volume_threshold"]
        rr = self.config["risk_reward"]
        direction = self.config["direction"]

        # Long breakout
        if direction in ("long", "both"):
            if price > orb_high:
                if vol_ratio is not None and vol_ratio >= vol_threshold:
                    if rsi_val is None or rsi_val < self.config["rsi_max_long"]:
                        risk = orb_high - orb_low
                        if risk > 0:
                            stop = orb_low
                            target = price + (risk * rr)
                            reason = f"ORB breakout above {orb_high:.2f}, vol ratio {vol_ratio:.1f}x"
                            if rsi_val is not None:
                                reason += f", RSI {rsi_val:.0f}"
                            return Signal(
                                timestamp=bar_time,
                                direction="long",
                                reason=reason,
                                stop_price=stop,
                                target_price=target,
                                metadata={"orb_high": orb_high, "orb_low": orb_low},
                            )

        # Short breakdown
        if direction in ("short", "both"):
            if price < orb_low:
                if vol_ratio is not None and vol_ratio >= vol_threshold:
                    if rsi_val is None or rsi_val > self.config["rsi_min_short"]:
                        risk = orb_high - orb_low
                        if risk > 0:
                            stop = orb_high
                            target = price - (risk * rr)
                            reason = f"ORB breakdown below {orb_low:.2f}, vol ratio {vol_ratio:.1f}x"
                            if rsi_val is not None:
                                reason += f", RSI {rsi_val:.0f}"
                            return Signal(
                                timestamp=bar_time,
                                direction="short",
                                reason=reason,
                                stop_price=stop,
                                target_price=target,
                                metadata={"orb_high": orb_high, "orb_low": orb_low},
                            )

        return None
