"""
Config-based strategy that interprets YAML strategy definitions at runtime.
"""

from __future__ import annotations

import pandas as pd

from engine.models import Signal, Position
from strategies.base import Strategy


class ConfigStrategy(Strategy):
    """Strategy defined by a YAML config file."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.name = config.get("name", "Custom Strategy")
        self.description = config.get("description", "")

    def on_bar(self, ticker: str, bar_idx: int, bars: pd.DataFrame,
               indicators: dict, position: Position | None) -> Signal | None:
        if position is not None:
            return None

        bar = bars.iloc[bar_idx]
        bar_time = bar["dt"]
        time_minutes = bar_time.hour * 60 + bar_time.minute

        # Check entry window
        entry_window = self.config.get("entry_window", {})
        if entry_window:
            start = entry_window.get("start", "09:30")
            end = entry_window.get("end", "15:30")
            start_h, start_m = map(int, start.split(":"))
            end_h, end_m = map(int, end.split(":"))
            if time_minutes < start_h * 60 + start_m or time_minutes > end_h * 60 + end_m:
                return None

        # Check all entry conditions
        conditions = self.config.get("entry_conditions", [])
        if not conditions:
            return None

        all_met = True
        reasons = []

        for cond in conditions:
            met, reason = self._check_condition(cond, bar, indicators)
            if not met:
                all_met = False
                break
            reasons.append(reason)

        if not all_met:
            return None

        direction = self.config.get("direction", "long")
        price = bar["c"]

        # Calculate stop and target
        stop_price = self._calc_stop(price, direction, bar, indicators)
        target_price = self._calc_target(price, direction, stop_price, indicators)

        return Signal(
            timestamp=bar_time,
            direction=direction,
            reason=" + ".join(reasons),
            stop_price=stop_price,
            target_price=target_price,
            metadata={},
        )

    def _check_condition(self, cond: dict, bar, indicators: dict) -> tuple[bool, str]:
        """Evaluate a single entry condition. Returns (met, reason)."""
        indicator_name = cond.get("indicator", "")
        operator = cond.get("operator", ">")

        # Get the left-hand value
        lhs = self._get_indicator_value(indicator_name, bar, indicators)
        if lhs is None:
            return False, ""

        # Get the right-hand value
        if "value" in cond:
            rhs = cond["value"]
        elif "reference" in cond:
            rhs = self._get_indicator_value(cond["reference"], bar, indicators)
            if rhs is None:
                return False, ""
        else:
            return False, ""

        # Compare
        result = False
        if operator == ">":
            result = lhs > rhs
        elif operator == ">=":
            result = lhs >= rhs
        elif operator == "<":
            result = lhs < rhs
        elif operator == "<=":
            result = lhs <= rhs
        elif operator == "==":
            result = lhs == rhs

        reason = f"{indicator_name} {operator} {rhs}" if result else ""
        return result, reason

    def _get_indicator_value(self, name: str, bar, indicators: dict):
        """Resolve an indicator name to a value."""
        # Direct scalar indicators
        val_key = f"{name}_val"
        if val_key in indicators:
            return indicators[val_key]

        # Static values from indicators dict
        if name in indicators:
            val = indicators[name]
            if isinstance(val, (int, float)):
                return val

        # Bar fields
        bar_map = {
            "price": "c", "close": "c", "open": "o",
            "high": "h", "low": "l", "volume": "v",
        }
        if name in bar_map:
            return bar[bar_map[name]]

        return None

    def _calc_stop(self, price: float, direction: str, bar, indicators: dict) -> float:
        """Calculate stop price from config."""
        stop_cfg = self.config.get("stop", {})
        stop_type = stop_cfg.get("type", "percent")

        if stop_type == "percent":
            pct = stop_cfg.get("value", 2.0) / 100
            if direction == "long":
                return price * (1 - pct)
            return price * (1 + pct)

        elif stop_type == "atr_multiple":
            atr_val = indicators.get("atr_14_val")
            mult = stop_cfg.get("value", 1.5)
            if atr_val:
                if direction == "long":
                    return price - (atr_val * mult)
                return price + (atr_val * mult)

        elif stop_type == "fixed_level":
            ref = stop_cfg.get("reference", "")
            level = self._get_indicator_value(ref, bar, indicators)
            if level is not None:
                return level

        # Fallback: 2% stop
        if direction == "long":
            return price * 0.98
        return price * 1.02

    def _calc_target(self, price: float, direction: str, stop_price: float,
                     indicators: dict) -> float | None:
        """Calculate target price from config."""
        target_cfg = self.config.get("target", {})
        target_type = target_cfg.get("type", "risk_multiple")

        if target_type == "risk_multiple":
            mult = target_cfg.get("value", 2.0)
            risk = abs(price - stop_price)
            if direction == "long":
                return price + (risk * mult)
            return price - (risk * mult)

        elif target_type == "fixed_level":
            ref = target_cfg.get("reference", "")
            val = indicators.get(ref) or indicators.get(f"{ref}_val")
            if val is not None:
                return val

        elif target_type == "percent":
            pct = target_cfg.get("value", 2.0) / 100
            if direction == "long":
                return price * (1 + pct)
            return price * (1 - pct)

        return None
