"""
Abstract base class for all backtest strategies.
"""

from abc import ABC, abstractmethod

import pandas as pd

from engine.models import Signal, Position


class Strategy(ABC):
    """Base class for all backtest strategies."""

    name: str = "Unnamed Strategy"
    description: str = ""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @abstractmethod
    def on_bar(self, ticker: str, bar_idx: int, bars: pd.DataFrame,
               indicators: dict, position: Position | None) -> Signal | None:
        """
        Called for every minute bar.

        Args:
            ticker: Current symbol
            bar_idx: Index into bars DataFrame (current bar)
            bars: Full day's minute bars up to and including current bar
            indicators: Pre-computed indicator values dict at this bar
            position: Current open position for this ticker, or None

        Returns:
            Signal to enter/exit, or None to do nothing
        """
        pass

    def should_exit(self, ticker: str, bar_idx: int, bars: pd.DataFrame,
                    indicators: dict, position: Position) -> Signal | None:
        """
        Called every bar while in a position. Default checks stop loss and target.
        Override for custom exit logic.
        """
        current_price = bars.iloc[bar_idx]["c"]
        current_time = bars.iloc[bar_idx]["dt"]

        # Stop loss
        if position.direction == "long" and current_price <= position.stop_price:
            return Signal(current_time, "exit", "stop_loss", 1.0, {
                "price": current_price, "stop": position.stop_price
            })
        if position.direction == "short" and current_price >= position.stop_price:
            return Signal(current_time, "exit", "stop_loss", 1.0, {
                "price": current_price, "stop": position.stop_price
            })

        # Target
        if position.target_price:
            if position.direction == "long" and current_price >= position.target_price:
                return Signal(current_time, "exit", "target_hit", 1.0, {
                    "price": current_price, "target": position.target_price
                })
            if position.direction == "short" and current_price <= position.target_price:
                return Signal(current_time, "exit", "target_hit", 1.0, {
                    "price": current_price, "target": position.target_price
                })

        return None

    def pre_market_scan(self, ticker: str, bars: pd.DataFrame, indicators: dict) -> dict:
        """
        Called once before market open. Return dict of levels/data for the strategy.
        """
        return {}
