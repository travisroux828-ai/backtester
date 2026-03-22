"""
Data models for the backtesting engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Signal:
    timestamp: datetime
    direction: str          # "long", "short", or "exit"
    reason: str             # Human-readable: "VWAP reclaim with volume > 2x avg"
    strength: float = 1.0   # 0.0 to 1.0
    metadata: dict = field(default_factory=dict)
    stop_price: float | None = None
    target_price: float | None = None


@dataclass
class Position:
    ticker: str
    direction: str          # "long" or "short"
    entry_time: datetime
    entry_price: float
    shares: int
    stop_price: float
    target_price: float | None = None
    max_favorable: float = 0.0
    max_adverse: float = 0.0


@dataclass
class Trade:
    ticker: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    shares: int
    gross_pnl: float
    signal_reason: str
    exit_reason: str
    stop_price: float = 0.0
    target_price: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def net_pnl(self):
        return self.gross_pnl

    @property
    def is_winner(self):
        return self.gross_pnl > 0


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: list[float]
    config: dict
    start_date: str
    end_date: str
    starting_capital: float = 25000.0

    @property
    def total_pnl(self):
        return sum(t.gross_pnl for t in self.trades)

    @property
    def win_rate(self):
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.is_winner) / len(self.trades) * 100

    @property
    def profit_factor(self):
        gross_wins = sum(t.gross_pnl for t in self.trades if t.gross_pnl > 0)
        gross_losses = abs(sum(t.gross_pnl for t in self.trades if t.gross_pnl < 0))
        if gross_losses == 0:
            return float("inf") if gross_wins > 0 else 0.0
        return gross_wins / gross_losses

    @property
    def max_drawdown(self):
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def avg_winner(self):
        winners = [t.gross_pnl for t in self.trades if t.gross_pnl > 0]
        return sum(winners) / len(winners) if winners else 0.0

    @property
    def avg_loser(self):
        losers = [t.gross_pnl for t in self.trades if t.gross_pnl < 0]
        return sum(losers) / len(losers) if losers else 0.0
