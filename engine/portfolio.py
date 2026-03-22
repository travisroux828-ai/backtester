"""
Portfolio and position management for backtesting.
"""

from engine.models import Position, Trade, Signal


class Portfolio:
    """Tracks positions, calculates sizing, and records P&L."""

    def __init__(self, starting_capital: float = 25000.0):
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.equity_history = [starting_capital]
        self.open_positions = {}  # ticker -> Position

    def calculate_position_size(self, price: float, signal: Signal, config: dict) -> int:
        """Calculate shares based on sizing config."""
        sizing = config.get("position_sizing", {})
        sizing_type = sizing.get("type", "fixed_shares")

        if sizing_type == "fixed_shares":
            shares = sizing.get("shares", 100)

        elif sizing_type == "fixed_dollar":
            dollar_amount = sizing.get("amount", 1000)
            shares = int(dollar_amount / price) if price > 0 else 0

        elif sizing_type == "risk_percent":
            risk_pct = sizing.get("risk_percent", 1.0) / 100
            stop = signal.stop_price
            if stop is None:
                stop = price * (0.98 if signal.direction == "long" else 1.02)
            risk_per_share = abs(price - stop)
            if risk_per_share <= 0:
                return 0
            risk_amount = self.cash * risk_pct
            shares = int(risk_amount / risk_per_share)
        else:
            shares = 100

        max_shares = sizing.get("max_shares", 10000)
        shares = min(shares, max_shares)

        # Don't exceed available cash
        max_affordable = int(self.cash / price) if price > 0 else 0
        shares = min(shares, max_affordable)

        return max(0, shares)

    def open_position(self, ticker: str, direction: str, price: float,
                      time, shares: int, stop_price: float,
                      target_price: float | None, signal_reason: str,
                      metadata: dict | None = None) -> Position:
        """Open a new position."""
        pos = Position(
            ticker=ticker,
            direction=direction,
            entry_time=time,
            entry_price=price,
            shares=shares,
            stop_price=stop_price,
            target_price=target_price,
            max_favorable=price,
            max_adverse=price,
        )
        pos._signal_reason = signal_reason
        pos._metadata = metadata or {}
        self.open_positions[ticker] = pos
        self.cash -= price * shares
        return pos

    def close_position(self, position: Position, price: float, time,
                       exit_reason: str) -> Trade:
        """Close a position and return the completed Trade."""
        if position.direction == "long":
            pnl = (price - position.entry_price) * position.shares
        else:
            pnl = (position.entry_price - price) * position.shares

        trade = Trade(
            ticker=position.ticker,
            direction=position.direction,
            entry_time=position.entry_time,
            exit_time=time,
            entry_price=position.entry_price,
            exit_price=price,
            shares=position.shares,
            gross_pnl=round(pnl, 2),
            signal_reason=getattr(position, "_signal_reason", ""),
            exit_reason=exit_reason,
            stop_price=position.stop_price,
            target_price=position.target_price,
            metadata=getattr(position, "_metadata", {}),
        )

        self.cash += price * position.shares + pnl
        self.equity_history.append(self.cash)

        if position.ticker in self.open_positions:
            del self.open_positions[position.ticker]

        return trade

    def update_position_extremes(self, position: Position, bar):
        """Update max favorable/adverse excursion."""
        if position.direction == "long":
            position.max_favorable = max(position.max_favorable, bar["h"])
            position.max_adverse = min(position.max_adverse, bar["l"])
        else:
            position.max_favorable = min(position.max_favorable, bar["l"])
            position.max_adverse = max(position.max_adverse, bar["h"])
