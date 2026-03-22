"""
Core backtesting simulation loop.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from data.polygon_client import PolygonClient
from engine.models import BacktestResult
from engine.portfolio import Portfolio
from indicators.core import compute_all_indicators
from strategies.base import Strategy


def get_trading_days(start_date: str, end_date: str) -> list[str]:
    """Generate list of weekday date strings between start and end."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon-Fri
            days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return days


def run_backtest(
    strategy: Strategy,
    tickers: list[str],
    start_date: str,
    end_date: str,
    account_size: float,
    polygon_client: PolygonClient,
    progress_callback=None,
) -> BacktestResult:
    """
    Run a backtest simulation.

    Args:
        strategy: Strategy instance to test
        tickers: List of ticker symbols
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        account_size: Starting capital
        polygon_client: PolygonClient instance
        progress_callback: Optional callable(current, total, message) for progress updates
    """
    portfolio = Portfolio(account_size)
    all_trades = []
    trading_days = get_trading_days(start_date, end_date)

    total_steps = len(trading_days) * len(tickers)
    step = 0

    for date_str in trading_days:
        for ticker in tickers:
            step += 1
            if progress_callback:
                progress_callback(step, total_steps, f"{ticker} on {date_str}")

            bars = polygon_client.get_minute_bars(ticker, date_str)
            if bars is None or len(bars) == 0:
                continue

            # Get prev close for gap calculation
            prev_close = polygon_client.get_prev_close(ticker, date_str)

            # Compute all indicators once for the full day
            indicators = compute_all_indicators(bars, prev_close)

            # Let strategy do pre-market analysis
            premarket_data = strategy.pre_market_scan(ticker, bars, indicators)
            indicators["premarket_data"] = premarket_data

            # Check filters from strategy config
            if not _passes_filters(bars, indicators, strategy.config):
                continue

            # Simulate bar by bar
            position = None
            for i in range(len(bars)):
                bar = bars.iloc[i]
                bar_time = bar["dt"]

                # Build indicator snapshot for this bar
                ind_snapshot = _slice_indicators(indicators, i)

                # Check exit first if in position
                if position is not None:
                    exit_signal = strategy.should_exit(
                        ticker, i, bars.iloc[:i + 1], ind_snapshot, position
                    )
                    if exit_signal:
                        trade = portfolio.close_position(
                            position, bar["c"], bar_time, exit_signal.reason
                        )
                        trade.metadata.update(ind_snapshot.get("_scalar_snapshot", {}))
                        all_trades.append(trade)
                        position = None

                # Check entry if flat
                if position is None:
                    signal = strategy.on_bar(
                        ticker, i, bars.iloc[:i + 1], ind_snapshot, None
                    )
                    if signal and signal.direction in ("long", "short"):
                        shares = portfolio.calculate_position_size(
                            bar["c"], signal, strategy.config
                        )
                        if shares > 0:
                            stop = signal.stop_price or (
                                bar["c"] * 0.98 if signal.direction == "long"
                                else bar["c"] * 1.02
                            )
                            target = signal.target_price

                            # Build entry metadata
                            entry_meta = {
                                "signal_reason": signal.reason,
                                "vwap_at_entry": ind_snapshot.get("vwap_val"),
                                "rsi_at_entry": ind_snapshot.get("rsi_val"),
                                "volume_ratio_at_entry": ind_snapshot.get("volume_ratio_val"),
                                "gap_percent": indicators.get("gap_percent"),
                                "orb_high": indicators.get("orb_high"),
                                "orb_low": indicators.get("orb_low"),
                                "pm_high": indicators.get("pm_high"),
                                "pm_low": indicators.get("pm_low"),
                            }
                            entry_meta.update(signal.metadata)

                            position = portfolio.open_position(
                                ticker, signal.direction, bar["c"], bar_time,
                                shares, stop, target, signal.reason, entry_meta
                            )

                # Update extremes
                if position is not None:
                    portfolio.update_position_extremes(position, bar)

            # Force close at EOD
            if position is not None:
                last_bar = bars.iloc[-1]
                trade = portfolio.close_position(
                    position, last_bar["c"], last_bar["dt"], "eod_close"
                )
                all_trades.append(trade)
                position = None

    return BacktestResult(
        trades=all_trades,
        equity_curve=portfolio.equity_history,
        config=strategy.config,
        start_date=start_date,
        end_date=end_date,
        starting_capital=account_size,
    )


def run_backtest_with_scanner(
    strategy: Strategy,
    start_date: str,
    end_date: str,
    account_size: float,
    polygon_client: PolygonClient,
    scanner_filters: dict,
    progress_callback=None,
) -> BacktestResult:
    """
    Run a backtest using the scanner to find tickers each day.

    Instead of a fixed ticker list, the scanner finds matching tickers
    per trading day based on price, volume, float, market cap criteria.
    """
    from data.scanner import scan_tickers

    portfolio = Portfolio(account_size)
    all_trades = []
    trading_days = get_trading_days(start_date, end_date)
    scanned_tickers = {}  # date -> [tickers] for reporting

    total_days = len(trading_days)
    tickers_processed = 0

    for day_idx, date_str in enumerate(trading_days):
        # Phase 1: Scan for tickers on this day
        if progress_callback:
            progress_callback(day_idx + 1, total_days * 2, f"Scanning {date_str}...")

        candidates = scan_tickers(
            client=polygon_client,
            date_str=date_str,
            min_price=scanner_filters.get("min_price", 0),
            max_price=scanner_filters.get("max_price", float("inf")),
            min_volume=scanner_filters.get("min_volume", 0),
            min_dollar_volume=scanner_filters.get("min_dollar_volume", 0),
            min_float=scanner_filters.get("min_float", 0),
            max_float=scanner_filters.get("max_float", float("inf")),
            min_market_cap=scanner_filters.get("min_market_cap", 0),
            max_market_cap=scanner_filters.get("max_market_cap", float("inf")),
            min_change_percent=scanner_filters.get("min_change_percent", 0),
            max_results=scanner_filters.get("max_results", 50),
        )

        day_tickers = [c["ticker"] for c in candidates]
        scanned_tickers[date_str] = day_tickers

        if not day_tickers:
            continue

        # Phase 2: Run backtest on scanned tickers for this day
        for ticker_idx, ticker in enumerate(day_tickers):
            tickers_processed += 1
            if progress_callback:
                progress_callback(
                    total_days + day_idx + 1, total_days * 2,
                    f"{ticker} on {date_str} ({ticker_idx+1}/{len(day_tickers)})"
                )

            bars = polygon_client.get_minute_bars(ticker, date_str)
            if bars is None or len(bars) == 0:
                continue

            prev_close = polygon_client.get_prev_close(ticker, date_str)
            indicators = compute_all_indicators(bars, prev_close)
            premarket_data = strategy.pre_market_scan(ticker, bars, indicators)
            indicators["premarket_data"] = premarket_data

            if not _passes_filters(bars, indicators, strategy.config):
                continue

            position = None
            for i in range(len(bars)):
                bar = bars.iloc[i]
                bar_time = bar["dt"]
                ind_snapshot = _slice_indicators(indicators, i)

                if position is not None:
                    exit_signal = strategy.should_exit(
                        ticker, i, bars.iloc[:i + 1], ind_snapshot, position
                    )
                    if exit_signal:
                        trade = portfolio.close_position(
                            position, bar["c"], bar_time, exit_signal.reason
                        )
                        trade.metadata.update(ind_snapshot.get("_scalar_snapshot", {}))
                        all_trades.append(trade)
                        position = None

                if position is None:
                    signal = strategy.on_bar(
                        ticker, i, bars.iloc[:i + 1], ind_snapshot, None
                    )
                    if signal and signal.direction in ("long", "short"):
                        shares = portfolio.calculate_position_size(
                            bar["c"], signal, strategy.config
                        )
                        if shares > 0:
                            stop = signal.stop_price or (
                                bar["c"] * 0.98 if signal.direction == "long"
                                else bar["c"] * 1.02
                            )
                            target = signal.target_price
                            entry_meta = {
                                "signal_reason": signal.reason,
                                "vwap_at_entry": ind_snapshot.get("vwap_val"),
                                "rsi_at_entry": ind_snapshot.get("rsi_val"),
                                "volume_ratio_at_entry": ind_snapshot.get("volume_ratio_val"),
                                "gap_percent": indicators.get("gap_percent"),
                                "orb_high": indicators.get("orb_high"),
                                "orb_low": indicators.get("orb_low"),
                                "pm_high": indicators.get("pm_high"),
                                "pm_low": indicators.get("pm_low"),
                            }
                            entry_meta.update(signal.metadata)
                            position = portfolio.open_position(
                                ticker, signal.direction, bar["c"], bar_time,
                                shares, stop, target, signal.reason, entry_meta
                            )

                if position is not None:
                    portfolio.update_position_extremes(position, bar)

            if position is not None:
                last_bar = bars.iloc[-1]
                trade = portfolio.close_position(
                    position, last_bar["c"], last_bar["dt"], "eod_close"
                )
                all_trades.append(trade)
                position = None

    # Include scanned ticker info in the result config
    result_config = {**strategy.config, "_scanned_tickers": scanned_tickers}

    return BacktestResult(
        trades=all_trades,
        equity_curve=portfolio.equity_history,
        config=result_config,
        start_date=start_date,
        end_date=end_date,
        starting_capital=account_size,
    )


def _slice_indicators(indicators: dict, bar_idx: int) -> dict:
    """Extract indicator values at a specific bar index."""
    snapshot = {}
    scalar_snapshot = {}

    for key, value in indicators.items():
        if isinstance(value, pd.Series):
            snapshot[key] = value.iloc[:bar_idx + 1]
            # Also store the current scalar value for metadata
            val = value.iloc[bar_idx]
            scalar_snapshot[f"{key}_val"] = round(float(val), 4) if pd.notna(val) else None
            snapshot[f"{key}_val"] = scalar_snapshot[f"{key}_val"]
        else:
            snapshot[key] = value

    snapshot["_scalar_snapshot"] = scalar_snapshot
    return snapshot


def _passes_filters(bars: pd.DataFrame, indicators: dict, config: dict) -> bool:
    """Check if a ticker/day passes the strategy's filters."""
    filters = config.get("filters", {})
    if not filters:
        return True

    # Get market open price
    market_bars = bars[bars["dt"].dt.hour * 60 + bars["dt"].dt.minute >= 570]
    if len(market_bars) == 0:
        return False

    open_price = market_bars.iloc[0]["o"]

    if "min_price" in filters and open_price < filters["min_price"]:
        return False
    if "max_price" in filters and open_price > filters["max_price"]:
        return False

    # Volume filter on pre-market
    if "min_volume_premarket" in filters:
        pre = bars[bars["dt"].dt.hour * 60 + bars["dt"].dt.minute < 570]
        if len(pre) > 0:
            pm_vol = pre["v"].sum()
            if pm_vol < filters["min_volume_premarket"]:
                return False

    # Gap filter
    if "min_gap_percent" in filters:
        gap = indicators.get("gap_percent")
        if gap is None or abs(gap) < filters["min_gap_percent"]:
            return False

    return True
