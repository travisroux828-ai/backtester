"""
Backtester Dashboard - Streamlit web interface for day trading backtesting.

Run with: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import yaml
from datetime import datetime, timedelta

from data.polygon_client import PolygonClient
from data.scanner import scan_tickers, format_number
from engine.backtest import run_backtest, run_backtest_with_scanner
from engine.models import BacktestResult
from strategies.loader import discover_strategies, load_strategy
from export.csv_export import result_to_dataframe, export_to_csv
from ai.strategy_builder import generate_config
from ai.market_stats import parse_stats_query, execute_stats_query, render_stats_results

# Page config
st.set_page_config(page_title="Backtester", layout="wide")
st.title("Day Trading Backtester")

# ─── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")

# API Key - reads from Streamlit secrets, with fallback to manual entry
try:
    default_key = st.secrets["POLYGON_API_KEY"]
except (KeyError, FileNotFoundError):
    default_key = ""
api_key = st.sidebar.text_input(
    "Polygon API Key",
    value=default_key,
    type="password",
)

# Anthropic API Key - for AI Strategy Builder
try:
    default_anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
except (KeyError, FileNotFoundError):
    default_anthropic_key = ""
anthropic_api_key = st.sidebar.text_input(
    "Anthropic API Key",
    value=default_anthropic_key,
    type="password",
)

# Discover strategies
strategies = discover_strategies()
strategy_names = list(strategies.keys())

if not strategy_names:
    st.sidebar.error("No strategies found!")
    st.stop()

selected_strategy = st.sidebar.selectbox("Strategy", strategy_names)
strat_info = strategies[selected_strategy]
st.sidebar.caption(strat_info["description"])

# ─── Ticker Selection Mode ────────────────────────────────────────────────
ticker_mode = st.sidebar.radio(
    "Ticker Selection",
    ["Manual", "Scanner"],
    horizontal=True,
    help="Manual: type specific tickers. Scanner: auto-find tickers matching your criteria.",
)

if ticker_mode == "Manual":
    tickers_input = st.sidebar.text_input(
        "Tickers (comma-separated)",
        value="AAPL",
    )
    scanner_filters = None
else:
    scanner_filters = {}
    with st.sidebar.expander("Scanner Filters", expanded=True):
        st.caption("Set criteria to auto-find tickers each day")

        # Price range
        pc1, pc2 = st.columns(2)
        scanner_filters["min_price"] = pc1.number_input("Min Price ($)", value=1.0, min_value=0.0, step=0.5)
        scanner_filters["max_price"] = pc2.number_input("Max Price ($)", value=20.0, min_value=0.0, step=1.0)

        # Volume
        scanner_filters["min_volume"] = st.number_input(
            "Min Daily Volume", value=500_000, min_value=0, step=100_000,
            help="Minimum total shares traded on the day",
        )

        # Dollar volume
        scanner_filters["min_dollar_volume"] = st.number_input(
            "Min Dollar Volume ($)", value=0, min_value=0, step=1_000_000,
            help="Minimum price * volume (filters out cheap illiquid stocks)",
        )

        # Float
        fc1, fc2 = st.columns(2)
        float_min_options = {"No min": 0, "1M": 1_000_000, "5M": 5_000_000, "10M": 10_000_000, "50M": 50_000_000}
        float_max_options = {"No max": 0, "5M": 5_000_000, "10M": 10_000_000, "20M": 20_000_000,
                             "50M": 50_000_000, "100M": 100_000_000, "500M": 500_000_000}

        float_min_label = fc1.selectbox("Min Float", list(float_min_options.keys()), index=0)
        float_max_label = fc2.selectbox("Max Float", list(float_max_options.keys()), index=0)
        scanner_filters["min_float"] = float_min_options[float_min_label]
        scanner_filters["max_float"] = float_max_options[float_max_label] or float("inf")

        # Market cap
        mc1, mc2 = st.columns(2)
        mcap_min_options = {"No min": 0, "1M": 1_000_000, "10M": 10_000_000, "50M": 50_000_000,
                            "100M": 100_000_000, "500M": 500_000_000, "1B": 1_000_000_000}
        mcap_max_options = {"No max": 0, "50M": 50_000_000, "100M": 100_000_000, "300M": 300_000_000,
                            "500M": 500_000_000, "1B": 1_000_000_000, "5B": 5_000_000_000, "10B": 10_000_000_000}

        mcap_min_label = mc1.selectbox("Min Market Cap", list(mcap_min_options.keys()), index=0)
        mcap_max_label = mc2.selectbox("Max Market Cap", list(mcap_max_options.keys()), index=0)
        scanner_filters["min_market_cap"] = mcap_min_options[mcap_min_label]
        scanner_filters["max_market_cap"] = mcap_max_options[mcap_max_label] or float("inf")

        # Change %
        scanner_filters["min_change_percent"] = st.number_input(
            "Min Intraday Change %", value=0.0, min_value=0.0, step=1.0,
            help="Minimum absolute intraday change (filters for movers)",
        )

        # Max results per day
        scanner_filters["max_results"] = st.number_input(
            "Max Tickers Per Day", value=20, min_value=1, max_value=100, step=5,
            help="Limit how many tickers to scan per day (top by volume)",
        )

    tickers_input = None

# Date range
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)
default_end = datetime.now().date()
default_start = default_end - timedelta(days=7)
start_date = col1.date_input("Start Date", value=default_start)
end_date = col2.date_input("End Date", value=default_end)

# Account size
account_size = st.sidebar.number_input(
    "Account Size ($)", value=25000, min_value=1000, step=1000
)

# Strategy config editor
with st.sidebar.expander("Strategy Parameters", expanded=False):
    default_config = strat_info.get("default_config", {})
    config_yaml = yaml.dump(default_config, default_flow_style=False, sort_keys=False)
    edited_yaml = st.text_area(
        "Edit config (YAML)",
        value=config_yaml,
        height=300,
        help="Modify strategy parameters. Changes apply on next backtest run.",
    )

# Run button
run_clicked = st.sidebar.button("Run Backtest", type="primary", use_container_width=True)

# ─── Run Backtest ──────────────────────────────────────────────────────────
if run_clicked:
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Parse config override
    try:
        config_override = yaml.safe_load(edited_yaml)
        if not isinstance(config_override, dict):
            config_override = {}
    except yaml.YAMLError:
        st.error("Invalid YAML in strategy config.")
        st.stop()

    # Load strategy
    try:
        strategy = load_strategy(selected_strategy, config_override)
    except Exception as e:
        st.error(f"Failed to load strategy: {e}")
        st.stop()

    client = PolygonClient(api_key)
    progress_bar = st.progress(0, text="Starting...")
    status_text = st.empty()

    def update_progress(current, total, msg):
        pct = min(current / total, 1.0) if total > 0 else 0
        progress_bar.progress(pct, text=f"Processing {msg}...")

    try:
        if ticker_mode == "Manual":
            # Manual tickers mode
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            if not tickers:
                st.error("Please enter at least one ticker.")
                st.stop()

            result = run_backtest(
                strategy=strategy,
                tickers=tickers,
                start_date=start_str,
                end_date=end_str,
                account_size=account_size,
                polygon_client=client,
                progress_callback=update_progress,
            )
        else:
            # Scanner mode
            status_text.info("Scanning for tickers matching your criteria...")
            result = run_backtest_with_scanner(
                strategy=strategy,
                start_date=start_str,
                end_date=end_str,
                account_size=account_size,
                polygon_client=client,
                scanner_filters=scanner_filters,
                progress_callback=update_progress,
            )

        st.session_state["result"] = result
        progress_bar.empty()
        status_text.empty()
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Backtest failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# ─── Display Results ───────────────────────────────────────────────────────
tab_ai, tab_stats, tab_summary, tab_trades, tab_analysis, tab_export = st.tabs(
    ["AI Strategy Builder", "Market Statistics", "Summary", "Trades", "Analysis", "Export"]
)

# ─── AI Strategy Builder Tab ──────────────────────────────────────────────
with tab_ai:
    st.subheader("AI Strategy Builder")
    st.caption("Describe your backtest in plain English and Claude will generate the configuration.")

    ai_prompt = st.text_area(
        "What do you want to backtest?",
        placeholder="e.g. Backtest a momentum breakout strategy on small-cap stocks under $10 with high volume for the last 2 weeks, long only",
        height=100,
    )

    generate_clicked = st.button("Generate Config", type="primary")

    if generate_clicked:
        if not anthropic_api_key:
            st.error("Please enter your Anthropic API key in the sidebar.")
        elif not ai_prompt.strip():
            st.warning("Please describe what you want to backtest.")
        else:
            with st.spinner("Claude is designing your backtest..."):
                try:
                    ai_config = generate_config(ai_prompt, anthropic_api_key)
                    st.session_state["ai_config"] = ai_config
                except ValueError as e:
                    st.error(str(e))

    if "ai_config" in st.session_state:
        ai_config = st.session_state["ai_config"]

        st.info(ai_config.get("explanation", ""))

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Strategy:** {ai_config['strategy']}")
            st.markdown(f"**Ticker Mode:** {ai_config['ticker_mode']}")
            if ai_config.get("tickers"):
                st.markdown(f"**Tickers:** {', '.join(ai_config['tickers'])}")
            st.markdown(f"**Dates:** {ai_config['start_date']} to {ai_config['end_date']}")
            st.markdown(f"**Account Size:** ${ai_config['account_size']:,}")

        with col_b:
            if ai_config.get("scanner_filters"):
                st.markdown("**Scanner Filters:**")
                for k, v in ai_config["scanner_filters"].items():
                    st.markdown(f"- {k}: {v}")

        # Editable strategy params
        params_yaml = yaml.dump(ai_config.get("strategy_params", {}), default_flow_style=False, sort_keys=False)
        edited_ai_params = st.text_area(
            "Strategy Parameters (edit if needed)",
            value=params_yaml,
            height=150,
        )

        if st.button("Run This Backtest", type="primary"):
            if not api_key:
                st.error("Please enter your Polygon API key in the sidebar to run the backtest.")
            else:
                try:
                    config_override = yaml.safe_load(edited_ai_params)
                    if not isinstance(config_override, dict):
                        config_override = {}
                except yaml.YAMLError:
                    st.error("Invalid YAML in strategy parameters.")
                    st.stop()

                try:
                    strategy = load_strategy(ai_config["strategy"], config_override)
                except Exception as e:
                    st.error(f"Failed to load strategy: {e}")
                    st.stop()

                client = PolygonClient(api_key)
                progress_bar = st.progress(0, text="Starting...")

                def ai_update_progress(current, total, msg):
                    pct = min(current / total, 1.0) if total > 0 else 0
                    progress_bar.progress(pct, text=f"Processing {msg}...")

                try:
                    if ai_config["ticker_mode"] == "manual":
                        tickers_list = ai_config.get("tickers", [])
                        if not tickers_list:
                            st.error("No tickers specified.")
                            st.stop()
                        ai_result = run_backtest(
                            strategy=strategy,
                            tickers=tickers_list,
                            start_date=ai_config["start_date"],
                            end_date=ai_config["end_date"],
                            account_size=ai_config["account_size"],
                            polygon_client=client,
                            progress_callback=ai_update_progress,
                        )
                    else:
                        filters = ai_config.get("scanner_filters", {})
                        ai_result = run_backtest_with_scanner(
                            strategy=strategy,
                            start_date=ai_config["start_date"],
                            end_date=ai_config["end_date"],
                            account_size=ai_config["account_size"],
                            polygon_client=client,
                            scanner_filters=filters,
                            progress_callback=ai_update_progress,
                        )

                    st.session_state["result"] = ai_result
                    progress_bar.empty()
                    st.rerun()
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Backtest failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# ─── Market Statistics Tab ────────────────────────────────────────────────
with tab_stats:
    st.subheader("Market Statistics")
    st.caption("Ask natural language questions about historical market behavior and get statistical answers.")

    # Example questions
    example_questions = [
        "If a stock is up 100% at 8am, what are the odds it closes below 20% up?",
        "What % of stocks that gap up 50%+ premarket close red?",
        "If volume is 10x average in the first 5 minutes, what happens by noon?",
    ]
    st.markdown("**Example questions:**")
    ex_cols = st.columns(len(example_questions))
    for i, eq in enumerate(example_questions):
        if ex_cols[i].button(eq, key=f"example_{i}"):
            st.session_state["stats_question"] = eq

    stats_question = st.text_area(
        "Your question:",
        value=st.session_state.get("stats_question", ""),
        placeholder="e.g. If a stock gaps up 50%+ premarket, what % close red?",
        height=80,
    )

    analyze_clicked = st.button("Analyze", type="primary")

    if analyze_clicked:
        if not anthropic_api_key:
            st.error("Please enter your Anthropic API key in the sidebar.")
        elif not api_key:
            st.error("Please enter your Polygon API key in the sidebar.")
        elif not stats_question.strip():
            st.warning("Please enter a question.")
        else:
            # Step 1: Parse question into structured query
            with st.spinner("Interpreting your question..."):
                try:
                    stats_query = parse_stats_query(stats_question, anthropic_api_key)
                    st.session_state["stats_query"] = stats_query
                except ValueError as e:
                    st.error(str(e))
                    stats_query = None

            if stats_query:
                st.info(stats_query.get("explanation", stats_query.get("description", "")))

                with st.expander("Query Details"):
                    st.json(stats_query)

                # Step 2: Execute the query
                client = PolygonClient(api_key)
                progress_bar = st.progress(0, text="Starting analysis...")
                status_text = st.empty()

                def stats_progress(current, total, msg):
                    pct = min(current / total, 1.0) if total > 0 else 0
                    progress_bar.progress(pct, text=msg)

                try:
                    stats_result = execute_stats_query(
                        stats_query, client, progress_callback=stats_progress
                    )
                    st.session_state["stats_result"] = stats_result
                    progress_bar.empty()
                    status_text.empty()
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Analysis failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    if "stats_result" in st.session_state:
        render_stats_results(st.session_state["stats_result"])

has_result = "result" in st.session_state

# ─── Summary Tab ───────────────────────────────────────────────────────────
with tab_summary:
    if not has_result:
        st.info("Configure your backtest in the sidebar and click **Run Backtest**, or use the **AI Strategy Builder** tab.")
    else:
        result: BacktestResult = st.session_state["result"]
        trades_df = result_to_dataframe(result)
        if not result.trades:
            st.warning("No trades were generated. Try adjusting strategy parameters, scanner filters, or date range.")
        else:
            # Show scanned tickers info if available
            scanned_info = result.config.get("_scanned_tickers")
            if scanned_info:
                unique_tickers = set()
                for day_tickers in scanned_info.values():
                    unique_tickers.update(day_tickers)
                st.caption(
                    f"Scanner found **{len(unique_tickers)}** unique tickers "
                    f"across **{len(scanned_info)}** trading days"
                )

            # Metric cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total P&L", f"${result.total_pnl:,.2f}")
            col2.metric("Win Rate", f"{result.win_rate:.1f}%")
            col3.metric("Profit Factor", f"{result.profit_factor:.2f}")
            col4.metric("Max Drawdown", f"{result.max_drawdown:.1f}%")

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Total Trades", len(result.trades))
            col6.metric("Avg Winner", f"${result.avg_winner:,.2f}")
            col7.metric("Avg Loser", f"${result.avg_loser:,.2f}")
            winners = sum(1 for t in result.trades if t.is_winner)
            losers = len(result.trades) - winners
            col8.metric("W / L", f"{winners} / {losers}")

            # Equity curve
            st.subheader("Equity Curve")
            eq_fig = go.Figure()
            eq_fig.add_trace(go.Scatter(
                y=result.equity_curve,
                mode="lines",
                name="Equity",
                line=dict(color="#2196F3", width=2),
            ))
            eq_fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="Account Value ($)",
                xaxis_title="Trade #",
            )
            st.plotly_chart(eq_fig, use_container_width=True)

            # Daily P&L
            st.subheader("Daily P&L")
            daily_pnl = trades_df.groupby("date")["gross_pnl"].sum().reset_index()
            colors = ["#4CAF50" if x > 0 else "#F44336" for x in daily_pnl["gross_pnl"]]
            daily_fig = go.Figure(go.Bar(
                x=daily_pnl["date"],
                y=daily_pnl["gross_pnl"],
                marker_color=colors,
            ))
            daily_fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="P&L ($)",
            )
            st.plotly_chart(daily_fig, use_container_width=True)

# ─── Trades Tab ────────────────────────────────────────────────────────────
with tab_trades:
    if not has_result:
        st.info("Run a backtest to see trades.")
    else:
        result = st.session_state["result"]
        trades_df = result_to_dataframe(result)
        if result.trades:
            st.subheader(f"Trade Log ({len(result.trades)} trades)")

            # Display columns
            display_cols = [
                "date", "ticker", "direction", "entry_time", "exit_time",
                "entry_price", "exit_price", "shares", "gross_pnl",
                "signal_reason", "exit_reason",
            ]
            display_df = trades_df[display_cols].copy()

            # Style P&L column
            st.dataframe(
                display_df.style.applymap(
                    lambda v: "color: #4CAF50" if isinstance(v, (int, float)) and v > 0
                    else ("color: #F44336" if isinstance(v, (int, float)) and v < 0 else ""),
                    subset=["gross_pnl"],
                ),
                use_container_width=True,
                height=500,
            )
        else:
            st.warning("No trades to display.")

# ─── Analysis Tab ──────────────────────────────────────────────────────────
with tab_analysis:
    if not has_result:
        st.info("Run a backtest to see analysis.")
    else:
        result = st.session_state["result"]
        trades_df = result_to_dataframe(result)
        if result.trades:
            col_left, col_right = st.columns(2)

            with col_left:
                # P&L by hour
                st.subheader("P&L by Entry Hour")
                trades_df["entry_hour"] = trades_df["entry_time"].str[:2].astype(int)
                hourly = trades_df.groupby("entry_hour")["gross_pnl"].sum().reset_index()
                colors_h = ["#4CAF50" if x > 0 else "#F44336" for x in hourly["gross_pnl"]]
                hour_fig = go.Figure(go.Bar(
                    x=hourly["entry_hour"],
                    y=hourly["gross_pnl"],
                    marker_color=colors_h,
                ))
                hour_fig.update_layout(
                    height=300, margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Hour (ET)", yaxis_title="P&L ($)",
                )
                st.plotly_chart(hour_fig, use_container_width=True)

            with col_right:
                # P&L by ticker
                st.subheader("P&L by Ticker")
                by_ticker = trades_df.groupby("ticker")["gross_pnl"].sum().reset_index()
                by_ticker = by_ticker.sort_values("gross_pnl", ascending=True)
                colors_t = ["#4CAF50" if x > 0 else "#F44336" for x in by_ticker["gross_pnl"]]
                ticker_fig = go.Figure(go.Bar(
                    x=by_ticker["gross_pnl"],
                    y=by_ticker["ticker"],
                    orientation="h",
                    marker_color=colors_t,
                ))
                ticker_fig.update_layout(
                    height=max(300, len(by_ticker) * 25), margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="P&L ($)",
                )
                st.plotly_chart(ticker_fig, use_container_width=True)

            col_left2, col_right2 = st.columns(2)

            with col_left2:
                # Win rate by direction
                st.subheader("Win Rate by Direction")
                for direction in trades_df["direction"].unique():
                    dir_trades = trades_df[trades_df["direction"] == direction]
                    wins = (dir_trades["gross_pnl"] > 0).sum()
                    total = len(dir_trades)
                    wr = wins / total * 100 if total > 0 else 0
                    st.metric(
                        f"{direction.title()} ({total} trades)",
                        f"{wr:.1f}%",
                    )

            with col_right2:
                # P&L distribution
                st.subheader("P&L Distribution")
                dist_fig = go.Figure(go.Histogram(
                    x=trades_df["gross_pnl"],
                    nbinsx=20,
                    marker_color="#2196F3",
                ))
                dist_fig.update_layout(
                    height=300, margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="P&L ($)", yaxis_title="Count",
                )
                st.plotly_chart(dist_fig, use_container_width=True)

            # Exit reason breakdown
            st.subheader("Exit Reasons")
            exit_counts = trades_df["exit_reason"].value_counts().reset_index()
            exit_counts.columns = ["Exit Reason", "Count"]
            st.dataframe(exit_counts, use_container_width=True, hide_index=True)

        else:
            st.warning("No trades to analyze.")

# ─── Export Tab ────────────────────────────────────────────────────────────
with tab_export:
    if not has_result:
        st.info("Run a backtest to export results.")
    else:
        result = st.session_state["result"]
        trades_df = result_to_dataframe(result)
        if result.trades:
            st.subheader("CSV Export")
            st.caption("Download trade data for review in TradingView or spreadsheets.")

            # Preview
            st.dataframe(trades_df, use_container_width=True, height=300)

            # Download
            csv_data = export_to_csv(result)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"backtest_{result.start_date}_{result.end_date}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.warning("No trades to export.")
