"""
Market Statistics - Natural language statistical queries on historical market data.

Uses Claude to interpret questions, then scans historical data via Polygon
to compute probabilities and distributions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import anthropic
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from ai.utils import parse_response
from data.polygon_client import PolygonClient
from data.scanner import scan_tickers
from engine.backtest import get_trading_days
from indicators.core import compute_all_indicators


# ─── Supported metrics ────────────────────────────────────────────────────────

SUPPORTED_METRICS = {
    "change_from_prev_close": "Percentage change from previous close at a given time",
    "change_from_open": "Percentage change from 9:30 open at a given time",
    "price": "Raw price at a given time",
    "gap_percent": "Gap percentage (9:30 open vs previous close)",
    "volume_ratio": "Volume ratio (current bar volume / 20-bar average) at a given time",
    "rsi": "RSI-14 value at a given time",
    "vwap_distance": "Percentage distance from VWAP at a given time",
    "close_red": "Whether the stock closed red (close < open)",
    "high_from_prev_close": "Intraday high as % change from previous close",
    "low_from_prev_close": "Intraday low as % change from previous close",
}

SUPPORTED_OPERATORS = [">=", "<=", ">", "<", "=="]


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class StatsResult:
    query: dict
    total_scanned: int = 0
    condition_matches: int = 0
    outcome_true: int = 0
    probability: float | None = None
    outcome_values: list[float] = field(default_factory=list)
    details: list[dict] = field(default_factory=list)


# ─── System prompt ────────────────────────────────────────────────────────────

def build_stats_system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    six_months_ago = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

    metrics_block = "\n".join(f"- `{k}`: {v}" for k, v in SUPPORTED_METRICS.items())

    return f"""You are a market statistics query generator. Convert natural language questions
about stock market behavior into structured JSON queries.

## Supported Metrics
{metrics_block}

## Time Format
- Use 24-hour ET time as "HH:MM" (e.g., "08:00", "09:35", "12:00", "16:00")
- Use null for whole-day metrics (gap_percent, close_red, high_from_prev_close, low_from_prev_close)

## Operators
Supported: >=, <=, >, <, ==

## Output Schema
Return ONLY a JSON object:
{{{{
  "description": "Human-readable restatement of the question",
  "universe": {{{{
    "date_range": {{{{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}}},
    "scanner_filters": {{{{
      "min_price": 0.5,
      "max_price": 100,
      "min_volume": 100000,
      "min_change_percent": 0,
      "max_results": 50
    }}}}
  }}}},
  "condition": {{{{
    "metric": "<metric_name>",
    "time": "HH:MM" or null,
    "operator": "<operator>",
    "value": <number>
  }}}},
  "outcome": {{{{
    "metric": "<metric_name>",
    "time": "HH:MM" or null,
    "operator": "<operator>" or null,
    "value": <number> or null
  }}}},
  "explanation": "2-3 sentences explaining the query setup"
}}}}

When outcome.operator and outcome.value are both null, the engine computes a full
distribution of outcome.metric (for "what happens" style questions).

## Examples

User: "If a stock is up 100% at 8am, what are the odds it closes below 20% up?"
{{{{
  "description": "Probability that stocks up 100%+ at 8:00 AM close below 20% up from prev close",
  "universe": {{{{"date_range": {{{{"start": "{six_months_ago}", "end": "{today}"}}}}, "scanner_filters": {{{{"min_price": 0.5, "max_price": 100, "min_volume": 100000, "min_change_percent": 20, "max_results": 50}}}}}}}},
  "condition": {{{{"metric": "change_from_prev_close", "time": "08:00", "operator": ">=", "value": 100.0}}}},
  "outcome": {{{{"metric": "change_from_prev_close", "time": "16:00", "operator": "<", "value": 20.0}}}},
  "explanation": "Scanning for stocks showing 100%+ gains in premarket at 8am, then checking if they fade below 20% gain by close"
}}}}

User: "What % of stocks that gap up 50%+ premarket close red?"
{{{{
  "description": "Percentage of 50%+ gap-up stocks that close red",
  "universe": {{{{"date_range": {{{{"start": "{six_months_ago}", "end": "{today}"}}}}, "scanner_filters": {{{{"min_price": 0.5, "max_price": 100, "min_volume": 200000, "min_change_percent": 10, "max_results": 50}}}}}}}},
  "condition": {{{{"metric": "gap_percent", "time": null, "operator": ">=", "value": 50.0}}}},
  "outcome": {{{{"metric": "close_red", "time": null, "operator": "==", "value": 1}}}},
  "explanation": "Finding stocks that gapped up 50%+ at the open and checking if they closed below their open price"
}}}}

User: "If volume is 10x average in the first 5 minutes, what happens by noon?"
{{{{
  "description": "Distribution of price change from open to noon for stocks with 10x volume in first 5 minutes",
  "universe": {{{{"date_range": {{{{"start": "{six_months_ago}", "end": "{today}"}}}}, "scanner_filters": {{{{"min_price": 1, "max_price": 50, "min_volume": 500000, "max_results": 50}}}}}}}},
  "condition": {{{{"metric": "volume_ratio", "time": "09:35", "operator": ">=", "value": 10.0}}}},
  "outcome": {{{{"metric": "change_from_open", "time": "12:00", "operator": null, "value": null}}}},
  "explanation": "Looking at stocks with extreme early volume (10x+ average) and measuring how price moves from open to noon"
}}}}

## Rules
- Default date range: last 6 months ({six_months_ago} to {today})
- Use reasonable scanner_filters to find relevant stocks (min_change_percent is key for big movers)
- For premarket questions, use times before "09:30"
- For "close" or "end of day", use time "16:00"
- For "open" or "at the open", use time "09:30"
- gap_percent, close_red, high_from_prev_close, low_from_prev_close use time: null
- Always generate a valid query even if the question is vague
- Put all explanations inside the "explanation" field

CRITICAL: Your entire response must be ONLY the JSON object. No text outside the JSON.
Start with {{{{ and end with }}}}."""


# ─── Query parsing ────────────────────────────────────────────────────────────

def parse_stats_query(question: str, api_key: str) -> dict:
    """Parse a natural language question into a stats query via Claude."""
    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=build_stats_system_prompt(),
            messages=[{"role": "user", "content": question}],
        )
    except anthropic.AuthenticationError:
        raise ValueError("Invalid Anthropic API key.")
    except anthropic.RateLimitError:
        raise ValueError("Rate limited. Please wait and try again.")
    except anthropic.APIError as e:
        raise ValueError(f"API error: {e}")

    raw_text = response.content[0].text

    try:
        query = parse_response(raw_text)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse response as JSON. Raw:\n{raw_text}")

    # Validate
    for section in ("condition", "outcome"):
        metric = query.get(section, {}).get("metric")
        if metric and metric not in SUPPORTED_METRICS:
            raise ValueError(f"Unknown metric '{metric}' in {section}. Supported: {list(SUPPORTED_METRICS.keys())}")

    return query


# ─── Metric evaluation ────────────────────────────────────────────────────────

def _find_bar_at_time(bars: pd.DataFrame, time_str: str | None) -> pd.Series | None:
    """Find the bar at or nearest before the given time."""
    if time_str is None:
        return None

    hour, minute = int(time_str[:2]), int(time_str[3:5])
    target_minutes = hour * 60 + minute

    bars_minutes = bars["dt"].dt.hour * 60 + bars["dt"].dt.minute
    valid = bars[bars_minutes <= target_minutes]

    if len(valid) == 0:
        return None
    return valid.iloc[-1]


def evaluate_metric(
    bars: pd.DataFrame,
    indicators: dict,
    prev_close: float | None,
    metric: str,
    time_str: str | None,
) -> float | None:
    """Evaluate a metric at a given time, returning a scalar value."""
    bar = _find_bar_at_time(bars, time_str)

    # Find open bar (first bar at or after 09:30)
    market_bars = bars[bars["dt"].dt.hour * 60 + bars["dt"].dt.minute >= 570]
    open_bar = market_bars.iloc[0] if len(market_bars) > 0 else None
    close_bar = market_bars.iloc[-1] if len(market_bars) > 0 else None

    if metric == "change_from_prev_close":
        if bar is None or prev_close is None or prev_close == 0:
            return None
        return ((bar["c"] - prev_close) / prev_close) * 100

    elif metric == "change_from_open":
        if bar is None or open_bar is None or open_bar["o"] == 0:
            return None
        return ((bar["c"] - open_bar["o"]) / open_bar["o"]) * 100

    elif metric == "price":
        return bar["c"] if bar is not None else None

    elif metric == "gap_percent":
        return indicators.get("gap_percent")

    elif metric == "volume_ratio":
        if bar is None:
            return None
        vr = indicators.get("volume_ratio")
        if vr is None:
            return None
        idx = bars.index.get_loc(bar.name) if bar.name in bars.index else None
        if idx is not None and idx < len(vr):
            val = vr.iloc[idx]
            return float(val) if not pd.isna(val) else None
        return None

    elif metric == "rsi":
        if bar is None:
            return None
        rsi_series = indicators.get("rsi_14")
        if rsi_series is None:
            return None
        idx = bars.index.get_loc(bar.name) if bar.name in bars.index else None
        if idx is not None and idx < len(rsi_series):
            val = rsi_series.iloc[idx]
            return float(val) if not pd.isna(val) else None
        return None

    elif metric == "vwap_distance":
        if bar is None:
            return None
        dist = indicators.get("dist_from_vwap")
        if dist is None:
            return None
        idx = bars.index.get_loc(bar.name) if bar.name in bars.index else None
        if idx is not None and idx < len(dist):
            val = dist.iloc[idx]
            return float(val) if not pd.isna(val) else None
        return None

    elif metric == "close_red":
        if open_bar is None or close_bar is None:
            return None
        return 1.0 if close_bar["c"] < open_bar["o"] else 0.0

    elif metric == "high_from_prev_close":
        if prev_close is None or prev_close == 0:
            return None
        return ((bars["h"].max() - prev_close) / prev_close) * 100

    elif metric == "low_from_prev_close":
        if prev_close is None or prev_close == 0:
            return None
        return ((bars["l"].min() - prev_close) / prev_close) * 100

    return None


def _check_condition(value: float | None, operator: str, target) -> bool:
    """Check if a value meets a condition."""
    if value is None:
        return False
    if operator == ">=":
        return value >= target
    elif operator == "<=":
        return value <= target
    elif operator == ">":
        return value > target
    elif operator == "<":
        return value < target
    elif operator == "==":
        return value == target
    return False


# ─── Query execution ──────────────────────────────────────────────────────────

def execute_stats_query(
    query: dict,
    polygon_client: PolygonClient,
    progress_callback=None,
) -> StatsResult:
    """Execute a statistical query against historical data."""
    result = StatsResult(query=query)

    universe = query["universe"]
    date_range = universe["date_range"]
    filters = universe.get("scanner_filters", {})

    condition = query["condition"]
    outcome = query["outcome"]
    is_distribution = outcome.get("operator") is None

    # Get trading days
    days = get_trading_days(date_range["start"], date_range["end"])

    # Phase 1: Scan for candidates across all days
    all_ticker_days = []
    for i, date_str in enumerate(days):
        if progress_callback:
            progress_callback(i + 1, len(days), f"Scanning {date_str}")

        candidates = scan_tickers(polygon_client, date_str, **filters)
        for cand in candidates:
            all_ticker_days.append((date_str, cand["ticker"]))

    result.total_scanned = len(all_ticker_days)

    if not all_ticker_days:
        return result

    # Phase 2: Evaluate condition and outcome for each ticker-day
    for i, (date_str, ticker) in enumerate(all_ticker_days):
        if progress_callback:
            progress_callback(
                len(days) + i + 1,
                len(days) + len(all_ticker_days),
                f"Analyzing {ticker} on {date_str}",
            )

        bars = polygon_client.get_minute_bars(ticker, date_str)
        if bars is None or len(bars) == 0:
            continue

        prev_close = polygon_client.get_prev_close(ticker, date_str)
        indicators = compute_all_indicators(bars, prev_close)

        # Evaluate condition
        cond_value = evaluate_metric(
            bars, indicators, prev_close,
            condition["metric"], condition.get("time"),
        )

        if not _check_condition(cond_value, condition["operator"], condition["value"]):
            continue

        result.condition_matches += 1

        # Evaluate outcome
        outcome_value = evaluate_metric(
            bars, indicators, prev_close,
            outcome["metric"], outcome.get("time"),
        )

        if outcome_value is None:
            continue

        outcome_met = False
        if not is_distribution:
            outcome_met = _check_condition(outcome_value, outcome["operator"], outcome["value"])
            if outcome_met:
                result.outcome_true += 1

        result.outcome_values.append(outcome_value)
        result.details.append({
            "date": date_str,
            "ticker": ticker,
            "condition_value": round(cond_value, 2) if cond_value is not None else None,
            "outcome_value": round(outcome_value, 2),
            "outcome_met": outcome_met if not is_distribution else None,
        })

    # Compute probability for binary queries
    if not is_distribution and result.condition_matches > 0:
        result.probability = result.outcome_true / result.condition_matches

    return result


# ─── Display ──────────────────────────────────────────────────────────────────

def render_stats_results(result: StatsResult):
    """Render statistical results in Streamlit."""
    query = result.query
    is_distribution = query["outcome"].get("operator") is None

    # Summary metrics
    st.markdown("---")
    st.subheader("Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Ticker-Days Scanned", f"{result.total_scanned:,}")
    col2.metric("Matched Condition", f"{result.condition_matches:,}")

    if result.condition_matches == 0:
        st.warning("No ticker-days matched the condition. Try broadening the date range or adjusting scanner filters.")
        return

    if not is_distribution:
        # Binary mode
        pct = result.probability * 100 if result.probability is not None else 0
        col3.metric("Outcome True", f"{result.outcome_true:,}")

        st.markdown(f"### {pct:.1f}% of the time")
        st.caption(
            f"Out of {result.condition_matches} instances where the condition was met, "
            f"{result.outcome_true} met the outcome ({pct:.1f}%)"
        )

        # Bar chart
        fig = go.Figure(go.Bar(
            x=["Outcome True", "Outcome False"],
            y=[result.outcome_true, result.condition_matches - result.outcome_true],
            marker_color=["#4CAF50", "#F44336"],
        ))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Distribution mode
        values = result.outcome_values
        if values:
            arr = np.array(values)
            col3.metric("Sample Size", f"{len(values):,}")

            scol1, scol2, scol3, scol4 = st.columns(4)
            scol1.metric("Mean", f"{arr.mean():.1f}%")
            scol2.metric("Median", f"{np.median(arr):.1f}%")
            scol3.metric("Std Dev", f"{arr.std():.1f}%")
            scol4.metric("Range", f"{arr.min():.1f}% to {arr.max():.1f}%")

            # Histogram
            st.subheader("Distribution")
            hist_fig = go.Figure(go.Histogram(
                x=values,
                nbinsx=30,
                marker_color="#2196F3",
            ))
            hist_fig.update_layout(
                height=350, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title=f"{query['outcome']['metric']} (%)",
                yaxis_title="Count",
            )
            st.plotly_chart(hist_fig, use_container_width=True)

    # Outcome distribution (also show for binary mode)
    if result.outcome_values and not is_distribution:
        with st.expander("Outcome Value Distribution"):
            hist_fig = go.Figure(go.Histogram(
                x=result.outcome_values,
                nbinsx=25,
                marker_color="#2196F3",
            ))
            hist_fig.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title=f"{query['outcome']['metric']}",
                yaxis_title="Count",
            )
            st.plotly_chart(hist_fig, use_container_width=True)

    # Detail table
    if result.details:
        st.subheader("Individual Results")
        df = pd.DataFrame(result.details)
        df = df.sort_values("outcome_value", ascending=True)
        st.dataframe(df, use_container_width=True, height=400)
