"""
AI Strategy Builder - Uses Claude API to generate backtest configurations
from natural language descriptions.
"""

from __future__ import annotations

import json
from datetime import datetime

import anthropic
import yaml

from ai.utils import parse_response
from strategies.loader import discover_strategies


def build_system_prompt() -> str:
    """Build the system prompt dynamically from discovered strategies."""
    strategies = discover_strategies()
    today = datetime.now().strftime("%Y-%m-%d")

    strategy_descriptions = []
    for name, info in strategies.items():
        config_yaml = yaml.dump(info.get("default_config", {}), default_flow_style=False, sort_keys=False)
        strategy_descriptions.append(
            f'- Name: "{name}"\n'
            f'  Description: {info["description"]}\n'
            f"  Type: {info['type']}\n"
            f"  Default Config:\n"
            + "\n".join(f"    {line}" for line in config_yaml.strip().split("\n"))
        )

    strategies_block = "\n\n".join(strategy_descriptions)
    strategy_names = ", ".join(f'"{n}"' for n in strategies.keys())

    return f"""You are a day trading backtest configuration assistant. Interpret the user's
natural language description and produce a JSON configuration for the backtesting engine.

## Available Strategies
{strategies_block}

## Available Indicators
The engine computes these on minute bars:
- VWAP (cumulative, resets at 09:30)
- EMA 9, EMA 20 (on close)
- RSI 14
- ATR 14
- volume_ratio (current bar volume / 20-bar average)
- cum_volume (cumulative from day start)
- dist_from_vwap (percentage distance)
- orb_high / orb_low (opening range high/low, configurable minutes)
- premarket high/low
- gap_percent (open vs previous close)
- prev_close

## Scanner Filters
When ticker_mode is "scanner", these filters find tickers automatically each day:
- min_price / max_price: Dollar price range (float)
- min_volume: Minimum daily shares traded (int)
- min_dollar_volume: Minimum price * volume (float)
- min_float / max_float: Float shares outstanding (float, 0 = no limit)
- min_market_cap / max_market_cap: Market capitalization (float, 0 = no limit)
- min_change_percent: Minimum absolute intraday change % (float)
- max_results: Maximum tickers per day (int, default 20)

## Output Format
Return ONLY a JSON object with this exact structure:
{{
  "strategy": "<one of: {strategy_names}>",
  "strategy_params": {{}},
  "ticker_mode": "manual" or "scanner",
  "tickers": ["AAPL", "TSLA"] or null,
  "scanner_filters": {{...}} or null,
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "account_size": 25000,
  "explanation": "2-3 sentence rationale for the choices"
}}

## Rules
- strategy MUST be one of the exact names listed above
- strategy_params should only override parameters that differ from defaults
- For relative dates like "last month", calculate from today: {today}
- Default account_size to 25000 if not mentioned
- Default max_results to 20 if using scanner
- If the user mentions specific tickers, use "manual" mode
- If the user describes criteria (price range, market cap, volume, etc.), use "scanner" mode
- If direction is mentioned (long only, short only), include it in strategy_params
- If the user's request is vague or not about trading, pick the most reasonable strategy and config anyway
- Put all commentary inside the "explanation" field

CRITICAL: Your entire response must be ONLY the JSON object. No introductory text,
no explanations outside the JSON, no markdown. Start with {{ and end with }}."""


def _validate_config(config: dict) -> dict:
    """Validate the parsed config against known constraints."""
    strategies = discover_strategies()

    # Check strategy name
    if config.get("strategy") not in strategies:
        available = list(strategies.keys())
        raise ValueError(
            f"Unknown strategy: '{config.get('strategy')}'. Available: {available}"
        )

    # Check ticker_mode
    if config.get("ticker_mode") not in ("manual", "scanner"):
        raise ValueError(f"Invalid ticker_mode: '{config.get('ticker_mode')}'")

    # Check dates parse
    for field in ("start_date", "end_date"):
        try:
            datetime.strptime(config[field], "%Y-%m-%d")
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid {field}: {e}")

    # Ensure required fields have defaults
    config.setdefault("strategy_params", {})
    config.setdefault("account_size", 25000)
    config.setdefault("explanation", "")

    return config


def generate_config(user_prompt: str, api_key: str) -> dict:
    """
    Generate a backtest configuration from a natural language description.

    Returns a validated config dict with keys: strategy, strategy_params,
    ticker_mode, tickers, scanner_filters, start_date, end_date,
    account_size, explanation.
    """
    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=build_system_prompt(),
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.AuthenticationError:
        raise ValueError("Invalid Anthropic API key. Please check and try again.")
    except anthropic.RateLimitError:
        raise ValueError("Rate limited by the Anthropic API. Please wait a moment and try again.")
    except anthropic.APIError as e:
        raise ValueError(f"Anthropic API error: {e}")

    raw_text = response.content[0].text

    try:
        config = parse_response(raw_text)
    except json.JSONDecodeError:
        raise ValueError(
            f"Failed to parse AI response as JSON. Raw response:\n{raw_text}"
        )

    return _validate_config(config)
