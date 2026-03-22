"""
Strategy loader - auto-discovers strategies from YAML configs and Python files.
"""

from __future__ import annotations

import importlib
import os
import sys

import yaml

from strategies.base import Strategy
from strategies.config_strategy import ConfigStrategy

STRATEGIES_DIR = os.path.dirname(__file__)
CONFIGS_DIR = os.path.join(STRATEGIES_DIR, "configs")


def discover_strategies() -> dict[str, dict]:
    """
    Discover all available strategies.

    Returns dict mapping strategy name to info:
        {"name": str, "description": str, "type": "python"|"yaml", "source": str,
         "factory": callable that returns Strategy instance}
    """
    strategies = {}

    # Discover Python built-in strategies
    for filename in sorted(os.listdir(STRATEGIES_DIR)):
        if filename.startswith("builtin_") and filename.endswith(".py"):
            module_name = f"strategies.{filename[:-3]}"
            try:
                mod = importlib.import_module(module_name)
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if (isinstance(attr, type) and issubclass(attr, Strategy)
                            and attr is not Strategy and not attr.__name__.startswith("_")):
                        strat = attr()
                        strategies[strat.name] = {
                            "name": strat.name,
                            "description": strat.description,
                            "type": "python",
                            "source": filename,
                            "factory": attr,
                            "default_config": getattr(attr, "DEFAULT_CONFIG", {}),
                        }
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")

    # Discover YAML configs
    if os.path.isdir(CONFIGS_DIR):
        for filename in sorted(os.listdir(CONFIGS_DIR)):
            if filename.endswith((".yaml", ".yml")):
                filepath = os.path.join(CONFIGS_DIR, filename)
                try:
                    with open(filepath) as f:
                        config = yaml.safe_load(f)
                    name = config.get("name", filename)
                    strategies[name] = {
                        "name": name,
                        "description": config.get("description", ""),
                        "type": "yaml",
                        "source": filename,
                        "factory": lambda c=config: ConfigStrategy(c),
                        "default_config": config,
                    }
                except Exception as e:
                    print(f"Warning: Failed to load {filename}: {e}")

    return strategies


def load_strategy(name: str, config_override: dict | None = None) -> Strategy:
    """Load a strategy by name, optionally overriding config."""
    strategies = discover_strategies()
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")

    info = strategies[name]
    if info["type"] == "yaml":
        config = {**info["default_config"], **(config_override or {})}
        return ConfigStrategy(config)
    else:
        return info["factory"](config_override)
