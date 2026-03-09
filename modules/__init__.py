"""
Forex Trading Bot modules package
"""

from .config import *  # noqa: F401,F403
from .logger import logger

__all__ = [
    "logger",
    "DataFetcher",
    "TradingEnvironment",
    "TradingModel",
    "PortfolioOptimizer",
    "CustomRewardWrapper",
    "train_ppo_model",
    "evaluate_model",
    "create_compatible_env",
    "LiveTradingEnvironment",
    "DebugLogger",
]


def __getattr__(name):
    if name == "DataFetcher":
        from .data_fetcher import DataFetcher
        return DataFetcher
    if name in {
        "TradingEnvironment",
        "TradingModel",
        "PortfolioOptimizer",
        "CustomRewardWrapper",
        "train_ppo_model",
        "evaluate_model",
        "create_compatible_env",
    }:
        from . import model as model_module
        return getattr(model_module, name)
    if name == "LiveTradingEnvironment":
        from .live_trading import LiveTradingEnvironment
        return LiveTradingEnvironment
    if name == "DebugLogger":
        from .debug import DebugLogger
        return DebugLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
