"""Execution adapters for paper, MT5, and GCAPI routing."""

from .base_adapter import ExecutionAdapter, OrderRequest, OrderResult, PositionState, Quote
from .gcapi_adapter import GCAPIAdapter
from .mt5_adapter import MT5Adapter
from .paper_adapter import PaperAdapter

__all__ = [
    "ExecutionAdapter",
    "OrderRequest",
    "OrderResult",
    "PositionState",
    "Quote",
    "PaperAdapter",
    "MT5Adapter",
    "GCAPIAdapter",
]
