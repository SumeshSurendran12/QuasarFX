from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    spread_pips: float
    ts_utc: str


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    order_type: str  # "market"
    sl: Optional[float]
    tp: Optional[float]
    comment: str


@dataclass(frozen=True)
class OrderResult:
    accepted: bool
    broker_order_id: str
    broker_position_id: str
    broker_deal_id: str
    requested_price: Optional[float]
    fill_price: Optional[float]
    retcode: str
    retcode_detail: str
    raw_status: str


@dataclass(frozen=True)
class PositionState:
    is_open: bool
    side: Optional[str]
    qty: float
    entry_price: Optional[float]
    broker_position_id: str
    broker_order_id: str
    entry_ts_utc: Optional[str]
    unrealized_pnl_usd: float


class ExecutionAdapter(Protocol):
    def connect(self) -> None: ...

    def shutdown(self) -> None: ...

    def get_quote(self, symbol: str) -> Quote: ...

    def submit_order(self, request: OrderRequest) -> OrderResult: ...

    def get_open_position(self, symbol: str) -> PositionState: ...

    def close_position(self, symbol: str) -> OrderResult: ...
