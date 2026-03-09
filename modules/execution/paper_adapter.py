from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

from .base_adapter import OrderRequest, OrderResult, PositionState, Quote
from .mt5_utils import MT5_AVAILABLE, bool_env, pip_size_for_symbol, safe_float, tick_timestamp_iso, utc_iso

if MT5_AVAILABLE:
    from .mt5_utils import get_mt5_module

    _mt5 = get_mt5_module()
else:  # pragma: no cover - covered via no-mt5 environments
    _mt5 = None


@dataclass
class _PaperPosition:
    side: str
    qty: float
    entry_price: float
    entry_ts_utc: str
    broker_position_id: str
    broker_order_id: str


class PaperAdapter:
    def __init__(
        self,
        *,
        symbol: str,
        pip_size: Optional[float] = None,
        start_price: float = 1.08500,
        spread: float = 0.00008,
        tick_step: float = 0.00003,
        use_mt5_feed: Optional[bool] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.symbol = str(symbol).upper()
        self.pip_size = float(pip_size if pip_size is not None else pip_size_for_symbol(self.symbol))
        self.paper_mid_price = float(start_price)
        self.paper_spread = max(self.pip_size * 0.2, float(spread))
        self.paper_tick_step = max(self.pip_size * 0.05, float(tick_step))
        self.use_mt5_feed = bool_env("FX_PAPER_USE_MT5_FEED", True) if use_mt5_feed is None else bool(use_mt5_feed)
        self.logger = logger or logging.getLogger(__name__)
        self.connected = False

        self._synthetic_tick_seq = 0
        self._order_seq = 0
        self._last_quote: Optional[Quote] = None
        self._position: Optional[_PaperPosition] = None

    def connect(self) -> None:
        self.connected = True

    def shutdown(self) -> None:
        self.connected = False

    def _ensure_connected(self) -> None:
        if not self.connected:
            self.connect()

    def _next_synthetic_quote(self) -> Quote:
        self._synthetic_tick_seq += 1
        phase = self._synthetic_tick_seq % 40
        direction = 1.0 if phase < 20 else -1.0
        self.paper_mid_price = max(self.pip_size * 10.0, self.paper_mid_price + (direction * self.paper_tick_step))
        half_spread = max(self.pip_size * 0.1, self.paper_spread / 2.0)
        bid = float(self.paper_mid_price - half_spread)
        ask = float(self.paper_mid_price + half_spread)
        return Quote(
            symbol=self.symbol,
            bid=bid,
            ask=ask,
            spread_pips=float((ask - bid) / self.pip_size),
            ts_utc=utc_iso(),
        )

    def _try_mt5_quote(self) -> Optional[Quote]:
        if not self.use_mt5_feed or not MT5_AVAILABLE or _mt5 is None:
            return None
        try:
            tick = _mt5.symbol_info_tick(self.symbol)
        except Exception:
            return None
        if tick is None:
            return None
        bid = safe_float(getattr(tick, "bid", 0.0), 0.0)
        ask = safe_float(getattr(tick, "ask", 0.0), 0.0)
        if bid <= 0 or ask <= 0:
            return None
        spread_pips = float((ask - bid) / self.pip_size)
        return Quote(
            symbol=self.symbol,
            bid=bid,
            ask=ask,
            spread_pips=spread_pips,
            ts_utc=tick_timestamp_iso(tick),
        )

    def get_quote(self, symbol: str) -> Quote:
        self._ensure_connected()
        quote = self._try_mt5_quote()
        if quote is None:
            quote = self._next_synthetic_quote()
        self._last_quote = quote
        return quote

    def submit_order(self, request: OrderRequest) -> OrderResult:
        self._ensure_connected()

        side = str(request.side).strip().lower()
        if side not in {"buy", "sell"}:
            return OrderResult(
                accepted=False,
                broker_order_id="",
                broker_position_id="",
                broker_deal_id="",
                requested_price=None,
                fill_price=None,
                retcode="PAPER_INVALID_SIDE",
                retcode_detail=f"Unsupported side '{request.side}'",
                raw_status="rejected",
            )

        if self._position is not None:
            return OrderResult(
                accepted=False,
                broker_order_id=self._position.broker_order_id,
                broker_position_id=self._position.broker_position_id,
                broker_deal_id="",
                requested_price=None,
                fill_price=None,
                retcode="PAPER_POSITION_EXISTS",
                retcode_detail="A paper position is already open.",
                raw_status="rejected",
            )

        quote = self._last_quote or self.get_quote(request.symbol)
        requested_price = float(quote.ask if side == "buy" else quote.bid)
        self._order_seq += 1
        broker_order_id = f"paper-order-{self._order_seq:06d}"
        broker_position_id = f"paper-pos-{self._order_seq:06d}"
        self._position = _PaperPosition(
            side="long" if side == "buy" else "short",
            qty=max(0.0, float(request.qty)),
            entry_price=requested_price,
            entry_ts_utc=utc_iso(),
            broker_position_id=broker_position_id,
            broker_order_id=broker_order_id,
        )
        return OrderResult(
            accepted=True,
            broker_order_id=broker_order_id,
            broker_position_id=broker_position_id,
            broker_deal_id=f"paper-deal-{self._order_seq:06d}",
            requested_price=requested_price,
            fill_price=requested_price,
            retcode="PAPER_FILLED",
            retcode_detail="filled",
            raw_status="filled",
        )

    def get_open_position(self, symbol: str) -> PositionState:
        self._ensure_connected()
        if self._position is None:
            return PositionState(
                is_open=False,
                side=None,
                qty=0.0,
                entry_price=None,
                broker_position_id="",
                broker_order_id="",
                entry_ts_utc=None,
                unrealized_pnl_usd=0.0,
            )
        return PositionState(
            is_open=True,
            side=self._position.side,
            qty=float(self._position.qty),
            entry_price=float(self._position.entry_price),
            broker_position_id=self._position.broker_position_id,
            broker_order_id=self._position.broker_order_id,
            entry_ts_utc=self._position.entry_ts_utc,
            unrealized_pnl_usd=0.0,
        )

    def close_position(self, symbol: str) -> OrderResult:
        self._ensure_connected()
        if self._position is None:
            return OrderResult(
                accepted=False,
                broker_order_id="",
                broker_position_id="",
                broker_deal_id="",
                requested_price=None,
                fill_price=None,
                retcode="PAPER_NO_POSITION",
                retcode_detail="No open paper position.",
                raw_status="rejected",
            )

        quote = self._last_quote or self.get_quote(symbol)
        fill_price = safe_float(quote.bid if self._position.side == "long" else quote.ask, 0.0)
        broker_order_id = self._position.broker_order_id
        broker_position_id = self._position.broker_position_id
        self._order_seq += 1
        broker_deal_id = f"paper-close-{self._order_seq:06d}"
        self._position = None
        return OrderResult(
            accepted=True,
            broker_order_id=broker_order_id,
            broker_position_id=broker_position_id,
            broker_deal_id=broker_deal_id,
            requested_price=fill_price,
            fill_price=fill_price,
            retcode="PAPER_CLOSED",
            retcode_detail="closed",
            raw_status="filled",
        )
