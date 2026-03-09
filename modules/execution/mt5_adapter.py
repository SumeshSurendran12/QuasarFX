from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .base_adapter import OrderRequest, OrderResult, PositionState, Quote
from .mt5_symbol_map import resolve_symbol
from .mt5_types import is_success_retcode, retcode_label
from .mt5_utils import (
    MT5_AVAILABLE,
    get_mt5_module,
    pip_size_for_symbol,
    safe_float,
    safe_int,
    tick_timestamp_iso,
    utc_iso,
)


class MT5Adapter:
    def __init__(
        self,
        *,
        symbol: str,
        mode: str,
        config: Dict[str, Any],
        magic_number: int,
        order_comment_prefix: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.requested_symbol = str(symbol).upper()
        self.mode = str(mode).lower()
        self.config = config
        self.magic_number = int(magic_number)
        self.order_comment_prefix = str(order_comment_prefix).strip() or "QFX-S1"
        self.logger = logger or logging.getLogger(__name__)

        self.connected = False
        self.resolved_symbol = self.requested_symbol
        self.account_id = ""
        self.server_name = ""

        self.deviation_points = safe_int(config.get("MT5_DEVIATION_POINTS", config.get("BASE_DEVIATION", 20)), 20)
        self.order_comment = str(config.get("MT5_ORDER_COMMENT", config.get("COMMENT", "QuasarFX_Strategy1"))).strip()
        self.terminal_path = str(config.get("MT5_TERMINAL_PATH", "")).strip()
        self.login = str(config.get("MT5LOGIN", "") or "").strip()
        self.password = str(config.get("MT5PASSWORD", "") or "").strip()
        self.server = str(config.get("MT5SERVER", "") or "").strip()

    def _build_comment(self, action: str) -> str:
        base = self.order_comment or self.order_comment_prefix
        return f"{base}|{action}"[:31]

    def _ensure_connected(self) -> None:
        if not self.connected:
            self.connect()

    def connect(self) -> None:
        if self.connected:
            return
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 package is required for mt5_demo/mt5_live execution modes.")

        mt5 = get_mt5_module()
        init_kwargs: Dict[str, Any] = {}
        if self.terminal_path:
            init_kwargs["path"] = self.terminal_path

        if not mt5.initialize(**init_kwargs):
            raise RuntimeError(f"Failed to initialize MT5: {mt5.last_error()}")

        if self.login and self.password and self.server:
            requested_login = safe_int(self.login, 0)
            current_account = mt5.account_info()
            current_login = safe_int(getattr(current_account, "login", 0), 0)
            current_server = str(getattr(current_account, "server", "") or "").strip().lower()
            requested_server = str(self.server or "").strip().lower()

            needs_login = current_account is None or current_login != requested_login
            if requested_server and current_server and current_server != requested_server:
                needs_login = True

            if needs_login:
                if not mt5.login(login=requested_login, password=self.password, server=self.server):
                    raise RuntimeError(f"Failed MT5 login: {mt5.last_error()}")

        account_info = mt5.account_info()
        if account_info is not None:
            self.account_id = str(getattr(account_info, "login", "") or "")
            self.server_name = str(getattr(account_info, "server", "") or "")

        terminal_info = mt5.terminal_info()
        terminal_trade_allowed = bool(getattr(terminal_info, "trade_allowed", False)) if terminal_info is not None else False
        terminal_tradeapi_disabled = bool(getattr(terminal_info, "tradeapi_disabled", False)) if terminal_info is not None else False
        account_trade_allowed = bool(getattr(account_info, "trade_allowed", False)) if account_info is not None else False
        account_trade_expert = bool(getattr(account_info, "trade_expert", False)) if account_info is not None else False

        if not terminal_trade_allowed:
            raise RuntimeError(
                "MT5 terminal Algo Trading is disabled (terminal.trade_allowed=False). "
                "Enable the toolbar Algo Trading button and Expert Advisors automation settings."
            )
        if terminal_tradeapi_disabled:
            raise RuntimeError(
                "MT5 terminal has external Python API trading disabled (terminal.tradeapi_disabled=True). "
                "Disable this restriction in Tools -> Options -> Expert Advisors."
            )
        if not account_trade_allowed or not account_trade_expert:
            raise RuntimeError(
                "MT5 account does not permit expert trading (trade_allowed/trade_expert is false). "
                "Verify broker account permissions and terminal Expert Advisors settings."
            )

        available_symbols: List[str] = []
        symbols = mt5.symbols_get()
        if symbols is not None:
            available_symbols = [str(getattr(s, "name", "") or "") for s in symbols]
        self.resolved_symbol = resolve_symbol(self.requested_symbol, available_symbols)

        if not mt5.symbol_select(self.resolved_symbol, True):
            raise RuntimeError(f"Failed to select MT5 symbol '{self.resolved_symbol}'")
        if mt5.symbol_info(self.resolved_symbol) is None:
            raise RuntimeError(f"Resolved MT5 symbol '{self.resolved_symbol}' is unavailable")

        self.connected = True
        self.logger.info(
            "MT5 adapter connected mode=%s account=%s server=%s symbol=%s",
            self.mode,
            self.account_id or "unknown",
            self.server_name or "unknown",
            self.resolved_symbol,
        )

    def shutdown(self) -> None:
        if not MT5_AVAILABLE:
            self.connected = False
            return
        if self.connected:
            mt5 = get_mt5_module()
            mt5.shutdown()
        self.connected = False

    def get_quote(self, symbol: str) -> Quote:
        self._ensure_connected()
        mt5 = get_mt5_module()
        tick = mt5.symbol_info_tick(self.resolved_symbol)
        if tick is None:
            raise RuntimeError(f"No tick data available for {self.resolved_symbol}")

        bid = safe_float(getattr(tick, "bid", 0.0), 0.0)
        ask = safe_float(getattr(tick, "ask", 0.0), 0.0)
        if bid <= 0.0 or ask <= 0.0:
            raise RuntimeError(f"Invalid tick values for {self.resolved_symbol}: bid={bid} ask={ask}")
        pip_size = pip_size_for_symbol(self.requested_symbol)
        spread_pips = float((ask - bid) / pip_size)
        return Quote(
            symbol=self.requested_symbol,
            bid=bid,
            ask=ask,
            spread_pips=spread_pips,
            ts_utc=tick_timestamp_iso(tick),
        )

    def _managed_positions(self) -> List[Any]:
        self._ensure_connected()
        mt5 = get_mt5_module()
        positions = mt5.positions_get(symbol=self.resolved_symbol)
        if positions is None:
            return []

        filtered: List[Any] = []
        for position in positions:
            p_magic = safe_int(getattr(position, "magic", 0), 0)
            p_comment = str(getattr(position, "comment", "") or "")
            if p_magic == self.magic_number or p_comment.startswith(self.order_comment_prefix):
                filtered.append(position)
        return filtered

    def submit_order(self, request: OrderRequest) -> OrderResult:
        self._ensure_connected()
        mt5 = get_mt5_module()
        quote = self.get_quote(request.symbol)

        side = str(request.side).strip().lower()
        if side not in {"buy", "sell"}:
            return OrderResult(
                accepted=False,
                broker_order_id="",
                broker_position_id="",
                broker_deal_id="",
                requested_price=None,
                fill_price=None,
                retcode="MT5_INVALID_SIDE",
                retcode_detail=f"Unsupported side '{request.side}'",
                raw_status="rejected",
            )

        order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
        request_price = quote.ask if side == "buy" else quote.bid
        payload: Dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.resolved_symbol,
            "volume": float(request.qty),
            "type": order_type,
            "price": float(request_price),
            "sl": float(request.sl) if request.sl is not None else 0.0,
            "tp": float(request.tp) if request.tp is not None else 0.0,
            "deviation": int(max(self.deviation_points, 1)),
            "magic": int(self.magic_number),
            "comment": str(request.comment or self._build_comment("open"))[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(payload)
        if result is None:
            return OrderResult(
                accepted=False,
                broker_order_id="",
                broker_position_id="",
                broker_deal_id="",
                requested_price=float(request_price),
                fill_price=None,
                retcode="MT5_ORDER_SEND_NONE",
                retcode_detail=str(mt5.last_error()),
                raw_status="rejected",
            )

        retcode = safe_int(getattr(result, "retcode", 0), 0)
        accepted = is_success_retcode(retcode)
        fill_price = safe_float(getattr(result, "price", 0.0), 0.0)
        return OrderResult(
            accepted=accepted,
            broker_order_id=str(getattr(result, "order", "") or ""),
            broker_position_id=str(getattr(result, "position", "") or ""),
            broker_deal_id=str(getattr(result, "deal", "") or ""),
            requested_price=float(request_price),
            fill_price=fill_price if fill_price > 0.0 else None,
            retcode=retcode_label(retcode),
            retcode_detail=str(getattr(result, "comment", "") or ""),
            raw_status="filled" if accepted else "rejected",
        )

    def get_open_position(self, symbol: str) -> PositionState:
        positions = self._managed_positions()
        if not positions:
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

        oldest = min(positions, key=lambda p: safe_int(getattr(p, "time", 0), 0))
        mt5 = get_mt5_module()
        side = "long" if safe_int(getattr(oldest, "type", 0), 0) == int(mt5.ORDER_TYPE_BUY) else "short"
        entry_ts = utc_iso(safe_float(getattr(oldest, "time", 0.0), 0.0))
        return PositionState(
            is_open=True,
            side=side,
            qty=safe_float(getattr(oldest, "volume", 0.0), 0.0),
            entry_price=safe_float(getattr(oldest, "price_open", 0.0), 0.0),
            broker_position_id=str(getattr(oldest, "ticket", "") or ""),
            broker_order_id=str(getattr(oldest, "identifier", "") or ""),
            entry_ts_utc=entry_ts,
            unrealized_pnl_usd=safe_float(getattr(oldest, "profit", 0.0), 0.0),
        )

    def close_position(self, symbol: str) -> OrderResult:
        positions = self._managed_positions()
        if not positions:
            return OrderResult(
                accepted=False,
                broker_order_id="",
                broker_position_id="",
                broker_deal_id="",
                requested_price=None,
                fill_price=None,
                retcode="MT5_NO_POSITION",
                retcode_detail="No managed position found.",
                raw_status="rejected",
            )

        mt5 = get_mt5_module()
        oldest = min(positions, key=lambda p: safe_int(getattr(p, "time", 0), 0))
        position_ticket = safe_int(getattr(oldest, "ticket", 0), 0)
        position_type = safe_int(getattr(oldest, "type", 0), 0)
        volume = safe_float(getattr(oldest, "volume", 0.0), 0.0)
        quote = self.get_quote(symbol)

        close_type = mt5.ORDER_TYPE_SELL if position_type == int(mt5.ORDER_TYPE_BUY) else mt5.ORDER_TYPE_BUY
        close_price = quote.bid if position_type == int(mt5.ORDER_TYPE_BUY) else quote.ask
        payload: Dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.resolved_symbol,
            "volume": float(volume),
            "type": close_type,
            "position": int(position_ticket),
            "price": float(close_price),
            "deviation": int(max(self.deviation_points, 1)),
            "magic": int(self.magic_number),
            "comment": self._build_comment("close"),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(payload)
        if result is None:
            return OrderResult(
                accepted=False,
                broker_order_id="",
                broker_position_id=str(position_ticket),
                broker_deal_id="",
                requested_price=float(close_price),
                fill_price=None,
                retcode="MT5_ORDER_SEND_NONE",
                retcode_detail=str(mt5.last_error()),
                raw_status="rejected",
            )

        retcode = safe_int(getattr(result, "retcode", 0), 0)
        accepted = is_success_retcode(retcode)
        fill_price = safe_float(getattr(result, "price", 0.0), 0.0)
        return OrderResult(
            accepted=accepted,
            broker_order_id=str(getattr(result, "order", "") or ""),
            broker_position_id=str(position_ticket),
            broker_deal_id=str(getattr(result, "deal", "") or ""),
            requested_price=float(close_price),
            fill_price=fill_price if fill_price > 0.0 else float(close_price),
            retcode=retcode_label(retcode),
            retcode_detail=str(getattr(result, "comment", "") or ""),
            raw_status="filled" if accepted else "rejected",
        )

    def get_account_balance(self) -> float:
        self._ensure_connected()
        mt5 = get_mt5_module()
        account_info = mt5.account_info()
        return safe_float(getattr(account_info, "balance", 0.0), 0.0)
