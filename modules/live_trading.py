"""
Adapter-driven live trading orchestrator.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
import pickle
import sys
import time
import zlib
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from .config import (
        DATA_FEED_HEARTBEAT_SECONDS,
        GCAPI_CONFIG,
        MAX_POSITION_SIZE,
        MIN_POSITION_SIZE,
        MT5_CONFIG,
        PROFIT_TARGET,
        STOP_LOSS,
        STRATEGY_1_LIVE_MAX_SPREAD,
        STRATEGY_1_NAME,
    )
except ImportError:  # pragma: no cover - script mode fallback
    from config import (  # type: ignore[no-redef]
        DATA_FEED_HEARTBEAT_SECONDS,
        GCAPI_CONFIG,
        MAX_POSITION_SIZE,
        MIN_POSITION_SIZE,
        MT5_CONFIG,
        PROFIT_TARGET,
        STOP_LOSS,
        STRATEGY_1_LIVE_MAX_SPREAD,
        STRATEGY_1_NAME,
    )

try:
    from .execution.base_adapter import ExecutionAdapter, OrderRequest, PositionState, Quote
    from .execution.gcapi_adapter import GCAPIAdapter
    from .execution.mt5_adapter import MT5Adapter
    from .execution.paper_adapter import PaperAdapter
except ImportError:  # pragma: no cover - script mode fallback
    from execution.base_adapter import ExecutionAdapter, OrderRequest, PositionState, Quote  # type: ignore[no-redef]
    from execution.gcapi_adapter import GCAPIAdapter  # type: ignore[no-redef]
    from execution.mt5_adapter import MT5Adapter  # type: ignore[no-redef]
    from execution.paper_adapter import PaperAdapter  # type: ignore[no-redef]

try:
    from .strategy_1_event_logger import Strategy1CanonicalEventLogger
except ImportError:  # pragma: no cover - script mode fallback
    from strategy_1_event_logger import Strategy1CanonicalEventLogger  # type: ignore[no-redef]


logger = logging.getLogger(__name__)

EXECUTION_MODE_ALIASES = {
    "live": "mt5_live",
    "mt5": "mt5_demo",
    "demo": "mt5_demo",
    "gcapi": "gcapi_demo",
    "forexcom": "gcapi_demo",
    "cityindex": "gcapi_demo",
}
SUPPORTED_EXECUTION_MODES = {"paper", "mt5_demo", "mt5_live", "gcapi_demo", "gcapi_live"}


class _FallbackPaperModel:
    """Deterministic policy used when model loading fails in paper mode."""

    def __init__(self) -> None:
        self._step = 0

    def predict(self, observation: Any, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        self._step += 1
        if self._step % 7 == 0:
            action = 3  # close
        elif self._step % 3 == 0:
            action = 2  # sell
        elif self._step % 2 == 0:
            action = 1  # buy
        else:
            action = 0  # hold
        return np.array([action], dtype=np.int32), None


class LiveTradingEnvironment:
    def __init__(
        self,
        symbol: str = MT5_CONFIG["MT5SYMBOL"],
        timeframe: str = MT5_CONFIG["MT5TIMEFRAME"],
        model_path: str = MT5_CONFIG["MODEL_PATH"],
    ) -> None:
        self.symbol = str(symbol).upper()
        self.timeframe = str(timeframe).upper()
        self.execution_mode = self._normalize_execution_mode(os.getenv("FX_EXECUTION_MODE", "paper"))
        self.event_timezone = str(os.getenv("FX_EVENT_TIMEZONE", "America/Chicago")).strip() or "America/Chicago"
        self.session_label = "".join(ch for ch in str(os.getenv("FX_SESSION_LABEL", "LONDON")).upper() if ch.isalnum()) or "LONDON"
        self.broker_name = str(os.getenv("FX_BROKER_NAME", self._default_broker_name())).strip()

        self.pip_size = 0.01 if self.symbol.endswith("JPY") else 0.0001
        self.position = 0  # -1 short, 0 flat, +1 long
        self.entry_price = 0.0
        self.position_size = float(MIN_POSITION_SIZE)
        self.trade_history: list[Dict[str, Any]] = []
        self.current_trade: Optional[Dict[str, Any]] = None
        self.trade_counter = 0
        self.session_trade_count = 0
        self.session_net_pnl_usd = 0.0
        self.max_trades_per_session = 1

        self.last_action: Optional[int] = None
        self.last_action_log = 0.0
        self.action_log_interval = 30.0
        self.last_feed_heartbeat = 0.0
        self.data_feed_heartbeat_seconds = max(1, int(DATA_FEED_HEARTBEAT_SECONDS))
        self.last_tick_time: Optional[str] = None
        self.last_quote: Optional[Quote] = None

        self.order_comment_prefix = str(os.getenv("FX_ORDER_COMMENT_PREFIX", "QFX-S1")).strip() or "QFX-S1"
        self.magic = self._resolve_magic_number()
        self.session_closed = False
        self._shutdown_done = False
        self.sb3_model_load_status = "unknown"

        self.event_logger = Strategy1CanonicalEventLogger(
            session_label=self.session_label,
            mode=self.execution_mode,
            symbol=self.symbol,
            broker=self.broker_name,
            timezone_name=self.event_timezone,
        )
        self.max_trades_per_session = int(self.event_logger.max_trades_per_session)

        self.adapter: ExecutionAdapter = self._build_adapter()
        self.adapter.connect()

        self.model = self._load_model(model_path)
        logger.info(
            "Runtime context python_executable=%s sb3_model_load=%s execution_mode=%s",
            sys.executable,
            self.sb3_model_load_status,
            self.execution_mode,
        )
        self._sync_position_state()

        session_start_fields: Dict[str, Any] = {
            "profile_name": self.event_logger.profile_name,
            "symbol": self.symbol,
            "mode": self.execution_mode,
            "execution_mode": self.execution_mode,
            "broker": self.broker_name,
            "timezone": self.event_timezone,
        }
        resolved_symbol = str(getattr(self.adapter, "resolved_symbol", self.symbol) or self.symbol)
        if resolved_symbol:
            session_start_fields["resolved_symbol"] = resolved_symbol
        account_id = str(getattr(self.adapter, "account_id", "") or "")
        if account_id:
            session_start_fields["account_id"] = account_id
        server_name = str(getattr(self.adapter, "server_name", "") or "")
        if server_name:
            session_start_fields["server"] = server_name

        self.event_logger.append("session_start", **session_start_fields)

        logger.info(
            "Initialized trading symbol=%s mode=%s model=%s run_id=%s",
            self.symbol,
            self.execution_mode,
            model_path,
            self.event_logger.run_id,
        )

    @staticmethod
    def _normalize_execution_mode(raw_mode: str) -> str:
        mode = str(raw_mode or "").strip().lower()
        mode = EXECUTION_MODE_ALIASES.get(mode, mode)
        if mode not in SUPPORTED_EXECUTION_MODES:
            return "paper"
        return mode

    def _default_broker_name(self) -> str:
        if self.execution_mode.startswith("gcapi"):
            return "forexcom_gcapi"
        if self.execution_mode == "paper":
            return "paper"
        return self.execution_mode

    def _resolve_magic_number(self) -> int:
        raw_magic = str(os.getenv("FX_MAGIC_NUMBER", "")).strip()
        if raw_magic:
            try:
                return int(raw_magic)
            except ValueError:
                logger.warning("Invalid FX_MAGIC_NUMBER=%r; using deterministic strategy magic.", raw_magic)
        strategy_key = f"{STRATEGY_1_NAME}:{self.symbol}".encode("utf-8")
        return int(MT5_CONFIG.get("MAGIC_BASE", 234000)) + int(zlib.crc32(strategy_key) % 1000)

    def _build_adapter(self) -> ExecutionAdapter:
        if self.execution_mode == "paper":
            start_price = float(str(os.getenv("FX_PAPER_START_PRICE", "1.08500")).strip() or "1.08500")
            spread = float(str(os.getenv("FX_PAPER_SPREAD", "0.00008")).strip() or "0.00008")
            tick_step = float(str(os.getenv("FX_PAPER_TICK_STEP", "0.00003")).strip() or "0.00003")
            return PaperAdapter(
                symbol=self.symbol,
                pip_size=self.pip_size,
                start_price=start_price,
                spread=spread,
                tick_step=tick_step,
                logger=logger,
            )

        if self.execution_mode in {"gcapi_demo", "gcapi_live"}:
            return GCAPIAdapter(
                symbol=self.symbol,
                mode=self.execution_mode,
                config=GCAPI_CONFIG,
                order_comment_prefix=self.order_comment_prefix,
                logger=logger,
            )

        return MT5Adapter(
            symbol=self.symbol,
            mode=self.execution_mode,
            config=MT5_CONFIG,
            magic_number=self.magic,
            order_comment_prefix=self.order_comment_prefix,
            logger=logger,
        )

    def _load_model(self, model_path: str) -> Any:
        flag = str(os.getenv("FX_DISABLE_SB3_MODEL_LOAD", "auto")).strip().lower()

        if flag == "true":
            disable_sb3 = True
        elif flag == "false":
            disable_sb3 = False
        else:  # auto
            disable_sb3 = sys.version_info >= (3, 13)

        if disable_sb3:
            logger.warning(
                "SB3 model load disabled (Python %s detected). Using fallback policy.",
                sys.version.split()[0],
            )
            if self.execution_mode in {"paper", "mt5_demo", "gcapi_demo"}:
                self.sb3_model_load_status = "disabled_fallback"
                return _FallbackPaperModel()
            raise RuntimeError(
                "SB3 model load is disabled for this runtime. Use Python 3.11 environment "
                "(.venv/.venv311) for live model execution, or set FX_DISABLE_SB3_MODEL_LOAD=false "
                "to force SB3 loading."
            )
        else:
            logger.info("Loading SB3 model: %s", model_path)

        try:
            from stable_baselines3 import PPO

            model = PPO.load(model_path)
            self.sb3_model_load_status = "enabled"
            return model
        except Exception as exc:
            logger.warning("Failed to load as stable-baselines3 model: %s", exc)
            try:
                with open(model_path, "rb") as handle:
                    model = pickle.load(handle)
                    self.sb3_model_load_status = "pickle_fallback"
                    return model
            except Exception as fallback_exc:
                if self.execution_mode in {"paper", "mt5_demo", "gcapi_demo"}:
                    logger.warning("Failed to load model from %s: %s", model_path, fallback_exc)
                    logger.warning("Using fallback policy for ingestion validation in %s mode.", self.execution_mode)
                    self.sb3_model_load_status = "load_failed_fallback"
                    return _FallbackPaperModel()
                raise RuntimeError(f"Cannot load model from {model_path}: {fallback_exc}") from fallback_exc

    @staticmethod
    def _utc_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _parse_utc(value: str) -> datetime:
        text = str(value or "").strip()
        if not text:
            return datetime.now(timezone.utc)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return datetime.now(timezone.utc)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _safe_get_quote(self, *, refresh: bool = True) -> Optional[Quote]:
        if not refresh and self.last_quote is not None:
            return self.last_quote
        try:
            quote = self.adapter.get_quote(self.symbol)
        except Exception as exc:
            logger.warning("Failed to fetch quote for %s: %s", self.symbol, exc)
            return None
        self.last_quote = quote
        return quote

    def _quote_last_price(self, quote: Optional[Quote]) -> Optional[float]:
        if quote is None:
            return None
        return float((quote.bid + quote.ask) / 2.0)

    def _spread_pips(self, quote: Optional[Quote]) -> float:
        if quote is None:
            return 0.0
        return float(quote.spread_pips)

    def _max_spread_pips(self) -> float:
        return float(float(STRATEGY_1_LIVE_MAX_SPREAD) / self.pip_size)

    def _next_trade_id(self) -> str:
        self.trade_counter += 1
        trade_day = datetime.now().strftime("%Y%m%d")
        return f"S1-{trade_day}-{self.trade_counter:03d}"

    def _calculate_position_size(self) -> float:
        if not self.trade_history:
            return float(MIN_POSITION_SIZE)
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for trade in self.trade_history if float(trade.get("profit", 0.0)) > 0.0)
        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0
        raw_size = float(MIN_POSITION_SIZE) + (float(MAX_POSITION_SIZE) - float(MIN_POSITION_SIZE)) * float(win_rate)
        return float(np.clip(raw_size, float(MIN_POSITION_SIZE), float(MAX_POSITION_SIZE)))

    def _emit_data_feed_alive(self, quote: Optional[Quote]) -> None:
        self.event_logger.append(
            "data_feed_alive",
            symbol=self.symbol,
            last_price=self._quote_last_price(quote),
        )

    def _emit_signal_candidate(self, action: int, quote: Optional[Quote]) -> None:
        if action not in {1, 2}:
            return
        side = "long" if action == 1 else "short"
        self.event_logger.append(
            "signal_evaluated",
            symbol=self.symbol,
            bar_ts=self._utc_iso(),
            side=side,
            signal_score=float(action),
            spread_pips=self._spread_pips(quote),
            decision="candidate",
            reason_code="signal_pass",
        )

    def _emit_skip(self, *, action: int, reason_code: str, quote: Optional[Quote], decision: str = "skip") -> None:
        side = "long" if action == 1 else "short"
        self.event_logger.append(
            "trade_skipped",
            symbol=self.symbol,
            bar_ts=self._utc_iso(),
            side=side,
            decision=decision,
            reason_code=reason_code,
            spread_pips=self._spread_pips(quote),
            max_spread_pips=self._max_spread_pips(),
        )

    def _sync_position_state(self) -> None:
        try:
            pos_state: PositionState = self.adapter.get_open_position(self.symbol)
        except Exception as exc:
            logger.warning("Failed to read open position: %s", exc)
            return

        if not pos_state.is_open:
            self.position = 0
            self.entry_price = 0.0
            self.current_trade = None
            return

        side = str(pos_state.side or "long").lower()
        self.position = 1 if side == "long" else -1
        self.entry_price = float(pos_state.entry_price or 0.0)

        if self.current_trade is None:
            derived_trade_id = f"S1-{datetime.now().strftime('%Y%m%d')}-{str(pos_state.broker_position_id or 'sync')}"
            self.current_trade = {
                "trade_id": derived_trade_id,
                "side": side,
                "qty": float(pos_state.qty or 0.0),
                "entry_price": float(pos_state.entry_price or 0.0),
                "entry_ts": self._parse_utc(str(pos_state.entry_ts_utc or self._utc_iso())),
                "broker_order_id": str(pos_state.broker_order_id or ""),
                "broker_position_id": str(pos_state.broker_position_id or ""),
            }
        else:
            self.current_trade["side"] = side
            self.current_trade["qty"] = float(pos_state.qty or self.current_trade.get("qty", 0.0))
            self.current_trade["entry_price"] = float(pos_state.entry_price or self.current_trade.get("entry_price", 0.0))
            self.current_trade["broker_order_id"] = str(pos_state.broker_order_id or self.current_trade.get("broker_order_id", ""))
            self.current_trade["broker_position_id"] = str(
                pos_state.broker_position_id or self.current_trade.get("broker_position_id", "")
            )

    def get_observation(self, quote: Optional[Quote] = None) -> np.ndarray:
        q = quote or self._safe_get_quote(refresh=True)
        if q is None:
            raise RuntimeError("Failed to obtain quote for observation")
        mid = float((q.bid + q.ask) / 2.0)
        return np.array([mid, mid + self.pip_size, mid - self.pip_size, mid, 100.0], dtype=np.float32)

    def _calculate_pnl_usd(self, side: str, entry_price: float, exit_price: float, qty: float) -> float:
        if side == "long":
            pips = (float(exit_price) - float(entry_price)) / self.pip_size
        else:
            pips = (float(entry_price) - float(exit_price)) / self.pip_size
        return float(pips * 10.0 * float(qty))

    def _open_position(self, side: str, position_size: float, quote: Optional[Quote]) -> bool:
        q = quote or self._safe_get_quote(refresh=True)
        if q is None:
            logger.error("No quote available; cannot open position.")
            return False

        is_buy = str(side).lower() == "buy"
        action = 1 if is_buy else 2
        expected_price = float(q.ask if is_buy else q.bid)
        canonical_side = "long" if is_buy else "short"
        trade_id = self._next_trade_id()
        client_order_id = f"{self.execution_mode}-S1-{self.trade_counter:03d}"

        if is_buy:
            sl = expected_price * (1 - float(STOP_LOSS))
            tp = expected_price * (1 + float(PROFIT_TARGET))
        else:
            sl = expected_price * (1 + float(STOP_LOSS))
            tp = expected_price * (1 - float(PROFIT_TARGET))

        self.event_logger.append(
            "order_submitted",
            symbol=self.symbol,
            trade_id=trade_id,
            client_order_id=client_order_id,
            side=canonical_side,
            qty=float(position_size),
            order_type="market",
            expected_entry=expected_price,
            sl=sl,
            tp=tp,
            risk_usd=0.0,
            broker_request_price=expected_price,
            deviation_points=int(MT5_CONFIG.get("MT5_DEVIATION_POINTS", MT5_CONFIG.get("BASE_DEVIATION", 20))),
            mt5_magic_number=int(self.magic),
        )

        result = self.adapter.submit_order(
            OrderRequest(
                symbol=self.symbol,
                side="buy" if is_buy else "sell",
                qty=float(position_size),
                order_type="market",
                sl=float(sl),
                tp=float(tp),
                comment=f"{self.order_comment_prefix}|open",
            )
        )

        if not result.accepted:
            self._emit_skip(action=action, reason_code="broker_api_failure", quote=q, decision="rejected")
            logger.error(
                "Order rejected side=%s retcode=%s detail=%s",
                side,
                result.retcode,
                result.retcode_detail,
            )
            return False

        fill_price = float(result.fill_price if result.fill_price is not None else expected_price)
        self.position = 1 if is_buy else -1
        self.entry_price = fill_price
        self.current_trade = {
            "trade_id": trade_id,
            "side": canonical_side,
            "qty": float(position_size),
            "entry_price": fill_price,
            "entry_ts": datetime.now(timezone.utc),
            "broker_order_id": str(result.broker_order_id or ""),
            "broker_position_id": str(result.broker_position_id or ""),
        }
        self.session_trade_count += 1

        self.event_logger.append(
            "order_filled",
            symbol=self.symbol,
            trade_id=trade_id,
            broker_order_id=str(result.broker_order_id or ""),
            broker_position_id=str(result.broker_position_id or ""),
            side=canonical_side,
            qty=float(position_size),
            fill_price=fill_price,
            slippage_pips=abs(fill_price - expected_price) / self.pip_size,
            spread_pips=self._spread_pips(q),
            retcode=result.retcode,
        )
        logger.info(
            "Opened %s position at %.5f size=%.2f trade_id=%s",
            canonical_side,
            fill_price,
            float(position_size),
            trade_id,
        )
        return True

    def _close_position(self, exit_reason: str = "manual_disable", quote: Optional[Quote] = None) -> bool:
        self._sync_position_state()
        if self.position == 0:
            return False
        if self.current_trade is None:
            logger.warning("Position exists but current_trade is missing; rebuilding from adapter state.")
            self._sync_position_state()
            if self.current_trade is None:
                return False

        result = self.adapter.close_position(self.symbol)
        if not result.accepted:
            logger.error("Failed to close position retcode=%s detail=%s", result.retcode, result.retcode_detail)
            return False

        q = quote or self._safe_get_quote(refresh=False) or self._safe_get_quote(refresh=True)
        side = str(self.current_trade.get("side", "long"))
        entry_price = float(self.current_trade.get("entry_price", self.entry_price))
        qty = float(self.current_trade.get("qty", self.position_size))
        trade_id = str(self.current_trade.get("trade_id", self._next_trade_id()))
        entry_ts_raw = self.current_trade.get("entry_ts", datetime.now(timezone.utc))
        entry_ts = entry_ts_raw if isinstance(entry_ts_raw, datetime) else self._parse_utc(str(entry_ts_raw))

        if result.fill_price is not None:
            exit_price = float(result.fill_price)
        elif q is not None:
            exit_price = float(q.bid if side == "long" else q.ask)
        else:
            exit_price = float(entry_price)

        pnl_usd = self._calculate_pnl_usd(side, entry_price, exit_price, qty)
        hold_seconds = max(0, int((datetime.now(timezone.utc) - entry_ts).total_seconds()))

        self.trade_history.append(
            {
                "type": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "profit": pnl_usd,
                "timestamp": datetime.now(),
            }
        )
        self.session_net_pnl_usd += pnl_usd
        self.event_logger.append(
            "position_closed",
            symbol=self.symbol,
            trade_id=trade_id,
            side=side,
            exit_reason=exit_reason,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_usd=pnl_usd,
            hold_seconds=hold_seconds,
            fees_usd=0.0,
            broker_order_id=str(result.broker_order_id or ""),
            broker_position_id=str(result.broker_position_id or ""),
            broker_deal_id=str(result.broker_deal_id or ""),
        )

        self.position = 0
        self.entry_price = 0.0
        self.current_trade = None
        logger.info("Closed %s at %.5f pnl_usd=%.2f trade_id=%s", side, exit_price, pnl_usd, trade_id)
        return True

    def _check_profit_targets(self, quote: Optional[Quote]) -> Tuple[bool, Optional[str], Optional[float]]:
        self._sync_position_state()
        if self.position == 0 or self.entry_price == 0.0 or quote is None:
            return False, None, None

        current_price = float(quote.bid if self.position > 0 else quote.ask)
        if self.position > 0:
            loss_price = self.entry_price * (1 - float(STOP_LOSS))
            profit_price = self.entry_price * (1 + float(PROFIT_TARGET))
            if current_price <= loss_price:
                return True, "stop_loss", current_price
            if current_price >= profit_price:
                return True, "take_profit", current_price
        else:
            loss_price = self.entry_price * (1 + float(STOP_LOSS))
            profit_price = self.entry_price * (1 - float(PROFIT_TARGET))
            if current_price >= loss_price:
                return True, "stop_loss", current_price
            if current_price <= profit_price:
                return True, "take_profit", current_price
        return False, None, current_price

    def execute_trade(self, action: int, quote: Optional[Quote] = None) -> None:
        q = quote or self._safe_get_quote(refresh=True)
        if q is None:
            logger.warning("No quote available; skipping action.")
            return

        self._sync_position_state()
        if action == 0:
            return

        live_spread = float(q.ask - q.bid)
        entry_blocked_by_spread = live_spread > float(STRATEGY_1_LIVE_MAX_SPREAD)

        if action in {1, 2} and self.position == 0 and self.session_trade_count >= self.max_trades_per_session:
            self._emit_skip(action=action, reason_code="session_cap", quote=q)
            logger.info("Skip entry due to Strategy 1 session cap.")
            return

        if action == 1 and self.position <= 0:
            if self.position < 0:
                if self._close_position(exit_reason="manual_disable", quote=q):
                    logger.info("Closed short; waiting before opening long.")
                return
            if entry_blocked_by_spread:
                self._emit_skip(action=action, reason_code="spread_gate", quote=q)
                logger.info(
                    "Skip BUY entry [%s] spread %.5f > %.5f",
                    STRATEGY_1_NAME,
                    live_spread,
                    float(STRATEGY_1_LIVE_MAX_SPREAD),
                )
                return
            self.position_size = self._calculate_position_size()
            self._open_position("buy", self.position_size, q)
            return

        if action == 1 and self.position > 0:
            self._emit_skip(action=action, reason_code="duplicate_signal", quote=q)
            logger.info("Action BUY ignored (already long)")
            return

        if action == 2 and self.position >= 0:
            if self.position > 0:
                if self._close_position(exit_reason="manual_disable", quote=q):
                    logger.info("Closed long; waiting before opening short.")
                return
            if entry_blocked_by_spread:
                self._emit_skip(action=action, reason_code="spread_gate", quote=q)
                logger.info(
                    "Skip SELL entry [%s] spread %.5f > %.5f",
                    STRATEGY_1_NAME,
                    live_spread,
                    float(STRATEGY_1_LIVE_MAX_SPREAD),
                )
                return
            self.position_size = self._calculate_position_size()
            self._open_position("sell", self.position_size, q)
            return

        if action == 2 and self.position < 0:
            self._emit_skip(action=action, reason_code="duplicate_signal", quote=q)
            logger.info("Action SELL ignored (already short)")
            return

        if action == 3 and self.position != 0:
            self._close_position(exit_reason="manual_disable", quote=q)
            return

        if action == 3 and self.position == 0:
            logger.info("Action CLOSE ignored (no open position)")

    def _log_action(self, action: int, price: float) -> None:
        now = time.time()
        if self.last_action is None or action != self.last_action or (now - self.last_action_log) >= self.action_log_interval:
            action_names = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}
            logger.info("Action: %s | position=%s | price=%.5f", action_names.get(action, action), self.position, price)
            self.last_action = action
            self.last_action_log = now

    @staticmethod
    def _action_to_int(action: Any) -> int:
        action_arr = np.asarray(action)
        if action_arr.size == 1:
            return int(action_arr.item())
        return int(action_arr.reshape(-1)[0])

    def run(self, max_duration_seconds: int = 1800) -> None:
        logger.info(
            "Starting trading symbol=%s mode=%s max_duration=%ss",
            self.symbol,
            self.execution_mode,
            max_duration_seconds,
        )
        start_time = time.time()

        while True:
            try:
                if (time.time() - start_time) >= max_duration_seconds:
                    logger.info("Max duration reached, stopping.")
                    break

                quote = self._safe_get_quote(refresh=True)
                loop_now = time.time()
                if (loop_now - self.last_feed_heartbeat) >= float(self.data_feed_heartbeat_seconds):
                    self._emit_data_feed_alive(quote)
                    self.last_feed_heartbeat = loop_now

                should_close, reason, price = self._check_profit_targets(quote)
                if should_close:
                    logger.info("Auto-close triggered (%s) at %.5f", reason, float(price or 0.0))
                    self._close_position(exit_reason=str(reason or "risk_rule"), quote=quote)
                    self.wait_for_next_tick()
                    time.sleep(0.1)
                    continue

                if quote is None:
                    self.wait_for_next_tick()
                    time.sleep(0.1)
                    continue

                observation = self.get_observation(quote)
                action_raw, _ = self.model.predict(observation, deterministic=True)
                action = self._action_to_int(action_raw)
                self._log_action(action, float(observation[3]))
                self._emit_signal_candidate(action=action, quote=quote)
                self.execute_trade(action, quote=quote)
                self.wait_for_next_tick()

            except Exception as exc:
                logger.error("Error in trading loop: %s", exc, exc_info=True)

            time.sleep(0.1)

    def wait_for_next_tick(self) -> None:
        max_wait_seconds = 1.0 if self.execution_mode == "paper" else 10.0
        deadline = time.time() + max_wait_seconds
        while True:
            quote = self._safe_get_quote(refresh=True)
            if quote is not None and (self.last_tick_time is None or quote.ts_utc != self.last_tick_time):
                self.last_tick_time = quote.ts_utc
                return
            if time.time() >= deadline:
                if quote is not None:
                    self.last_tick_time = quote.ts_utc
                return
            time.sleep(0.1)

    def _get_account_balance(self) -> float:
        if hasattr(self.adapter, "get_account_balance"):
            try:
                return float(getattr(self.adapter, "get_account_balance")())
            except Exception:
                return 0.0
        return 0.0

    def close_session(self, status: str = "completed") -> None:
        if self.session_closed:
            return
        self.session_closed = True
        self.event_logger.append(
            "session_end",
            symbol=self.symbol,
            trades=int(self.session_trade_count),
            net_pnl_usd=float(self.session_net_pnl_usd),
            status=status,
        )

    def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        try:
            self.adapter.shutdown()
        except Exception as exc:
            logger.error("Adapter shutdown failed: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    trader: Optional[LiveTradingEnvironment] = None
    session_status = "completed"
    try:
        trader = LiveTradingEnvironment(
            symbol=str(MT5_CONFIG["MT5SYMBOL"]),
            timeframe=str(MT5_CONFIG["MT5TIMEFRAME"]),
            model_path=str(MT5_CONFIG["MODEL_PATH"]),
        )
        max_duration = int(str(os.getenv("FX_MAX_DURATION_SECONDS", "1800")).strip() or "1800")
        trader.run(max_duration_seconds=max(30, max_duration))
    except KeyboardInterrupt:
        logger.info("Trading stopped by user")
        session_status = "stopped"
    except Exception as exc:
        logger.error("Error in main loop: %s", exc, exc_info=True)
        session_status = "error"
    finally:
        if trader is not None:
            try:
                trader.close_session(status=session_status)
            except Exception as close_exc:
                logger.error("Failed to emit session_end event: %s", close_exc)
            trader.shutdown()
