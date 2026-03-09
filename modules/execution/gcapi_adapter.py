from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import ParseResult, urlparse

import requests

from .base_adapter import OrderRequest, OrderResult, PositionState, Quote
from .mt5_utils import pip_size_for_symbol, safe_float, safe_int, utc_iso


def _normalize_symbol(value: str) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def _market_search_queries(symbol: str) -> List[str]:
    norm = _normalize_symbol(symbol)
    if len(norm) == 6:
        base = norm[:3]
        quote = norm[3:]
        return [f"{base}/{quote}", norm, f"{base} {quote}"]
    return [norm]


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _as_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("Markets", "OpenPositions", "Positions", "Orders", "Data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


class GCAPIAdapter:
    def __init__(
        self,
        *,
        symbol: str,
        mode: str,
        config: Dict[str, Any],
        order_comment_prefix: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.requested_symbol = _normalize_symbol(symbol)
        self.mode = str(mode or "").strip().lower()
        self.config = dict(config or {})
        self.order_comment_prefix = str(order_comment_prefix or "QFX-S1").strip() or "QFX-S1"
        self.logger = logger or logging.getLogger(__name__)

        self.base_url = self._normalize_base_url(str(self.config.get("GCAPI_BASE_URL", "")).strip())
        self.username = str(self.config.get("GCAPI_USERNAME", "")).strip()
        self.password = str(self.config.get("GCAPI_PASSWORD", "")).strip()
        self.app_key = str(self.config.get("GCAPI_APP_KEY", "")).strip()
        self.app_version = str(self.config.get("GCAPI_APP_VERSION", "1")).strip() or "1"
        self.app_comments = str(self.config.get("GCAPI_APP_COMMENTS", "QuasarFX_Strategy1")).strip() or "QuasarFX_Strategy1"
        self.order_reference = str(self.config.get("GCAPI_ORDER_REFERENCE", "")).strip()
        self.source = str(self.config.get("GCAPI_SOURCE", "")).strip()
        self.timeout_seconds = max(3, safe_int(self.config.get("GCAPI_TIMEOUT_SECONDS", 15), 15))
        self.quantity_multiplier = max(1.0, safe_float(self.config.get("GCAPI_QUANTITY_MULTIPLIER", 100000), 100000.0))

        self.client_account_id = safe_int(self.config.get("GCAPI_CLIENT_ACCOUNT_ID", 0), 0)
        self.trading_account_id = safe_int(self.config.get("GCAPI_TRADING_ACCOUNT_ID", 0), 0)

        self.connected = False
        self.session_token = ""
        self.session_user_name = ""
        self.resolved_symbol = self.requested_symbol
        self.resolved_market_id = 0
        self.account_id = ""
        self.server_name = ""

        self._http = requests.Session()
        self._last_quote: Optional[Quote] = None

    @staticmethod
    def _normalize_base_url(raw_url: str) -> str:
        default_url = "https://ciapi.cityindex.com/TradingAPI"
        value = (raw_url or default_url).strip()
        if "://" not in value:
            value = f"https://{value.lstrip('/')}"

        parsed = urlparse(value)
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc
        path = (parsed.path or "").rstrip("/")
        if not path.lower().endswith("/tradingapi"):
            path = f"{path}/TradingAPI" if path else "/TradingAPI"
        rebuilt = ParseResult(
            scheme=scheme,
            netloc=netloc,
            path=path,
            params="",
            query="",
            fragment="",
        )
        return rebuilt.geturl().rstrip("/")

    def _absolute_url(self, path: str) -> str:
        segment = f"/{str(path or '').lstrip('/')}"
        return f"{self.base_url}{segment}"

    def _headers(self, *, include_auth: bool) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
        }
        if self.app_key:
            headers["AppKey"] = self.app_key
        if include_auth and self.session_token:
            headers["Session"] = self.session_token
            headers["UserName"] = self.session_user_name or self.username
        return headers

    def _request_json(
        self,
        *,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_payload: Optional[Dict[str, Any]] = None,
        include_auth: bool,
        retry_on_401: bool = True,
    ) -> Any:
        url = self._absolute_url(path)
        try:
            response = self._http.request(
                method=str(method).upper(),
                url=url,
                params=params,
                json=json_payload,
                headers=self._headers(include_auth=include_auth),
                timeout=float(self.timeout_seconds),
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"GCAPI request failed {method.upper()} {path}: {exc}") from exc

        if response.status_code == 401 and include_auth and retry_on_401:
            self.logger.warning("GCAPI session expired; re-authenticating and retrying %s %s", method.upper(), path)
            self._login()
            return self._request_json(
                method=method,
                path=path,
                params=params,
                json_payload=json_payload,
                include_auth=include_auth,
                retry_on_401=False,
            )

        parsed: Any
        body_text = response.text or ""
        try:
            parsed = response.json() if body_text else {}
        except ValueError:
            parsed = {"raw_body": body_text[:1000]}

        if response.status_code >= 400:
            reason = _safe_text(getattr(response, "reason", ""))
            parsed_text = _safe_text(parsed)
            body_text_trimmed = (body_text or "").strip()[:1000]
            detail = " | ".join(part for part in (reason, parsed_text, body_text_trimmed) if part)
            raise RuntimeError(f"GCAPI HTTP {response.status_code} on {method.upper()} {path}: {detail}")
        return parsed

    def _ensure_connected(self) -> None:
        if not self.connected:
            self.connect()

    def _login(self) -> None:
        if not self.username or not self.password or not self.app_key:
            raise RuntimeError(
                "GCAPI credentials missing. Set GCAPI_USERNAME, GCAPI_PASSWORD, and GCAPI_APP_KEY."
            )
        payload = {
            "UserName": self.username,
            "Password": self.password,
            "AppVersion": self.app_version,
            "AppComments": self.app_comments,
            "AppKey": self.app_key,
        }
        response = self._request_json(
            method="POST",
            path="/session",
            json_payload=payload,
            include_auth=False,
            retry_on_401=False,
        )
        if not isinstance(response, dict):
            raise RuntimeError("GCAPI session login returned an invalid payload.")

        status_code = safe_int(response.get("StatusCode", 1), 1)
        if status_code != 1:
            message = _safe_text(response.get("ErrorMessage")) or _safe_text(response.get("StatusMessage")) or _safe_text(response)
            raise RuntimeError(f"GCAPI session login failed: {message}")

        session = _safe_text(response.get("Session"))
        if not session:
            raise RuntimeError("GCAPI session login succeeded but did not return a Session token.")
        self.session_token = session
        self.session_user_name = _safe_text(response.get("UserName")) or self.username

    def _resolve_account_context(self) -> None:
        response = self._request_json(
            method="GET",
            path="/UserAccount/ClientAndTradingAccount",
            include_auth=True,
        )
        if not isinstance(response, dict):
            raise RuntimeError("GCAPI account context returned an invalid payload.")

        if self.client_account_id <= 0:
            self.client_account_id = safe_int(response.get("ClientAccountId", 0), 0)

        trading_accounts = response.get("TradingAccounts")
        candidates = [x for x in trading_accounts if isinstance(x, dict)] if isinstance(trading_accounts, list) else []
        if self.trading_account_id <= 0:
            for account in candidates:
                candidate_id = safe_int(account.get("TradingAccountId", 0), 0)
                if candidate_id > 0:
                    self.trading_account_id = candidate_id
                    break

        if self.client_account_id <= 0:
            for account in candidates:
                candidate_id = safe_int(account.get("ClientAccountId", 0), 0)
                if candidate_id > 0:
                    self.client_account_id = candidate_id
                    break

        if self.trading_account_id <= 0:
            raise RuntimeError("GCAPI did not return a valid TradingAccountId.")

        self.account_id = str(self.trading_account_id)
        self.server_name = urlparse(self.base_url).netloc

    @staticmethod
    def _market_identity_strings(market: Dict[str, Any]) -> Iterable[str]:
        for key in ("Name", "DisplayName", "MarketName", "Symbol", "Epic"):
            value = _safe_text(market.get(key))
            if value:
                yield value

    def _market_score(self, market: Dict[str, Any], target: str) -> int:
        score = 0
        target_norm = _normalize_symbol(target)
        symbols = list(self._market_identity_strings(market))
        for item in symbols:
            item_norm = _normalize_symbol(item)
            if item_norm == target_norm:
                score = max(score, 100)
            elif target_norm and target_norm in item_norm:
                score = max(score, 70)
        if len(target_norm) == 6:
            base = target_norm[:3]
            quote = target_norm[3:]
            joined = " ".join(symbols).upper()
            if base in joined and quote in joined:
                score = max(score, 60)
        return score

    def _resolve_market(self) -> Tuple[int, str]:
        best_market: Optional[Dict[str, Any]] = None
        best_score = -1
        for query in _market_search_queries(self.requested_symbol):
            response = self._request_json(
                method="GET",
                path="/market/search",
                params={"Query": query, "MaxResults": 60},
                include_auth=True,
            )
            for market in _as_list(response):
                market_id = safe_int(market.get("MarketId", 0), 0)
                if market_id <= 0:
                    continue
                score = self._market_score(market, self.requested_symbol)
                if score > best_score:
                    best_score = score
                    best_market = market

            if best_market is not None and best_score >= 100:
                break

        if best_market is None:
            raise RuntimeError(f"GCAPI could not resolve market for symbol '{self.requested_symbol}'.")

        market_id = safe_int(best_market.get("MarketId", 0), 0)
        resolved_symbol = ""
        for identity in self._market_identity_strings(best_market):
            resolved_symbol = identity
            break
        return market_id, resolved_symbol or self.requested_symbol

    def connect(self) -> None:
        if self.connected:
            return

        self._login()
        self._resolve_account_context()
        market_id, symbol_name = self._resolve_market()
        if market_id <= 0:
            raise RuntimeError(f"Failed to resolve GCAPI market id for {self.requested_symbol}.")

        self.resolved_market_id = market_id
        self.resolved_symbol = symbol_name
        self.connected = True
        self.logger.info(
            "GCAPI adapter connected mode=%s account=%s symbol=%s market_id=%s",
            self.mode,
            self.account_id or "unknown",
            self.resolved_symbol,
            self.resolved_market_id,
        )

    def shutdown(self) -> None:
        try:
            if self.session_token:
                self._request_json(
                    method="DELETE",
                    path="/session",
                    include_auth=True,
                    retry_on_401=False,
                )
        except Exception:
            pass
        finally:
            try:
                self._http.close()
            except Exception:
                pass
            self.connected = False
            self.session_token = ""
            self.session_user_name = ""

    def _extract_quote_from_information_extended(self, payload: Any) -> Optional[Tuple[float, float, str]]:
        if not isinstance(payload, dict):
            return None
        prices: Any = None
        if isinstance(payload.get("prices"), dict):
            prices = payload.get("prices")
        elif isinstance(payload.get("marketInformation"), dict) and isinstance(payload.get("marketInformation", {}).get("prices"), dict):
            prices = payload.get("marketInformation", {}).get("prices")
        else:
            prices = payload
        if not isinstance(prices, dict):
            return None

        bid = safe_float(prices.get("bidPrice", prices.get("Bid")), 0.0)
        ask = safe_float(prices.get("offerPrice", prices.get("Ask")), 0.0)
        if bid <= 0.0 or ask <= 0.0:
            return None

        ts_raw = (
            _safe_text(prices.get("lastPriceTimestampUTC"))
            or _safe_text(prices.get("lastUpdateTime"))
            or _safe_text(payload.get("PriceTimestampUTC"))
            or _safe_text(payload.get("marketInformation", {}).get("lastChangedDateTimeUTC") if isinstance(payload.get("marketInformation"), dict) else "")
        )
        ts_value = ts_raw if ts_raw else utc_iso()
        return bid, ask, ts_value

    @staticmethod
    def _extract_tick_price(payload: Any) -> Optional[Tuple[float, str]]:
        if not isinstance(payload, dict):
            return None

        tick_rows = payload.get("PriceTicks")
        if not isinstance(tick_rows, list) or not tick_rows:
            return None
        first = tick_rows[0]
        if not isinstance(first, dict):
            return None

        price = safe_float(first.get("Price", first.get("price")), 0.0)
        if price <= 0.0:
            return None
        ts_raw = _safe_text(first.get("TickDate", first.get("tickDate")))
        return price, (ts_raw or utc_iso())

    def _tick_price(self, price_type: str) -> Optional[Tuple[float, str]]:
        response = self._request_json(
            method="GET",
            path=f"/market/{self.resolved_market_id}/tickhistory",
            params={"PriceTicks": 1, "priceType": str(price_type).upper()},
            include_auth=True,
        )
        return self._extract_tick_price(response)

    def get_quote(self, symbol: str) -> Quote:
        self._ensure_connected()

        params: Dict[str, Any] = {}
        if self.client_account_id > 0:
            params["clientAccountId"] = self.client_account_id

        payload = self._request_json(
            method="GET",
            path=f"/v2/market/{self.resolved_market_id}/informationExtended",
            params=params,
            include_auth=True,
        )
        quote_tuple = self._extract_quote_from_information_extended(payload)

        if quote_tuple is None:
            bid_tick = self._tick_price("BID")
            ask_tick = self._tick_price("ASK")
            if bid_tick is None or ask_tick is None:
                raise RuntimeError(f"GCAPI quote unavailable for {self.requested_symbol} market_id={self.resolved_market_id}.")
            bid, bid_ts = bid_tick
            ask, ask_ts = ask_tick
            ts_utc = ask_ts or bid_ts or utc_iso()
        else:
            bid, ask, ts_utc = quote_tuple

        pip_size = pip_size_for_symbol(self.requested_symbol)
        spread_pips = float((ask - bid) / pip_size)
        quote = Quote(
            symbol=self.requested_symbol,
            bid=float(bid),
            ask=float(ask),
            spread_pips=spread_pips,
            ts_utc=ts_utc,
        )
        self._last_quote = quote
        return quote

    def _to_units(self, qty_lots: float) -> float:
        return float(max(0.0, float(qty_lots)) * self.quantity_multiplier)

    def _to_lots(self, qty_units: float) -> float:
        if self.quantity_multiplier <= 0:
            return float(qty_units)
        return float(qty_units) / self.quantity_multiplier

    @staticmethod
    def _status_payload(response: Dict[str, Any]) -> Dict[str, Any]:
        orders = response.get("Orders")
        if isinstance(orders, list) and orders and isinstance(orders[0], dict):
            return orders[0]
        return response

    @staticmethod
    def _format_order_status(response: Dict[str, Any], order_payload: Dict[str, Any]) -> str:
        status = _safe_text(order_payload.get("Status")) or _safe_text(response.get("Status"))
        reason = _safe_text(order_payload.get("StatusReason")) or _safe_text(response.get("StatusReason"))
        error = _safe_text(response.get("ErrorMessage"))
        parts = [x for x in (status, reason, error) if x]
        return " | ".join(parts)

    @staticmethod
    def _is_rejection(status_text: str) -> bool:
        text = status_text.lower()
        rejection_tokens = (
            "reject",
            "error",
            "failed",
            "invalid",
            "disable",
            "closed",
            "insufficient",
            "denied",
            "unable",
        )
        return any(token in text for token in rejection_tokens)

    def _build_order_result(
        self,
        *,
        response: Dict[str, Any],
        requested_price: Optional[float],
        broker_position_id_hint: str = "",
    ) -> OrderResult:
        order_payload = self._status_payload(response)
        order_id = safe_int(response.get("OrderId", order_payload.get("OrderId", 0)), 0)
        status_text = self._format_order_status(response, order_payload)
        accepted = order_id > 0 and not self._is_rejection(status_text)

        deal_id = _safe_text(order_payload.get("DealId")) or _safe_text(order_payload.get("TradeOrderId"))
        position_id = _safe_text(order_payload.get("PositionId")) or broker_position_id_hint
        if not position_id and order_id > 0:
            position_id = str(order_id)

        fill_price_raw = safe_float(order_payload.get("Price", response.get("Price", 0.0)), 0.0)
        fill_price = fill_price_raw if fill_price_raw > 0.0 else None
        if accepted and fill_price is None and requested_price is not None:
            fill_price = float(requested_price)

        retcode_raw = order_payload.get("Status", response.get("StatusCode", "0"))
        retcode = str(retcode_raw)
        retcode_detail = status_text or "No status details returned by GCAPI."
        raw_status = "filled" if accepted else "rejected"

        return OrderResult(
            accepted=accepted,
            broker_order_id=str(order_id) if order_id > 0 else "",
            broker_position_id=str(position_id or ""),
            broker_deal_id=str(deal_id or ""),
            requested_price=requested_price,
            fill_price=fill_price,
            retcode=retcode,
            retcode_detail=retcode_detail,
            raw_status=raw_status,
        )

    def submit_order(self, request: OrderRequest) -> OrderResult:
        self._ensure_connected()

        side = str(request.side or "").strip().lower()
        if side not in {"buy", "sell"}:
            return OrderResult(
                accepted=False,
                broker_order_id="",
                broker_position_id="",
                broker_deal_id="",
                requested_price=None,
                fill_price=None,
                retcode="GCAPI_INVALID_SIDE",
                retcode_detail=f"Unsupported side '{request.side}'",
                raw_status="rejected",
            )

        if self.trading_account_id <= 0 or self.resolved_market_id <= 0:
            return OrderResult(
                accepted=False,
                broker_order_id="",
                broker_position_id="",
                broker_deal_id="",
                requested_price=None,
                fill_price=None,
                retcode="GCAPI_NOT_CONNECTED",
                retcode_detail="Adapter is missing account or market context.",
                raw_status="rejected",
            )

        quantity_units = self._to_units(float(request.qty))
        if quantity_units <= 0.0:
            return OrderResult(
                accepted=False,
                broker_order_id="",
                broker_position_id="",
                broker_deal_id="",
                requested_price=None,
                fill_price=None,
                retcode="GCAPI_INVALID_QTY",
                retcode_detail=f"Calculated quantity is invalid for qty={request.qty}",
                raw_status="rejected",
            )

        quote = self.get_quote(request.symbol)
        requested_price = float(quote.ask if side == "buy" else quote.bid)
        quantity_payload = int(round(quantity_units))
        payload: Dict[str, Any] = {
            "MarketId": int(self.resolved_market_id),
            "Direction": "Buy" if side == "buy" else "Sell",
            "Quantity": quantity_payload,
            "TradingAccountId": int(self.trading_account_id),
            "BidPrice": float(quote.bid),
            "OfferPrice": float(quote.ask),
        }
        if quantity_payload <= 0:
            return OrderResult(
                accepted=False,
                broker_order_id="",
                broker_position_id="",
                broker_deal_id="",
                requested_price=requested_price,
                fill_price=None,
                retcode="GCAPI_INVALID_QTY",
                retcode_detail=f"Calculated quantity is invalid for qty={request.qty}",
                raw_status="rejected",
            )

        try:
            response = self._request_json(
                method="POST",
                path="/order/newtradeorder",
                json_payload=payload,
                include_auth=True,
            )
        except Exception as exc:
            return OrderResult(
                accepted=False,
                broker_order_id="",
                broker_position_id="",
                broker_deal_id="",
                requested_price=requested_price,
                fill_price=None,
                retcode="GCAPI_ORDER_HTTP_ERROR",
                retcode_detail=str(exc),
                raw_status="rejected",
            )
        if not isinstance(response, dict):
            return OrderResult(
                accepted=False,
                broker_order_id="",
                broker_position_id="",
                broker_deal_id="",
                requested_price=requested_price,
                fill_price=None,
                retcode="GCAPI_ORDER_INVALID_RESPONSE",
                retcode_detail="GCAPI returned a non-object response for order submission.",
                raw_status="rejected",
            )

        return self._build_order_result(response=response, requested_price=requested_price)

    def _open_positions(self) -> List[Dict[str, Any]]:
        self._ensure_connected()
        params: Dict[str, Any] = {}
        if self.trading_account_id > 0:
            params["TradingAccountId"] = int(self.trading_account_id)

        response = self._request_json(
            method="GET",
            path="/order/openpositions",
            params=params or None,
            include_auth=True,
        )
        positions = _as_list(response)
        if not positions:
            return []

        filtered: List[Dict[str, Any]] = []
        for position in positions:
            market_id = safe_int(position.get("MarketId", 0), 0)
            if self.resolved_market_id > 0 and market_id != self.resolved_market_id:
                continue
            filtered.append(position)
        return filtered

    @staticmethod
    def _position_sort_key(position: Dict[str, Any]) -> datetime:
        ts_text = _safe_text(position.get("OpenDateTimeUTC")) or _safe_text(position.get("OpenDateTime")) or _safe_text(position.get("CreatedDateTime"))
        if not ts_text:
            return datetime.now(timezone.utc)
        if ts_text.endswith("Z"):
            ts_text = ts_text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(ts_text)
        except ValueError:
            return datetime.now(timezone.utc)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def get_open_position(self, symbol: str) -> PositionState:
        positions = self._open_positions()
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

        oldest = min(positions, key=self._position_sort_key)
        direction = _safe_text(oldest.get("Direction")).lower()
        side = "long" if direction.startswith("buy") else "short"
        quantity_units = safe_float(oldest.get("Quantity", 0.0), 0.0)
        quantity_lots = self._to_lots(quantity_units)

        entry_price = safe_float(oldest.get("Price", oldest.get("OpenPrice")), 0.0)
        entry_price_opt = float(entry_price) if entry_price > 0.0 else None
        entry_ts_utc = _safe_text(oldest.get("OpenDateTimeUTC")) or _safe_text(oldest.get("OpenDateTime")) or None
        position_id = _safe_text(oldest.get("PositionId")) or _safe_text(oldest.get("OrderId"))
        order_id = _safe_text(oldest.get("OrderId"))
        pnl = safe_float(
            oldest.get("ProfitAndLoss", oldest.get("NetProfitAndLoss", oldest.get("UnrealisedPnl", 0.0))),
            0.0,
        )

        return PositionState(
            is_open=True,
            side=side,
            qty=float(quantity_lots),
            entry_price=entry_price_opt,
            broker_position_id=position_id,
            broker_order_id=order_id,
            entry_ts_utc=entry_ts_utc,
            unrealized_pnl_usd=float(pnl),
        )

    def close_position(self, symbol: str) -> OrderResult:
        positions = self._open_positions()
        if not positions:
            return OrderResult(
                accepted=False,
                broker_order_id="",
                broker_position_id="",
                broker_deal_id="",
                requested_price=None,
                fill_price=None,
                retcode="GCAPI_NO_POSITION",
                retcode_detail="No open managed position found.",
                raw_status="rejected",
            )

        oldest = min(positions, key=self._position_sort_key)
        position_order_id = safe_int(oldest.get("OrderId", 0), 0)
        direction = _safe_text(oldest.get("Direction")).lower()
        quantity_units = safe_float(oldest.get("Quantity", 0.0), 0.0)
        if quantity_units <= 0.0 or position_order_id <= 0:
            return OrderResult(
                accepted=False,
                broker_order_id="",
                broker_position_id=_safe_text(oldest.get("PositionId") or oldest.get("OrderId")),
                broker_deal_id="",
                requested_price=None,
                fill_price=None,
                retcode="GCAPI_POSITION_INVALID",
                retcode_detail="Open position does not have a closeable order id or quantity.",
                raw_status="rejected",
            )

        close_side = "Sell" if direction.startswith("buy") else "Buy"
        quote = self.get_quote(symbol)
        requested_price = float(quote.bid if close_side == "Sell" else quote.ask)
        quantity_payload = int(round(quantity_units))

        payload: Dict[str, Any] = {
            "MarketId": int(self.resolved_market_id),
            "Direction": close_side,
            "Quantity": quantity_payload,
            "TradingAccountId": int(self.trading_account_id),
            "Close": [int(position_order_id)],
            "BidPrice": float(quote.bid),
            "OfferPrice": float(quote.ask),
        }
        if quantity_payload <= 0:
            return OrderResult(
                accepted=False,
                broker_order_id="",
                broker_position_id=_safe_text(oldest.get("PositionId") or oldest.get("OrderId")),
                broker_deal_id="",
                requested_price=requested_price,
                fill_price=None,
                retcode="GCAPI_POSITION_INVALID",
                retcode_detail="Open position quantity rounded to zero for close request.",
                raw_status="rejected",
            )

        try:
            response = self._request_json(
                method="POST",
                path="/order/newtradeorder",
                json_payload=payload,
                include_auth=True,
            )
        except Exception as exc:
            return OrderResult(
                accepted=False,
                broker_order_id="",
                broker_position_id=_safe_text(oldest.get("PositionId") or oldest.get("OrderId")),
                broker_deal_id="",
                requested_price=requested_price,
                fill_price=None,
                retcode="GCAPI_CLOSE_HTTP_ERROR",
                retcode_detail=str(exc),
                raw_status="rejected",
            )
        if not isinstance(response, dict):
            return OrderResult(
                accepted=False,
                broker_order_id="",
                broker_position_id=_safe_text(oldest.get("PositionId") or oldest.get("OrderId")),
                broker_deal_id="",
                requested_price=requested_price,
                fill_price=None,
                retcode="GCAPI_CLOSE_INVALID_RESPONSE",
                retcode_detail="GCAPI returned a non-object response for close request.",
                raw_status="rejected",
            )

        return self._build_order_result(
            response=response,
            requested_price=requested_price,
            broker_position_id_hint=_safe_text(oldest.get("PositionId") or oldest.get("OrderId")),
        )

    def get_account_balance(self) -> float:
        self._ensure_connected()
        response = self._request_json(
            method="GET",
            path="/margin/ClientAccountMargin",
            include_auth=True,
        )
        if not isinstance(response, dict):
            return 0.0

        balance = safe_float(response.get("NetEquity", 0.0), 0.0)
        if balance <= 0.0:
            balance = safe_float(response.get("Cash", 0.0), 0.0)
        if balance <= 0.0:
            balance = safe_float(response.get("Balance", 0.0), 0.0)
        return float(balance)
