from __future__ import annotations

from typing import Dict, Set

from .mt5_utils import MT5_AVAILABLE, get_mt5_module


if MT5_AVAILABLE:
    _mt5 = get_mt5_module()
    RETCODE_LABELS: Dict[int, str] = {
        int(getattr(_mt5, "TRADE_RETCODE_DONE", 10009)): "TRADE_RETCODE_DONE",
        int(getattr(_mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010)): "TRADE_RETCODE_DONE_PARTIAL",
        int(getattr(_mt5, "TRADE_RETCODE_REQUOTE", 10004)): "TRADE_RETCODE_REQUOTE",
        int(getattr(_mt5, "TRADE_RETCODE_REJECT", 10006)): "TRADE_RETCODE_REJECT",
        int(getattr(_mt5, "TRADE_RETCODE_INVALID", 10013)): "TRADE_RETCODE_INVALID",
        int(getattr(_mt5, "TRADE_RETCODE_INVALID_VOLUME", 10014)): "TRADE_RETCODE_INVALID_VOLUME",
        int(getattr(_mt5, "TRADE_RETCODE_INVALID_PRICE", 10015)): "TRADE_RETCODE_INVALID_PRICE",
        int(getattr(_mt5, "TRADE_RETCODE_MARKET_CLOSED", 10018)): "TRADE_RETCODE_MARKET_CLOSED",
        int(getattr(_mt5, "TRADE_RETCODE_NO_MONEY", 10019)): "TRADE_RETCODE_NO_MONEY",
        int(getattr(_mt5, "TRADE_RETCODE_PRICE_CHANGED", 10020)): "TRADE_RETCODE_PRICE_CHANGED",
        int(getattr(_mt5, "TRADE_RETCODE_PRICE_OFF", 10021)): "TRADE_RETCODE_PRICE_OFF",
    }
    SUCCESS_RETCODES: Set[int] = {
        int(getattr(_mt5, "TRADE_RETCODE_DONE", 10009)),
        int(getattr(_mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010)),
    }
else:
    RETCODE_LABELS = {
        10009: "TRADE_RETCODE_DONE",
        10010: "TRADE_RETCODE_DONE_PARTIAL",
    }
    SUCCESS_RETCODES = {10009, 10010}


def retcode_label(retcode: int) -> str:
    code = int(retcode)
    return RETCODE_LABELS.get(code, f"TRADE_RETCODE_{code}")


def is_success_retcode(retcode: int) -> bool:
    return int(retcode) in SUCCESS_RETCODES
