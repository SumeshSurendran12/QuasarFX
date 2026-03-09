from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

try:
    import MetaTrader5 as mt5
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    mt5 = None  # type: ignore[assignment]


MT5_AVAILABLE = mt5 is not None


def get_mt5_module() -> Any:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is not installed. Install with `pip install MetaTrader5`.")
    return mt5


def utc_iso(ts_seconds: Optional[float] = None) -> str:
    if ts_seconds is None:
        ts = datetime.now(timezone.utc)
    else:
        ts = datetime.fromtimestamp(float(ts_seconds), tz=timezone.utc)
    return ts.isoformat().replace("+00:00", "Z")


def pip_size_for_symbol(symbol: str) -> float:
    return 0.01 if str(symbol).upper().endswith("JPY") else 0.0001


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def bool_env(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def tick_timestamp_iso(tick: Any) -> str:
    if tick is None:
        return utc_iso()
    msec = safe_float(getattr(tick, "time_msc", 0.0), 0.0)
    if msec > 0:
        return utc_iso(msec / 1000.0)
    sec = safe_float(getattr(tick, "time", 0.0), 0.0)
    if sec > 0:
        return utc_iso(sec)
    return utc_iso()
