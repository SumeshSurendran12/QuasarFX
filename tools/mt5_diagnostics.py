from __future__ import annotations

import json
import os
import subprocess
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment]

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package is not installed. Run: pip install MetaTrader5")
    sys.exit(1)


# ============================================================
# Config
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
if load_dotenv is not None:
    # Prefer repo .env values for this diagnostics workflow.
    load_dotenv(ROOT / ".env", override=True)


def _first_env(names: List[str], default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _env_int(names: List[str], default: int = 0) -> int:
    raw = _first_env(names, "")
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(names: List[str], default: float = 0.0) -> float:
    raw = _first_env(names, "")
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _running_terminal_path() -> str:
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        "$p=Get-Process terminal64 -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Path; if ($p) { Write-Output $p }",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=3)
        path_text = str(result.stdout or "").strip()
        if path_text and Path(path_text).exists():
            return path_text
    except Exception:
        pass
    return ""


def _resolve_terminal_path() -> str:
    explicit = str(os.getenv("MT5_TERMINAL_PATH", "")).strip()
    if explicit and Path(explicit).exists():
        return explicit

    running = _running_terminal_path()
    if running:
        return running

    common_candidates = [
        r"C:\Program Files\FOREX.com US\terminal64.exe",
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        r"C:\Program Files\MetaTrader5\terminal64.exe",
        r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
    ]
    for candidate in common_candidates:
        if Path(candidate).exists():
            return candidate
    return explicit or r"C:\Program Files\MetaTrader 5\terminal64.exe"


LOGIN_RAW = _first_env(["MT5_LOGIN", "MT5LOGIN"], "0")
LOGIN = _env_int(["MT5_LOGIN", "MT5LOGIN"], 0)
LOGIN_PARSE_WARNING = bool(str(LOGIN_RAW).strip() not in {"", "0"} and LOGIN == 0)
PASSWORD = _first_env(["MT5_PASSWORD", "MT5PASSWORD"], "")
SERVER = _first_env(["MT5_SERVER", "MT5SERVER"], "")
TERMINAL_PATH = _resolve_terminal_path()
BASE_SYMBOL = _first_env(["FX_SYMBOL", "MT5_SYMBOL", "MT5SYMBOL"], "EURUSD")
ORDER_VOLUME = _env_float(["FX_ORDER_QTY"], 0.01)
DEVIATION_POINTS = _env_int(["MT5_DEVIATION_POINTS"], 20)
MAGIC_NUMBER = _env_int(["MT5_MAGIC_NUMBER"], 51001)
LATENCY_SAMPLES = max(1, _env_int(["MT5_LATENCY_SAMPLES"], 5))
DIAG_JSON_PATH = _first_env(["MT5_DIAG_JSON_PATH"], "mt5_diagnostics_report.json")

SYMBOL_CANDIDATES = [
    BASE_SYMBOL,
    f"{BASE_SYMBOL}.a",
    f"{BASE_SYMBOL}m",
    f"{BASE_SYMBOL}.m",
    f"{BASE_SYMBOL}.r",
    f"{BASE_SYMBOL}_i",
    f"{BASE_SYMBOL}pro",
    f"{BASE_SYMBOL}-pro",
]


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    data: Optional[Dict[str, Any]] = None


@dataclass
class DiagnosticsReport:
    generated_at_utc: str
    terminal_connected: bool
    account_logged_in: bool
    resolved_symbol: Optional[str]
    summary_status: str
    checks: List[Dict[str, Any]]


# ============================================================
# Helpers
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def namedtuple_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "_asdict"):
        return dict(obj._asdict())
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": str(obj)}


def pip_size_from_digits(digits: int) -> float:
    return 0.01 if digits in (2, 3) else 0.0001


def spread_in_pips(bid: float, ask: float, digits: int) -> float:
    pip_size = pip_size_from_digits(digits)
    if pip_size <= 0:
        return 0.0
    return (ask - bid) / pip_size


def print_section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")


def add_check(
    results: List[CheckResult],
    name: str,
    ok: bool,
    detail: str,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    results.append(CheckResult(name=name, ok=ok, detail=detail, data=data))


def shutdown_and_exit(code: int) -> None:
    try:
        mt5.shutdown()
    except Exception:
        pass
    sys.exit(code)


# ============================================================
# MT5 diagnostics
# ============================================================

def initialize_terminal(results: List[CheckResult]) -> bool:
    print_section("1) TERMINAL INITIALIZATION")

    if not mt5.initialize(path=TERMINAL_PATH):
        err = mt5.last_error()
        add_check(
            results,
            "terminal_initialize",
            False,
            f"mt5.initialize failed: {err}",
            {"terminal_path": TERMINAL_PATH, "last_error": str(err)},
        )
        print(f"FAIL: initialize failed -> {err}")
        return False

    term_info = mt5.terminal_info()
    version_info = mt5.version()

    data = {
        "terminal_path": TERMINAL_PATH,
        "terminal_info": namedtuple_to_dict(term_info),
        "version": version_info,
    }

    add_check(results, "terminal_initialize", True, "MT5 terminal initialized", data)
    print("PASS: terminal initialized")
    print(f"Terminal path: {TERMINAL_PATH}")
    print(f"Version: {version_info}")
    return True


def login_account(results: List[CheckResult]) -> bool:
    print_section("2) ACCOUNT LOGIN")

    if LOGIN:
        if not mt5.login(login=LOGIN, password=PASSWORD, server=SERVER):
            err = mt5.last_error()
            add_check(
                results,
                "account_login",
                False,
                f"mt5.login failed: {err}",
                {"login": LOGIN, "server": SERVER, "last_error": str(err)},
            )
            print(f"FAIL: login failed -> {err}")
            return False

    account = mt5.account_info()
    if account is None:
        err = mt5.last_error()
        add_check(
            results,
            "account_info",
            False,
            f"mt5.account_info returned None: {err}",
            {"last_error": str(err)},
        )
        print(f"FAIL: account_info unavailable -> {err}")
        return False

    acc = namedtuple_to_dict(account)
    detail = f"Logged into account {acc.get('login')} on {acc.get('server')}"

    add_check(results, "account_login", True, detail, acc)
    print("PASS:", detail)
    print(f"Balance: {acc.get('balance')}")
    print(f"Equity: {acc.get('equity')}")
    print(f"Leverage: {acc.get('leverage')}")
    print(f"Trade allowed: {acc.get('trade_allowed')}")
    print(f"Trade expert: {acc.get('trade_expert')}")
    return True


def resolve_symbol(results: List[CheckResult]) -> Optional[str]:
    print_section("3) SYMBOL MAPPING")

    all_symbols = mt5.symbols_get()
    if all_symbols is None:
        err = mt5.last_error()
        add_check(
            results,
            "symbols_get",
            False,
            f"mt5.symbols_get failed: {err}",
            {"last_error": str(err)},
        )
        print(f"FAIL: symbols_get failed -> {err}")
        return None

    symbol_names = {s.name for s in all_symbols}
    tried: List[str] = []
    resolved: Optional[str] = None

    for candidate in SYMBOL_CANDIDATES:
        if candidate in tried:
            continue
        tried.append(candidate)
        if candidate in symbol_names:
            resolved = candidate
            break

    if resolved is None:
        add_check(
            results,
            "symbol_mapping",
            False,
            f"No symbol match found for {BASE_SYMBOL}",
            {
                "base_symbol": BASE_SYMBOL,
                "tried": tried,
                "sample_symbols": sorted(list(symbol_names))[:50],
            },
        )
        print(f"FAIL: could not resolve symbol for {BASE_SYMBOL}")
        print("Tried:", tried)
        return None

    selected = mt5.symbol_select(resolved, True)
    info = mt5.symbol_info(resolved)
    data = {
        "base_symbol": BASE_SYMBOL,
        "resolved_symbol": resolved,
        "tried": tried,
        "selected": selected,
        "symbol_info": namedtuple_to_dict(info),
    }

    ok = bool(selected and info is not None)
    detail = f"Resolved {BASE_SYMBOL} -> {resolved}"

    add_check(results, "symbol_mapping", ok, detail, data)

    if ok:
        print("PASS:", detail)
        print(f"Visible: {getattr(info, 'visible', None)}")
        print(f"Digits: {getattr(info, 'digits', None)}")
        print(f"Trade mode: {getattr(info, 'trade_mode', None)}")
    else:
        print("FAIL:", detail)

    return resolved if ok else None


def collect_tick_latency_and_spread(results: List[CheckResult], symbol: str) -> bool:
    print_section("4) TICK / SPREAD / LATENCY")

    latencies_ms: List[float] = []
    spreads_pips: List[float] = []
    last_tick_dict: Dict[str, Any] = {}

    info = mt5.symbol_info(symbol)
    if info is None:
        err = mt5.last_error()
        add_check(
            results,
            "symbol_info",
            False,
            f"symbol_info failed for {symbol}: {err}",
            {"symbol": symbol, "last_error": str(err)},
        )
        print(f"FAIL: symbol_info unavailable -> {err}")
        return False

    digits = safe_int(getattr(info, "digits", 5), 5)

    for i in range(LATENCY_SAMPLES):
        t0 = time.perf_counter()
        tick = mt5.symbol_info_tick(symbol)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        latencies_ms.append(latency_ms)

        if tick is None:
            err = mt5.last_error()
            add_check(
                results,
                "tick_fetch",
                False,
                f"symbol_info_tick failed on sample {i + 1}: {err}",
                {"symbol": symbol, "sample_index": i + 1, "last_error": str(err)},
            )
            print(f"FAIL: tick fetch failed on sample {i + 1} -> {err}")
            return False

        bid = safe_float(getattr(tick, "bid", 0.0))
        ask = safe_float(getattr(tick, "ask", 0.0))
        spreads_pips.append(spread_in_pips(bid, ask, digits))
        last_tick_dict = namedtuple_to_dict(tick)
        time.sleep(0.15)

    data = {
        "symbol": symbol,
        "latency_ms_samples": latencies_ms,
        "latency_ms_min": min(latencies_ms),
        "latency_ms_max": max(latencies_ms),
        "latency_ms_avg": statistics.mean(latencies_ms),
        "spread_pips_samples": spreads_pips,
        "spread_pips_min": min(spreads_pips),
        "spread_pips_max": max(spreads_pips),
        "spread_pips_avg": statistics.mean(spreads_pips),
        "last_tick": last_tick_dict,
        "digits": digits,
        "pip_size": pip_size_from_digits(digits),
    }

    add_check(
        results,
        "tick_latency_spread",
        True,
        f"Collected {LATENCY_SAMPLES} tick samples",
        data,
    )

    print("PASS: tick data collected")
    print(f"Latency avg: {data['latency_ms_avg']:.3f} ms")
    print(f"Latency min/max: {data['latency_ms_min']:.3f} / {data['latency_ms_max']:.3f} ms")
    print(f"Spread avg: {data['spread_pips_avg']:.3f} pips")
    print(f"Spread min/max: {data['spread_pips_min']:.3f} / {data['spread_pips_max']:.3f} pips")
    print(f"Bid/Ask: {last_tick_dict.get('bid')} / {last_tick_dict.get('ask')}")
    return True


def inspect_account_limits(results: List[CheckResult]) -> bool:
    print_section("5) ACCOUNT / TERMINAL LIMITS")

    account = mt5.account_info()
    terminal = mt5.terminal_info()

    if account is None or terminal is None:
        err = mt5.last_error()
        add_check(
            results,
            "account_terminal_limits",
            False,
            f"account_info or terminal_info unavailable: {err}",
            {"last_error": str(err)},
        )
        print(f"FAIL: account/terminal info unavailable -> {err}")
        return False

    acc = namedtuple_to_dict(account)
    term = namedtuple_to_dict(terminal)

    trade_allowed = bool(acc.get("trade_allowed", False))
    trade_expert = bool(acc.get("trade_expert", False))
    connected = bool(term.get("connected", False))
    terminal_trade_allowed = bool(term.get("trade_allowed", False))
    terminal_tradeapi_disabled = bool(term.get("tradeapi_disabled", False))

    ok = connected and trade_allowed and trade_expert and terminal_trade_allowed and (not terminal_tradeapi_disabled)
    detail = "Collected account and terminal trading permissions"
    if not ok:
        detail = (
            "Trading permissions require attention "
            f"(connected={connected}, account_trade_allowed={trade_allowed}, "
            f"account_trade_expert={trade_expert}, terminal_trade_allowed={terminal_trade_allowed}, "
            f"terminal_tradeapi_disabled={terminal_tradeapi_disabled})"
        )

    data = {
        "account_login": acc.get("login"),
        "server": acc.get("server"),
        "balance": acc.get("balance"),
        "equity": acc.get("equity"),
        "margin": acc.get("margin"),
        "margin_free": acc.get("margin_free"),
        "margin_level": acc.get("margin_level"),
        "leverage": acc.get("leverage"),
        "trade_allowed": trade_allowed,
        "trade_expert": trade_expert,
        "terminal_connected": term.get("connected"),
        "terminal_trade_allowed": term.get("trade_allowed"),
        "terminal_dlls_allowed": term.get("dlls_allowed"),
        "terminal_tradeapi_disabled": term.get("tradeapi_disabled"),
        "company": term.get("company"),
        "name": term.get("name"),
        "community_account": term.get("community_account"),
    }

    add_check(results, "account_terminal_limits", ok, detail, data)

    print("PASS: account and terminal info collected")
    print(f"Trade allowed: {trade_allowed}")
    print(f"Expert trading allowed: {trade_expert}")
    print(f"Terminal connected: {term.get('connected')}")
    print(f"Terminal trade allowed: {term.get('trade_allowed')}")
    print(f"DLLs allowed: {term.get('dlls_allowed')}")
    print(f"tradeapi_disabled: {term.get('tradeapi_disabled')}")
    return True


def inspect_symbol_limits(results: List[CheckResult], symbol: str) -> bool:
    print_section("6) SYMBOL TRADING LIMITS")

    info = mt5.symbol_info(symbol)
    if info is None:
        err = mt5.last_error()
        add_check(
            results,
            "symbol_limits",
            False,
            f"symbol_info failed for {symbol}: {err}",
            {"symbol": symbol, "last_error": str(err)},
        )
        print(f"FAIL: symbol info unavailable -> {err}")
        return False

    d = namedtuple_to_dict(info)

    volume_min = d.get("volume_min")
    volume_max = d.get("volume_max")
    volume_step = d.get("volume_step")
    trade_stops_level = d.get("trade_stops_level")
    trade_freeze_level = d.get("trade_freeze_level")
    filling_mode = d.get("filling_mode")
    order_mode = d.get("order_mode")
    trade_mode = d.get("trade_mode")
    digits = d.get("digits")
    point = d.get("point")
    spread = d.get("spread")
    spread_float = d.get("spread_float")

    data = {
        "symbol": symbol,
        "volume_min": volume_min,
        "volume_max": volume_max,
        "volume_step": volume_step,
        "trade_stops_level": trade_stops_level,
        "trade_freeze_level": trade_freeze_level,
        "filling_mode": filling_mode,
        "order_mode": order_mode,
        "trade_mode": trade_mode,
        "digits": digits,
        "point": point,
        "spread_points": spread,
        "spread_float": spread_float,
        "currency_base": d.get("currency_base"),
        "currency_profit": d.get("currency_profit"),
        "currency_margin": d.get("currency_margin"),
        "contract_size": d.get("trade_contract_size"),
    }

    ok = True
    detail = f"Collected symbol limits for {symbol}"
    add_check(results, "symbol_limits", ok, detail, data)

    print("PASS: symbol trading limits collected")
    print(f"Volume min/max/step: {volume_min} / {volume_max} / {volume_step}")
    print(f"Stops level: {trade_stops_level}")
    print(f"Freeze level: {trade_freeze_level}")
    print(f"Trade mode: {trade_mode}")
    return True


def order_capability_check(results: List[CheckResult], symbol: str) -> bool:
    print_section("7) ORDER CAPABILITY CHECK")

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        err = mt5.last_error()
        add_check(
            results,
            "order_check_prereq_tick",
            False,
            f"symbol_info_tick failed before order_check: {err}",
            {"symbol": symbol, "last_error": str(err)},
        )
        print(f"FAIL: tick unavailable for order check -> {err}")
        return False

    info = mt5.symbol_info(symbol)
    if info is None:
        err = mt5.last_error()
        add_check(
            results,
            "order_check_prereq_symbol",
            False,
            f"symbol_info failed before order_check: {err}",
            {"symbol": symbol, "last_error": str(err)},
        )
        print(f"FAIL: symbol info unavailable for order check -> {err}")
        return False

    symbol_fill_flags = safe_int(getattr(info, "filling_mode", 0), 0)
    filling_candidates: List[Dict[str, Any]] = []
    flag_to_candidate = [
        (1, "FOK", mt5.ORDER_FILLING_FOK),
        (2, "IOC", mt5.ORDER_FILLING_IOC),
        (4, "RETURN", mt5.ORDER_FILLING_RETURN),
    ]
    for flag, label, value in flag_to_candidate:
        if symbol_fill_flags & flag:
            filling_candidates.append({"label": label, "value": value, "flag": flag})
    # Fallback coverage when broker flags are absent/unreliable.
    for label, value in [
        ("FOK", mt5.ORDER_FILLING_FOK),
        ("IOC", mt5.ORDER_FILLING_IOC),
        ("RETURN", mt5.ORDER_FILLING_RETURN),
    ]:
        if all(c["value"] != value for c in filling_candidates):
            filling_candidates.append({"label": label, "value": value, "flag": None})

    attempted: List[Dict[str, Any]] = []
    selected: Optional[Dict[str, Any]] = None

    for filling in filling_candidates:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": ORDER_VOLUME,
            "type": mt5.ORDER_TYPE_BUY,
            "price": float(tick.ask),
            "deviation": DEVIATION_POINTS,
            "magic": MAGIC_NUMBER,
            "comment": "quasarfx_mt5_diagnostics",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling["value"],
        }

        check = mt5.order_check(request)
        if check is None:
            err = mt5.last_error()
            attempted.append(
                {
                    "filling_label": filling["label"],
                    "request": request,
                    "retcode": None,
                    "comment": "",
                    "last_error": str(err),
                }
            )
            continue

        d = namedtuple_to_dict(check)
        retcode = d.get("retcode")
        comment = d.get("comment", "")
        attempt = {
            "filling_label": filling["label"],
            "request": request,
            "retcode": retcode,
            "comment": comment,
            "order_check": d,
        }
        attempted.append(attempt)

        if retcode == 0 or str(retcode) == "0":
            selected = attempt
            break

    if selected is not None:
        detail = (
            f"order_check retcode={selected.get('retcode')}, "
            f"filling={selected.get('filling_label')}, comment={selected.get('comment', '')}"
        )
        add_check(
            results,
            "order_check",
            True,
            detail,
            {
                "selected_attempt": selected,
                "attempts": attempted,
                "symbol_filling_mode_flags": symbol_fill_flags,
            },
        )
        print("PASS:", detail)
        return True

    last_attempt = attempted[-1] if attempted else {}
    detail = (
        f"order_check failed for all filling modes; "
        f"last retcode={last_attempt.get('retcode')}, comment={last_attempt.get('comment', '')}"
    )
    add_check(
        results,
        "order_check",
        False,
        detail,
        {
            "attempts": attempted,
            "symbol_filling_mode_flags": symbol_fill_flags,
        },
    )
    print("WARN:", detail)
    return False


def existing_positions_orders_snapshot(results: List[CheckResult], symbol: str) -> bool:
    print_section("8) EXISTING POSITIONS / ORDERS SNAPSHOT")

    positions = mt5.positions_get(symbol=symbol)
    orders = mt5.orders_get(symbol=symbol)

    pos_list = [namedtuple_to_dict(p) for p in positions] if positions else []
    ord_list = [namedtuple_to_dict(o) for o in orders] if orders else []

    data = {
        "symbol": symbol,
        "open_positions_count": len(pos_list),
        "open_orders_count": len(ord_list),
        "positions": pos_list,
        "orders": ord_list,
    }

    add_check(
        results,
        "positions_orders_snapshot",
        True,
        f"Found {len(pos_list)} open positions and {len(ord_list)} open orders for {symbol}",
        data,
    )

    print("PASS: snapshot collected")
    print(f"Open positions: {len(pos_list)}")
    print(f"Open orders: {len(ord_list)}")
    return True


def compute_summary_status(results: List[CheckResult]) -> str:
    required_failures = {
        "terminal_initialize",
        "account_login",
        "symbol_mapping",
        "tick_latency_spread",
        "account_terminal_limits",
    }

    failed_required = any(r.name in required_failures and not r.ok for r in results)
    any_failure = any(not r.ok for r in results)

    if failed_required:
        return "FAIL"
    if any_failure:
        return "ATTENTION"
    return "PASS"


def write_report(report: DiagnosticsReport) -> None:
    out = Path(DIAG_JSON_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("\nMT5 FULL DIAGNOSTICS REPORT")
    print(f"UTC time: {utc_now_iso()}")
    print(f"Base symbol: {BASE_SYMBOL}")
    print(f"Requested volume: {ORDER_VOLUME}")
    print(f"Latency samples: {LATENCY_SAMPLES}")
    if LOGIN_PARSE_WARNING:
        print(f"WARNING: MT5_LOGIN is not a valid integer ({LOGIN_RAW!r}); falling back to LOGIN=0.")

    results: List[CheckResult] = []
    terminal_connected = False
    account_logged_in = False
    resolved_symbol: Optional[str] = None

    try:
        terminal_connected = initialize_terminal(results)
        if not terminal_connected:
            summary = compute_summary_status(results)
            report = DiagnosticsReport(
                generated_at_utc=utc_now_iso(),
                terminal_connected=False,
                account_logged_in=False,
                resolved_symbol=None,
                summary_status=summary,
                checks=[asdict(r) for r in results],
            )
            write_report(report)
            print(f"\nSummary: {summary}")
            print(f"JSON report written to: {DIAG_JSON_PATH}")
            shutdown_and_exit(1)

        account_logged_in = login_account(results)
        if not account_logged_in:
            summary = compute_summary_status(results)
            report = DiagnosticsReport(
                generated_at_utc=utc_now_iso(),
                terminal_connected=True,
                account_logged_in=False,
                resolved_symbol=None,
                summary_status=summary,
                checks=[asdict(r) for r in results],
            )
            write_report(report)
            print(f"\nSummary: {summary}")
            print(f"JSON report written to: {DIAG_JSON_PATH}")
            shutdown_and_exit(1)

        resolved_symbol = resolve_symbol(results)
        if resolved_symbol is None:
            summary = compute_summary_status(results)
            report = DiagnosticsReport(
                generated_at_utc=utc_now_iso(),
                terminal_connected=True,
                account_logged_in=True,
                resolved_symbol=None,
                summary_status=summary,
                checks=[asdict(r) for r in results],
            )
            write_report(report)
            print(f"\nSummary: {summary}")
            print(f"JSON report written to: {DIAG_JSON_PATH}")
            shutdown_and_exit(1)

        collect_tick_latency_and_spread(results, resolved_symbol)
        inspect_account_limits(results)
        inspect_symbol_limits(results, resolved_symbol)
        order_capability_check(results, resolved_symbol)
        existing_positions_orders_snapshot(results, resolved_symbol)

        summary = compute_summary_status(results)

        report = DiagnosticsReport(
            generated_at_utc=utc_now_iso(),
            terminal_connected=terminal_connected,
            account_logged_in=account_logged_in,
            resolved_symbol=resolved_symbol,
            summary_status=summary,
            checks=[asdict(r) for r in results],
        )
        write_report(report)

        print_section("FINAL SUMMARY")
        print(f"Status: {summary}")
        print(f"Resolved symbol: {resolved_symbol}")
        print(f"JSON report: {DIAG_JSON_PATH}")

        for r in results:
            badge = "PASS" if r.ok else "FAIL"
            print(f"[{badge}] {r.name}: {r.detail}")

        if summary == "PASS":
            print("\nReady to run QuasarFX with MT5 demo.")
        elif summary == "ATTENTION":
            print("\nUsable, but review warnings before running live execution.")
        else:
            print("\nDo not run live_trading.py until the failing checks are fixed.")

    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
