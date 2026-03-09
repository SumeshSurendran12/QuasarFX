#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_COMMON_FIELDS = [
    "ts",
    "run_id",
    "event",
    "stage",
    "strategy_id",
    "profile_hash",
    "manifest_hash",
    "schema_version",
    "manifest_version",
]
DEFAULT_SCHEMA_VERSION = "1.0.0"
DEFAULT_MANIFEST_VERSION = "1.0.0"
RUN_ID_REGEX = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<session>[A-Z0-9]+)_sha(?P<hash>[a-f0-9]{8})$")
DEFAULT_EVENTS = {
    "session_start",
    "signal_evaluated",
    "trade_skipped",
    "order_submitted",
    "order_filled",
    "position_closed",
    "daily_risk_update",
    "health_status",
    "kill_switch_check",
    "kill_switch_triggered",
    "session_end",
    "market_closed",
    "data_feed_alive",
}
DEFAULT_REASON_CODES = {
    "signal_pass",
    "spread_gate",
    "session_cap",
    "daily_loss_limit",
    "max_open_positions",
    "outside_session",
    "cooldown_active",
    "duplicate_signal",
    "no_liquidity",
    "manual_disable",
    "policy_breach",
    "within_limits",
    "broker_api_failure",
}
DEFAULT_EVENT_REQUIRED_FIELDS: Dict[str, List[str]] = {
    "session_start": ["symbol", "mode"],
    "signal_evaluated": ["symbol", "bar_ts", "side", "decision", "reason_code"],
    "trade_skipped": ["symbol", "bar_ts", "side", "decision", "reason_code"],
    "order_submitted": ["symbol", "trade_id", "side", "qty", "order_type"],
    "order_filled": ["symbol", "trade_id", "side", "qty", "fill_price"],
    "position_closed": ["symbol", "trade_id", "exit_reason", "entry_price", "exit_price", "pnl_usd", "hold_seconds"],
    "daily_risk_update": [
        "symbol",
        "trades_today",
        "wins",
        "losses",
        "net_pnl_usd",
        "gross_profit_usd",
        "gross_loss_usd",
        "profit_factor",
        "max_drawdown_usd",
        "spread_gate_skips",
        "session_cap_skips",
    ],
    "health_status": ["status", "event_log_present", "kill_switch_armed", "kill_switch_triggered"],
    "kill_switch_check": ["status", "reason_code"],
    "kill_switch_triggered": ["status", "reason_code"],
    "session_end": ["symbol", "trades", "net_pnl_usd", "status"],
    "market_closed": ["symbol"],
    "data_feed_alive": ["symbol"],
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_ts(value: str) -> Optional[datetime]:
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Events file not found: {path}")
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]
        raise ValueError("JSON events file must contain a list of objects.")

    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def event_ts(ev: Dict[str, Any]) -> Optional[datetime]:
    for key in ("ts", "timestamp", "time", "event_time"):
        if key in ev:
            dt = parse_ts(str(ev.get(key, "")))
            if dt is not None:
                return dt
    return None


def max_drawdown(values: List[float]) -> float:
    if not values:
        return 0.0
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for v in values:
        equity += float(v)
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > worst:
            worst = dd
    return float(worst)


def daily_pnl_series(rows: List[Tuple[Optional[datetime], float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for ts, pnl in rows:
        if ts is None:
            continue
        day = ts.date().isoformat()
        out[day] = float(out.get(day, 0.0) + float(pnl))
    return out


def day_key(ts: Optional[datetime]) -> str:
    if ts is None:
        return "unknown"
    return ts.date().isoformat()


def session_key(ev: Dict[str, Any], ts: Optional[datetime]) -> str:
    run_id = str(ev.get("run_id", "")).strip()
    if run_id:
        return run_id
    if ts is None:
        return "unknown_session"
    return ts.date().isoformat()


def event_missing_fields(ev: Dict[str, Any], required_fields: List[str]) -> List[str]:
    missing: List[str] = []
    for field in required_fields:
        if field not in ev:
            missing.append(field)
            continue
        value = ev.get(field)
        if value is None:
            missing.append(field)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(field)
    return missing


def normalize_profile_hash(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if text.startswith("sha256:"):
        return text
    return f"sha256:{text}"


def normalize_manifest_hash(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if text.startswith("sha256:"):
        return text
    return f"sha256:{text}"


def canonical_json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def spread_threshold_pips(controls: Dict[str, Any]) -> float:
    max_spread_pips = as_float(controls.get("live_max_spread_pips"), 0.0)
    if max_spread_pips > 0:
        return max_spread_pips

    raw = as_float(controls.get("live_max_spread"), 0.0)
    if raw <= 0:
        return 0.0
    if raw < 0.01:
        return raw / 0.0001
    return raw


def get_required_common_fields(manifest: Dict[str, Any], profile: Dict[str, Any]) -> List[str]:
    from_manifest = (manifest.get("event_contract", {}) or {}).get("required_common_fields", [])
    if isinstance(from_manifest, list) and from_manifest:
        return [str(x) for x in from_manifest]

    from_profile = (profile.get("event_contract", {}) or {}).get("required_common_fields", [])
    if isinstance(from_profile, list) and from_profile:
        return [str(x) for x in from_profile]

    return list(DEFAULT_COMMON_FIELDS)


def get_expected_version_fields(manifest: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[str, str]:
    schema_version = str(
        manifest.get("schema_version")
        or profile.get("schema_version")
        or DEFAULT_SCHEMA_VERSION
    ).strip() or DEFAULT_SCHEMA_VERSION
    manifest_version = str(
        manifest.get("manifest_version")
        or profile.get("manifest_version")
        or DEFAULT_MANIFEST_VERSION
    ).strip() or DEFAULT_MANIFEST_VERSION
    return schema_version, manifest_version


def extract_schema_contract(schema: Dict[str, Any]) -> Tuple[Set[str], Set[str], Dict[str, List[str]]]:
    events: Set[str] = set(DEFAULT_EVENTS)
    reasons: Set[str] = set(DEFAULT_REASON_CODES)
    event_required: Dict[str, List[str]] = {k: list(v) for k, v in DEFAULT_EVENT_REQUIRED_FIELDS.items()}

    if not schema:
        return events, reasons, event_required

    props = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}

    event_prop = props.get("event", {}) if isinstance(props.get("event"), dict) else {}
    schema_events = event_prop.get("enum", [])
    if isinstance(schema_events, list) and schema_events:
        events = {str(x) for x in schema_events}

    reason_prop = props.get("reason_code", {}) if isinstance(props.get("reason_code"), dict) else {}
    schema_reasons = reason_prop.get("enum", [])
    if isinstance(schema_reasons, list) and schema_reasons:
        reasons = {str(x) for x in schema_reasons}

    all_of = schema.get("allOf", []) if isinstance(schema.get("allOf"), list) else []
    for block in all_of:
        if not isinstance(block, dict):
            continue
        condition = block.get("if", {}) if isinstance(block.get("if"), dict) else {}
        then = block.get("then", {}) if isinstance(block.get("then"), dict) else {}
        cond_props = condition.get("properties", {}) if isinstance(condition.get("properties"), dict) else {}
        event_cond = cond_props.get("event", {}) if isinstance(cond_props.get("event"), dict) else {}
        event_name = str(event_cond.get("const", "")).strip()
        req = then.get("required", []) if isinstance(then.get("required"), list) else []
        if event_name and req:
            event_required[event_name] = [str(x) for x in req]

    return events, reasons, event_required


def evaluate_kill_switch(summary: Dict[str, Any], profile: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    ks = profile.get("kill_switch", {}) if isinstance(profile.get("kill_switch"), dict) else {}
    triggered_rules: List[str] = []
    triggered_from_events: List[str] = []

    for ev in events:
        if str(ev.get("event", "")).strip() != "kill_switch_triggered":
            continue
        reason = str(ev.get("reason_code", "")).strip()
        if reason:
            triggered_from_events.append(reason)

    spread_cfg = ks.get("spread_too_high", {}) if isinstance(ks.get("spread_too_high"), dict) else {}
    if bool(spread_cfg.get("enabled", False)):
        spread_trig = as_int(spread_cfg.get("trigger_on_entry_violation_count_ge"), 1)
        if as_int(summary.get("spread_gate_violations"), 0) >= max(1, spread_trig):
            triggered_rules.append("spread_too_high")

    skip_cfg = ks.get("too_many_skipped_bars", {}) if isinstance(ks.get("too_many_skipped_bars"), dict) else {}
    if bool(skip_cfg.get("enabled", False)):
        max_skip = as_int(skip_cfg.get("max_consecutive_skipped_bars"), 0)
        if max_skip > 0 and as_int(summary.get("max_consecutive_skipped_bars"), 0) > max_skip:
            triggered_rules.append("too_many_skipped_bars")

    api_cfg = ks.get("broker_api_failure", {}) if isinstance(ks.get("broker_api_failure"), dict) else {}
    if bool(api_cfg.get("enabled", False)):
        max_fail = as_int(api_cfg.get("max_consecutive_failures"), 0)
        if max_fail > 0 and as_int(summary.get("max_consecutive_api_failures"), 0) >= max_fail:
            triggered_rules.append("broker_api_failure")

    dd_cfg = ks.get("daily_drawdown_breach", {}) if isinstance(ks.get("daily_drawdown_breach"), dict) else {}
    if bool(dd_cfg.get("enabled", False)):
        max_daily_dd = abs(as_float(dd_cfg.get("max_daily_drawdown_usd"), 0.0))
        if max_daily_dd > 0:
            breaches = [
                d
                for d, v in (summary.get("daily_pnl", {}) or {}).items()
                if as_float(v) <= -max_daily_dd
            ]
            if breaches:
                summary["daily_drawdown_breach_days"] = breaches
                triggered_rules.append("daily_drawdown_breach")

    triggered = sorted(set(triggered_from_events + triggered_rules))
    return {
        "triggered": triggered,
        "triggered_from_events": sorted(set(triggered_from_events)),
        "triggered_from_rules": sorted(set(triggered_rules)),
        "is_halted": bool(triggered),
    }


def to_markdown(payload: Dict[str, Any]) -> str:
    s = payload.get("summary", {})
    checks = payload.get("checks", {})
    ks = payload.get("kill_switch", {})
    labels = payload.get("deployment_labels", {})
    lines = [
        "# Strategy 1 Paper Trading Mode Report",
        "",
        f"- Generated (UTC): `{payload.get('generated_utc', '')}`",
        f"- Profile: `{payload.get('profile_path', '')}`",
        f"- Manifest: `{payload.get('manifest_path', '')}`",
        f"- Schema: `{payload.get('schema_path', '')}`",
        f"- Stage: `{labels.get('strategy_1', '')}` | Promotion target: `{labels.get('promotion_target', '')}`",
        f"- Strategy 2 deterministic stage: `{labels.get('strategy_2_deterministic', '')}`",
        f"- RL/RLM stage: `{labels.get('rlm_rl', '')}`",
        "",
        "## Window",
        "",
        f"- Canonical window start (UTC): `{payload.get('canonical_window_start_utc', '')}`",
        f"- Start (UTC): `{payload.get('window_start_utc', '')}`",
        f"- End (UTC): `{payload.get('window_end_utc', '')}`",
        f"- Days covered: `{payload.get('paper_days', 0)}`",
        "",
        "## Summary",
        "",
        f"- status: `{payload.get('status', '')}`",
        f"- events: `{as_int(s.get('total_events'))}` | trades: `{as_int(s.get('trade_count'))}`",
        f"- source_events_total_unfiltered: `{as_int(s.get('source_events_total_unfiltered'))}` | source_events_excluded_before_window: `{as_int(s.get('source_events_excluded_before_window'))}`",
        f"- ingestion_state: `{s.get('ingestion_state', '')}` | first_event_ts_utc: `{s.get('first_event_ts_utc', '')}` | last_event_ts_utc: `{s.get('last_event_ts_utc', '')}`",
        f"- net_pnl_usd: `{as_float(s.get('net_pnl_usd')):.2f}` | win_rate: `{as_float(s.get('win_rate')):.2f}` | pf: `{as_float(s.get('profit_factor')):.2f}`",
        f"- max_dd_usd: `{as_float(s.get('max_drawdown_usd')):.2f}` | avg_slippage_pips: `{as_float(s.get('avg_slippage_pips')):.3f}` | avg_fill_spread_pips: `{as_float(s.get('avg_fill_spread_pips')):.3f}`",
        f"- spread_gate_skips: `{as_int(s.get('spread_gate_skips'))}` | session_cap_skips: `{as_int(s.get('session_cap_skips'))}`",
        f"- spread_gate_violations: `{as_int(s.get('spread_gate_violations'))}` | session_cap_violations: `{as_int(s.get('session_cap_violations'))}`",
        f"- max_consecutive_skipped_bars: `{as_int(s.get('max_consecutive_skipped_bars'))}` | max_consecutive_api_failures: `{as_int(s.get('max_consecutive_api_failures'))}`",
        f"- last_trade_age_minutes: `{s.get('last_trade_age_minutes', 'n/a')}`",
        "",
        "## Contract Integrity",
        "",
        f"- missing_common_field_events: `{as_int(s.get('missing_common_field_events'))}`",
        f"- missing_event_field_events: `{as_int(s.get('missing_event_field_events'))}`",
        f"- unknown_event_value_events: `{as_int(s.get('unknown_event_value_events'))}`",
        f"- unknown_reason_code_events: `{as_int(s.get('unknown_reason_code_events'))}`",
        f"- profile_hash_missing_events: `{as_int(s.get('profile_hash_missing_events'))}` | profile_hash_mismatch_events: `{as_int(s.get('profile_hash_mismatch_events'))}`",
        f"- manifest_hash_missing_events: `{as_int(s.get('manifest_hash_missing_events'))}` | manifest_hash_mismatch_events: `{as_int(s.get('manifest_hash_mismatch_events'))}` | manifest_hash_legacy_fallback_events: `{as_int(s.get('manifest_hash_legacy_fallback_events'))}`",
        f"- schema_version_mismatch_events: `{as_int(s.get('schema_version_mismatch_events'))}` | manifest_version_mismatch_events: `{as_int(s.get('manifest_version_mismatch_events'))}`",
        f"- invalid_run_id_events: `{as_int(s.get('invalid_run_id_events'))}` | run_id_hash_prefix_mismatch_events: `{as_int(s.get('run_id_hash_prefix_mismatch_events'))}`",
        f"- monotonic_order_violations: `{as_int(s.get('monotonic_order_violations'))}`",
        f"- lifecycle_fill_without_submit: `{as_int(s.get('lifecycle_fill_without_submit_count'))}` | lifecycle_close_without_fill: `{as_int(s.get('lifecycle_close_without_fill_count'))}`",
        f"- lifecycle_duplicate_close: `{as_int(s.get('lifecycle_duplicate_close_count'))}` | lifecycle_mixed_side: `{as_int(s.get('lifecycle_mixed_side_count'))}`",
        f"- contract_violation_days: `{len((s.get('contract_violation_counters_by_day') or {}))}`",
        "",
        "## Checks",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- {name}: `{bool(value)}`")

    lines += [
        "",
        "## Kill Switch",
        "",
        f"- halted: `{bool(ks.get('is_halted', False))}`",
        f"- triggered: `{', '.join(ks.get('triggered', [])) if ks.get('triggered') else 'none'}`",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Strategy 1 paper-trading mode report.")
    p.add_argument("--profile", default="strategy_1_profile.json")
    p.add_argument("--manifest", default="manifest.json")
    p.add_argument("--schema", default="", help="Optional schema override path.")
    p.add_argument("--events-jsonl", default="", help="Optional JSONL/JSON events file.")
    p.add_argument("--window-start", default="", help="Optional ISO-8601 UTC window start.")
    p.add_argument("--window-end", default="", help="Optional ISO-8601 UTC window end.")
    p.add_argument("--allow-non-monotonic", action="store_true", help="Allow non-monotonic event timestamps within a run.")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--out-prefix", default="strategy_1_paper_mode")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    profile_path = Path(args.profile)
    if not profile_path.is_absolute():
        profile_path = ROOT / profile_path
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")
    profile = load_json(profile_path)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        manifest = load_json(manifest_path)

    schema_path: Optional[Path] = None
    if str(args.schema).strip():
        schema_path = Path(args.schema)
        if not schema_path.is_absolute():
            schema_path = ROOT / schema_path
    else:
        schema_ref = str((manifest.get("event_contract", {}) or {}).get("schema_path", "")).strip()
        if not schema_ref:
            schema_ref = str((profile.get("event_contract", {}) or {}).get("schema_path", "")).strip()
        if schema_ref:
            schema_path = Path(schema_ref)
            if not schema_path.is_absolute():
                schema_path = ROOT / schema_path

    schema: Dict[str, Any] = {}
    if schema_path is not None and schema_path.exists():
        schema = load_json(schema_path)

    required_common_fields = get_required_common_fields(manifest=manifest, profile=profile)
    expected_schema_version, expected_manifest_version = get_expected_version_fields(manifest=manifest, profile=profile)
    run_id_pattern_text = str((manifest.get("event_contract", {}) or {}).get("run_id_pattern", "")).strip()
    if not run_id_pattern_text:
        run_id_pattern_text = str((profile.get("event_contract", {}) or {}).get("run_id_pattern", "")).strip()
    run_id_regex = RUN_ID_REGEX
    if run_id_pattern_text:
        try:
            run_id_regex = re.compile(run_id_pattern_text)
        except re.error:
            run_id_regex = RUN_ID_REGEX
    schema_events, schema_reasons, event_required_fields = extract_schema_contract(schema=schema)

    expected_manifest_hash = ""
    expected_profile_hash = ""
    if manifest_path.exists():
        expected_manifest_hash = f"sha256:{canonical_json_sha256(manifest_path)}"
    if profile_path.exists():
        expected_profile_hash = f"sha256:{canonical_json_sha256(profile_path)}"
    expected_profile_hashes: Set[str] = set()
    if expected_profile_hash:
        expected_profile_hashes.add(expected_profile_hash)
    if expected_manifest_hash:
        # Compatibility: older logs used manifest hash in profile_hash.
        expected_profile_hashes.add(expected_manifest_hash)

    events_path: Optional[Path] = None
    if str(args.events_jsonl).strip():
        events_path = Path(args.events_jsonl)
        if not events_path.is_absolute():
            events_path = ROOT / events_path

    source_events = load_events(events_path)
    source_events_total_unfiltered = int(len(source_events))

    configured_window_start = str(args.window_start).strip()
    if not configured_window_start:
        configured_window_start = str((profile.get("reporting", {}) or {}).get("canonical_window_start_utc", "")).strip()
    if not configured_window_start:
        configured_window_start = str((manifest.get("event_contract", {}) or {}).get("canonical_window_start_utc", "")).strip()
    canonical_window_start = parse_ts(configured_window_start) if configured_window_start else None
    if configured_window_start and canonical_window_start is None:
        raise ValueError(f"Invalid --window-start / canonical_window_start_utc: {configured_window_start}")

    if canonical_window_start is None:
        events = source_events
    else:
        events = []
        for ev in source_events:
            ts = event_ts(ev)
            if ts is not None and ts >= canonical_window_start:
                events.append(ev)
    source_events_excluded_before_window = int(source_events_total_unfiltered - len(events))

    violation_counts_by_day: Dict[str, Counter[str]] = defaultdict(Counter)
    monotonic_order_violations = 0
    monotonic_order_examples: List[Dict[str, Any]] = []
    if events:
        last_ts_by_run: Dict[str, datetime] = {}
        for idx, ev in enumerate(events):
            run_id = str(ev.get("run_id", "")).strip() or "__missing_run_id__"
            ts = event_ts(ev)
            if ts is None:
                continue
            prev = last_ts_by_run.get(run_id)
            if prev is not None and ts < prev:
                monotonic_order_violations += 1
                violation_counts_by_day[day_key(ts)]["monotonic_order_violation"] += 1
                if len(monotonic_order_examples) < 5:
                    monotonic_order_examples.append(
                        {
                            "index": idx,
                            "run_id": run_id,
                            "prev_ts": prev.isoformat(),
                            "ts": ts.isoformat(),
                        }
                    )
            else:
                last_ts_by_run[run_id] = ts

    enriched: List[Tuple[Optional[datetime], Dict[str, Any]]] = [(event_ts(ev), ev) for ev in events]
    enriched.sort(key=lambda x: x[0] or datetime.min.replace(tzinfo=timezone.utc))

    controls = profile.get("controls", {}) if isinstance(profile.get("controls"), dict) else {}
    max_trades_per_session = max(1, as_int(controls.get("max_trades_per_session"), 1))
    max_spread_pips = spread_threshold_pips(controls)

    event_counts: Counter[str] = Counter()
    skip_reason_counts: Counter[str] = Counter()
    profile_hash_values: Counter[str] = Counter()
    manifest_hash_values: Counter[str] = Counter()
    stage_values: Counter[str] = Counter()
    strategy_values: Counter[str] = Counter()
    run_id_values: Counter[str] = Counter()

    missing_common_field_events = 0
    missing_event_field_events = 0
    unknown_event_value_events = 0
    unknown_reason_code_events = 0
    profile_hash_missing_events = 0
    profile_hash_mismatch_events = 0
    manifest_hash_missing_events = 0
    manifest_hash_mismatch_events = 0
    manifest_hash_legacy_fallback_events = 0
    schema_version_mismatch_events = 0
    manifest_version_mismatch_events = 0
    invalid_run_id_events = 0
    run_id_hash_prefix_mismatch_events = 0

    pnl_rows: List[Tuple[Optional[datetime], float]] = []
    session_trade_ids: Dict[str, Set[str]] = defaultdict(set)
    slippage_pips: List[float] = []
    fill_spread_pips: List[float] = []
    submit_ids: Set[Tuple[str, str]] = set()
    fill_ids: Set[Tuple[str, str]] = set()
    close_ids: Set[Tuple[str, str]] = set()
    close_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    trade_sides: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    trade_first_day: Dict[Tuple[str, str], str] = {}
    fill_trade_day: Dict[Tuple[str, str], str] = {}
    close_trade_day: Dict[Tuple[str, str], str] = {}

    spread_gate_violations = 0
    spread_gate_skips = 0
    session_cap_skips = 0

    skipped_total = 0
    skipped_run = 0
    max_skipped_run = 0
    api_failure_total = 0
    api_fail_run = 0
    max_api_fail_run = 0

    kill_switch_check_pass = 0
    kill_switch_check_fail = 0

    last_trade_ts: Optional[datetime] = None

    for idx, (ts, ev) in enumerate(enriched):
        event_name = str(ev.get("event", "")).strip()
        reason_code = str(ev.get("reason_code", "")).strip()
        event_day = day_key(ts)

        if event_name:
            event_counts[event_name] += 1
            if schema_events and event_name not in schema_events:
                unknown_event_value_events += 1
                violation_counts_by_day[event_day]["unknown_event_value"] += 1
        else:
            unknown_event_value_events += 1
            violation_counts_by_day[event_day]["unknown_event_value"] += 1

        if reason_code:
            if schema_reasons and reason_code not in schema_reasons:
                unknown_reason_code_events += 1
                violation_counts_by_day[event_day]["unknown_reason_code"] += 1

        missing_common_fields = event_missing_fields(ev, required_common_fields)
        if "manifest_hash" in missing_common_fields:
            legacy_profile_hash = normalize_profile_hash(ev.get("profile_hash"))
            if expected_manifest_hash and legacy_profile_hash == expected_manifest_hash:
                missing_common_fields = [f for f in missing_common_fields if f != "manifest_hash"]
        if missing_common_fields:
            missing_common_field_events += 1
            violation_counts_by_day[event_day]["missing_common_fields"] += 1

        required_for_event = event_required_fields.get(event_name, [])
        if required_for_event and event_missing_fields(ev, required_for_event):
            missing_event_field_events += 1
            violation_counts_by_day[event_day]["missing_event_fields"] += 1

        normalized_hash = normalize_profile_hash(ev.get("profile_hash"))
        if normalized_hash:
            profile_hash_values[normalized_hash] += 1
            if expected_profile_hashes and normalized_hash not in expected_profile_hashes:
                profile_hash_mismatch_events += 1
                violation_counts_by_day[event_day]["profile_hash_mismatch"] += 1
        else:
            profile_hash_missing_events += 1
            violation_counts_by_day[event_day]["profile_hash_missing"] += 1

        normalized_manifest_hash = normalize_manifest_hash(ev.get("manifest_hash"))
        if not normalized_manifest_hash:
            legacy_profile_hash = normalize_profile_hash(ev.get("profile_hash"))
            if expected_manifest_hash and legacy_profile_hash == expected_manifest_hash:
                normalized_manifest_hash = expected_manifest_hash
                manifest_hash_legacy_fallback_events += 1
        if normalized_manifest_hash:
            manifest_hash_values[normalized_manifest_hash] += 1
            if expected_manifest_hash and normalized_manifest_hash != expected_manifest_hash:
                manifest_hash_mismatch_events += 1
                violation_counts_by_day[event_day]["manifest_hash_mismatch"] += 1
        else:
            manifest_hash_missing_events += 1
            violation_counts_by_day[event_day]["manifest_hash_missing"] += 1

        stage = str(ev.get("stage", "")).strip()
        if stage:
            stage_values[stage] += 1

        strategy_id = str(ev.get("strategy_id", "")).strip()
        if strategy_id:
            strategy_values[strategy_id] += 1

        run_id = str(ev.get("run_id", "")).strip()
        if run_id:
            run_id_values[run_id] += 1
            run_match = run_id_regex.match(run_id)
            if not run_match:
                invalid_run_id_events += 1
                violation_counts_by_day[event_day]["invalid_run_id"] += 1
            else:
                run_hash_prefix = str(run_match.groupdict().get("hash", "")).strip().lower()
                normalized_hash = normalize_profile_hash(ev.get("profile_hash"))
                if normalized_hash and run_hash_prefix:
                    event_hash_prefix = normalized_hash.replace("sha256:", "")[:8]
                    if run_hash_prefix != event_hash_prefix:
                        run_id_hash_prefix_mismatch_events += 1
                        violation_counts_by_day[event_day]["run_id_hash_prefix_mismatch"] += 1

        schema_version = str(ev.get("schema_version", "")).strip()
        if schema_version and schema_version != expected_schema_version:
            schema_version_mismatch_events += 1
            violation_counts_by_day[event_day]["schema_version_mismatch"] += 1
        manifest_version = str(ev.get("manifest_version", "")).strip()
        if manifest_version and manifest_version != expected_manifest_version:
            manifest_version_mismatch_events += 1
            violation_counts_by_day[event_day]["manifest_version_mismatch"] += 1

        if event_name == "position_closed" and "pnl_usd" in ev:
            pnl_rows.append((ts, as_float(ev.get("pnl_usd"), 0.0)))

        if event_name in {"order_submitted", "order_filled", "position_closed"} and ts is not None:
            last_trade_ts = ts if last_trade_ts is None or ts > last_trade_ts else last_trade_ts

        if event_name in {"order_submitted", "order_filled"}:
            sess = session_key(ev, ts)
            trade_id = str(ev.get("trade_id", "")).strip()
            if not trade_id:
                trade_id = f"{sess}::idx{idx}"
            session_trade_ids[sess].add(trade_id)
            trade_key = (sess, trade_id)
            if trade_key not in trade_first_day:
                trade_first_day[trade_key] = event_day
            side = str(ev.get("side", "")).strip().lower()
            if side in {"long", "short"}:
                trade_sides[trade_key].add(side)
            if event_name == "order_submitted":
                submit_ids.add(trade_key)
            elif event_name == "order_filled":
                fill_ids.add(trade_key)
                fill_trade_day[trade_key] = event_day

            spread_pips = as_float(ev.get("spread_pips"), 0.0)
            local_max_spread_pips = as_float(ev.get("max_spread_pips"), max_spread_pips)
            if local_max_spread_pips > 0 and spread_pips > local_max_spread_pips:
                spread_gate_violations += 1

        if event_name == "order_filled":
            slippage_pips.append(as_float(ev.get("slippage_pips"), 0.0))
            if "spread_pips" in ev:
                fill_spread_pips.append(as_float(ev.get("spread_pips"), 0.0))

        if event_name == "trade_skipped":
            skipped_total += 1
            skipped_run += 1
            max_skipped_run = max(max_skipped_run, skipped_run)
            if reason_code:
                skip_reason_counts[reason_code] += 1
            if reason_code == "spread_gate":
                spread_gate_skips += 1
            if reason_code == "session_cap":
                session_cap_skips += 1
        else:
            skipped_run = 0

        api_failure_event = (
            reason_code == "broker_api_failure"
            or (event_name == "kill_switch_triggered" and reason_code == "broker_api_failure")
            or str(ev.get("api_status", "")).strip().lower() in {"error", "failed", "failure", "disconnected"}
        )
        if api_failure_event:
            api_failure_total += 1
            api_fail_run += 1
            max_api_fail_run = max(max_api_fail_run, api_fail_run)
        else:
            api_fail_run = 0

        if event_name == "kill_switch_check":
            status = str(ev.get("status", "")).strip().lower()
            if status in {"pass", "ok", "healthy", "within_limits"}:
                kill_switch_check_pass += 1
            else:
                kill_switch_check_fail += 1

        if event_name == "position_closed":
            sess = session_key(ev, ts)
            trade_id = str(ev.get("trade_id", "")).strip()
            if not trade_id:
                trade_id = f"{sess}::idx{idx}"
            trade_key = (sess, trade_id)
            close_ids.add(trade_key)
            close_counts[trade_key] = int(close_counts.get(trade_key, 0) + 1)
            close_trade_day[trade_key] = event_day
            if trade_key not in trade_first_day:
                trade_first_day[trade_key] = event_day
            side = str(ev.get("side", "")).strip().lower()
            if side in {"long", "short"}:
                trade_sides[trade_key].add(side)

    session_cap_violations = int(
        sum(max(0, len(trade_ids) - max_trades_per_session) for trade_ids in session_trade_ids.values())
    )
    fill_without_submit_ids = sorted([f"{run}::{tid}" for run, tid in (fill_ids - submit_ids)])
    close_without_fill_ids = sorted([f"{run}::{tid}" for run, tid in (close_ids - fill_ids)])
    duplicate_close_ids = sorted([f"{run}::{tid}" for (run, tid), n in close_counts.items() if int(n) > 1])
    mixed_side_ids = sorted(
        [f"{run}::{tid}" for (run, tid), sides in trade_sides.items() if len({s for s in sides if s}) > 1]
    )
    for trade_key in (fill_ids - submit_ids):
        violation_counts_by_day[fill_trade_day.get(trade_key, trade_first_day.get(trade_key, "unknown"))][
            "lifecycle_fill_without_submit"
        ] += 1
    for trade_key in (close_ids - fill_ids):
        violation_counts_by_day[close_trade_day.get(trade_key, trade_first_day.get(trade_key, "unknown"))][
            "lifecycle_close_without_fill"
        ] += 1
    for trade_key, count in close_counts.items():
        if int(count) > 1:
            violation_counts_by_day[close_trade_day.get(trade_key, trade_first_day.get(trade_key, "unknown"))][
                "lifecycle_duplicate_close"
            ] += 1
    for trade_key, sides in trade_sides.items():
        if len({s for s in sides if s}) > 1:
            violation_counts_by_day[trade_first_day.get(trade_key, "unknown")]["lifecycle_mixed_side"] += 1

    pnl_values = [p for _, p in pnl_rows]
    trade_count = len(pnl_values)
    wins = int(sum(1 for p in pnl_values if p > 0.0))
    losses = int(sum(1 for p in pnl_values if p < 0.0))
    net_pnl = float(sum(pnl_values))
    gross_profit = float(sum(p for p in pnl_values if p > 0.0))
    gross_loss = float(abs(sum(p for p in pnl_values if p < 0.0)))
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    win_rate = float(wins / trade_count) if trade_count > 0 else 0.0
    dd = max_drawdown(pnl_values)
    daily_pnl = daily_pnl_series(pnl_rows)

    event_times = [ts for ts, _ in enriched if ts is not None]
    first_event_ts = min(event_times) if event_times else None
    last_event_ts = max(event_times) if event_times else None
    cli_start = canonical_window_start
    cli_end = parse_ts(str(args.window_end)) if str(args.window_end).strip() else None
    window_start = cli_start or (min(event_times) if event_times else None)
    window_end = cli_end or (max(event_times) if event_times else None)

    paper_days = 0
    if window_start is not None and window_end is not None and window_end >= window_start:
        paper_days = int((window_end.date() - window_start.date()).days + 1)

    last_trade_age_minutes: Optional[int] = None
    reference_end = window_end or (utc_now() if event_times else None)
    if last_trade_ts is not None and reference_end is not None and reference_end >= last_trade_ts:
        last_trade_age_minutes = int((reference_end - last_trade_ts).total_seconds() // 60)
    signal_count = as_int(event_counts.get("signal_evaluated"), 0)
    closed_count = as_int(event_counts.get("position_closed"), 0)
    if len(events) == 0:
        ingestion_state = "NO_PIPELINE_INPUT"
    elif signal_count == 0:
        ingestion_state = "NO_SIGNALS"
    elif closed_count == 0:
        ingestion_state = "NO_TRADES"
    else:
        ingestion_state = "ACTIVE"

    paper_rules = profile.get("paper_validation_window", {}) if isinstance(profile.get("paper_validation_window"), dict) else {}
    min_days = as_int(paper_rules.get("min_days"), 0)
    min_trades = as_int(paper_rules.get("min_trades"), 0)

    expected_stage = str(manifest.get("stage", profile.get("stage", ""))).strip()
    expected_strategy_id = str(manifest.get("strategy_id", profile.get("strategy_id", "strategy_1"))).strip() or "strategy_1"

    stage_consistent = all(stage == expected_stage for stage in stage_values) if stage_values else True
    strategy_id_consistent = all(sid == expected_strategy_id for sid in strategy_values) if strategy_values else True
    has_events = bool(events)
    # Day-0/no-input runs should be handled by ingestion checks, not treated as hash violations.
    profile_hash_consistent = profile_hash_missing_events == 0 and (
        not has_events
        or len(profile_hash_values) == 1
        or (expected_profile_hashes and set(profile_hash_values.keys()).issubset(expected_profile_hashes))
    )
    profile_hash_matches_expected = (
        profile_hash_missing_events == 0
        and (
            not has_events
            or not expected_profile_hashes
            or profile_hash_mismatch_events == 0
        )
    )
    manifest_hash_consistent = manifest_hash_missing_events == 0 and (not has_events or len(manifest_hash_values) == 1)
    manifest_hash_matches_expected = (
        manifest_hash_missing_events == 0
        and (
            not has_events
            or not expected_manifest_hash
            or manifest_hash_mismatch_events == 0
        )
    )
    schema_version_matches_expected = schema_version_mismatch_events == 0
    manifest_version_matches_expected = manifest_version_mismatch_events == 0
    run_id_format_valid = invalid_run_id_events == 0
    run_id_hash_prefix_valid = run_id_hash_prefix_mismatch_events == 0
    monotonic_event_order_pass = bool(args.allow_non_monotonic or monotonic_order_violations == 0)
    trade_lifecycle_complete = (
        len(fill_without_submit_ids) == 0
        and len(close_without_fill_ids) == 0
        and len(duplicate_close_ids) == 0
        and len(mixed_side_ids) == 0
    )
    contract_violation_counters = {
        "schema_version_mismatch": int(schema_version_mismatch_events),
        "manifest_version_mismatch": int(manifest_version_mismatch_events),
        "invalid_run_id": int(invalid_run_id_events),
        "run_id_hash_prefix_mismatch": int(run_id_hash_prefix_mismatch_events),
        "profile_hash_mismatch": int(profile_hash_mismatch_events),
        "profile_hash_missing": int(profile_hash_missing_events),
        "manifest_hash_mismatch": int(manifest_hash_mismatch_events),
        "manifest_hash_missing": int(manifest_hash_missing_events),
        "monotonic_order_violations": int(monotonic_order_violations),
        "lifecycle_fill_without_submit": int(len(fill_without_submit_ids)),
        "lifecycle_close_without_fill": int(len(close_without_fill_ids)),
        "lifecycle_duplicate_close": int(len(duplicate_close_ids)),
        "lifecycle_mixed_side": int(len(mixed_side_ids)),
    }
    contract_violation_counters_by_day = {
        day: dict(sorted(counter.items()))
        for day, counter in sorted(violation_counts_by_day.items())
    }

    summary: Dict[str, Any] = {
        "total_events": int(len(events)),
        "source_events_total_unfiltered": int(source_events_total_unfiltered),
        "source_events_excluded_before_window": int(source_events_excluded_before_window),
        "canonical_window_start_utc": canonical_window_start.isoformat() if canonical_window_start else "",
        "first_event_ts_utc": first_event_ts.isoformat() if first_event_ts else "",
        "last_event_ts_utc": last_event_ts.isoformat() if last_event_ts else "",
        "ingestion_state": ingestion_state,
        "trade_count": int(trade_count),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": float(win_rate),
        "net_pnl_usd": float(net_pnl),
        "gross_profit_usd": float(gross_profit),
        "gross_loss_usd": float(gross_loss),
        "profit_factor": float(profit_factor),
        "max_drawdown_usd": float(dd),
        "daily_pnl": daily_pnl,
        "event_counts": dict(sorted(event_counts.items())),
        "skip_reason_counts": dict(sorted(skip_reason_counts.items())),
        "spread_gate_skips": int(spread_gate_skips),
        "session_cap_skips": int(session_cap_skips),
        "spread_gate_violations": int(spread_gate_violations),
        "session_cap_violations": int(session_cap_violations),
        "skipped_bars": int(skipped_total),
        "max_consecutive_skipped_bars": int(max_skipped_run),
        "api_failure_count": int(api_failure_total),
        "max_consecutive_api_failures": int(max_api_fail_run),
        "avg_slippage_pips": float(safe_mean(slippage_pips)),
        "avg_fill_spread_pips": float(safe_mean(fill_spread_pips)),
        "last_trade_age_minutes": last_trade_age_minutes,
        "kill_switch_check_pass": int(kill_switch_check_pass),
        "kill_switch_check_fail": int(kill_switch_check_fail),
        "missing_common_field_events": int(missing_common_field_events),
        "missing_event_field_events": int(missing_event_field_events),
        "unknown_event_value_events": int(unknown_event_value_events),
        "unknown_reason_code_events": int(unknown_reason_code_events),
        "profile_hash_missing_events": int(profile_hash_missing_events),
        "profile_hash_mismatch_events": int(profile_hash_mismatch_events),
        "manifest_hash_missing_events": int(manifest_hash_missing_events),
        "manifest_hash_mismatch_events": int(manifest_hash_mismatch_events),
        "manifest_hash_legacy_fallback_events": int(manifest_hash_legacy_fallback_events),
        "schema_version_expected": expected_schema_version,
        "manifest_version_expected": expected_manifest_version,
        "schema_version_mismatch_events": int(schema_version_mismatch_events),
        "manifest_version_mismatch_events": int(manifest_version_mismatch_events),
        "invalid_run_id_events": int(invalid_run_id_events),
        "run_id_hash_prefix_mismatch_events": int(run_id_hash_prefix_mismatch_events),
        "run_ids_seen": dict(run_id_values),
        "monotonic_order_violations": int(monotonic_order_violations),
        "monotonic_order_violation_examples": monotonic_order_examples,
        "lifecycle_fill_without_submit_count": int(len(fill_without_submit_ids)),
        "lifecycle_close_without_fill_count": int(len(close_without_fill_ids)),
        "lifecycle_duplicate_close_count": int(len(duplicate_close_ids)),
        "lifecycle_mixed_side_count": int(len(mixed_side_ids)),
        "lifecycle_fill_without_submit_ids": fill_without_submit_ids[:20],
        "lifecycle_close_without_fill_ids": close_without_fill_ids[:20],
        "lifecycle_duplicate_close_ids": duplicate_close_ids[:20],
        "lifecycle_mixed_side_ids": mixed_side_ids[:20],
        "contract_violation_counters": contract_violation_counters,
        "contract_violation_counters_by_day": contract_violation_counters_by_day,
        "profile_hash_values_seen": dict(profile_hash_values),
        "manifest_hash_expected": expected_manifest_hash,
        "manifest_hash_values_seen": dict(manifest_hash_values),
        "stage_values_seen": dict(stage_values),
        "strategy_values_seen": dict(strategy_values),
    }

    checks = {
        "spread_gate_active": bool(controls.get("live_spread_gate_active", False)),
        "one_trade_per_session_cap": int(max_trades_per_session) == 1,
        "frozen_parameters_recorded": bool((manifest.get("frozen_parameters") or profile.get("frozen_parameters"))),
        "paper_window_pass": bool(paper_days >= min_days and trade_count >= min_trades),
        "event_contract_common_fields_pass": missing_common_field_events == 0,
        "event_contract_event_fields_pass": missing_event_field_events == 0 and unknown_event_value_events == 0,
        "schema_event_values_valid": unknown_event_value_events == 0,
        "schema_reason_values_valid": unknown_reason_code_events == 0,
        "profile_hash_consistent": profile_hash_consistent,
        "profile_hash_matches_expected": profile_hash_matches_expected,
        "manifest_hash_consistent": manifest_hash_consistent,
        "manifest_hash_matches_expected": manifest_hash_matches_expected,
        "schema_version_matches_expected": schema_version_matches_expected,
        "manifest_version_matches_expected": manifest_version_matches_expected,
        "run_id_format_valid": run_id_format_valid,
        "run_id_hash_prefix_valid": run_id_hash_prefix_valid,
        "monotonic_event_order_pass": monotonic_event_order_pass,
        "trade_lifecycle_complete": trade_lifecycle_complete,
        "stage_consistent": stage_consistent,
        "strategy_id_consistent": strategy_id_consistent,
    }

    kill_switch = evaluate_kill_switch(summary=summary, profile=profile, events=events)

    critical_checks = [
        "spread_gate_active",
        "one_trade_per_session_cap",
        "frozen_parameters_recorded",
        "event_contract_common_fields_pass",
        "event_contract_event_fields_pass",
        "schema_event_values_valid",
        "schema_reason_values_valid",
        "profile_hash_consistent",
        "profile_hash_matches_expected",
        "manifest_hash_consistent",
        "manifest_hash_matches_expected",
        "schema_version_matches_expected",
        "manifest_version_matches_expected",
        "run_id_format_valid",
        "run_id_hash_prefix_valid",
        "monotonic_event_order_pass",
        "trade_lifecycle_complete",
        "stage_consistent",
        "strategy_id_consistent",
    ]

    status = "PAPER_IN_PROGRESS"
    if bool(kill_switch.get("is_halted", False)):
        status = "HALT"
    elif bool(checks["paper_window_pass"]) and all(bool(checks[name]) for name in critical_checks):
        status = "PAPER_PASS"

    deployment_labels = {
        "strategy_1": str(profile.get("stage", manifest.get("stage", "PAPER_CANDIDATE"))),
        "strategy_2_deterministic": str((profile.get("branch_stages", {}) or {}).get("strategy_2_deterministic", "RESEARCH")),
        "rlm_rl": str((profile.get("branch_stages", {}) or {}).get("rlm_rl", "EXPERIMENTAL_ONLY")),
        "promotion_target": str(profile.get("promotion_target", manifest.get("promotion_target", "LIVE_GATED"))),
    }

    payload = {
        "generated_utc": utc_now().isoformat(),
        "profile_path": str(profile_path.resolve()),
        "manifest_path": str(manifest_path.resolve()) if manifest_path.exists() else "",
        "schema_path": str(schema_path.resolve()) if schema_path and schema_path.exists() else "",
        "events_path": str(events_path.resolve()) if events_path else "",
        "status": status,
        "deployment_labels": deployment_labels,
        "canonical_window_start_utc": canonical_window_start.isoformat() if canonical_window_start else "",
        "window_start_utc": window_start.isoformat() if window_start else "",
        "window_end_utc": window_end.isoformat() if window_end else "",
        "paper_days": int(paper_days),
        "summary": summary,
        "checks": checks,
        "kill_switch": kill_switch,
        "paper_validation_window": {
            "min_days": min_days,
            "min_trades": min_trades,
        },
        "event_contract": {
            "required_common_fields": required_common_fields,
            "run_id_pattern": run_id_regex.pattern,
            "schema_events": sorted(schema_events),
            "schema_reason_codes": sorted(schema_reasons),
        },
        "contract_versions": {
            "schema_version": expected_schema_version,
            "manifest_version": expected_manifest_version,
        },
        "expected_hashes": {
            "manifest_hash": expected_manifest_hash,
            "profile_hash": expected_profile_hash,
            "accepted_profile_hashes": sorted(expected_profile_hashes),
        },
        # Backward-compatible alias retained for existing downstream consumers.
        "expected_profile_hashes": {
            "manifest_hash": expected_manifest_hash,
            "profile_hash": expected_profile_hash,
        },
    }

    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = ROOT / reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = utc_now().strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"{args.out_prefix}_{ts}.json"
    md_path = reports_dir / f"{args.out_prefix}_{ts}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(payload), encoding="utf-8")

    print("=== DONE ===")
    print(f"status={status}")
    print(f"report_json={json_path}")
    print(f"report_md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
