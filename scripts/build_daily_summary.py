#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMON_FIELDS = ["ts", "run_id", "event", "stage", "strategy_id", "profile_hash", "schema_version", "manifest_version"]
DEFAULT_SCHEMA_VERSION = "1.0.0"
DEFAULT_MANIFEST_VERSION = "1.0.0"
RUN_ID_REGEX = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<session>[A-Z0-9]+)_sha(?P<hash>[a-f0-9]{8})$")


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


def load_events(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Events file not found: {path}")

    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("JSON events file must contain a list of objects.")
        return [x for x in raw if isinstance(x, dict)]

    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        obj = json.loads(text)
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


def resolve_common_fields(manifest: Dict[str, Any]) -> List[str]:
    req = (manifest.get("event_contract", {}) or {}).get("required_common_fields", [])
    if isinstance(req, list) and req:
        return [str(x) for x in req]
    return list(DEFAULT_COMMON_FIELDS)


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


def get_expected_versions(manifest: Dict[str, Any]) -> Tuple[str, str]:
    schema_version = str(manifest.get("schema_version") or DEFAULT_SCHEMA_VERSION).strip() or DEFAULT_SCHEMA_VERSION
    manifest_version = str(manifest.get("manifest_version") or DEFAULT_MANIFEST_VERSION).strip() or DEFAULT_MANIFEST_VERSION
    return schema_version, manifest_version


def normalize_profile_hash(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if text.startswith("sha256:"):
        return text
    return f"sha256:{text}"


def canonical_hash(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build daily_summary.json from canonical events.jsonl.")
    p.add_argument("--events-jsonl", default="events.jsonl")
    p.add_argument("--manifest", default="manifest.json")
    p.add_argument(
        "--window-start",
        default="",
        help="Optional ISO-8601 UTC lower bound; events older than this are excluded.",
    )
    p.add_argument("--date", default="", help="YYYY-MM-DD. Defaults to latest event day in UTC.")
    p.add_argument("--run-id", default="", help="Optional run_id filter.")
    p.add_argument("--strategy-id", default="strategy_1")
    p.add_argument("--out", default="daily_summary.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        manifest = load_json(manifest_path)
    expected_schema_version, expected_manifest_version = get_expected_versions(manifest)
    run_id_pattern_text = str((manifest.get("event_contract", {}) or {}).get("run_id_pattern", "")).strip()
    run_id_regex = RUN_ID_REGEX
    if run_id_pattern_text:
        try:
            run_id_regex = re.compile(run_id_pattern_text)
        except re.error:
            run_id_regex = RUN_ID_REGEX

    events_path = Path(args.events_jsonl)
    if not events_path.is_absolute():
        events_path = ROOT / events_path
    source_events = load_events(events_path)
    source_events_total_unfiltered = int(len(source_events))

    configured_window_start = str(args.window_start).strip()
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

    target_run_id = str(args.run_id).strip()
    target_strategy_id = str(args.strategy_id).strip() or "strategy_1"

    def contract_day(ts: datetime, ev: Dict[str, Any]) -> str:
        run_id = str(ev.get("run_id", "")).strip()
        if run_id:
            run_day_prefix = run_id.split("_", 1)[0].strip()
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", run_day_prefix):
                return run_day_prefix
            match = run_id_regex.match(run_id)
            if match:
                run_day = str(match.groupdict().get("date", "")).strip()
                if run_day:
                    return run_day
        return ts.date().isoformat()

    dated_events: List[Tuple[datetime, Dict[str, Any]]] = []
    for ev in events:
        ts = event_ts(ev)
        if ts is None:
            continue
        if target_run_id and str(ev.get("run_id", "")).strip() != target_run_id:
            continue
        if target_strategy_id and str(ev.get("strategy_id", "")).strip() != target_strategy_id:
            continue
        dated_events.append((ts, ev))

    if not dated_events:
        target_day = str(args.date).strip() or utc_now().date().isoformat()
        filtered_file_order: List[Tuple[datetime, Dict[str, Any]]] = []
        filtered: List[Tuple[datetime, Dict[str, Any]]] = []
    else:
        if str(args.date).strip():
            target_day = str(args.date).strip()
        else:
            target_day = max(contract_day(ts, ev) for ts, ev in dated_events)
        filtered_file_order = [(ts, ev) for ts, ev in dated_events if contract_day(ts, ev) == target_day]
        filtered = sorted(filtered_file_order, key=lambda x: x[0])

    monotonic_order_violations = 0
    last_ts_by_run: Dict[str, datetime] = {}
    for idx, (ts, ev) in enumerate(filtered_file_order):
        run_id = str(ev.get("run_id", "")).strip() or "__missing_run_id__"
        prev = last_ts_by_run.get(run_id)
        if prev is not None and ts < prev:
            monotonic_order_violations += 1
        else:
            last_ts_by_run[run_id] = ts

    required_common_fields = resolve_common_fields(manifest)
    first_event_ts = filtered[0][0] if filtered else None
    last_event_ts = filtered[-1][0] if filtered else None

    event_counts: Counter[str] = Counter()
    skip_reason_counts: Counter[str] = Counter()
    profile_hash_values: Counter[str] = Counter()
    run_id_values: Counter[str] = Counter()
    process_start_values: Counter[str] = Counter()

    pnl_values: List[float] = []
    slippage_values: List[float] = []
    fill_spread_values: List[float] = []

    missing_common_field_events = 0
    schema_version_mismatch_events = 0
    manifest_version_mismatch_events = 0
    invalid_run_id_events = 0
    run_id_hash_prefix_mismatch_events = 0
    spread_gate_skips = 0
    session_cap_skips = 0
    kill_switch_trigger_count = 0
    kill_switch_reasons: Counter[str] = Counter()
    last_process_start_ts: Optional[datetime] = None

    submitted_trade_ids: Set[Tuple[str, str]] = set()
    filled_trade_ids: Set[Tuple[str, str]] = set()
    closed_trade_ids: Set[Tuple[str, str]] = set()
    close_counts: Dict[Tuple[str, str], int] = {}
    trade_sides: Dict[Tuple[str, str], Set[str]] = {}

    for ts, ev in filtered:
        event_name = str(ev.get("event", "")).strip()
        reason_code = str(ev.get("reason_code", "")).strip()
        event_counts[event_name] += 1
        run_id = str(ev.get("run_id", "")).strip()
        if run_id:
            run_id_values[run_id] += 1
            run_match = run_id_regex.match(run_id)
            if not run_match:
                invalid_run_id_events += 1
            else:
                run_hash_prefix = str(run_match.groupdict().get("hash", "")).strip().lower()
                normalized_hash = normalize_profile_hash(ev.get("profile_hash"))
                if normalized_hash and run_hash_prefix:
                    event_hash_prefix = normalized_hash.replace("sha256:", "")[:8]
                    if run_hash_prefix != event_hash_prefix:
                        run_id_hash_prefix_mismatch_events += 1

        if event_missing_fields(ev, required_common_fields):
            missing_common_field_events += 1

        process_start_ts = parse_ts(str(ev.get("process_start_ts", "")))
        if process_start_ts is not None:
            process_start_values[process_start_ts.isoformat()] += 1
            if last_process_start_ts is None or process_start_ts > last_process_start_ts:
                last_process_start_ts = process_start_ts

        profile_hash = str(ev.get("profile_hash", "")).strip().lower()
        if profile_hash:
            profile_hash_values[profile_hash] += 1

        schema_version = str(ev.get("schema_version", "")).strip()
        if schema_version and schema_version != expected_schema_version:
            schema_version_mismatch_events += 1
        manifest_version = str(ev.get("manifest_version", "")).strip()
        if manifest_version and manifest_version != expected_manifest_version:
            manifest_version_mismatch_events += 1

        if event_name == "trade_skipped":
            if reason_code:
                skip_reason_counts[reason_code] += 1
            if reason_code == "spread_gate":
                spread_gate_skips += 1
            if reason_code == "session_cap":
                session_cap_skips += 1

        if event_name == "position_closed":
            pnl_values.append(as_float(ev.get("pnl_usd"), 0.0))
            trade_id = str(ev.get("trade_id", "")).strip()
            if trade_id:
                trade_key = (run_id, trade_id)
                closed_trade_ids.add(trade_key)
                close_counts[trade_key] = int(close_counts.get(trade_key, 0) + 1)
                side = str(ev.get("side", "")).strip().lower()
                if side in {"long", "short"}:
                    trade_sides.setdefault(trade_key, set()).add(side)

        if event_name == "order_submitted":
            trade_id = str(ev.get("trade_id", "")).strip()
            if trade_id:
                trade_key = (run_id, trade_id)
                submitted_trade_ids.add(trade_key)
                side = str(ev.get("side", "")).strip().lower()
                if side in {"long", "short"}:
                    trade_sides.setdefault(trade_key, set()).add(side)

        if event_name == "order_filled":
            trade_id = str(ev.get("trade_id", "")).strip()
            if trade_id:
                trade_key = (run_id, trade_id)
                filled_trade_ids.add(trade_key)
                side = str(ev.get("side", "")).strip().lower()
                if side in {"long", "short"}:
                    trade_sides.setdefault(trade_key, set()).add(side)
            if "slippage_pips" in ev:
                slippage_values.append(as_float(ev.get("slippage_pips"), 0.0))
            if "spread_pips" in ev:
                fill_spread_values.append(as_float(ev.get("spread_pips"), 0.0))

        if event_name == "kill_switch_triggered":
            kill_switch_trigger_count += 1
            if reason_code:
                kill_switch_reasons[reason_code] += 1

    trade_count = len(pnl_values)
    wins = int(sum(1 for p in pnl_values if p > 0.0))
    losses = int(sum(1 for p in pnl_values if p < 0.0))
    net_pnl_usd = float(sum(pnl_values))
    gross_profit_usd = float(sum(p for p in pnl_values if p > 0.0))
    gross_loss_usd = float(abs(sum(p for p in pnl_values if p < 0.0)))
    profit_factor = float(gross_profit_usd / gross_loss_usd) if gross_loss_usd > 0 else (float("inf") if gross_profit_usd > 0 else 0.0)

    filled_without_submit = sorted([f"{run}::{tid}" for run, tid in (filled_trade_ids - submitted_trade_ids)])
    closed_without_fill = sorted([f"{run}::{tid}" for run, tid in (closed_trade_ids - filled_trade_ids)])
    submitted_not_closed = sorted([f"{run}::{tid}" for run, tid in (submitted_trade_ids - closed_trade_ids)])
    duplicate_close = sorted([f"{run}::{tid}" for (run, tid), n in close_counts.items() if int(n) > 1])
    mixed_side = sorted(
        [f"{run}::{tid}" for (run, tid), sides in trade_sides.items() if len({s for s in sides if s}) > 1]
    )

    unmatched_trade_lifecycle = {
        "filled_without_submit": filled_without_submit,
        "closed_without_fill": closed_without_fill,
        "submitted_not_closed": submitted_not_closed,
        "duplicate_close": duplicate_close,
        "mixed_side": mixed_side,
    }
    contract_violation_counters = {
        "schema_version_mismatch": int(schema_version_mismatch_events),
        "manifest_version_mismatch": int(manifest_version_mismatch_events),
        "invalid_run_id": int(invalid_run_id_events),
        "run_id_hash_prefix_mismatch": int(run_id_hash_prefix_mismatch_events),
        "missing_common_fields": int(missing_common_field_events),
        "monotonic_order_violations": int(monotonic_order_violations),
        "lifecycle_fill_without_submit": int(len(filled_without_submit)),
        "lifecycle_close_without_fill": int(len(closed_without_fill)),
        "lifecycle_duplicate_close": int(len(duplicate_close)),
        "lifecycle_mixed_side": int(len(mixed_side)),
    }
    signal_count = as_int(event_counts.get("signal_evaluated"), 0)
    closed_count = as_int(event_counts.get("position_closed"), 0)
    process_start_count = int(len(process_start_values))
    if process_start_count == 0 and run_id_values:
        # Backward-compatible fallback for historical events written before process_start_ts existed.
        process_start_count = int(len(run_id_values))
    last_process_start_age_minutes: Optional[int] = None
    if last_process_start_ts is not None:
        delta_seconds = (utc_now() - last_process_start_ts).total_seconds()
        last_process_start_age_minutes = int(max(0.0, delta_seconds) // 60)
    if len(events) == 0:
        ingestion_state = "NO_PIPELINE_INPUT"
    elif len(filtered) == 0:
        ingestion_state = "NO_EVENTS_FOR_TARGET_DAY"
    elif signal_count == 0:
        ingestion_state = "NO_SIGNALS"
    elif closed_count == 0:
        ingestion_state = "NO_TRADES"
    else:
        ingestion_state = "ACTIVE"

    payload: Dict[str, Any] = {
        "generated_utc": utc_now().isoformat(),
        "schema_version": expected_schema_version,
        "manifest_version": expected_manifest_version,
        "canonical_window_start_utc": canonical_window_start.isoformat() if canonical_window_start else "",
        "date": target_day,
        "strategy_id": target_strategy_id,
        "run_id": target_run_id,
        "events_path": str(events_path.resolve()),
        "manifest_path": str(manifest_path.resolve()) if manifest_path.exists() else "",
        "summary": {
            "events_total": int(len(filtered)),
            "source_events_total": int(len(events)),
            "source_events_total_unfiltered": int(source_events_total_unfiltered),
            "source_events_excluded_before_window": int(source_events_excluded_before_window),
            "canonical_window_start_utc": canonical_window_start.isoformat() if canonical_window_start else "",
            "first_event_ts_utc": first_event_ts.isoformat() if first_event_ts else "",
            "last_event_ts_utc": last_event_ts.isoformat() if last_event_ts else "",
            "ingestion_state": ingestion_state,
            "event_counts": dict(sorted(event_counts.items())),
            "missing_common_field_events": int(missing_common_field_events),
            "schema_version_mismatch_events": int(schema_version_mismatch_events),
            "manifest_version_mismatch_events": int(manifest_version_mismatch_events),
            "invalid_run_id_events": int(invalid_run_id_events),
            "run_id_hash_prefix_mismatch_events": int(run_id_hash_prefix_mismatch_events),
            "run_ids_seen": dict(run_id_values),
            "process_start_count": int(process_start_count),
            "process_start_values_seen": dict(process_start_values),
            "last_process_start_ts_utc": last_process_start_ts.isoformat() if last_process_start_ts else "",
            "last_process_start_age_minutes": last_process_start_age_minutes,
            "monotonic_order_violations": int(monotonic_order_violations),
            "profile_hash_values_seen": dict(profile_hash_values),
            "trade_count": int(trade_count),
            "wins": int(wins),
            "losses": int(losses),
            "net_pnl_usd": net_pnl_usd,
            "gross_profit_usd": gross_profit_usd,
            "gross_loss_usd": gross_loss_usd,
            "profit_factor": float(profit_factor),
            "skip_reason_counts": dict(sorted(skip_reason_counts.items())),
            "spread_gate_skips": int(spread_gate_skips),
            "session_cap_skips": int(session_cap_skips),
            "avg_slippage_pips": float(safe_mean(slippage_values)),
            "avg_fill_spread_pips": float(safe_mean(fill_spread_values)),
            "kill_switch_trigger_count": int(kill_switch_trigger_count),
            "kill_switch_reasons": dict(sorted(kill_switch_reasons.items())),
            "contract_violation_counters": contract_violation_counters,
            "unmatched_trade_lifecycle": unmatched_trade_lifecycle,
            "trade_lifecycle_complete": bool(
                len(filled_without_submit) == 0
                and len(closed_without_fill) == 0
                and len(duplicate_close) == 0
                and len(mixed_side) == 0
            ),
        },
    }
    payload["reconciliation_hash"] = canonical_hash(
        {
            "schema_version": payload["schema_version"],
            "manifest_version": payload["manifest_version"],
            "date": payload["date"],
            "strategy_id": payload["strategy_id"],
            "run_id": payload["run_id"],
            "summary": payload["summary"],
        }
    )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== DONE ===")
    print(f"date={target_day}")
    print(f"events={len(filtered)}")
    print(f"summary_json={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
