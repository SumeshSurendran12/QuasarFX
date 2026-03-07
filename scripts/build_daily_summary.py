#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMON_FIELDS = ["ts", "run_id", "event", "stage", "strategy_id", "profile_hash"]


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build daily_summary.json from canonical events.jsonl.")
    p.add_argument("--events-jsonl", default="events.jsonl")
    p.add_argument("--manifest", default="manifest.json")
    p.add_argument("--date", default="", help="YYYY-MM-DD. Defaults to latest event day in UTC.")
    p.add_argument("--run-id", default="", help="Optional run_id filter.")
    p.add_argument("--strategy-id", default="strategy_1")
    p.add_argument("--out", default="daily_summary.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    events_path = Path(args.events_jsonl)
    if not events_path.is_absolute():
        events_path = ROOT / events_path
    events = load_events(events_path)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        manifest = load_json(manifest_path)

    target_run_id = str(args.run_id).strip()
    target_strategy_id = str(args.strategy_id).strip() or "strategy_1"

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
        raise ValueError("No matching events with valid timestamps for daily summary.")

    if str(args.date).strip():
        target_day = str(args.date).strip()
    else:
        target_day = max(ts.date().isoformat() for ts, _ in dated_events)

    filtered = [(ts, ev) for ts, ev in dated_events if ts.date().isoformat() == target_day]
    filtered.sort(key=lambda x: x[0])

    required_common_fields = resolve_common_fields(manifest)

    event_counts: Counter[str] = Counter()
    skip_reason_counts: Counter[str] = Counter()
    profile_hash_values: Counter[str] = Counter()

    pnl_values: List[float] = []
    slippage_values: List[float] = []
    fill_spread_values: List[float] = []

    missing_common_field_events = 0
    spread_gate_skips = 0
    session_cap_skips = 0
    kill_switch_trigger_count = 0
    kill_switch_reasons: Counter[str] = Counter()

    submitted_trade_ids: Set[str] = set()
    filled_trade_ids: Set[str] = set()
    closed_trade_ids: Set[str] = set()

    for ts, ev in filtered:
        event_name = str(ev.get("event", "")).strip()
        reason_code = str(ev.get("reason_code", "")).strip()
        event_counts[event_name] += 1

        if event_missing_fields(ev, required_common_fields):
            missing_common_field_events += 1

        profile_hash = str(ev.get("profile_hash", "")).strip().lower()
        if profile_hash:
            profile_hash_values[profile_hash] += 1

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
                closed_trade_ids.add(trade_id)

        if event_name == "order_submitted":
            trade_id = str(ev.get("trade_id", "")).strip()
            if trade_id:
                submitted_trade_ids.add(trade_id)

        if event_name == "order_filled":
            trade_id = str(ev.get("trade_id", "")).strip()
            if trade_id:
                filled_trade_ids.add(trade_id)
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

    unmatched_trade_lifecycle = {
        "filled_without_submit": sorted(list(filled_trade_ids - submitted_trade_ids)),
        "closed_without_fill": sorted(list(closed_trade_ids - filled_trade_ids)),
        "submitted_not_closed": sorted(list(submitted_trade_ids - closed_trade_ids)),
    }

    payload: Dict[str, Any] = {
        "generated_utc": utc_now().isoformat(),
        "date": target_day,
        "strategy_id": target_strategy_id,
        "run_id": target_run_id,
        "events_path": str(events_path.resolve()),
        "manifest_path": str(manifest_path.resolve()) if manifest_path.exists() else "",
        "summary": {
            "events_total": int(len(filtered)),
            "event_counts": dict(sorted(event_counts.items())),
            "missing_common_field_events": int(missing_common_field_events),
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
            "unmatched_trade_lifecycle": unmatched_trade_lifecycle,
        },
    }

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
