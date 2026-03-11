#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]


def parse_ts(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
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


def safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def load_events(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    malformed = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows, malformed


def event_ts(ev: Dict[str, Any]) -> Optional[datetime]:
    for key in ("ts", "timestamp", "time", "event_time"):
        if key in ev:
            ts = parse_ts(ev.get(key))
            if ts is not None:
                return ts
    return None


def pick_latest_run_id(events: Iterable[Dict[str, Any]]) -> Optional[str]:
    latest_run: Optional[str] = None
    latest_ts: Optional[datetime] = None
    for ev in events:
        run_id = str(ev.get("run_id") or "").strip()
        if not run_id:
            continue
        ts = event_ts(ev)
        if ts is None:
            continue
        if latest_ts is None or ts > latest_ts:
            latest_ts = ts
            latest_run = run_id
    return latest_run


@dataclass
class CycleMetrics:
    run_id: str
    strategy_id: str
    total_events: int
    event_counts: Dict[str, int]
    submitted_orders: int
    filled_orders: int
    closed_positions: int
    unique_filled_trade_ids: int
    unique_closed_trade_ids: int
    completed_cycles: int
    open_cycles: int
    orphan_closes: int
    reverse_time_cycles: int
    completion_rate: float
    win_rate: float
    total_realized_pnl_usd: float
    avg_pnl_per_closed_trade_usd: float
    avg_hold_seconds: float
    max_hold_seconds: float
    lifecycle_quality: str
    outcome_quality: str


def classify_lifecycle(*, completed: int, open_cycles: int, orphan_closes: int, reverse_time: int, filled: int, closed: int) -> str:
    if filled == 0 and closed == 0:
        return "NO_TRADES"
    if completed == 0 and filled > 0:
        return "INCOMPLETE"
    if open_cycles == 0 and orphan_closes == 0 and reverse_time == 0:
        return "PASS"
    return "PARTIAL"


def classify_outcome(*, completed: int, total_pnl: float, avg_pnl: float, win_rate: float) -> str:
    if completed == 0:
        return "N/A"
    if total_pnl > 0 and win_rate >= 0.5:
        return "GOOD"
    if avg_pnl >= -0.1:
        return "NEUTRAL"
    return "WEAK"


def compute_metrics(run_id: str, events: List[Dict[str, Any]]) -> CycleMetrics:
    event_counts: Counter[str] = Counter()
    fill_ts_by_id: Dict[str, datetime] = {}
    close_ts_by_id: Dict[str, datetime] = {}
    hold_seconds: List[float] = []
    pnl_values: List[float] = []
    wins = 0

    strategy_id = ""
    submitted = 0
    filled = 0
    closed = 0

    for ev in events:
        event_name = str(ev.get("event") or "").strip()
        event_counts[event_name] += 1
        if not strategy_id:
            strategy_id = str(ev.get("strategy_id") or "").strip()
        trade_id = str(ev.get("trade_id") or "").strip()
        ts = event_ts(ev)

        if event_name == "order_submitted":
            submitted += 1
        elif event_name == "order_filled":
            filled += 1
            if trade_id and ts is not None:
                current = fill_ts_by_id.get(trade_id)
                if current is None or ts < current:
                    fill_ts_by_id[trade_id] = ts
        elif event_name == "position_closed":
            closed += 1
            pnl = as_float(ev.get("pnl_usd"), 0.0)
            pnl_values.append(pnl)
            if pnl > 0:
                wins += 1
            hold = ev.get("hold_seconds")
            if hold is not None:
                hold_seconds.append(as_float(hold, 0.0))
            if trade_id and ts is not None:
                current = close_ts_by_id.get(trade_id)
                if current is None or ts < current:
                    close_ts_by_id[trade_id] = ts

    filled_ids = set(fill_ts_by_id.keys())
    closed_ids = set(close_ts_by_id.keys())
    completed_ids = filled_ids & closed_ids
    open_ids = filled_ids - closed_ids
    orphan_close_ids = closed_ids - filled_ids
    reverse_time_cycles = sum(1 for tid in completed_ids if close_ts_by_id[tid] < fill_ts_by_id[tid])

    completion_rate = float(len(completed_ids) / len(filled_ids)) if filled_ids else 0.0
    total_pnl = float(sum(pnl_values))
    avg_pnl = safe_mean(pnl_values)
    win_rate = float(wins / len(pnl_values)) if pnl_values else 0.0

    lifecycle_quality = classify_lifecycle(
        completed=len(completed_ids),
        open_cycles=len(open_ids),
        orphan_closes=len(orphan_close_ids),
        reverse_time=reverse_time_cycles,
        filled=len(filled_ids),
        closed=len(closed_ids),
    )
    outcome_quality = classify_outcome(
        completed=len(completed_ids),
        total_pnl=total_pnl,
        avg_pnl=avg_pnl,
        win_rate=win_rate,
    )

    return CycleMetrics(
        run_id=run_id,
        strategy_id=strategy_id,
        total_events=len(events),
        event_counts=dict(event_counts),
        submitted_orders=submitted,
        filled_orders=filled,
        closed_positions=closed,
        unique_filled_trade_ids=len(filled_ids),
        unique_closed_trade_ids=len(closed_ids),
        completed_cycles=len(completed_ids),
        open_cycles=len(open_ids),
        orphan_closes=len(orphan_close_ids),
        reverse_time_cycles=reverse_time_cycles,
        completion_rate=completion_rate,
        win_rate=win_rate,
        total_realized_pnl_usd=total_pnl,
        avg_pnl_per_closed_trade_usd=avg_pnl,
        avg_hold_seconds=safe_mean(hold_seconds),
        max_hold_seconds=max(hold_seconds) if hold_seconds else 0.0,
        lifecycle_quality=lifecycle_quality,
        outcome_quality=outcome_quality,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score completed trade cycle quality from events.jsonl.")
    parser.add_argument("--events-jsonl", default="events.jsonl", help="Path to canonical events JSONL file.")
    parser.add_argument("--run-id", default="", help="Specific run_id to inspect.")
    parser.add_argument("--strategy-id", default="", help="Optional strategy_id filter.")
    parser.add_argument("--all-runs", action="store_true", help="Report all run_ids instead of only latest.")
    parser.add_argument("--json-out", default="", help="Optional output path for JSON metrics.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events_path = Path(args.events_jsonl)
    if not events_path.is_absolute():
        events_path = ROOT / events_path
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    rows, malformed_rows = load_events(events_path)
    if args.strategy_id:
        rows = [ev for ev in rows if str(ev.get("strategy_id") or "").strip() == str(args.strategy_id).strip()]

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ev in rows:
        run_id = str(ev.get("run_id") or "").strip()
        if run_id:
            grouped[run_id].append(ev)

    if not grouped:
        print("No events with run_id found for the selected filters.")
        return 1

    selected_run_ids: List[str]
    if args.all_runs:
        selected_run_ids = sorted(grouped.keys())
    elif args.run_id:
        if args.run_id not in grouped:
            print(f"run_id not found: {args.run_id}")
            return 1
        selected_run_ids = [args.run_id]
    else:
        latest = pick_latest_run_id(rows)
        if latest is None or latest not in grouped:
            print("Could not infer latest run_id.")
            return 1
        selected_run_ids = [latest]

    reports = [compute_metrics(run_id, grouped[run_id]) for run_id in selected_run_ids]

    print(f"events_file={events_path}")
    print(f"malformed_rows_skipped={malformed_rows}")
    for report in reports:
        print("")
        print(f"run_id={report.run_id}")
        print(f"strategy_id={report.strategy_id or 'unknown'}")
        print(f"lifecycle_quality={report.lifecycle_quality}")
        print(f"outcome_quality={report.outcome_quality}")
        print(f"completed_cycles={report.completed_cycles} | open_cycles={report.open_cycles} | orphan_closes={report.orphan_closes}")
        print(
            "completion_rate="
            f"{report.completion_rate:.2%} | win_rate={report.win_rate:.2%} | total_realized_pnl_usd={report.total_realized_pnl_usd:.2f}"
        )
        print(
            "avg_pnl_per_closed_trade_usd="
            f"{report.avg_pnl_per_closed_trade_usd:.4f} | avg_hold_seconds={report.avg_hold_seconds:.2f} | max_hold_seconds={report.max_hold_seconds:.2f}"
        )

    if args.json_out:
        out_path = Path(args.json_out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "events_file": str(events_path),
            "malformed_rows_skipped": malformed_rows,
            "reports": [asdict(report) for report in reports],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"json_report={out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
