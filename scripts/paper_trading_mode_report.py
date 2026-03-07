#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]


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
    for key in ("timestamp", "time", "event_time", "ts"):
        if key in ev:
            dt = parse_ts(str(ev.get(key, "")))
            if dt is not None:
                return dt
    return None


def is_entry(ev: Dict[str, Any]) -> bool:
    action = str(ev.get("action", "")).strip().upper()
    if action in {"BUY", "SELL", "OPEN_LONG", "OPEN_SHORT", "ENTRY"}:
        return True
    return bool(ev.get("is_entry", False))


def is_skip(ev: Dict[str, Any]) -> bool:
    action = str(ev.get("action", "")).strip().upper()
    if action in {"SKIP", "NO_TRADE"}:
        return True
    return bool(str(ev.get("skip_reason", "")).strip())


def is_api_failure(ev: Dict[str, Any]) -> bool:
    broker_ok = ev.get("broker_ok")
    if broker_ok is False:
        return True
    status = str(ev.get("api_status", "")).strip().lower()
    return status in {"error", "failed", "failure", "disconnected"}


def session_key(ev: Dict[str, Any], ts: Optional[datetime]) -> str:
    if str(ev.get("session_key", "")).strip():
        return str(ev["session_key"]).strip()
    if ts is None:
        return "unknown_session"
    sess = str(ev.get("session", "default")).strip() or "default"
    return f"{ts.date().isoformat()}::{sess}"


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


def evaluate_kill_switch(summary: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    ks = profile.get("kill_switch", {}) if isinstance(profile.get("kill_switch"), dict) else {}
    triggered: List[str] = []

    spread_cfg = ks.get("spread_too_high", {}) if isinstance(ks.get("spread_too_high"), dict) else {}
    if bool(spread_cfg.get("enabled", False)):
        spread_trig = int(as_float(spread_cfg.get("trigger_on_entry_violation_count_ge"), 1.0))
        if int(summary.get("spread_gate_violations", 0)) >= max(1, spread_trig):
            triggered.append("spread_too_high")

    skip_cfg = ks.get("too_many_skipped_bars", {}) if isinstance(ks.get("too_many_skipped_bars"), dict) else {}
    if bool(skip_cfg.get("enabled", False)):
        max_skip = int(as_float(skip_cfg.get("max_consecutive_skipped_bars"), 0.0))
        if max_skip > 0 and int(summary.get("max_consecutive_skipped_bars", 0)) > max_skip:
            triggered.append("too_many_skipped_bars")

    api_cfg = ks.get("broker_api_failure", {}) if isinstance(ks.get("broker_api_failure"), dict) else {}
    if bool(api_cfg.get("enabled", False)):
        max_fail = int(as_float(api_cfg.get("max_consecutive_failures"), 0.0))
        if max_fail > 0 and int(summary.get("max_consecutive_api_failures", 0)) >= max_fail:
            triggered.append("broker_api_failure")

    dd_cfg = ks.get("daily_drawdown_breach", {}) if isinstance(ks.get("daily_drawdown_breach"), dict) else {}
    if bool(dd_cfg.get("enabled", False)):
        max_daily_dd = abs(as_float(dd_cfg.get("max_daily_drawdown_usd"), 0.0))
        if max_daily_dd > 0:
            breaches = [d for d, v in (summary.get("daily_pnl", {}) or {}).items() if as_float(v) <= -max_daily_dd]
            if breaches:
                triggered.append("daily_drawdown_breach")
                summary["daily_drawdown_breach_days"] = breaches

    return {"triggered": triggered, "is_halted": bool(triggered)}


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
        f"- Stage: `{labels.get('strategy_1', '')}` | Promotion target: `{labels.get('promotion_target', '')}`",
        f"- Strategy 2 deterministic stage: `{labels.get('strategy_2_deterministic', '')}`",
        f"- RL/RLM stage: `{labels.get('rlm_rl', '')}`",
        "",
        "## Window",
        "",
        f"- Start (UTC): `{payload.get('window_start_utc', '')}`",
        f"- End (UTC): `{payload.get('window_end_utc', '')}`",
        f"- Days covered: `{payload.get('paper_days', 0)}`",
        "",
        "## Summary",
        "",
        f"- status: `{payload.get('status', '')}`",
        f"- events: `{int(s.get('total_events', 0))}` | trades: `{int(s.get('trade_count', 0))}`",
        f"- net_pnl_usd: `{float(s.get('net_pnl_usd', 0.0)):.2f}` | win_rate: `{float(s.get('win_rate', 0.0)):.2f}` | max_dd_usd: `{float(s.get('max_drawdown_usd', 0.0)):.2f}`",
        f"- spread_gate_violations: `{int(s.get('spread_gate_violations', 0))}` | session_cap_violations: `{int(s.get('session_cap_violations', 0))}`",
        f"- max_consecutive_skipped_bars: `{int(s.get('max_consecutive_skipped_bars', 0))}` | max_consecutive_api_failures: `{int(s.get('max_consecutive_api_failures', 0))}`",
        "",
        "## Checks",
        "",
        f"- spread_gate_active: `{bool(checks.get('spread_gate_active', False))}`",
        f"- one_trade_per_session_cap: `{bool(checks.get('one_trade_per_session_cap', False))}`",
        f"- frozen_parameters_recorded: `{bool(checks.get('frozen_parameters_recorded', False))}`",
        f"- paper_window_pass: `{bool(checks.get('paper_window_pass', False))}`",
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
    p.add_argument("--events-jsonl", default="", help="Optional JSONL/JSON events file.")
    p.add_argument("--window-start", default="", help="Optional ISO-8601 UTC window start.")
    p.add_argument("--window-end", default="", help="Optional ISO-8601 UTC window end.")
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

    events_path: Optional[Path] = None
    if str(args.events_jsonl).strip():
        events_path = Path(args.events_jsonl)
        if not events_path.is_absolute():
            events_path = ROOT / events_path

    events = load_events(events_path)
    enriched: List[Tuple[Optional[datetime], Dict[str, Any]]] = [(event_ts(ev), ev) for ev in events]
    enriched.sort(key=lambda x: x[0] or datetime.min.replace(tzinfo=timezone.utc))

    pnl_rows: List[Tuple[Optional[datetime], float]] = []
    entry_per_session: Dict[str, int] = {}
    spread_gate_violations = 0
    skipped_total = 0
    skipped_run = 0
    max_skipped_run = 0
    api_failure_total = 0
    api_fail_run = 0
    max_api_fail_run = 0

    live_max_spread = as_float((profile.get("controls", {}) or {}).get("live_max_spread"), 0.0)
    max_trades_per_session = int(as_float((profile.get("controls", {}) or {}).get("max_trades_per_session"), 1.0))

    for ts, ev in enriched:
        if "pnl_usd" in ev:
            pnl_rows.append((ts, as_float(ev.get("pnl_usd"))))
        if is_entry(ev):
            key = session_key(ev, ts)
            entry_per_session[key] = int(entry_per_session.get(key, 0) + 1)
            spread = as_float(ev.get("spread"), 0.0)
            if live_max_spread > 0.0 and spread > live_max_spread:
                spread_gate_violations += 1

        if is_skip(ev):
            skipped_total += 1
            skipped_run += 1
            if skipped_run > max_skipped_run:
                max_skipped_run = skipped_run
        else:
            skipped_run = 0

        if is_api_failure(ev):
            api_failure_total += 1
            api_fail_run += 1
            if api_fail_run > max_api_fail_run:
                max_api_fail_run = api_fail_run
        else:
            api_fail_run = 0

    session_cap_violations = int(sum(max(0, cnt - max(max_trades_per_session, 1)) for cnt in entry_per_session.values()))
    pnl_values = [p for _, p in pnl_rows]
    trade_count = len(pnl_values)
    wins = int(sum(1 for p in pnl_values if p > 0.0))
    losses = int(sum(1 for p in pnl_values if p < 0.0))
    net_pnl = float(sum(pnl_values))
    win_rate = float(wins / trade_count) if trade_count > 0 else 0.0
    dd = max_drawdown(pnl_values)
    daily_pnl = daily_pnl_series(pnl_rows)

    event_times = [ts for ts, _ in enriched if ts is not None]
    cli_start = parse_ts(str(args.window_start)) if str(args.window_start).strip() else None
    cli_end = parse_ts(str(args.window_end)) if str(args.window_end).strip() else None
    window_start = cli_start or (min(event_times) if event_times else None)
    window_end = cli_end or (max(event_times) if event_times else None)
    paper_days = 0
    if window_start is not None and window_end is not None and window_end >= window_start:
        paper_days = int((window_end.date() - window_start.date()).days + 1)

    paper_rules = profile.get("paper_validation_window", {}) if isinstance(profile.get("paper_validation_window"), dict) else {}
    min_days = int(as_float(paper_rules.get("min_days"), 0.0))
    min_trades = int(as_float(paper_rules.get("min_trades"), 0.0))

    summary: Dict[str, Any] = {
        "total_events": int(len(events)),
        "trade_count": int(trade_count),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": float(win_rate),
        "net_pnl_usd": float(net_pnl),
        "max_drawdown_usd": float(dd),
        "spread_gate_violations": int(spread_gate_violations),
        "session_cap_violations": int(session_cap_violations),
        "skipped_bars": int(skipped_total),
        "max_consecutive_skipped_bars": int(max_skipped_run),
        "api_failure_count": int(api_failure_total),
        "max_consecutive_api_failures": int(max_api_fail_run),
        "daily_pnl": daily_pnl,
    }

    checks = {
        "spread_gate_active": bool((profile.get("controls", {}) or {}).get("live_spread_gate_active", False)),
        "one_trade_per_session_cap": int(max_trades_per_session) == 1,
        "frozen_parameters_recorded": bool(profile.get("frozen_parameters")),
        "paper_window_pass": bool(paper_days >= min_days and trade_count >= min_trades),
    }
    kill_switch = evaluate_kill_switch(summary=summary, profile=profile)

    status = "PAPER_IN_PROGRESS"
    if bool(kill_switch.get("is_halted", False)):
        status = "HALT"
    elif bool(checks["paper_window_pass"]):
        status = "PAPER_PASS"

    deployment_labels = {
        "strategy_1": str(profile.get("stage", "PAPER_CANDIDATE")),
        "strategy_2_deterministic": str((profile.get("branch_stages", {}) or {}).get("strategy_2_deterministic", "RESEARCH")),
        "rlm_rl": str((profile.get("branch_stages", {}) or {}).get("rlm_rl", "EXPERIMENTAL_ONLY")),
        "promotion_target": str(profile.get("promotion_target", "LIVE_GATED")),
    }

    payload = {
        "generated_utc": utc_now().isoformat(),
        "profile_path": str(profile_path.resolve()),
        "events_path": str(events_path.resolve()) if events_path else "",
        "status": status,
        "deployment_labels": deployment_labels,
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
