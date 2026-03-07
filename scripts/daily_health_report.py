#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def latest_report(reports_dir: Path, prefix: str) -> Optional[Path]:
    files = sorted(reports_dir.glob(f"{prefix}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    return files[0]


def to_markdown(payload: Dict[str, Any]) -> str:
    checks = payload.get("checks", {})
    labels = payload.get("deployment_labels", {})
    lines = [
        "# Strategy 1 Daily Health Report",
        "",
        f"- Generated (UTC): `{payload.get('generated_utc', '')}`",
        f"- Status: `{payload.get('status', '')}`",
        f"- Profile: `{payload.get('profile_path', '')}`",
        f"- Paper report: `{payload.get('paper_report_path', '')}`",
        "",
        "## Deployment Labels",
        "",
        f"- Strategy 1: `{labels.get('strategy_1', '')}`",
        f"- Strategy 2 deterministic: `{labels.get('strategy_2_deterministic', '')}`",
        f"- RLM/RL: `{labels.get('rlm_rl', '')}`",
        f"- Promotion target: `{labels.get('promotion_target', '')}`",
        "",
        "## Checks",
        "",
    ]
    for c in checks:
        lines.append(f"- {c['name']}: `{c['pass']}` ({c['detail']})")
    actions = payload.get("recommended_actions", [])
    lines += ["", "## Recommended Actions", ""]
    if actions:
        for a in actions:
            lines.append(f"- {a}")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Strategy 1 daily health report.")
    p.add_argument("--profile", default="strategy_1_profile.json")
    p.add_argument("--paper-report-json", default="", help="Optional paper report JSON. Defaults to latest strategy_1_paper_mode report.")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--paper-prefix", default="strategy_1_paper_mode")
    p.add_argument("--out-prefix", default="strategy_1_daily_health")
    p.add_argument("--max-paper-report-age-hours", type=float, default=36.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    profile_path = Path(args.profile)
    if not profile_path.is_absolute():
        profile_path = ROOT / profile_path
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")
    profile = load_json(profile_path)

    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = ROOT / reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    paper_path: Optional[Path] = None
    if str(args.paper_report_json).strip():
        paper_path = Path(args.paper_report_json)
        if not paper_path.is_absolute():
            paper_path = ROOT / paper_path
        if not paper_path.exists():
            raise FileNotFoundError(f"Paper report not found: {paper_path}")
    else:
        paper_path = latest_report(reports_dir, str(args.paper_prefix))

    paper: Dict[str, Any] = {}
    if paper_path is not None and paper_path.exists():
        paper = load_json(paper_path)

    labels = {
        "strategy_1": str(profile.get("stage", "PAPER_CANDIDATE")),
        "strategy_2_deterministic": str((profile.get("branch_stages", {}) or {}).get("strategy_2_deterministic", "RESEARCH")),
        "rlm_rl": str((profile.get("branch_stages", {}) or {}).get("rlm_rl", "EXPERIMENTAL_ONLY")),
        "promotion_target": str(profile.get("promotion_target", "LIVE_GATED")),
    }

    expected_labels_ok = (
        labels["strategy_1"] == "PAPER_CANDIDATE"
        and labels["strategy_2_deterministic"] == "RESEARCH"
        and labels["rlm_rl"] == "EXPERIMENTAL_ONLY"
        and labels["promotion_target"] == "LIVE_GATED"
    )
    spread_gate_active = bool((profile.get("controls", {}) or {}).get("live_spread_gate_active", False))
    one_trade_cap = int(as_float((profile.get("controls", {}) or {}).get("max_trades_per_session"), 0.0)) == 1
    frozen_profile = bool(profile.get("frozen_parameters"))

    has_paper = bool(paper)
    paper_generated = parse_ts(str(paper.get("generated_utc", ""))) if has_paper else None
    age_hours = None
    if paper_generated is not None:
        age_hours = float((utc_now() - paper_generated).total_seconds() / 3600.0)
    paper_fresh = bool(age_hours is not None and age_hours <= float(args.max_paper_report_age_hours))

    summary = paper.get("summary", {}) if isinstance(paper.get("summary"), dict) else {}
    paper_checks = paper.get("checks", {}) if isinstance(paper.get("checks"), dict) else {}
    kill_switch = paper.get("kill_switch", {}) if isinstance(paper.get("kill_switch"), dict) else {}
    max_consec_skip = int(as_float(summary.get("max_consecutive_skipped_bars"), 0.0))
    max_consec_fail = int(as_float(summary.get("max_consecutive_api_failures"), 0.0))

    ks_cfg = profile.get("kill_switch", {}) if isinstance(profile.get("kill_switch"), dict) else {}
    skip_limit = int(as_float((ks_cfg.get("too_many_skipped_bars", {}) or {}).get("max_consecutive_skipped_bars"), 0.0))
    fail_limit = int(as_float((ks_cfg.get("broker_api_failure", {}) or {}).get("max_consecutive_failures"), 0.0))

    kill_switch_clear = not bool(kill_switch.get("is_halted", False))
    paper_window_pass = bool(paper_checks.get("paper_window_pass", False))
    skip_health = bool(skip_limit <= 0 or max_consec_skip <= skip_limit)
    broker_health = bool(fail_limit <= 0 or max_consec_fail < fail_limit)
    drawdown_health = "daily_drawdown_breach" not in list(kill_switch.get("triggered", []))

    checks: List[Dict[str, Any]] = [
        {"name": "deployment_labels_explicit", "pass": expected_labels_ok, "detail": f"{labels}"},
        {"name": "spread_gate_active", "pass": spread_gate_active, "detail": "profile.controls.live_spread_gate_active"},
        {"name": "one_trade_per_session_cap", "pass": one_trade_cap, "detail": "profile.controls.max_trades_per_session == 1"},
        {"name": "frozen_parameters_recorded", "pass": frozen_profile, "detail": "strategy_1_profile.json contains frozen_parameters"},
        {"name": "paper_report_available", "pass": has_paper, "detail": str(paper_path) if paper_path else "not found"},
        {"name": "paper_report_fresh", "pass": paper_fresh, "detail": f"age_hours={age_hours if age_hours is not None else 'n/a'}"},
        {"name": "paper_window_pass", "pass": paper_window_pass, "detail": "paper report checks.paper_window_pass"},
        {"name": "kill_switch_clear", "pass": kill_switch_clear, "detail": f"triggered={kill_switch.get('triggered', [])}"},
        {"name": "skip_bar_health", "pass": skip_health, "detail": f"{max_consec_skip}/{skip_limit}"},
        {"name": "broker_api_health", "pass": broker_health, "detail": f"{max_consec_fail}/{fail_limit}"},
        {"name": "daily_drawdown_health", "pass": drawdown_health, "detail": "daily_drawdown_breach not triggered"},
    ]

    recommended_actions: List[str] = []
    for check in checks:
        if bool(check["pass"]):
            continue
        name = str(check["name"])
        if name == "paper_report_available":
            recommended_actions.append("Generate a paper-trading mode report before considering promotion.")
        elif name == "paper_report_fresh":
            recommended_actions.append("Refresh paper report (report is stale).")
        elif name == "paper_window_pass":
            recommended_actions.append("Continue paper trading until window and trade-count requirements pass.")
        elif name == "kill_switch_clear":
            recommended_actions.append("Keep trading halted until kill-switch trigger is cleared and reviewed.")
        elif name == "skip_bar_health":
            recommended_actions.append("Investigate data/feed conditions causing excessive skipped bars.")
        elif name == "broker_api_health":
            recommended_actions.append("Investigate broker/API stability; hold trading until failures stabilize.")
        elif name == "daily_drawdown_health":
            recommended_actions.append("Daily drawdown breach detected; pause until next session and review risk.")
        else:
            recommended_actions.append(f"Fix failing check: {name}.")

    if not has_paper:
        status = "NO_DATA"
    elif not kill_switch_clear:
        status = "HALT"
    elif not paper_window_pass:
        status = "ATTENTION"
    elif all(bool(c["pass"]) for c in checks):
        status = "HEALTHY"
    else:
        status = "ATTENTION"

    payload = {
        "generated_utc": utc_now().isoformat(),
        "status": status,
        "profile_path": str(profile_path.resolve()),
        "paper_report_path": str(paper_path.resolve()) if paper_path else "",
        "deployment_labels": labels,
        "checks": checks,
        "recommended_actions": recommended_actions,
    }

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
