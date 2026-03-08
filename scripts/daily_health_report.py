#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
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


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def as_optional_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_report(reports_dir: Path, prefix: str) -> Optional[Path]:
    files = sorted(reports_dir.glob(f"{prefix}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    return files[0]


def normalize_summary_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    normalized.pop("generated_utc", None)
    return normalized


def reconcile_daily_summary(
    *,
    builder_script: Path,
    events_path: Path,
    manifest_path: Path,
    window_start_value: str,
    date_value: str,
    run_id_value: str,
    strategy_id: str,
    existing_payload: Dict[str, Any],
) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as handle:
        temp_path = Path(handle.name)

    cmd = [
        sys.executable,
        str(builder_script),
        "--events-jsonl",
        str(events_path),
        "--manifest",
        str(manifest_path),
        "--date",
        date_value,
        "--strategy-id",
        strategy_id,
        "--out",
        str(temp_path),
    ]
    if window_start_value:
        cmd.extend(["--window-start", window_start_value])
    if run_id_value:
        cmd.extend(["--run-id", run_id_value])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(ROOT))
        if result.returncode != 0:
            return {
                "ok": False,
                "error": f"builder_failed rc={result.returncode}",
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        rebuilt = load_json(temp_path)
    finally:
        if temp_path.exists():
            os.unlink(temp_path)

    lhs = normalize_summary_payload(existing_payload)
    rhs = normalize_summary_payload(rebuilt)
    return {
        "ok": lhs == rhs,
        "error": "" if lhs == rhs else "summary_mismatch",
        "expected": rhs if lhs != rhs else {},
    }


def to_markdown(payload: Dict[str, Any]) -> str:
    checks = payload.get("checks", [])
    labels = payload.get("deployment_labels", {})
    lines = [
        "# Strategy 1 Daily Health Report",
        "",
        f"- Generated (UTC): `{payload.get('generated_utc', '')}`",
        f"- Status: `{payload.get('status', '')}`",
        f"- Profile: `{payload.get('profile_path', '')}`",
        f"- Manifest: `{payload.get('manifest_path', '')}`",
        f"- Paper report: `{payload.get('paper_report_path', '')}`",
        f"- Daily summary: `{payload.get('daily_summary_path', '')}`",
        f"- Events: `{payload.get('events_path', '')}`",
        "",
        "## Heartbeat",
        "",
        f"- ingestion_state: `{(payload.get('heartbeat', {}) or {}).get('ingestion_state', '')}`",
        f"- events_total: `{(payload.get('heartbeat', {}) or {}).get('events_total', 0)}` | source_events_total: `{(payload.get('heartbeat', {}) or {}).get('source_events_total', 0)}`",
        f"- source_events_total_unfiltered: `{(payload.get('heartbeat', {}) or {}).get('source_events_total_unfiltered', 0)}` | source_events_excluded_before_window: `{(payload.get('heartbeat', {}) or {}).get('source_events_excluded_before_window', 0)}`",
        f"- canonical_window_start_utc: `{(payload.get('heartbeat', {}) or {}).get('canonical_window_start_utc', '')}`",
        f"- first_event_ts_utc: `{(payload.get('heartbeat', {}) or {}).get('first_event_ts_utc', '')}`",
        f"- last_event_ts_utc: `{(payload.get('heartbeat', {}) or {}).get('last_event_ts_utc', '')}`",
        f"- process_start_count: `{(payload.get('heartbeat', {}) or {}).get('process_start_count', 0)}`",
        f"- last_process_start_ts_utc: `{(payload.get('heartbeat', {}) or {}).get('last_process_start_ts_utc', '')}`",
        f"- last_process_start_age_minutes: `{(payload.get('heartbeat', {}) or {}).get('last_process_start_age_minutes', None)}`",
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
        for action in actions:
            lines.append(f"- {action}")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Strategy 1 daily health report.")
    p.add_argument("--profile", default="strategy_1_profile.json")
    p.add_argument("--manifest", default="manifest.json")
    p.add_argument(
        "--paper-report-json",
        default="",
        help="Optional paper report JSON. Defaults to latest strategy_1_paper_mode report.",
    )
    p.add_argument("--daily-summary-json", default="", help="Optional daily summary JSON path.")
    p.add_argument("--events-jsonl", default="", help="Optional events path override.")
    p.add_argument("--daily-summary-script", default="scripts/build_daily_summary.py")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--paper-prefix", default="strategy_1_paper_mode")
    p.add_argument("--out-prefix", default="strategy_1_daily_health")
    p.add_argument("--max-paper-report-age-hours", type=float, default=36.0)
    p.add_argument(
        "--max-process-starts-per-day",
        type=int,
        default=3,
        help="Maximum distinct process starts allowed for the target report day before restart-frequency check fails.",
    )
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

    reporting = profile.get("reporting", {}) if isinstance(profile.get("reporting"), dict) else {}
    summary_path_value = str(args.daily_summary_json).strip() or str(reporting.get("daily_summary_file", "daily_summary.json")).strip()
    daily_summary_path = Path(summary_path_value or "daily_summary.json")
    if not daily_summary_path.is_absolute():
        daily_summary_path = ROOT / daily_summary_path

    events_path_value = str(args.events_jsonl).strip()
    if not events_path_value:
        events_path_value = str(reporting.get("events_file", "")).strip()
    if not events_path_value:
        events_path_value = str((manifest.get("event_contract", {}) or {}).get("events_file", "events.jsonl")).strip()
    events_path = Path(events_path_value or "events.jsonl")
    if not events_path.is_absolute():
        events_path = ROOT / events_path

    builder_script = Path(args.daily_summary_script)
    if not builder_script.is_absolute():
        builder_script = ROOT / builder_script

    daily_summary: Dict[str, Any] = {}
    if daily_summary_path.exists():
        daily_summary = load_json(daily_summary_path)

    labels = {
        "strategy_1": str(profile.get("stage", manifest.get("stage", "PAPER_CANDIDATE"))),
        "strategy_2_deterministic": str((profile.get("branch_stages", {}) or {}).get("strategy_2_deterministic", "RESEARCH")),
        "rlm_rl": str((profile.get("branch_stages", {}) or {}).get("rlm_rl", "EXPERIMENTAL_ONLY")),
        "promotion_target": str(profile.get("promotion_target", manifest.get("promotion_target", "LIVE_GATED"))),
    }

    expected_labels_ok = (
        labels["strategy_1"] == "PAPER_CANDIDATE"
        and labels["strategy_2_deterministic"] == "RESEARCH"
        and labels["rlm_rl"] == "EXPERIMENTAL_ONLY"
        and labels["promotion_target"] == "LIVE_GATED"
    )

    controls = profile.get("controls", {}) if isinstance(profile.get("controls"), dict) else {}
    spread_gate_active = bool(controls.get("live_spread_gate_active", False))
    one_trade_cap = as_int(controls.get("max_trades_per_session"), 0) == 1
    frozen_profile = bool(manifest.get("frozen_parameters") or profile.get("frozen_parameters"))

    has_paper = bool(paper)
    paper_generated = parse_ts(str(paper.get("generated_utc", ""))) if has_paper else None
    age_hours = None
    if paper_generated is not None:
        age_hours = float((utc_now() - paper_generated).total_seconds() / 3600.0)
    paper_fresh = bool(age_hours is not None and age_hours <= float(args.max_paper_report_age_hours))

    summary = paper.get("summary", {}) if isinstance(paper.get("summary"), dict) else {}
    paper_checks = paper.get("checks", {}) if isinstance(paper.get("checks"), dict) else {}
    kill_switch = paper.get("kill_switch", {}) if isinstance(paper.get("kill_switch"), dict) else {}

    max_consec_skip = as_int(summary.get("max_consecutive_skipped_bars"), 0)
    max_consec_fail = as_int(summary.get("max_consecutive_api_failures"), 0)

    ks_cfg = profile.get("kill_switch", {}) if isinstance(profile.get("kill_switch"), dict) else {}
    skip_limit = as_int((ks_cfg.get("too_many_skipped_bars", {}) or {}).get("max_consecutive_skipped_bars"), 0)
    fail_limit = as_int((ks_cfg.get("broker_api_failure", {}) or {}).get("max_consecutive_failures"), 0)

    kill_switch_clear = not bool(kill_switch.get("is_halted", False))
    paper_window_pass = bool(paper_checks.get("paper_window_pass", False))
    skip_health = bool(skip_limit <= 0 or max_consec_skip <= skip_limit)
    broker_health = bool(fail_limit <= 0 or max_consec_fail < fail_limit)
    drawdown_health = "daily_drawdown_breach" not in list(kill_switch.get("triggered", []))

    event_contract_common_fields_pass = bool(paper_checks.get("event_contract_common_fields_pass", False))
    event_contract_event_fields_pass = bool(paper_checks.get("event_contract_event_fields_pass", False))
    schema_event_values_valid = bool(paper_checks.get("schema_event_values_valid", False))
    schema_reason_values_valid = bool(paper_checks.get("schema_reason_values_valid", False))
    profile_hash_consistent = bool(paper_checks.get("profile_hash_consistent", False))
    profile_hash_matches_expected = bool(paper_checks.get("profile_hash_matches_expected", False))
    schema_version_matches_expected = bool(paper_checks.get("schema_version_matches_expected", False))
    manifest_version_matches_expected = bool(paper_checks.get("manifest_version_matches_expected", False))
    run_id_format_valid = bool(paper_checks.get("run_id_format_valid", False))
    run_id_hash_prefix_valid = bool(paper_checks.get("run_id_hash_prefix_valid", False))
    monotonic_event_order_pass = bool(paper_checks.get("monotonic_event_order_pass", False))
    trade_lifecycle_complete = bool(paper_checks.get("trade_lifecycle_complete", False))
    stage_consistent = bool(paper_checks.get("stage_consistent", False))
    strategy_id_consistent = bool(paper_checks.get("strategy_id_consistent", False))

    has_daily_summary = bool(daily_summary)
    daily_summary_summary = daily_summary.get("summary", {}) if isinstance(daily_summary.get("summary"), dict) else {}
    heartbeat = {
        "ingestion_state": str(daily_summary_summary.get("ingestion_state", "NO_DAILY_SUMMARY")),
        "events_total": as_int(daily_summary_summary.get("events_total"), 0),
        "source_events_total": as_int(daily_summary_summary.get("source_events_total"), 0),
        "source_events_total_unfiltered": as_int(daily_summary_summary.get("source_events_total_unfiltered"), 0),
        "source_events_excluded_before_window": as_int(daily_summary_summary.get("source_events_excluded_before_window"), 0),
        "canonical_window_start_utc": str(
            daily_summary_summary.get("canonical_window_start_utc")
            or daily_summary.get("canonical_window_start_utc")
            or paper.get("canonical_window_start_utc")
            or ""
        ),
        "first_event_ts_utc": str(daily_summary_summary.get("first_event_ts_utc", "")),
        "last_event_ts_utc": str(daily_summary_summary.get("last_event_ts_utc", "")),
        "process_start_count": as_int(daily_summary_summary.get("process_start_count"), 0),
        "last_process_start_ts_utc": str(daily_summary_summary.get("last_process_start_ts_utc", "")),
        "last_process_start_age_minutes": as_optional_int(daily_summary_summary.get("last_process_start_age_minutes")),
    }
    daily_summary_strategy = str(daily_summary.get("strategy_id", profile.get("strategy_id", "strategy_1"))).strip() or "strategy_1"
    daily_summary_run_id = str(daily_summary.get("run_id", "")).strip()
    daily_summary_date = str(daily_summary.get("date", "")).strip()
    if not daily_summary_date:
        window_end_ts = parse_ts(str(paper.get("window_end_utc", ""))) if has_paper else None
        daily_summary_date = window_end_ts.date().isoformat() if window_end_ts else utc_now().date().isoformat()

    reconciliation = {
        "ok": False,
        "error": "summary_missing",
        "stdout": "",
        "stderr": "",
    }
    if has_daily_summary and builder_script.exists() and events_path.exists() and manifest_path.exists():
        reconciliation = reconcile_daily_summary(
            builder_script=builder_script,
            events_path=events_path,
            manifest_path=manifest_path,
            window_start_value=str(heartbeat.get("canonical_window_start_utc", "")).strip(),
            date_value=daily_summary_date,
            run_id_value=daily_summary_run_id,
            strategy_id=daily_summary_strategy,
            existing_payload=daily_summary,
        )
    daily_summary_reconciled = bool(reconciliation.get("ok", False))

    checks: List[Dict[str, Any]] = [
        {"name": "deployment_labels_explicit", "pass": expected_labels_ok, "detail": f"{labels}"},
        {"name": "spread_gate_active", "pass": spread_gate_active, "detail": "profile.controls.live_spread_gate_active"},
        {"name": "one_trade_per_session_cap", "pass": one_trade_cap, "detail": "profile.controls.max_trades_per_session == 1"},
        {"name": "frozen_parameters_recorded", "pass": frozen_profile, "detail": "manifest/profile contains frozen_parameters"},
        {"name": "paper_report_available", "pass": has_paper, "detail": str(paper_path) if paper_path else "not found"},
        {"name": "paper_report_fresh", "pass": paper_fresh, "detail": f"age_hours={age_hours if age_hours is not None else 'n/a'}"},
        {"name": "paper_window_pass", "pass": paper_window_pass, "detail": "paper report checks.paper_window_pass"},
        {"name": "kill_switch_clear", "pass": kill_switch_clear, "detail": f"triggered={kill_switch.get('triggered', [])}"},
        {
            "name": "event_contract_common_fields_pass",
            "pass": event_contract_common_fields_pass,
            "detail": f"missing_common_field_events={as_int(summary.get('missing_common_field_events'), 0)}",
        },
        {
            "name": "event_contract_event_fields_pass",
            "pass": event_contract_event_fields_pass,
            "detail": f"missing_event_field_events={as_int(summary.get('missing_event_field_events'), 0)}",
        },
        {
            "name": "schema_event_values_valid",
            "pass": schema_event_values_valid,
            "detail": f"unknown_event_value_events={as_int(summary.get('unknown_event_value_events'), 0)}",
        },
        {
            "name": "schema_reason_values_valid",
            "pass": schema_reason_values_valid,
            "detail": f"unknown_reason_code_events={as_int(summary.get('unknown_reason_code_events'), 0)}",
        },
        {
            "name": "profile_hash_consistent",
            "pass": profile_hash_consistent,
            "detail": f"profile_hash_values_seen={summary.get('profile_hash_values_seen', {})}",
        },
        {
            "name": "profile_hash_matches_expected",
            "pass": profile_hash_matches_expected,
            "detail": f"profile_hash_mismatch_events={as_int(summary.get('profile_hash_mismatch_events'), 0)}",
        },
        {
            "name": "schema_version_matches_expected",
            "pass": schema_version_matches_expected,
            "detail": f"schema_version_mismatch_events={as_int(summary.get('schema_version_mismatch_events'), 0)}",
        },
        {
            "name": "manifest_version_matches_expected",
            "pass": manifest_version_matches_expected,
            "detail": f"manifest_version_mismatch_events={as_int(summary.get('manifest_version_mismatch_events'), 0)}",
        },
        {
            "name": "run_id_format_valid",
            "pass": run_id_format_valid,
            "detail": f"invalid_run_id_events={as_int(summary.get('invalid_run_id_events'), 0)}",
        },
        {
            "name": "run_id_hash_prefix_valid",
            "pass": run_id_hash_prefix_valid,
            "detail": f"run_id_hash_prefix_mismatch_events={as_int(summary.get('run_id_hash_prefix_mismatch_events'), 0)}",
        },
        {
            "name": "monotonic_event_order_pass",
            "pass": monotonic_event_order_pass,
            "detail": f"monotonic_order_violations={as_int(summary.get('monotonic_order_violations'), 0)}",
        },
        {
            "name": "trade_lifecycle_complete",
            "pass": trade_lifecycle_complete,
            "detail": (
                "fills_without_submit="
                f"{as_int(summary.get('lifecycle_fill_without_submit_count'), 0)}, "
                "close_without_fill="
                f"{as_int(summary.get('lifecycle_close_without_fill_count'), 0)}, "
                "duplicate_close="
                f"{as_int(summary.get('lifecycle_duplicate_close_count'), 0)}, "
                "mixed_side="
                f"{as_int(summary.get('lifecycle_mixed_side_count'), 0)}"
            ),
        },
        {"name": "stage_consistent", "pass": stage_consistent, "detail": f"stage_values_seen={summary.get('stage_values_seen', {})}"},
        {
            "name": "strategy_id_consistent",
            "pass": strategy_id_consistent,
            "detail": f"strategy_values_seen={summary.get('strategy_values_seen', {})}",
        },
        {
            "name": "daily_summary_available",
            "pass": has_daily_summary,
            "detail": str(daily_summary_path),
        },
        {
            "name": "ingestion_input_present",
            "pass": heartbeat["source_events_total"] > 0,
            "detail": f"source_events_total={heartbeat['source_events_total']}",
        },
        {
            "name": "ingestion_events_for_day",
            "pass": heartbeat["events_total"] > 0,
            "detail": f"events_total={heartbeat['events_total']}, ingestion_state={heartbeat['ingestion_state']}",
        },
        {
            "name": "daily_summary_reconciled",
            "pass": daily_summary_reconciled,
            "detail": str(reconciliation.get("error", "")),
        },
        {
            "name": "process_start_metadata_present",
            "pass": (
                heartbeat["events_total"] == 0
                or (
                    bool(heartbeat["last_process_start_ts_utc"])
                    and heartbeat["last_process_start_age_minutes"] is not None
                )
            ),
            "detail": (
                "last_process_start_ts_utc="
                f"{heartbeat['last_process_start_ts_utc']}, "
                "last_process_start_age_minutes="
                f"{heartbeat['last_process_start_age_minutes']}"
            ),
        },
        {
            "name": "restart_frequency_healthy",
            "pass": (
                heartbeat["events_total"] == 0
                or heartbeat["process_start_count"] <= int(args.max_process_starts_per_day)
            ),
            "detail": (
                "process_start_count="
                f"{heartbeat['process_start_count']}, "
                "max_process_starts_per_day="
                f"{int(args.max_process_starts_per_day)}"
            ),
        },
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
            recommended_actions.append("Continue paper trading until minimum days and trades are met.")
        elif name == "kill_switch_clear":
            recommended_actions.append("Keep trading halted until kill-switch trigger is cleared and reviewed.")
        elif name in {
            "event_contract_common_fields_pass",
            "event_contract_event_fields_pass",
            "schema_event_values_valid",
            "schema_reason_values_valid",
            "profile_hash_consistent",
            "profile_hash_matches_expected",
            "schema_version_matches_expected",
            "manifest_version_matches_expected",
            "run_id_format_valid",
            "run_id_hash_prefix_valid",
            "monotonic_event_order_pass",
            "trade_lifecycle_complete",
            "stage_consistent",
            "strategy_id_consistent",
        }:
            recommended_actions.append("Fix event logging contract violations before promotion decisions.")
        elif name == "daily_summary_available":
            recommended_actions.append("Generate daily_summary.json before health evaluation.")
        elif name == "ingestion_input_present":
            recommended_actions.append("No pipeline input detected; verify events.jsonl ingestion is active.")
        elif name == "ingestion_events_for_day":
            recommended_actions.append("No events for target day; verify session run and date/run_id filters.")
        elif name == "daily_summary_reconciled":
            recommended_actions.append("Rebuild daily summary from events and investigate reconciliation mismatch.")
        elif name == "process_start_metadata_present":
            recommended_actions.append("Ensure process_start_ts is emitted on canonical events for restart observability.")
        elif name == "restart_frequency_healthy":
            recommended_actions.append("Frequent restarts detected; investigate crash/reconnect loops before promotion.")
        elif name == "skip_bar_health":
            recommended_actions.append("Investigate feed/latency conditions causing excessive skipped bars.")
        elif name == "broker_api_health":
            recommended_actions.append("Investigate broker/API stability and hold trading until failures stabilize.")
        elif name == "daily_drawdown_health":
            recommended_actions.append("Daily drawdown breach detected; pause until next session and review risk.")
        else:
            recommended_actions.append(f"Fix failing check: {name}.")

    all_checks_pass = all(bool(c["pass"]) for c in checks)
    if not has_paper:
        status = "NO_DATA"
    elif not kill_switch_clear:
        status = "HALT"
    elif has_daily_summary and not daily_summary_reconciled:
        status = "UNHEALTHY"
    elif all_checks_pass and paper_window_pass:
        status = "HEALTHY"
    else:
        status = "ATTENTION"

    payload = {
        "generated_utc": utc_now().isoformat(),
        "status": status,
        "profile_path": str(profile_path.resolve()),
        "manifest_path": str(manifest_path.resolve()) if manifest_path.exists() else "",
        "paper_report_path": str(paper_path.resolve()) if paper_path else "",
        "daily_summary_path": str(daily_summary_path.resolve()),
        "events_path": str(events_path.resolve()),
        "deployment_labels": labels,
        "heartbeat": heartbeat,
        "checks": checks,
        "recommended_actions": sorted(set(recommended_actions)),
        "daily_summary_reconciliation": {
            "ok": bool(reconciliation.get("ok", False)),
            "error": str(reconciliation.get("error", "")),
            "stdout": str(reconciliation.get("stdout", ""))[:500],
            "stderr": str(reconciliation.get("stderr", ""))[:500],
        },
        "paper_snapshot": {
            "paper_status": str(paper.get("status", "")),
            "paper_days": as_int(paper.get("paper_days"), 0),
            "trade_count": as_int(summary.get("trade_count"), 0),
            "net_pnl_usd": as_float(summary.get("net_pnl_usd"), 0.0),
            "profit_factor": as_float(summary.get("profit_factor"), 0.0),
            "last_trade_age_minutes": summary.get("last_trade_age_minutes"),
        },
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
