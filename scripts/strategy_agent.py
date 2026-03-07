#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPORT_JSON_RE = re.compile(r"report_json\s*=\s*(.+)$", re.IGNORECASE | re.MULTILINE)
BOOL_FLAG_KEYS = {"--require-close-confirm"}
DEFAULT_WF_MIN_TRADES_TOTAL = 40.0
DEFAULT_WF_MIN_TRADES_PER_FOLD = 3.0
DEFAULT_WF_MIN_FOLDS_MEETING_TRADES = 6
DEFAULT_LOCKBOX_MIN_TRADES_FOR_PF = 30


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x: Any, d: float = 0.0) -> float:
    try:
        return d if x is None else float(x)
    except Exception:
        return d


def parse_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {"1", "true", "yes", "y", "on"}:
            return True
        if t in {"0", "false", "no", "n", "off"}:
            return False
    return None


def run_cmd(cmd: Sequence[str], cwd: Path) -> Tuple[int, str]:
    p = subprocess.run(list(cmd), cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=False)
    return p.returncode, p.stdout


def extract_report_path(stdout: str) -> Optional[Path]:
    m = REPORT_JSON_RE.search(stdout)
    if not m:
        return None
    raw = m.group(1).strip().strip('"').strip("'").strip()
    if not raw:
        return None
    try:
        return Path(raw)
    except Exception:
        return None


def find_report(stdout: str, out_prefix: str, reports_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    rp = extract_report_path(stdout)
    if rp and rp.exists():
        return rp, None
    cands = sorted(reports_dir.glob(f"{out_prefix}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        return cands[0], None
    return None, "could_not_find_report_json_in_stdout"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def pick_summary(report: Dict[str, Any], profile: str = "base") -> Optional[Dict[str, Any]]:
    s = report.get("summaries")
    if not isinstance(s, dict) or not s:
        return None
    if isinstance(s.get(profile), dict):
        return s[profile]
    first = next(iter(s.values()))
    return first if isinstance(first, dict) else None


def pick_mode_agg(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    m = report.get("modes")
    if not isinstance(m, dict) or not m:
        return None
    block = m.get("conservative") if isinstance(m.get("conservative"), dict) else next(iter(m.values()))
    if not isinstance(block, dict):
        return None
    agg = block.get("agg")
    return agg if isinstance(agg, dict) else None


def normalize_wf(report: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not report:
        return {"pos_fold_rate": 0.0, "worst_fold_net": 0.0, "pf_mean": 0.0, "net_sum": 0.0, "worst_dd": 0.0, "trades_total": 0.0}
    s = pick_summary(report, "base")
    if s:
        return {
            "pos_fold_rate": safe_float(s.get("pos_fold_rate")),
            "worst_fold_net": safe_float(s.get("worst_fold_net")),
            "pf_mean": safe_float(s.get("pf_mean")),
            "net_sum": safe_float(s.get("net_sum")),
            "worst_dd": safe_float(s.get("dd_worst")),
            "trades_total": safe_float(s.get("trades_total")),
        }
    agg = pick_mode_agg(report)
    if agg:
        n = agg.get("net_usd", {}) if isinstance(agg.get("net_usd"), dict) else {}
        p = agg.get("profit_factor", {}) if isinstance(agg.get("profit_factor"), dict) else {}
        d = agg.get("max_drawdown_usd", {}) if isinstance(agg.get("max_drawdown_usd"), dict) else {}
        t = agg.get("trades", {}) if isinstance(agg.get("trades"), dict) else {}
        return {
            "pos_fold_rate": safe_float(n.get("positive_fold_rate")),
            "worst_fold_net": safe_float(n.get("worst")),
            "pf_mean": safe_float(p.get("mean")),
            "net_sum": safe_float(n.get("sum")),
            "worst_dd": safe_float(d.get("worst")),
            "trades_total": safe_float(t.get("sum")),
        }
    return {
        "pos_fold_rate": safe_float(report.get("pos_fold_rate")),
        "worst_fold_net": safe_float(report.get("worst_fold_net")),
        "pf_mean": safe_float(report.get("pf_mean")),
        "net_sum": safe_float(report.get("net_sum")),
        "worst_dd": safe_float(report.get("worst_dd")),
        "trades_total": safe_float(report.get("trades_total")),
    }


def wf_fold_trade_counts(report: Optional[Dict[str, Any]]) -> List[float]:
    if not report:
        return []
    rows = report.get("fold_rows")
    if isinstance(rows, list) and rows:
        by_fold: Dict[int, float] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            profile = row.get("profile")
            if profile is not None and str(profile) != "base":
                continue
            fold_id = int(safe_float(row.get("fold"), d=0.0))
            trades = safe_float(row.get("trades"))
            by_fold[fold_id] = by_fold.get(fold_id, 0.0) + trades
        return [float(v) for _, v in sorted(by_fold.items(), key=lambda kv: kv[0])]

    modes = report.get("modes")
    if isinstance(modes, dict) and modes:
        block = modes.get("conservative") if isinstance(modes.get("conservative"), dict) else next(iter(modes.values()))
        if isinstance(block, dict):
            per_fold = block.get("per_fold")
            if isinstance(per_fold, list) and per_fold:
                return [safe_float(r.get("trades")) for r in per_fold if isinstance(r, dict)]

    folds = report.get("folds")
    if isinstance(folds, list) and folds:
        vals: List[float] = []
        for row in folds:
            if not isinstance(row, dict):
                continue
            vals.append(safe_float(row.get("total_trades", row.get("trades"))))
        return vals
    return []


def wf_sample_validity(
    wf_report: Optional[Dict[str, Any]],
    min_total_trades: float = DEFAULT_WF_MIN_TRADES_TOTAL,
    min_trades_per_fold: float = DEFAULT_WF_MIN_TRADES_PER_FOLD,
    min_folds_meeting_trades: int = DEFAULT_WF_MIN_FOLDS_MEETING_TRADES,
) -> Dict[str, float]:
    wf = normalize_wf(wf_report)
    fold_trades = wf_fold_trade_counts(wf_report)
    folds_meeting = int(sum(1 for t in fold_trades if t >= float(min_trades_per_fold)))
    min_fold_trades = float(min(fold_trades)) if fold_trades else 0.0
    valid_total = wf["trades_total"] >= float(min_total_trades)
    valid_fold_depth = folds_meeting >= int(min_folds_meeting_trades)
    return {
        "trades_total": float(wf["trades_total"]),
        "fold_count": float(len(fold_trades)),
        "min_fold_trades": float(min_fold_trades),
        "folds_meeting_trades": float(folds_meeting),
        "valid_total": 1.0 if valid_total else 0.0,
        "valid_fold_depth": 1.0 if valid_fold_depth else 0.0,
        "is_sufficient": 1.0 if (valid_total or valid_fold_depth) else 0.0,
    }


def normalize_lockbox(report: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not report:
        return {"pf": 0.0, "net": 0.0, "dd": 0.0, "trades": 0.0}
    s = pick_summary(report, "base")
    if s:
        return {
            "pf": safe_float(s.get("pf_mean", s.get("pf"))),
            "net": safe_float(s.get("net_sum", s.get("net"))),
            "dd": safe_float(s.get("dd_worst", s.get("max_dd_usd"))),
            "trades": safe_float(s.get("trades_total", s.get("trades"))),
        }
    agg = pick_mode_agg(report)
    if agg:
        n = agg.get("net_usd", {}) if isinstance(agg.get("net_usd"), dict) else {}
        p = agg.get("profit_factor", {}) if isinstance(agg.get("profit_factor"), dict) else {}
        d = agg.get("max_drawdown_usd", {}) if isinstance(agg.get("max_drawdown_usd"), dict) else {}
        t = agg.get("trades", {}) if isinstance(agg.get("trades"), dict) else {}
        return {"pf": safe_float(p.get("mean")), "net": safe_float(n.get("sum")), "dd": safe_float(d.get("worst")), "trades": safe_float(t.get("sum"))}
    return {
        "pf": safe_float(report.get("pf", report.get("profit_factor", report.get("pf_mean")))),
        "net": safe_float(report.get("net", report.get("net_sum"))),
        "dd": safe_float(report.get("dd", report.get("max_dd", report.get("max_dd_usd")))),
        "trades": safe_float(report.get("trades", report.get("trades_total"))),
    }


def lockbox_pf_validity(
    lockbox_report: Optional[Dict[str, Any]],
    min_lockbox_trades: float = DEFAULT_LOCKBOX_MIN_TRADES_FOR_PF,
) -> Dict[str, float | str]:
    lb = normalize_lockbox(lockbox_report)
    trades = safe_float(lb.get("trades"))
    ok = trades >= float(min_lockbox_trades)
    return {
        "trades": float(trades),
        "min_trades_required": float(min_lockbox_trades),
        "is_sufficient": 1.0 if ok else 0.0,
        "status": "LOCKBOX_PF_OK" if ok else "LOCKBOX_TOO_FEW_TRADES",
    }

def stability_score(wf_report: Optional[Dict[str, Any]]) -> float:
    m = normalize_wf(wf_report)
    pos = clamp(m["pos_fold_rate"], 0.0, 1.0)
    pf = clamp((m["pf_mean"] - 1.0) / 1.0, -1.0, 2.0)
    tail = clamp((m["worst_fold_net"] + 100.0) / 100.0, -3.0, 2.0)
    dd_pen = clamp(m["worst_dd"] / 200.0, 0.0, 5.0)
    trades_pen = clamp(m["trades_total"] / 5000.0, 0.0, 3.0)
    return 45.0 * pos + 30.0 * pf + 20.0 * tail - 15.0 * dd_pen - 10.0 * trades_pen


def sample_penalty(
    wf_report: Optional[Dict[str, Any]],
    min_total_trades: float = DEFAULT_WF_MIN_TRADES_TOTAL,
    min_trades_per_fold: float = DEFAULT_WF_MIN_TRADES_PER_FOLD,
    min_folds_meeting_trades: int = DEFAULT_WF_MIN_FOLDS_MEETING_TRADES,
) -> float:
    s = wf_sample_validity(
        wf_report,
        min_total_trades=min_total_trades,
        min_trades_per_fold=min_trades_per_fold,
        min_folds_meeting_trades=min_folds_meeting_trades,
    )
    if s["is_sufficient"] >= 0.5:
        return 0.0
    return -50.0


def robustness_score(
    lockbox_base_report: Optional[Dict[str, Any]],
    stress_reports: Dict[str, Optional[Dict[str, Any]]],
    min_lockbox_trades: float = DEFAULT_LOCKBOX_MIN_TRADES_FOR_PF,
) -> float:
    lb_valid = lockbox_pf_validity(lockbox_base_report, min_lockbox_trades=min_lockbox_trades)
    if lb_valid["is_sufficient"] < 0.5:
        return 0.0
    b = normalize_lockbox(lockbox_base_report)
    base_pf = b["pf"]
    base_net = b["net"]
    pf25 = normalize_lockbox(stress_reports.get("spread25_slip2x"))["pf"]
    pf30 = normalize_lockbox(stress_reports.get("spread30_slip2x"))["pf"]
    drop25 = max(0.0, base_pf - pf25) if pf25 > 0 else 1.0
    drop30 = max(0.0, base_pf - pf30) if pf30 > 0 else 1.2
    net25 = normalize_lockbox(stress_reports.get("spread25_slip2x"))["net"]
    net30 = normalize_lockbox(stress_reports.get("spread30_slip2x"))["net"]
    pos_stress = (1 if net25 > 0 else 0) + (1 if net30 > 0 else 0)
    pf_term = clamp((base_pf - 1.0) / 1.0, -1.0, 2.0)
    drop_term = clamp(1.0 - (drop25 + drop30) / 1.0, -2.0, 1.0)
    net_term = clamp(base_net / 100.0, -2.0, 2.0)
    return 40.0 * pf_term + 20.0 * net_term + 25.0 * (pos_stress / 2.0) + 25.0 * drop_term


def overall_score(stability: float, robustness: float) -> float:
    return 0.60 * stability + 0.40 * robustness


def gate_decision(
    wf_report: Optional[Dict[str, Any]],
    lockbox_base_report: Optional[Dict[str, Any]],
    stress_reports: Dict[str, Optional[Dict[str, Any]]],
    min_total_trades: float = DEFAULT_WF_MIN_TRADES_TOTAL,
    min_trades_per_fold: float = DEFAULT_WF_MIN_TRADES_PER_FOLD,
    min_folds_meeting_trades: int = DEFAULT_WF_MIN_FOLDS_MEETING_TRADES,
    min_lockbox_trades: float = DEFAULT_LOCKBOX_MIN_TRADES_FOR_PF,
) -> str:
    if not wf_report or not lockbox_base_report:
        return "INCOMPLETE"
    sample = wf_sample_validity(
        wf_report,
        min_total_trades=min_total_trades,
        min_trades_per_fold=min_trades_per_fold,
        min_folds_meeting_trades=min_folds_meeting_trades,
    )
    if sample["is_sufficient"] < 0.5:
        return "INSUFFICIENT_SAMPLE"
    lb_valid = lockbox_pf_validity(lockbox_base_report, min_lockbox_trades=min_lockbox_trades)
    if lb_valid["is_sufficient"] < 0.5:
        return "LOCKBOX_TOO_FEW_TRADES"
    wf = normalize_wf(wf_report)
    lb = normalize_lockbox(lockbox_base_report)
    pf25 = normalize_lockbox(stress_reports.get("spread25_slip2x"))["pf"]
    pf30 = normalize_lockbox(stress_reports.get("spread30_slip2x"))["pf"]
    stable_pass = (wf["pos_fold_rate"] >= 0.60) and (wf["worst_fold_net"] >= -70.0) and (wf["pf_mean"] >= 1.0)
    lockbox_pass = (lb["pf"] >= 1.30) and (lb["net"] > 0.0)
    stress_pass = (pf25 >= 1.30) and (pf30 >= 1.10)
    if stable_pass and lockbox_pass and stress_pass:
        return "PASS"
    if lockbox_pass and not stable_pass:
        return "LOCKBOX_ONLY"
    if stable_pass and not lockbox_pass:
        return "STABLE_ONLY"
    return "FAIL"


def suggest_patch(
    strategy: str,
    wf_report: Optional[Dict[str, Any]],
    lockbox_base_report: Optional[Dict[str, Any]],
    stress_reports: Dict[str, Optional[Dict[str, Any]]],
    min_total_trades: float = DEFAULT_WF_MIN_TRADES_TOTAL,
    min_trades_per_fold: float = DEFAULT_WF_MIN_TRADES_PER_FOLD,
    min_folds_meeting_trades: int = DEFAULT_WF_MIN_FOLDS_MEETING_TRADES,
    min_lockbox_trades: float = DEFAULT_LOCKBOX_MIN_TRADES_FOR_PF,
) -> str:
    if not wf_report or not lockbox_base_report:
        return "Add lockbox support and ensure report_json is emitted."
    sample = wf_sample_validity(
        wf_report,
        min_total_trades=min_total_trades,
        min_trades_per_fold=min_trades_per_fold,
        min_folds_meeting_trades=min_folds_meeting_trades,
    )
    if sample["is_sufficient"] < 0.5:
        return "Insufficient WF sample: increase deterministic trade count via session/rule adjustments; skip RL gate retuning."
    lb_valid = lockbox_pf_validity(lockbox_base_report, min_lockbox_trades=min_lockbox_trades)
    if lb_valid["is_sufficient"] < 0.5:
        return "LOCKBOX_TOO_FEW_TRADES: do not infer robustness PF until lockbox trades >= 30."
    wf = normalize_wf(wf_report)
    lb = normalize_lockbox(lockbox_base_report)
    pf25 = normalize_lockbox(stress_reports.get("spread25_slip2x"))["pf"]
    pf30 = normalize_lockbox(stress_reports.get("spread30_slip2x"))["pf"]
    if pf30 > 0 and pf30 < 1.0:
        return "Cost-fragile at 0.00030 spread: add stronger expected-move gate and reduce churn."
    if wf["pos_fold_rate"] < 0.60 or wf["worst_fold_net"] < -70.0:
        return "Fold stability weak: tighten one filter only (compression, hours, or close-confirm delay)."
    if 1.30 <= lb["pf"] < 1.50 and (pf25 > 0 and pf25 < 1.30):
        return "Lockbox base passes but stress drops: increase per-trade expectancy (slightly higher TP or entry buffer)."
    if strategy.lower().startswith("strategy 2"):
        return "For Strategy 2, confirm trend continuation with HTF alignment and pullback-depth constraints."
    return "No single obvious patch; sweep one axis only with all else fixed."


DEFAULT_CANDIDATES: Dict[str, List[Dict[str, Any]]] = {
    "strategy_1": [
        {
            "name": "S1_London_Compression_Breakout_SL12_TP24",
            "script": "scripts/vol_breakout_backtest_wf.py",
            "args": {
                "--horizon-bars": "6", "--move-threshold-pips": "40", "--prob-th": "0.62",
                "--train-years": "3", "--test-months": "6", "--step-months": "6", "--modes": "conservative",
                "--session-filter": "london_ny", "--hour-windows": "7-10", "--regime-filter": "trend_or_range", "--trend-min": "0.0001",
                "--lookback": "24", "--buffer-pips": "1", "--sl-pips": "12", "--tp-pips": "24", "--time-stop-bars": "12",
                "--compression-max-quantile": "0.35", "--compression-window": "24", "--atr-norm-max-quantile": "0.60", "--atr-window": "24",
                "--require-close-confirm": "true", "--max-trades-per-session": "1",
                "--spread": "0.0002", "--slippage": "0.00005", "--commission": "0.0001",
            },
        },
        {
            "name": "S1_London_Compression_Breakout_SL12_TP26",
            "script": "scripts/vol_breakout_backtest_wf.py",
            "args": {
                "--horizon-bars": "6", "--move-threshold-pips": "40", "--prob-th": "0.62",
                "--train-years": "3", "--test-months": "6", "--step-months": "6", "--modes": "conservative",
                "--session-filter": "london_ny", "--hour-windows": "7-10", "--regime-filter": "trend_or_range", "--trend-min": "0.0001",
                "--lookback": "24", "--buffer-pips": "1", "--sl-pips": "12", "--tp-pips": "26", "--time-stop-bars": "12",
                "--compression-max-quantile": "0.35", "--compression-window": "24", "--atr-norm-max-quantile": "0.60", "--atr-window": "24",
                "--require-close-confirm": "true", "--max-trades-per-session": "1",
                "--spread": "0.0002", "--slippage": "0.00005", "--commission": "0.0001",
            },
        },
    ],
    "strategy_2": [
        {
            "name": "S2_NY_Pullback_Trend_Continuation",
            "script": "scripts/ny_pullback_trend_continuation_wf.py",
            "args": {
                "--train-years": "3", "--test-months": "6", "--step-months": "6",
                "--session-filter": "ny_only", "--hour-windows": "12-17", "--max-trades-per-session": "1",
                "--sl-atr": "1.0", "--tp-atr": "1.6", "--time-stop-bars": "10",
                "--spread": "0.0002", "--slippage": "0.00005", "--commission": "0.0001",
            },
        }
    ],
}

EXPERIMENTAL_CANDIDATES: Dict[str, List[Dict[str, Any]]] = {
    "strategy_1": [
        {
            "name": "S1_London_Compression_Breakout_RLM_Gate",
            "script": "scripts/rlm_eval_wf.py",
            "args": {
                "--horizon-bars": "6", "--move-threshold-pips": "40", "--prob-th": "0.62",
                "--exec-mode": "conservative",
                "--train-years": "3", "--test-months": "6", "--step-months": "6",
                "--session-filter": "london_ny", "--hour-windows": "7-10", "--regime-filter": "trend_or_range", "--trend-min": "0.0001",
                "--lookback": "24", "--buffer-pips": "1", "--sl-pips": "12", "--tp-pips": "24", "--time-stop-bars": "12",
                "--compression-max-quantile": "0.35", "--compression-window": "24", "--atr-norm-max-quantile": "0.60", "--atr-window": "24",
                "--require-close-confirm": "true", "--max-trades-per-session": "1",
                "--rl-algo": "ppo", "--rl-train-timesteps": "30000", "--rl-min-train-events": "30",
                "--rl-target-take-rate": "0.30", "--rl-take-rate-min": "0.15", "--rl-take-rate-max": "0.60",
                "--spread": "0.0002", "--slippage": "0.00005", "--commission": "0.0001",
            },
        },
    ]
}


def stress_profiles(base_spread: float, base_slippage: float) -> List[Dict[str, Any]]:
    return [
        {"name": "base", "spread": base_spread, "slippage": base_slippage},
        {"name": "spread25_slip2x", "spread": 0.00025, "slippage": base_slippage * 2.0},
        {"name": "spread30_slip2x", "spread": 0.00030, "slippage": base_slippage * 2.0},
    ]

def lockbox_args(script_name: str, train_end: str, test_start: str, test_end: str) -> Dict[str, str]:
    n = script_name.lower()
    if n == "vol_breakout_backtest_wf.py":
        return {"--lockbox-train-end": train_end, "--lockbox-test-start": test_start, "--lockbox-test-end": test_end}
    if n == "ny_pullback_trend_continuation_wf.py":
        return {"--mode": "lockbox", "--train-end-utc": train_end, "--test-start-utc": test_start, "--test-end-utc": test_end}
    return {"--mode": "lockbox", "--lockbox-train-end": train_end, "--lockbox-test-start": test_start, "--lockbox-test-end": test_end}


def wf_extra_args(script_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(args)
    if script_name.lower() == "ny_pullback_trend_continuation_wf.py":
        out.setdefault("--mode", "wf")
    return out


def build_cmd(python_exe: str, script: Path, data_csv: Path, out_prefix: str, extra: Dict[str, Any], max_folds: Optional[int], lgbm_device: Optional[str], lb_args: Optional[Dict[str, str]] = None) -> List[str]:
    cmd = [python_exe, str(script), "--data-csv", str(data_csv), "--out-prefix", out_prefix]
    if max_folds is not None:
        cmd += ["--max-folds", str(max_folds)]
    if lgbm_device is not None:
        cmd += ["--lgbm-device", str(lgbm_device)]
    for k, v in extra.items():
        if v is None:
            continue
        b = parse_bool(v)
        if k in BOOL_FLAG_KEYS and b is not None:
            if b:
                cmd.append(str(k))
            continue
        cmd += [str(k), str(v)]
    if lb_args:
        for k, v in lb_args.items():
            cmd += [str(k), str(v)]
    return cmd


def run_one(kind: str, strategy: str, candidate: str, cmd: List[str], cwd: Path, logs_dir: Path, out_prefix: str, reports_dir: Path) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    ensure_dir(logs_dir)
    rc, out = run_cmd(cmd, cwd)
    log_path = logs_dir / f"{out_prefix}__{kind}.log"
    log_path.write_text(out, encoding="utf-8", errors="replace")
    rep, err = (None, None)
    parsed = None
    if rc == 0:
        rep, err = find_report(out, out_prefix, reports_dir)
        if rep:
            try:
                parsed = read_json(rep)
            except Exception as e:
                err = f"failed_to_parse_report_json: {e}"
    else:
        err = f"nonzero_exit rc={rc}"
    art = {
        "kind": kind,
        "strategy": strategy,
        "candidate": candidate,
        "cmd": cmd,
        "rc": rc,
        "stdout_log": str(log_path),
        "report_json": str(rep) if rep else None,
        "error": err,
    }
    return art, parsed


def render_md(results: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    lines = [
        "# Strategy Agent Ranking Report",
        "",
        f"- Generated (UTC): `{meta['generated_utc']}`",
        f"- Data: `{meta['data_csv']}`",
        f"- Lockbox: `{meta['lockbox_train_end']}` -> `{meta['lockbox_test_start']}` to `{meta['lockbox_test_end']}`",
        f"- WF sample rule: total_trades>={meta['wf_min_trades_total']} OR (folds_with_trades>={meta['wf_min_folds_meeting_trades']} at >= {meta['wf_min_trades_per_fold']}/fold)",
        f"- Lockbox PF rule: trades>={meta['lockbox_min_trades_for_pf']} required for PF-based robustness",
        f"- Experimental RLM candidates included: `{bool(meta.get('include_experimental_rlm', False))}`",
        "",
        "| Rank | Strategy | Candidate | Decision | Overall | Stability | Robustness | WF trades | Sample OK | LB PF OK | WF pos_fold | WF PF | WF worst_fold | LB PF | PF25 | PF30 |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, r in enumerate(results, 1):
        wm = r["wf_metrics"] or {}
        ws = r.get("wf_sample") or {}
        lb = r["lockbox_base_metrics"] or {}
        lbs = r.get("lockbox_sample") or {}
        s25 = (r["lockbox_stress_metrics"] or {}).get("spread25_slip2x") or {}
        s30 = (r["lockbox_stress_metrics"] or {}).get("spread30_slip2x") or {}
        lines.append(
            f"| {i} | {r['strategy']} | {r['candidate']} | **{r['decision']}** | {r['scores']['overall']:.2f} | {r['scores']['stability']:.2f} | {r['scores']['robustness']:.2f} | {safe_float(wm.get('trades_total')):.0f} | {bool(ws.get('is_sufficient', 0.0) >= 0.5)} | {bool(lbs.get('is_sufficient', 0.0) >= 0.5)} | {safe_float(wm.get('pos_fold_rate')):.2f} | {safe_float(wm.get('pf_mean')):.3f} | {safe_float(wm.get('worst_fold_net')):.2f} | {safe_float(lb.get('pf')):.3f} | {safe_float(s25.get('pf')):.3f} | {safe_float(s30.get('pf')):.3f} |"
        )
    lines += ["", "## Next Patches", ""]
    for r in results:
        lines.append(f"- `{r['strategy']} / {r['candidate']}`: {r['next_patch']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--python-exe", default=sys.executable)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--max-folds", type=int, default=10)
    ap.add_argument("--lgbm-device", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--lockbox-train-end", default="2024-12-31 23:00:00+00:00")
    ap.add_argument("--lockbox-test-start", default="2025-01-01 00:00:00+00:00")
    ap.add_argument("--lockbox-test-end", default="2026-02-04 23:00:00+00:00")
    ap.add_argument("--candidates-json", default=None)
    ap.add_argument("--base-spread", type=float, default=0.00020)
    ap.add_argument("--base-slippage", type=float, default=0.00005)
    ap.add_argument("--base-commission", type=float, default=0.00010)
    ap.add_argument("--wf-min-trades-total", type=float, default=DEFAULT_WF_MIN_TRADES_TOTAL)
    ap.add_argument("--wf-min-trades-per-fold", type=float, default=DEFAULT_WF_MIN_TRADES_PER_FOLD)
    ap.add_argument("--wf-min-folds-meeting-trades", type=int, default=DEFAULT_WF_MIN_FOLDS_MEETING_TRADES)
    ap.add_argument("--lockbox-min-trades-for-pf", type=float, default=DEFAULT_LOCKBOX_MIN_TRADES_FOR_PF)
    ap.add_argument("--include-experimental-rlm", action="store_true", help="Include RLM gate candidates (off by default).")
    args = ap.parse_args()
    if float(args.lockbox_min_trades_for_pf) < 0.0:
        raise SystemExit("--lockbox-min-trades-for-pf must be >= 0.")

    repo_root = Path(args.repo_root).resolve()
    data_csv = Path(args.data_csv).resolve()
    reports_dir = (repo_root / args.reports_dir).resolve()
    logs_dir = reports_dir / "agent_logs"
    ensure_dir(reports_dir)
    ensure_dir(logs_dir)

    candidates = DEFAULT_CANDIDATES
    if args.candidates_json:
        candidates = read_json(Path(args.candidates_json).resolve())
    elif bool(args.include_experimental_rlm):
        for k, items in EXPERIMENTAL_CANDIDATES.items():
            candidates.setdefault(k, [])
            candidates[k].extend(items)

    meta = {
        "generated_utc": now_iso(),
        "data_csv": str(data_csv),
        "lockbox_train_end": args.lockbox_train_end,
        "lockbox_test_start": args.lockbox_test_start,
        "lockbox_test_end": args.lockbox_test_end,
        "max_folds": args.max_folds,
        "lgbm_device": args.lgbm_device,
        "wf_min_trades_total": args.wf_min_trades_total,
        "wf_min_trades_per_fold": args.wf_min_trades_per_fold,
        "wf_min_folds_meeting_trades": args.wf_min_folds_meeting_trades,
        "lockbox_min_trades_for_pf": args.lockbox_min_trades_for_pf,
        "include_experimental_rlm": bool(args.include_experimental_rlm),
    }

    ranked: List[Dict[str, Any]] = []
    stresses = stress_profiles(args.base_spread, args.base_slippage)

    for sk, items in candidates.items():
        strategy = "Strategy 1" if sk == "strategy_1" else "Strategy 2"
        for item in items:
            name = str(item["name"])
            script = (repo_root / str(item["script"])).resolve()
            args_map = dict(item.get("args", {}))
            args_map.setdefault("--spread", str(args.base_spread))
            args_map.setdefault("--slippage", str(args.base_slippage))
            args_map.setdefault("--commission", str(args.base_commission))

            artifacts: List[Dict[str, Any]] = []
            wf_prefix = f"{args.out_prefix}__{name}__wf"
            wf_cmd = build_cmd(args.python_exe, script, data_csv, wf_prefix, wf_extra_args(script.name, args_map), args.max_folds, args.lgbm_device)
            wf_art, wf_json = run_one("wf", strategy, name, wf_cmd, repo_root, logs_dir, wf_prefix, reports_dir)
            artifacts.append(wf_art)

            lb_base = None
            lb_stress: Dict[str, Optional[Dict[str, Any]]] = {}
            lb_map = lockbox_args(script.name, args.lockbox_train_end, args.lockbox_test_start, args.lockbox_test_end)

            base_prefix = f"{args.out_prefix}__{name}__lockbox__base"
            base_cmd = build_cmd(
                args.python_exe,
                script,
                data_csv,
                base_prefix,
                {**args_map, "--spread": str(args.base_spread), "--slippage": str(args.base_slippage), "--commission": str(args.base_commission)},
                1,
                args.lgbm_device,
                lb_map,
            )
            base_art, lb_base = run_one("lockbox_base", strategy, name, base_cmd, repo_root, logs_dir, base_prefix, reports_dir)
            artifacts.append(base_art)

            for st in stresses:
                if st["name"] == "base":
                    continue
                pfx = f"{args.out_prefix}__{name}__lockbox__{st['name']}"
                cmd = build_cmd(
                    args.python_exe,
                    script,
                    data_csv,
                    pfx,
                    {**args_map, "--spread": str(st["spread"]), "--slippage": str(st["slippage"]), "--commission": str(args.base_commission)},
                    1,
                    args.lgbm_device,
                    lb_map,
                )
                art, js = run_one("lockbox_stress", strategy, name, cmd, repo_root, logs_dir, pfx, reports_dir)
                artifacts.append(art)
                lb_stress[str(st["name"])] = js

            wf_sample = wf_sample_validity(
                wf_json,
                min_total_trades=float(args.wf_min_trades_total),
                min_trades_per_fold=float(args.wf_min_trades_per_fold),
                min_folds_meeting_trades=int(args.wf_min_folds_meeting_trades),
            )
            lb_sample = lockbox_pf_validity(
                lb_base,
                min_lockbox_trades=float(args.lockbox_min_trades_for_pf),
            )
            stab = stability_score(wf_json) + sample_penalty(
                wf_json,
                min_total_trades=float(args.wf_min_trades_total),
                min_trades_per_fold=float(args.wf_min_trades_per_fold),
                min_folds_meeting_trades=int(args.wf_min_folds_meeting_trades),
            )
            rob = robustness_score(
                lb_base,
                lb_stress,
                min_lockbox_trades=float(args.lockbox_min_trades_for_pf),
            )
            ov = overall_score(stab, rob)
            dec = gate_decision(
                wf_json,
                lb_base,
                lb_stress,
                min_total_trades=float(args.wf_min_trades_total),
                min_trades_per_fold=float(args.wf_min_trades_per_fold),
                min_folds_meeting_trades=int(args.wf_min_folds_meeting_trades),
                min_lockbox_trades=float(args.lockbox_min_trades_for_pf),
            )
            nxt = suggest_patch(
                strategy,
                wf_json,
                lb_base,
                lb_stress,
                min_total_trades=float(args.wf_min_trades_total),
                min_trades_per_fold=float(args.wf_min_trades_per_fold),
                min_folds_meeting_trades=int(args.wf_min_folds_meeting_trades),
                min_lockbox_trades=float(args.lockbox_min_trades_for_pf),
            )

            ranked.append(
                {
                    "strategy": strategy,
                    "candidate": name,
                    "decision": dec,
                    "scores": {"stability": stab, "robustness": rob, "overall": ov},
                    "wf_metrics": normalize_wf(wf_json),
                    "wf_sample": wf_sample,
                    "lockbox_sample": lb_sample,
                    "lockbox_base_metrics": normalize_lockbox(lb_base),
                    "lockbox_stress_metrics": {k: normalize_lockbox(v) for k, v in lb_stress.items()},
                    "next_patch": nxt,
                    "artifacts": artifacts,
                }
            )

    decision_rank = {
        "PASS": 5,
        "STABLE_ONLY": 4,
        "LOCKBOX_ONLY": 3,
        "FAIL": 2,
        "LOCKBOX_TOO_FEW_TRADES": 1,
        "INSUFFICIENT_SAMPLE": 1,
        "INCOMPLETE": 0,
    }
    ranked.sort(
        key=lambda r: (
            decision_rank.get(str(r.get("decision")), 0),
            r["scores"]["overall"],
            r["scores"]["robustness"],
            r["scores"]["stability"],
        ),
        reverse=True,
    )

    ts = now_stamp()
    out_json = reports_dir / f"{args.out_prefix}_agent_rank_{ts}.json"
    out_md = reports_dir / f"{args.out_prefix}_agent_rank_{ts}.md"
    payload = {"meta": meta, "ranked": ranked}
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md.write_text(render_md(ranked, meta), encoding="utf-8")

    print("=== AGENT DONE ===")
    print(f"rank_report_json={out_json}")
    print(f"rank_report_md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
