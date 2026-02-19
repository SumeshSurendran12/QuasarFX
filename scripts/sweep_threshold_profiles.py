from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class CandidateConfig:
    candidate_id: int
    trend_min: float
    atr_min: float
    atr_max: float
    expectancy_threshold: float
    expectancy_reward_weight: float
    low_expectancy_entry_penalty: float
    low_expectancy_close_penalty_scale: float


def parse_float_list(raw: str) -> List[float]:
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("Empty float list.")
    return values


def parse_atr_bands(raw: str) -> List[Tuple[float, float]]:
    bands: List[Tuple[float, float]] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid ATR band token: {token}")
        atr_min = float(parts[0].strip())
        atr_max = float(parts[1].strip())
        if atr_max <= atr_min:
            raise ValueError(f"ATR max must be > min: {token}")
        bands.append((atr_min, atr_max))
    if not bands:
        raise ValueError("Empty ATR bands list.")
    return bands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Targeted threshold sweep (trend/ATR/expectancy) with auto walk-forward ranking."
    )
    parser.add_argument("--profile", default="gpu_quality")
    parser.add_argument("--timesteps", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-csv", default="")
    parser.add_argument("--trend-values", default="0.00010,0.00012")
    parser.add_argument("--atr-bands", default="0.00030:0.00180,0.00035:0.00190")
    parser.add_argument("--expectancy-values", default="0.00004,0.00005")
    parser.add_argument("--expectancy-reward-values", default="0.50,0.70")
    parser.add_argument("--low-entry-penalty-values", default="-0.0015,-0.0025")
    parser.add_argument("--low-close-penalty-scale-values", default="0.60,1.00")
    parser.add_argument("--train-block-low-expectancy", type=int, choices=[0, 1], default=1)
    parser.add_argument("--eval-block-low-expectancy", type=int, choices=[0, 1], default=1)
    parser.add_argument("--eval-block-in-eval", type=int, choices=[0, 1], default=1)
    parser.add_argument("--min-train-frac", type=float, default=0.60)
    parser.add_argument("--test-frac", type=float, default=0.04)
    parser.add_argument("--max-folds", type=int, default=10)
    parser.add_argument("--min-trades-per-fold", type=int, default=1)
    parser.add_argument("--out-prefix", default="threshold_sweep_gpu_quality")
    parser.add_argument("--keep-models", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: List[str], env: Dict[str, str]) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        tail = "\n".join(output.splitlines()[-80:])
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{tail}")
    return output


def extract_path(text: str, key: str) -> Path:
    pattern = re.compile(rf"{re.escape(key)}=(.+)")
    for line in text.splitlines():
        m = pattern.search(line.strip())
        if m:
            return Path(m.group(1).strip())
    raise ValueError(f"Could not find {key}=... in command output.")


def clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def selection_score(summary: Dict[str, float]) -> float:
    score = 0.0
    if str(summary.get("decision", "FAIL")).upper() == "PASS":
        score += 1000.0
    score += float(summary.get("pass_rate", 0.0)) * 300.0
    score += clamp(float(summary.get("overall_net_profit", 0.0)) / 40.0, -500.0, 500.0)
    score += (float(summary.get("overall_profit_factor", 0.0)) - 1.0) * 300.0
    score += float(summary.get("mean_fold_return_pct", 0.0)) * 5.0
    score -= float(summary.get("worst_fold_drawdown_pct", 0.0)) * 2.5
    if bool(summary.get("collapse_detected", False)):
        score -= 500.0
    return float(score)


def to_markdown(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Threshold Sweep Report")
    lines.append("")
    lines.append(f"- Generated: {payload['generated_at']}")
    lines.append(f"- Profile: `{payload['profile']}`")
    lines.append(f"- Timesteps per candidate: {payload['timesteps']}")
    lines.append(f"- Candidate count: {payload['candidate_count']}")
    policy = payload["policy"]
    lines.append(f"- Train block low expectancy: `{policy['train_block_low_expectancy']}`")
    lines.append(f"- Eval block low expectancy: `{policy['eval_block_low_expectancy']}`")
    lines.append(f"- Eval block in eval mode: `{policy['eval_block_in_eval']}`")
    lines.append("")
    best = payload["best_candidate"]
    lines.append("## Best Candidate")
    lines.append(
        f"- Candidate: `{best['candidate_id']}` | score={best['selection_score']:.3f} | "
        f"decision={best['walk_forward_summary']['decision']}"
    )
    lines.append(
        f"- trend_min={best['trend_min']:.8f}, atr_min={best['atr_min']:.8f}, "
        f"atr_max={best['atr_max']:.8f}, expectancy_threshold={best['expectancy_threshold']:.8f}, "
        f"entry_weight={best['expectancy_reward_weight']:.4f}, "
        f"entry_penalty={best['low_expectancy_entry_penalty']:.4f}, "
        f"close_penalty_scale={best['low_expectancy_close_penalty_scale']:.4f}"
    )
    lines.append(
        f"- WF net={best['walk_forward_summary']['overall_net_profit']:.2f}, "
        f"PF={best['walk_forward_summary']['overall_profit_factor']:.3f}, "
        f"worst_dd={best['walk_forward_summary']['worst_fold_drawdown_pct']:.2f}, "
        f"pass_rate={best['walk_forward_summary']['pass_rate']*100:.1f}%"
    )
    lines.append("")
    lines.append("| ID | trend_min | atr_min | atr_max | exp_th | entry_w | entry_pen | close_scale | score | decision | net | PF | DD% | pass_rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |")
    for c in payload["candidates_ranked"]:
        s = c["walk_forward_summary"]
        lines.append(
            "| "
            f"{c['candidate_id']} | {c['trend_min']:.8f} | {c['atr_min']:.8f} | {c['atr_max']:.8f} | "
            f"{c['expectancy_threshold']:.8f} | {c['expectancy_reward_weight']:.4f} | "
            f"{c['low_expectancy_entry_penalty']:.4f} | {c['low_expectancy_close_penalty_scale']:.4f} | "
            f"{c['selection_score']:.3f} | {s['decision']} | "
            f"{s['overall_net_profit']:.2f} | {s['overall_profit_factor']:.3f} | "
            f"{s['worst_fold_drawdown_pct']:.2f} | {s['pass_rate']*100:.1f}% |"
        )
    lines.append("")
    return "\n".join(lines)


def write_best_env(
    best: Dict[str, object],
    out_path: Path,
    eval_block_low_expectancy: int,
    eval_block_in_eval: int,
) -> None:
    lines = [
        "# Auto-selected best threshold profile from sweep",
        "FX_PROFILE=gpu_quality",
        "FX_REGIME_FILTER_ENABLED=1",
        f"FX_REGIME_BLOCK_LOW_EXPECTANCY={int(eval_block_low_expectancy)}",
        f"FX_REGIME_BLOCK_IN_EVAL={int(eval_block_in_eval)}",
        f"FX_REGIME_TREND_MIN_STRENGTH={best['trend_min']:.8f}",
        f"FX_REGIME_VOL_MIN_ATR_NORM={best['atr_min']:.8f}",
        f"FX_REGIME_VOL_MAX_ATR_NORM={best['atr_max']:.8f}",
        f"FX_ENTRY_EXPECTANCY_THRESHOLD={best['expectancy_threshold']:.8f}",
        f"FX_ENTRY_EXPECTANCY_REWARD_WEIGHT={best['expectancy_reward_weight']:.6f}",
        f"FX_LOW_EXPECTANCY_ENTRY_PENALTY={best['low_expectancy_entry_penalty']:.6f}",
        f"FX_LOW_EXPECTANCY_CLOSE_PENALTY_SCALE={best['low_expectancy_close_penalty_scale']:.6f}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    trend_values = parse_float_list(args.trend_values)
    atr_bands = parse_atr_bands(args.atr_bands)
    expectancy_values = parse_float_list(args.expectancy_values)
    expectancy_reward_values = parse_float_list(args.expectancy_reward_values)
    low_entry_penalty_values = parse_float_list(args.low_entry_penalty_values)
    low_close_penalty_scale_values = parse_float_list(args.low_close_penalty_scale_values)

    candidates: List[CandidateConfig] = []
    candidate_id = 1
    for trend_min, (atr_min, atr_max), expectancy, entry_weight, entry_penalty, close_penalty_scale in itertools.product(
        trend_values,
        atr_bands,
        expectancy_values,
        expectancy_reward_values,
        low_entry_penalty_values,
        low_close_penalty_scale_values,
    ):
        candidates.append(
            CandidateConfig(
                candidate_id=candidate_id,
                trend_min=float(trend_min),
                atr_min=float(atr_min),
                atr_max=float(atr_max),
                expectancy_threshold=float(expectancy),
                expectancy_reward_weight=float(entry_weight),
                low_expectancy_entry_penalty=float(entry_penalty),
                low_expectancy_close_penalty_scale=float(close_penalty_scale),
            )
        )
        candidate_id += 1

    results: List[Dict[str, object]] = []
    py_exec = str(ROOT / ".venv" / "Scripts" / "python.exe")

    for c in candidates:
        model_rel = Path("models") / f"{args.out_prefix}_c{c.candidate_id:02d}.zip"
        wf_prefix = f"{args.out_prefix}_c{c.candidate_id:02d}"

        env_common = os.environ.copy()
        env_common["FX_PROFILE"] = str(args.profile)
        env_common["WANDB_MODE"] = env_common.get("WANDB_MODE", "disabled")
        env_common["FX_REGIME_FILTER_ENABLED"] = "1"
        env_common["FX_REGIME_TREND_MIN_STRENGTH"] = f"{c.trend_min:.8f}"
        env_common["FX_REGIME_VOL_MIN_ATR_NORM"] = f"{c.atr_min:.8f}"
        env_common["FX_REGIME_VOL_MAX_ATR_NORM"] = f"{c.atr_max:.8f}"
        env_common["FX_ENTRY_EXPECTANCY_THRESHOLD"] = f"{c.expectancy_threshold:.8f}"
        env_common["FX_ENTRY_EXPECTANCY_REWARD_WEIGHT"] = f"{c.expectancy_reward_weight:.8f}"
        env_common["FX_LOW_EXPECTANCY_ENTRY_PENALTY"] = f"{c.low_expectancy_entry_penalty:.8f}"
        env_common["FX_LOW_EXPECTANCY_CLOSE_PENALTY_SCALE"] = f"{c.low_expectancy_close_penalty_scale:.8f}"

        train_env = env_common.copy()
        train_env["FX_REGIME_BLOCK_LOW_EXPECTANCY"] = str(int(args.train_block_low_expectancy))
        train_env["FX_REGIME_BLOCK_IN_EVAL"] = str(int(args.eval_block_in_eval))

        eval_env = env_common.copy()
        eval_env["FX_REGIME_BLOCK_LOW_EXPECTANCY"] = str(int(args.eval_block_low_expectancy))
        eval_env["FX_REGIME_BLOCK_IN_EVAL"] = str(int(args.eval_block_in_eval))

        train_cmd = [
            py_exec,
            "scripts/train_profile_quick.py",
            "--timesteps",
            str(int(args.timesteps)),
            "--seed",
            str(int(args.seed)),
            "--output",
            str(model_rel),
        ]
        if args.data_csv:
            train_cmd.extend(["--data-csv", str(args.data_csv)])
        print(
            f"[candidate {c.candidate_id}/{len(candidates)}] train trend={c.trend_min:.8f} "
            f"atr=({c.atr_min:.8f},{c.atr_max:.8f}) exp={c.expectancy_threshold:.8f} "
            f"entry_w={c.expectancy_reward_weight:.4f} entry_pen={c.low_expectancy_entry_penalty:.4f} "
            f"close_scale={c.low_expectancy_close_penalty_scale:.4f} "
            f"train_block={int(args.train_block_low_expectancy)} eval_block={int(args.eval_block_low_expectancy)}",
            flush=True,
        )
        run_cmd(train_cmd, env=train_env)

        wf_cmd = [
            py_exec,
            "scripts/walk_forward_eval.py",
            "--models",
            str(model_rel),
            "--min-train-frac",
            str(args.min_train_frac),
            "--test-frac",
            str(args.test_frac),
            "--max-folds",
            str(int(args.max_folds)),
            "--min-trades-per-fold",
            str(int(args.min_trades_per_fold)),
            "--out-prefix",
            wf_prefix,
        ]
        if args.data_csv:
            wf_cmd.extend(["--data-csv", str(args.data_csv)])
        wf_output = run_cmd(wf_cmd, env=eval_env)
        wf_json_path = extract_path(wf_output, "report_json")
        wf_payload = json.loads(wf_json_path.read_text(encoding="utf-8"))
        if not wf_payload.get("reports"):
            raise RuntimeError(f"No reports found in {wf_json_path}")
        summary = wf_payload["reports"][0]["summary"]
        score = selection_score(summary)

        metrics_path = (ROOT / model_rel).with_suffix(".metrics.json")
        quick_metrics = {}
        if metrics_path.exists():
            quick_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            quick_metrics = quick_payload.get("metrics", {})

        record = {
            **asdict(c),
            "model_path": str(ROOT / model_rel),
            "metrics_path": str(metrics_path),
            "walk_forward_report_json": str(wf_json_path),
            "quick_metrics": quick_metrics,
            "walk_forward_summary": summary,
            "selection_score": score,
        }
        results.append(record)
        print(
            f"[candidate {c.candidate_id}] score={score:.3f} "
            f"decision={summary['decision']} net={summary['overall_net_profit']:.2f} "
            f"pf={summary['overall_profit_factor']:.3f} dd={summary['worst_fold_drawdown_pct']:.2f} "
            f"entry_w={c.expectancy_reward_weight:.4f} entry_pen={c.low_expectancy_entry_penalty:.4f} "
            f"close_scale={c.low_expectancy_close_penalty_scale:.4f}",
            flush=True,
        )

    if not results:
        raise SystemExit("No sweep results generated.")

    ranked = sorted(results, key=lambda x: float(x["selection_score"]), reverse=True)
    best = ranked[0]

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = reports_dir / f"{args.out_prefix}_summary_{ts}.json"
    out_md = reports_dir / f"{args.out_prefix}_summary_{ts}.md"
    out_env = reports_dir / f"{args.out_prefix}_best_profile_{ts}.env"

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "profile": args.profile,
        "timesteps": int(args.timesteps),
        "candidate_count": len(results),
        "sweep_space": {
            "trend_values": trend_values,
            "atr_bands": [{"atr_min": mn, "atr_max": mx} for mn, mx in atr_bands],
            "expectancy_values": expectancy_values,
            "expectancy_reward_values": expectancy_reward_values,
            "low_entry_penalty_values": low_entry_penalty_values,
            "low_close_penalty_scale_values": low_close_penalty_scale_values,
        },
        "policy": {
            "train_block_low_expectancy": int(args.train_block_low_expectancy),
            "eval_block_low_expectancy": int(args.eval_block_low_expectancy),
            "eval_block_in_eval": int(args.eval_block_in_eval),
        },
        "best_candidate": best,
        "candidates_ranked": ranked,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md.write_text(to_markdown(payload), encoding="utf-8")
    write_best_env(
        best,
        out_env,
        eval_block_low_expectancy=int(args.eval_block_low_expectancy),
        eval_block_in_eval=int(args.eval_block_in_eval),
    )

    if not args.keep_models:
        for c in results:
            model_path = Path(c["model_path"])
            metrics_path = Path(c["metrics_path"])
            try:
                if model_path.exists():
                    model_path.unlink()
            except Exception:
                pass
            try:
                if metrics_path.exists():
                    metrics_path.unlink()
            except Exception:
                pass

    print(f"sweep_json={out_json}")
    print(f"sweep_md={out_md}")
    print(f"sweep_best_env={out_env}")
    print(
        "best "
        f"candidate={best['candidate_id']} score={best['selection_score']:.3f} "
        f"trend={best['trend_min']:.8f} atr_min={best['atr_min']:.8f} "
        f"atr_max={best['atr_max']:.8f} exp={best['expectancy_threshold']:.8f} "
        f"entry_w={best['expectancy_reward_weight']:.4f} "
        f"entry_pen={best['low_expectancy_entry_penalty']:.4f} "
        f"close_scale={best['low_expectancy_close_penalty_scale']:.4f} "
        f"decision={best['walk_forward_summary']['decision']} "
        f"net={best['walk_forward_summary']['overall_net_profit']:.2f} "
        f"pf={best['walk_forward_summary']['overall_profit_factor']:.3f} "
        f"dd={best['walk_forward_summary']['worst_fold_drawdown_pct']:.2f}"
    )


if __name__ == "__main__":
    main()
