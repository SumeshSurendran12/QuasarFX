from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.model import TradingEnvironment
from modules.config import EVAL_ACTION_SHAPING


DEFAULT_MIN_TRAIN_FRAC = 0.60
DEFAULT_TEST_FRAC = 0.04
DEFAULT_MAX_FOLDS = 10
DEFAULT_MIN_TRADES_PER_FOLD = 1


@dataclass
class FoldMetrics:
    fold_id: int
    train_rows: int
    test_rows: int
    test_start: str
    test_end: str
    total_reward: float
    total_trades: int
    win_rate: float
    profit_factor: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    avg_trade_pnl: float
    avg_bars_in_trade: float
    avg_hours_in_trade: float
    return_pct: float
    max_drawdown_pct: float
    pass_flag: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward evaluation for saved PPO Forex models.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["models/best_model.zip", "models/final_model.zip"],
        help="Model paths relative to repo root or absolute paths.",
    )
    parser.add_argument(
        "--data-csv",
        default="",
        help="Optional data CSV path. If omitted, latest file in data/*.csv is used.",
    )
    parser.add_argument("--min-train-frac", type=float, default=DEFAULT_MIN_TRAIN_FRAC)
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    parser.add_argument("--max-folds", type=int, default=DEFAULT_MAX_FOLDS)
    parser.add_argument(
        "--min-trades-per-fold",
        type=int,
        default=DEFAULT_MIN_TRADES_PER_FOLD,
        help="Minimum number of trades required per fold; folds below this fail immediately.",
    )
    parser.add_argument(
        "--out-prefix",
        default="walk_forward_eval",
        help="Prefix for output files in reports/ directory.",
    )
    return parser.parse_args()


def find_data_csv(user_value: str) -> Path:
    if user_value:
        path = Path(user_value)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Data CSV not found: {path}")
        return path

    data_dir = ROOT / "data"
    csvs = sorted(data_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return csvs[0]


def load_ohlcv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        raise ValueError(f"CSV missing time/date column: {csv_path}")

    needed = ["open", "high", "low", "close", "volume"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df[needed].copy()


def make_folds(
    df: pd.DataFrame,
    min_train_frac: float,
    test_frac: float,
    max_folds: int,
) -> List[Tuple[int, int, int, int]]:
    n = len(df)
    if n < 2000:
        raise ValueError("Dataset too small for strict walk-forward evaluation.")

    min_train = int(n * min_train_frac)
    test_size = max(int(n * test_frac), 1000)
    if min_train + test_size > n:
        raise ValueError("Not enough rows for the requested min_train_frac/test_frac.")

    folds: List[Tuple[int, int, int, int]] = []
    test_start = min_train
    fold_id = 1
    while test_start + test_size <= n and fold_id <= max_folds:
        test_end = test_start + test_size
        folds.append((fold_id, 0, test_start, test_end))
        test_start = test_end
        fold_id += 1
    return folds


def max_drawdown_pct(equity_curve: List[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        if peak > 0:
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
    return max_dd * 100.0


def eval_on_slice(model: PPO, eval_df: pd.DataFrame) -> Dict[str, float]:
    env = TradingEnvironment(
        eval_df.copy(),
        mode="eval",
        apply_action_shaping=EVAL_ACTION_SHAPING,
    )
    obs, _ = env.reset()
    data_len = len(env.data["close"])

    total_reward = 0.0
    episode_reward = 0.0
    episode_rewards: List[float] = []
    episode_count = 0
    equity_curve = [env.balance]

    while env.current_step + 1 < data_len:
        action, _ = model.predict(obs, deterministic=True)
        action_arr = np.asarray(action)
        if action_arr.size == 1:
            action_value = int(action_arr.item())
        else:
            action_value = int(action_arr.reshape(-1)[0])
        obs, reward, done, _, _ = env.step(action_value)
        total_reward += float(reward)
        episode_reward += float(reward)
        equity_curve.append(float(env.balance))
        if done:
            episode_count += 1
            episode_rewards.append(episode_reward)
            episode_reward = 0.0

    if episode_reward != 0.0:
        episode_rewards.append(episode_reward)

    trade_history = env.trade_history
    total_trades = len(trade_history)
    profits = [float(t["reward"]) for t in trade_history if float(t.get("reward", 0.0)) > 0]
    losses = [abs(float(t["reward"])) for t in trade_history if float(t.get("reward", 0.0)) < 0]
    holding_bars: List[float] = []
    for t in trade_history:
        if float(t.get("reward", 0.0)) == 0.0:
            continue
        entry_step = t.get("entry_step")
        close_step = t.get("close_step")
        if entry_step is None or close_step is None:
            continue
        try:
            span = int(close_step) - int(entry_step)
            if span >= 0:
                holding_bars.append(float(span))
        except Exception:
            continue
    gross_profit = float(sum(profits))
    gross_loss = float(sum(losses))
    net_profit = gross_profit - gross_loss
    win_rate = (len(profits) / total_trades) if total_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    avg_trade_pnl = (net_profit / total_trades) if total_trades > 0 else 0.0
    avg_bars_in_trade = float(np.mean(holding_bars)) if holding_bars else 0.0
    bar_hours = 24.0 / max(float(getattr(env, "bars_per_day", 24)), 1.0)
    avg_hours_in_trade = avg_bars_in_trade * bar_hours
    return_pct = ((equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100.0) if equity_curve else 0.0
    max_dd = max_drawdown_pct(equity_curve)

    return {
        "episodes": float(episode_count),
        "total_reward": float(total_reward),
        "avg_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "total_trades": float(total_trades),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "net_profit": float(net_profit),
        "avg_trade_pnl": float(avg_trade_pnl),
        "avg_bars_in_trade": float(avg_bars_in_trade),
        "avg_hours_in_trade": float(avg_hours_in_trade),
        "return_pct": float(return_pct),
        "max_drawdown_pct": float(max_dd),
    }


def fold_pass(metrics: Dict[str, float], min_trades_per_fold: int) -> bool:
    return (
        metrics["total_trades"] >= float(min_trades_per_fold)
        and metrics["net_profit"] > 0.0
        and metrics["profit_factor"] >= 1.0
        and metrics["max_drawdown_pct"] <= 40.0
    )


def model_pass(summary: Dict[str, float], min_trades_per_fold: int) -> bool:
    return (
        summary["overall_net_profit"] > 0.0
        and summary["overall_profit_factor"] >= 1.05
        and summary["pass_rate"] >= 0.60
        and summary["worst_fold_drawdown_pct"] <= 35.0
        and summary["min_fold_trades"] >= int(min_trades_per_fold)
    )


def evaluate_model_walk_forward(
    model_path: Path,
    df: pd.DataFrame,
    folds: List[Tuple[int, int, int, int]],
    min_trades_per_fold: int,
) -> Dict[str, object]:
    model = PPO.load(str(model_path))
    results: List[FoldMetrics] = []

    gross_profit_total = 0.0
    gross_loss_total = 0.0
    net_profit_total = 0.0
    returns: List[float] = []
    rewards: List[float] = []
    drawdowns: List[float] = []
    trade_counts: List[int] = []
    weighted_holding_bars_total = 0.0
    weighted_holding_hours_total = 0.0

    for fold_id, train_start, test_start, test_end in folds:
        test_df = df.iloc[test_start:test_end].copy()
        metrics = eval_on_slice(model, test_df)
        is_pass = fold_pass(metrics, min_trades_per_fold=min_trades_per_fold)
        fold_result = FoldMetrics(
            fold_id=fold_id,
            train_rows=test_start - train_start,
            test_rows=test_end - test_start,
            test_start=str(test_df.index.min()),
            test_end=str(test_df.index.max()),
            total_reward=metrics["total_reward"],
            total_trades=int(metrics["total_trades"]),
            win_rate=metrics["win_rate"],
            profit_factor=metrics["profit_factor"],
            gross_profit=metrics["gross_profit"],
            gross_loss=metrics["gross_loss"],
            net_profit=metrics["net_profit"],
            avg_trade_pnl=metrics["avg_trade_pnl"],
            avg_bars_in_trade=metrics["avg_bars_in_trade"],
            avg_hours_in_trade=metrics["avg_hours_in_trade"],
            return_pct=metrics["return_pct"],
            max_drawdown_pct=metrics["max_drawdown_pct"],
            pass_flag=is_pass,
        )
        results.append(fold_result)

        gross_profit_total += metrics["gross_profit"]
        gross_loss_total += metrics["gross_loss"]
        net_profit_total += metrics["net_profit"]
        returns.append(metrics["return_pct"])
        rewards.append(metrics["total_reward"])
        drawdowns.append(metrics["max_drawdown_pct"])
        trade_counts.append(int(metrics["total_trades"]))
        weighted_holding_bars_total += float(metrics["avg_bars_in_trade"]) * float(metrics["total_trades"])
        weighted_holding_hours_total += float(metrics["avg_hours_in_trade"]) * float(metrics["total_trades"])

    pass_count = sum(1 for r in results if r.pass_flag)
    pass_rate = (pass_count / len(results)) if results else 0.0
    overall_profit_factor = (
        gross_profit_total / gross_loss_total
        if gross_loss_total > 0
        else (float("inf") if gross_profit_total > 0 else 0.0)
    )

    summary = {
        "model_path": str(model_path),
        "folds": len(results),
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "total_trades": int(sum(trade_counts)),
        "min_fold_trades": int(min(trade_counts)) if trade_counts else 0,
        "mean_fold_trades": float(np.mean(trade_counts)) if trade_counts else 0.0,
        "overall_net_profit": net_profit_total,
        "overall_profit_factor": overall_profit_factor,
        "mean_fold_return_pct": float(np.mean(returns)) if returns else 0.0,
        "median_fold_return_pct": float(np.median(returns)) if returns else 0.0,
        "mean_fold_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_trade_holding_bars": (
            float(weighted_holding_bars_total / max(sum(trade_counts), 1))
            if trade_counts
            else 0.0
        ),
        "mean_trade_holding_hours": (
            float(weighted_holding_hours_total / max(sum(trade_counts), 1))
            if trade_counts
            else 0.0
        ),
        "worst_fold_drawdown_pct": float(max(drawdowns)) if drawdowns else 0.0,
        "min_trades_per_fold_required": int(min_trades_per_fold),
    }
    summary["decision"] = "PASS" if model_pass(summary, min_trades_per_fold=min_trades_per_fold) else "FAIL"
    summary["retrain_recommended"] = summary["decision"] == "FAIL"
    summary["collapse_detected"] = summary["min_fold_trades"] < int(min_trades_per_fold)

    return {
        "summary": summary,
        "folds": [asdict(r) for r in results],
    }


def to_markdown(
    data_csv: Path,
    config: Dict[str, float],
    model_reports: List[Dict[str, object]],
) -> str:
    lines: List[str] = []
    lines.append("# Walk-Forward Evaluation Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Data: `{data_csv}`")
    lines.append(
        f"- Config: min_train_frac={config['min_train_frac']}, test_frac={config['test_frac']}, max_folds={config['max_folds']}, min_trades_per_fold={config['min_trades_per_fold']}, eval_action_shaping={config['eval_action_shaping']}"
    )
    lines.append("")

    for report in model_reports:
        summary = report["summary"]
        folds = report["folds"]
        lines.append(f"## Model: `{summary['model_path']}`")
        lines.append(f"- Decision: **{summary['decision']}**")
        lines.append(f"- Retrain recommended: **{summary['retrain_recommended']}**")
        lines.append(f"- Fold pass rate: {summary['pass_count']}/{summary['folds']} ({summary['pass_rate']*100:.1f}%)")
        lines.append(f"- Total trades: {summary['total_trades']}")
        lines.append(f"- Min trades/fold: {summary['min_fold_trades']} (required: {summary['min_trades_per_fold_required']})")
        lines.append(f"- Collapse detected: **{summary['collapse_detected']}**")
        lines.append(f"- Overall net profit: {summary['overall_net_profit']:.2f}")
        lines.append(f"- Overall profit factor: {summary['overall_profit_factor']:.3f}")
        lines.append(f"- Mean fold return %: {summary['mean_fold_return_pct']:.2f}")
        lines.append(
            f"- Avg holding time: {summary['mean_trade_holding_bars']:.2f} bars ({summary['mean_trade_holding_hours']:.2f} hours)"
        )
        lines.append(f"- Worst fold drawdown %: {summary['worst_fold_drawdown_pct']:.2f}")
        lines.append("")
        lines.append("| Fold | Period Start | Period End | Trades | Net Profit | PF | Win Rate | Avg Bars | Avg Hours | Max DD % | Pass |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |")
        for f in folds:
            lines.append(
                "| "
                f"{f['fold_id']} | {f['test_start']} | {f['test_end']} | "
                f"{f['total_trades']} | {f['net_profit']:.2f} | {f['profit_factor']:.3f} | {f['win_rate']*100:.1f}% | "
                f"{f['avg_bars_in_trade']:.2f} | {f['avg_hours_in_trade']:.2f} | "
                f"{f['max_drawdown_pct']:.2f} | {'Y' if f['pass_flag'] else 'N'} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    data_csv = find_data_csv(args.data_csv)
    df = load_ohlcv(data_csv)
    folds = make_folds(
        df=df,
        min_train_frac=args.min_train_frac,
        test_frac=args.test_frac,
        max_folds=args.max_folds,
    )
    if not folds:
        raise SystemExit("No folds generated.")

    reports: List[Dict[str, object]] = []
    min_trades_per_fold = max(int(args.min_trades_per_fold), 0)
    for model_value in args.models:
        model_path = Path(model_value)
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        if not model_path.exists():
            print(f"skip_missing_model={model_path}")
            continue
        print(f"evaluating_model={model_path}")
        try:
            reports.append(
                evaluate_model_walk_forward(
                    model_path,
                    df,
                    folds,
                    min_trades_per_fold=min_trades_per_fold,
                )
            )
        except Exception as exc:
            print(f"skip_model_error={model_path} error={exc}")
            continue

    if not reports:
        raise SystemExit("No valid model paths found.")

    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{args.out_prefix}_{ts}.json"
    md_path = out_dir / f"{args.out_prefix}_{ts}.md"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "data_csv": str(data_csv),
        "config": {
            "min_train_frac": args.min_train_frac,
            "test_frac": args.test_frac,
            "max_folds": args.max_folds,
            "min_trades_per_fold": min_trades_per_fold,
            "eval_action_shaping": bool(EVAL_ACTION_SHAPING),
            "folds_generated": len(folds),
        },
        "reports": reports,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(data_csv, payload["config"], reports), encoding="utf-8")

    print(f"report_json={json_path}")
    print(f"report_md={md_path}")
    for r in reports:
        s = r["summary"]
        print(
            "summary "
            f"model={s['model_path']} decision={s['decision']} "
            f"pass_rate={s['pass_rate']*100:.1f}% net_profit={s['overall_net_profit']:.2f} "
            f"pf={s['overall_profit_factor']:.3f} worst_dd={s['worst_fold_drawdown_pct']:.2f} "
            f"min_fold_trades={s['min_fold_trades']} collapse={s['collapse_detected']}"
        )


if __name__ == "__main__":
    main()
