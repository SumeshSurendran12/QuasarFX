from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CellResult:
    horizon: int
    move_threshold_pips: int
    folds_used: int
    samples_scored: int
    events_true: int
    events_pred: int
    event_base_rate: float
    event_precision: float
    event_recall: float
    auc: float
    baseline_mean_abs_move_bps: float
    baseline_est_round_trip_cost_bps: float
    baseline_net_bps: float
    mean_abs_move_bps_when_predicted: float
    est_round_trip_cost_bps: float
    model_net_bps: float
    delta_net_vs_baseline_bps: float
    net_abs_move_minus_cost_bps: float
    folds_with_delta: int
    positive_fold_rate: float
    mean_delta_per_fold: float
    std_delta_per_fold: float
    worst_fold_delta: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Volatility strategy diagnostic: predict whether |return| > X pips over next H bars."
    )
    parser.add_argument("--data-csv", default="", help="Optional CSV path. Defaults to latest in data/*.csv")
    parser.add_argument("--horizons", default="1,3,5,10,12,24,48,72")
    parser.add_argument("--move-threshold-pips", default="2,4,6,8,10")
    parser.add_argument("--event-prob-th", type=float, default=0.62)
    parser.add_argument("--min-train-frac", type=float, default=0.60)
    parser.add_argument("--test-frac", type=float, default=0.04)
    parser.add_argument("--max-folds", type=int, default=10)
    parser.add_argument("--use-structural-features", action="store_true", default=True)
    parser.add_argument("--no-structural-features", dest="use_structural_features", action="store_false")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--num-leaves", type=int, default=64)
    parser.add_argument("--min-child-samples", type=int, default=200)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--subsample-freq", type=int, default=5)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-prefix", default="volatility_strategy_diagnostic")
    return parser.parse_args()


def parse_int_list(raw: str) -> List[int]:
    vals: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    vals = sorted(set(v for v in vals if v > 0))
    if not vals:
        raise ValueError("No valid integers parsed from list.")
    return vals


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
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).set_index("time")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
    else:
        raise ValueError(f"CSV missing time/date column: {csv_path}")

    needed = ["open", "high", "low", "close", "volume"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=needed)
    return df[needed].sort_index().copy()


def make_folds(df: pd.DataFrame, min_train_frac: float, test_frac: float, max_folds: int) -> List[Tuple[int, int, int, int]]:
    n = len(df)
    if n < 2000:
        raise ValueError("Dataset too small for walk-forward evaluation.")
    min_train = int(n * min_train_frac)
    test_size = max(int(n * test_frac), 1000)
    if min_train + test_size > n:
        raise ValueError("Not enough rows for fold settings.")

    folds: List[Tuple[int, int, int, int]] = []
    fold_id = 1
    test_start = min_train
    while test_start + test_size <= n and fold_id <= max_folds:
        folds.append((fold_id, 0, test_start, test_start + test_size))
        fold_id += 1
        test_start += test_size
    return folds


def infer_bars_per_day(index: pd.DatetimeIndex) -> int:
    if len(index) < 4:
        return 24
    diffs = index.to_series().diff().dropna().dt.total_seconds().astype(float)
    if diffs.empty:
        return 24
    median = float(np.nanmedian(diffs.values))
    if not np.isfinite(median) or median <= 0:
        return 24
    bars = int(round(86400.0 / median))
    return int(np.clip(bars, 1, 1440))


def build_features(df: pd.DataFrame, use_structural: bool) -> pd.DataFrame:
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    close_safe = close.replace(0.0, np.nan)
    open_safe = open_.replace(0.0, np.nan)
    vol_mean = volume.rolling(24, min_periods=1).mean()
    vol_std = volume.rolling(24, min_periods=1).std().replace(0.0, np.nan)

    feat = pd.DataFrame(index=df.index)
    feat["ret_1"] = close.pct_change(1)
    feat["ret_3"] = close.pct_change(3)
    feat["ret_6"] = close.pct_change(6)
    feat["ret_12"] = close.pct_change(12)
    feat["ret_24"] = close.pct_change(24)
    feat["hl_spread"] = (high - low) / close_safe
    feat["oc_delta"] = (close - open_) / open_safe
    feat["ma_fast_slow"] = (close.rolling(12, min_periods=1).mean() - close.rolling(48, min_periods=1).mean()) / close_safe
    feat["atr_norm"] = (high - low).rolling(14, min_periods=1).mean() / close_safe
    feat["vol_z"] = (volume - vol_mean) / vol_std

    if use_structural:
        bars_per_day = infer_bars_per_day(df.index)
        week = max(5 * bars_per_day, 10)
        month = max(21 * bars_per_day, 30)
        qtr = max(63 * bars_per_day, 90)
        y52w = max(252 * bars_per_day, 250)

        ret1 = close.pct_change(1)
        feat["weekly_return"] = close.pct_change(week)
        feat["monthly_return"] = close.pct_change(month)
        feat["rolling_3m_vol"] = ret1.rolling(qtr, min_periods=10).std()
        range_short = high.rolling(week, min_periods=10).max() - low.rolling(week, min_periods=10).min()
        range_long = high.rolling(qtr, min_periods=20).max() - low.rolling(qtr, min_periods=20).min()
        feat["range_compression"] = range_short / range_long.replace(0.0, np.nan)
        high_52w = high.rolling(y52w, min_periods=50).max()
        low_52w = low.rolling(y52w, min_periods=50).min()
        feat["dist_52w_high"] = (close / high_52w) - 1.0
        feat["dist_52w_low"] = (close / low_52w) - 1.0

    feat = feat.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return feat.astype(np.float32)


def make_vol_labels(close: pd.Series, horizon: int, threshold_pips: int) -> Tuple[pd.Series, pd.Series]:
    fwd_close = close.shift(-horizon)
    fwd_ret = (fwd_close - close) / close
    abs_move_pips = np.abs(fwd_close - close) / 0.0001
    y = (abs_move_pips > float(threshold_pips)).astype(int)
    return y, fwd_ret


def scan_cell(
    df: pd.DataFrame,
    features: pd.DataFrame,
    folds: List[Tuple[int, int, int, int]],
    horizon: int,
    threshold_pips: int,
    event_prob_th: float,
    model_params: Dict[str, float],
) -> CellResult:
    close = df["close"].astype(float)
    y, fwd_ret = make_vol_labels(close, horizon=horizon, threshold_pips=threshold_pips)

    total_samples = 0
    total_events_true = 0
    total_events_pred = 0
    total_tp = 0
    total_abs_move_bps_all = 0.0
    total_cost_bps_all = 0.0
    sum_abs_move_bps = 0.0
    sum_cost_bps = 0.0
    auc_vals: List[float] = []
    fold_deltas: List[float] = []
    folds_used = 0

    for _, train_start, test_start, test_end in folds:
        train_end = max(train_start, test_start - horizon)
        test_last = test_end - horizon
        train_idx = np.arange(train_start, train_end, dtype=int)
        test_idx = np.arange(test_start, max(test_start, test_last), dtype=int)
        if train_idx.size < 500 or test_idx.size < 50:
            continue

        X_train = features.iloc[train_idx]
        y_train = y.iloc[train_idx].to_numpy(dtype=int)
        if np.unique(y_train).size < 2:
            continue

        model = LGBMClassifier(**model_params)
        model.fit(X_train, y_train)

        X_test = features.iloc[test_idx]
        y_test = y.iloc[test_idx].to_numpy(dtype=int)
        p_event = model.predict_proba(X_test)[:, 1]
        pred_event = (p_event >= float(event_prob_th)).astype(np.int8)

        total_samples += int(test_idx.size)
        total_events_true += int(np.sum(y_test == 1))
        total_events_pred += int(np.sum(pred_event == 1))
        total_tp += int(np.sum((pred_event == 1) & (y_test == 1)))
        folds_used += 1

        if np.unique(y_test).size == 2:
            auc_vals.append(float(roc_auc_score(y_test, p_event)))

        abs_move_bps_all = np.abs(fwd_ret.iloc[test_idx].to_numpy(dtype=float)) * 10000.0
        price_all = close.iloc[test_idx].to_numpy(dtype=float)
        rt_bps_all = 2.0 * (((0.0002 + 0.00005) / np.clip(price_all, 1e-9, None)) + 0.0001) * 10000.0
        fold_baseline_net = float(np.mean(abs_move_bps_all - rt_bps_all))
        total_abs_move_bps_all += float(np.sum(abs_move_bps_all))
        total_cost_bps_all += float(np.sum(rt_bps_all))

        pred_idx = test_idx[pred_event == 1]
        if pred_idx.size > 0:
            abs_move_bps = np.abs(fwd_ret.iloc[pred_idx].to_numpy(dtype=float)) * 10000.0
            sum_abs_move_bps += float(np.sum(abs_move_bps))
            price = close.iloc[pred_idx].to_numpy(dtype=float)
            rt_bps = 2.0 * (((0.0002 + 0.00005) / np.clip(price, 1e-9, None)) + 0.0001) * 10000.0
            sum_cost_bps += float(np.sum(rt_bps))
            fold_model_net = float(np.mean(abs_move_bps - rt_bps))
            fold_deltas.append(float(fold_model_net - fold_baseline_net))

    if total_samples == 0:
        return CellResult(
            horizon=int(horizon),
            move_threshold_pips=int(threshold_pips),
            folds_used=0,
            samples_scored=0,
            events_true=0,
            events_pred=0,
            event_base_rate=float("nan"),
            event_precision=float("nan"),
            event_recall=float("nan"),
            auc=float("nan"),
            baseline_mean_abs_move_bps=float("nan"),
            baseline_est_round_trip_cost_bps=float("nan"),
            baseline_net_bps=float("nan"),
            mean_abs_move_bps_when_predicted=float("nan"),
            est_round_trip_cost_bps=float("nan"),
            model_net_bps=float("nan"),
            delta_net_vs_baseline_bps=float("nan"),
            net_abs_move_minus_cost_bps=float("nan"),
            folds_with_delta=0,
            positive_fold_rate=float("nan"),
            mean_delta_per_fold=float("nan"),
            std_delta_per_fold=float("nan"),
            worst_fold_delta=float("nan"),
        )

    base_rate = float(total_events_true / total_samples)
    precision = float(total_tp / total_events_pred) if total_events_pred > 0 else float("nan")
    recall = float(total_tp / total_events_true) if total_events_true > 0 else float("nan")
    auc = float(np.mean(auc_vals)) if auc_vals else float("nan")
    baseline_mean_abs = float(total_abs_move_bps_all / total_samples) if total_samples > 0 else float("nan")
    baseline_mean_cost = float(total_cost_bps_all / total_samples) if total_samples > 0 else float("nan")
    baseline_net = float(baseline_mean_abs - baseline_mean_cost) if total_samples > 0 else float("nan")
    mean_abs = float(sum_abs_move_bps / total_events_pred) if total_events_pred > 0 else float("nan")
    mean_cost = float(sum_cost_bps / total_events_pred) if total_events_pred > 0 else float("nan")
    model_net = float(mean_abs - mean_cost) if total_events_pred > 0 else float("nan")
    delta_net = float(model_net - baseline_net) if (model_net == model_net and baseline_net == baseline_net) else float("nan")
    folds_with_delta = int(len(fold_deltas))
    positive_fold_rate = float(np.mean(np.asarray(fold_deltas) > 0.0)) if fold_deltas else float("nan")
    mean_delta_per_fold = float(np.mean(fold_deltas)) if fold_deltas else float("nan")
    std_delta_per_fold = float(np.std(fold_deltas)) if fold_deltas else float("nan")
    worst_fold_delta = float(np.min(fold_deltas)) if fold_deltas else float("nan")

    return CellResult(
        horizon=int(horizon),
        move_threshold_pips=int(threshold_pips),
        folds_used=int(folds_used),
        samples_scored=int(total_samples),
        events_true=int(total_events_true),
        events_pred=int(total_events_pred),
        event_base_rate=base_rate,
        event_precision=precision,
        event_recall=recall,
        auc=auc,
        baseline_mean_abs_move_bps=baseline_mean_abs,
        baseline_est_round_trip_cost_bps=baseline_mean_cost,
        baseline_net_bps=baseline_net,
        mean_abs_move_bps_when_predicted=mean_abs,
        est_round_trip_cost_bps=mean_cost,
        model_net_bps=model_net,
        delta_net_vs_baseline_bps=delta_net,
        net_abs_move_minus_cost_bps=model_net,
        folds_with_delta=folds_with_delta,
        positive_fold_rate=positive_fold_rate,
        mean_delta_per_fold=mean_delta_per_fold,
        std_delta_per_fold=std_delta_per_fold,
        worst_fold_delta=worst_fold_delta,
    )


def to_markdown(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Volatility Strategy Diagnostic")
    lines.append("")
    lines.append(f"- Generated: {payload['generated_at']}")
    lines.append(f"- Data: `{payload['data_csv']}`")
    lines.append(f"- Event probability threshold: {payload['config']['event_prob_th']}")
    lines.append("")
    lines.append("| H | X pips | Prev | Trades(pred) | Precision | Recall | Baseline Net (bps) | Model Net (bps) | Delta (bps) | PosFoldRate | WorstFoldDelta | StdFoldDelta |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload["results"]:
        lines.append(
            f"| {row['horizon']} | {row['move_threshold_pips']} | "
            f"{row['event_base_rate'] if row['event_base_rate'] == row['event_base_rate'] else float('nan'):.4f} | "
            f"{row['events_pred']} | "
            f"{row['event_precision'] if row['event_precision'] == row['event_precision'] else float('nan'):.4f} | "
            f"{row['event_recall'] if row['event_recall'] == row['event_recall'] else float('nan'):.4f} | "
            f"{row['baseline_net_bps'] if row['baseline_net_bps'] == row['baseline_net_bps'] else float('nan'):.3f} | "
            f"{row['model_net_bps'] if row['model_net_bps'] == row['model_net_bps'] else float('nan'):.3f} | "
            f"{row['delta_net_vs_baseline_bps'] if row['delta_net_vs_baseline_bps'] == row['delta_net_vs_baseline_bps'] else float('nan'):.3f} | "
            f"{row['positive_fold_rate'] if row['positive_fold_rate'] == row['positive_fold_rate'] else float('nan'):.3f} | "
            f"{row['worst_fold_delta'] if row['worst_fold_delta'] == row['worst_fold_delta'] else float('nan'):.3f} | "
            f"{row['std_delta_per_fold'] if row['std_delta_per_fold'] == row['std_delta_per_fold'] else float('nan'):.3f} |"
        )
    lines.append("")
    lines.append(f"- Positive cells (net edge > 0): {payload['summary']['positive_cells']}/{payload['summary']['total_cells']}")
    lines.append(f"- Recommendation: {payload['summary']['recommendation']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    horizons = parse_int_list(args.horizons)
    move_thresholds = parse_int_list(args.move_threshold_pips)

    data_csv = find_data_csv(args.data_csv)
    df = load_ohlcv(data_csv)
    folds = make_folds(df, min_train_frac=float(args.min_train_frac), test_frac=float(args.test_frac), max_folds=int(args.max_folds))
    features = build_features(df, use_structural=bool(args.use_structural_features))

    model_params: Dict[str, float] = {
        "n_estimators": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
        "max_depth": int(args.max_depth),
        "num_leaves": int(args.num_leaves),
        "min_child_samples": int(args.min_child_samples),
        "subsample": float(args.subsample),
        "subsample_freq": int(args.subsample_freq),
        "colsample_bytree": float(args.colsample_bytree),
        "reg_lambda": float(args.reg_lambda),
        "n_jobs": -1,
        "random_state": int(args.random_state),
        "class_weight": "balanced",
        "verbose": -1,
    }

    results: List[CellResult] = []
    for h in horizons:
        for x in move_thresholds:
            row = scan_cell(
                df=df,
                features=features,
                folds=folds,
                horizon=int(h),
                threshold_pips=int(x),
                event_prob_th=float(args.event_prob_th),
                model_params=model_params,
            )
            results.append(row)
            print(
                f"cell h={h} x={x} prev={row.event_base_rate if row.event_base_rate == row.event_base_rate else float('nan'):.4f} "
                f"pred={row.events_pred} precision={row.event_precision if row.event_precision == row.event_precision else float('nan'):.4f} "
                f"recall={row.event_recall if row.event_recall == row.event_recall else float('nan'):.4f} "
                f"baseline_net_bps={row.baseline_net_bps if row.baseline_net_bps == row.baseline_net_bps else float('nan'):.3f} "
                f"model_net_bps={row.model_net_bps if row.model_net_bps == row.model_net_bps else float('nan'):.3f} "
                f"delta_bps={row.delta_net_vs_baseline_bps if row.delta_net_vs_baseline_bps == row.delta_net_vs_baseline_bps else float('nan'):.3f} "
                f"pos_fold_rate={row.positive_fold_rate if row.positive_fold_rate == row.positive_fold_rate else float('nan'):.3f} "
                f"worst_fold_delta={row.worst_fold_delta if row.worst_fold_delta == row.worst_fold_delta else float('nan'):.3f} "
                f"std_fold_delta={row.std_delta_per_fold if row.std_delta_per_fold == row.std_delta_per_fold else float('nan'):.3f}"
            )

    total_cells = len(results)
    positive_cells = sum(1 for r in results if r.net_abs_move_minus_cost_bps == r.net_abs_move_minus_cost_bps and r.net_abs_move_minus_cost_bps > 0.0)
    recommendation = (
        "Some volatility cells are positive after cost; refine around those H/X zones."
        if positive_cells > 0
        else "No positive volatility cells after cost; pivot strategy type or instrument."
    )

    payload: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_csv": str(data_csv),
        "config": {
            "horizons": horizons,
            "move_threshold_pips": move_thresholds,
            "event_prob_th": float(args.event_prob_th),
            "min_train_frac": float(args.min_train_frac),
            "test_frac": float(args.test_frac),
            "max_folds": int(args.max_folds),
            "use_structural_features": bool(args.use_structural_features),
            "model_params": model_params,
        },
        "results": [asdict(r) for r in results],
        "summary": {
            "total_cells": int(total_cells),
            "positive_cells": int(positive_cells),
            "recommendation": recommendation,
        },
    }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"{args.out_prefix}_{ts}.json"
    md_path = reports_dir / f"{args.out_prefix}_{ts}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(payload), encoding="utf-8")

    print(f"report_json={json_path}")
    print(f"report_md={md_path}")
    print(f"summary positive_cells={positive_cells}/{total_cells} recommendation={recommendation}")


if __name__ == "__main__":
    main()
