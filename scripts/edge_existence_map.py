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


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CellResult:
    horizon: int
    label_threshold_pips: int
    folds_used: int
    total_trades: int
    trade_precision: float
    mean_signed_forward_return_bps: float
    est_round_trip_cost_bps: float
    net_edge_bps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Horizon x label-threshold edge existence scan for directional LightGBM."
    )
    parser.add_argument("--data-csv", default="", help="Optional CSV path. Defaults to latest in data/*.csv")
    parser.add_argument("--horizons", default="1,3,5,10,12,24,48,72")
    parser.add_argument("--label-threshold-pips", default="0,2,4,6,8,10")
    parser.add_argument("--long-th", type=float, default=0.62)
    parser.add_argument("--short-th", type=float, default=0.38)
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
    parser.add_argument("--out-prefix", default="edge_existence_map")
    return parser.parse_args()


def parse_int_list(raw: str) -> List[int]:
    vals: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
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


def make_labels(close: pd.Series, horizon: int, threshold_pips: int) -> Tuple[pd.Series, pd.Series]:
    fwd_close = close.shift(-horizon)
    fwd_ret = (fwd_close - close) / close
    fwd_pips = (fwd_close - close) / 0.0001
    y = pd.Series(np.nan, index=close.index, dtype="float64")
    y[fwd_pips > float(threshold_pips)] = 1.0
    y[fwd_pips < -float(threshold_pips)] = 0.0
    return y, fwd_ret


def scan_cell(
    df: pd.DataFrame,
    features: pd.DataFrame,
    folds: List[Tuple[int, int, int, int]],
    horizon: int,
    threshold_pips: int,
    long_th: float,
    short_th: float,
    model_params: Dict[str, float],
) -> CellResult:
    close = df["close"].astype(float)
    y, fwd_ret = make_labels(close, horizon=horizon, threshold_pips=threshold_pips)

    total_trades = 0
    total_correct = 0
    sum_signed_bps = 0.0
    sum_cost_bps = 0.0
    folds_used = 0

    for _, train_start, test_start, test_end in folds:
        train_end = max(train_start, test_start - horizon)
        test_last = test_end - horizon
        train_idx = np.arange(train_start, train_end, dtype=int)
        test_idx = np.arange(test_start, max(test_start, test_last), dtype=int)
        if train_idx.size < 500 or test_idx.size < 50:
            continue

        y_train = y.iloc[train_idx]
        mask_train = y_train.notna().to_numpy()
        train_idx_used = train_idx[mask_train]
        if train_idx_used.size < 300:
            continue

        X_train = features.iloc[train_idx_used]
        y_train_used = y.iloc[train_idx_used].astype(int)
        if y_train_used.nunique() < 2:
            continue

        model = LGBMClassifier(**model_params)
        model.fit(X_train, y_train_used)

        X_test = features.iloc[test_idx]
        p_up = model.predict_proba(X_test)[:, 1]
        action = np.zeros_like(p_up, dtype=np.int8)
        action[p_up > float(long_th)] = 1
        action[p_up < float(short_th)] = -1

        traded_mask = action != 0
        if not np.any(traded_mask):
            continue

        trade_idx = test_idx[traded_mask]
        direction = action[traded_mask].astype(float)
        fwd_vals = fwd_ret.iloc[trade_idx].to_numpy(dtype=float)

        signed_bps = direction * fwd_vals * 10000.0
        correct = (direction * np.sign(fwd_vals)) > 0.0

        price = close.iloc[trade_idx].to_numpy(dtype=float)
        # Return-space round-trip cost approximation in bps.
        round_trip_bps = 2.0 * (((0.0002 + 0.00005) / np.clip(price, 1e-9, None)) + 0.0001) * 10000.0

        total_trades += int(trade_idx.size)
        total_correct += int(np.sum(correct))
        sum_signed_bps += float(np.sum(signed_bps))
        sum_cost_bps += float(np.sum(round_trip_bps))
        folds_used += 1

    if total_trades == 0:
        return CellResult(
            horizon=int(horizon),
            label_threshold_pips=int(threshold_pips),
            folds_used=int(folds_used),
            total_trades=0,
            trade_precision=float("nan"),
            mean_signed_forward_return_bps=float("nan"),
            est_round_trip_cost_bps=float("nan"),
            net_edge_bps=float("nan"),
        )

    precision = float(total_correct / total_trades)
    mean_signed = float(sum_signed_bps / total_trades)
    mean_cost = float(sum_cost_bps / total_trades)
    net_edge = float(mean_signed - mean_cost)

    return CellResult(
        horizon=int(horizon),
        label_threshold_pips=int(threshold_pips),
        folds_used=int(folds_used),
        total_trades=int(total_trades),
        trade_precision=precision,
        mean_signed_forward_return_bps=mean_signed,
        est_round_trip_cost_bps=mean_cost,
        net_edge_bps=net_edge,
    )


def to_markdown(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Edge Existence Map")
    lines.append("")
    lines.append(f"- Generated: {payload['generated_at']}")
    lines.append(f"- Data: `{payload['data_csv']}`")
    lines.append(
        f"- Prob thresholds: long={payload['config']['long_th']}, short={payload['config']['short_th']}"
    )
    lines.append("")
    lines.append("| Horizon | Label Pips | Trades | Precision | Mean Signed Fwd (bps) | Est RT Cost (bps) | Net Edge (bps) |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload["results"]:
        lines.append(
            f"| {row['horizon']} | {row['label_threshold_pips']} | {row['total_trades']} | "
            f"{row['trade_precision'] if row['trade_precision'] == row['trade_precision'] else float('nan'):.4f} | "
            f"{row['mean_signed_forward_return_bps'] if row['mean_signed_forward_return_bps'] == row['mean_signed_forward_return_bps'] else float('nan'):.3f} | "
            f"{row['est_round_trip_cost_bps'] if row['est_round_trip_cost_bps'] == row['est_round_trip_cost_bps'] else float('nan'):.3f} | "
            f"{row['net_edge_bps'] if row['net_edge_bps'] == row['net_edge_bps'] else float('nan'):.3f} |"
        )
    lines.append("")
    lines.append(f"- Positive-edge cells (net_edge_bps > 0): {payload['summary']['positive_cells']}/{payload['summary']['total_cells']}")
    lines.append(f"- All cells non-positive: **{payload['summary']['all_cells_non_positive']}**")
    lines.append(f"- Recommendation: {payload['summary']['recommendation']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    horizons = parse_int_list(args.horizons)
    label_pips = parse_int_list(args.label_threshold_pips)

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
        for t in label_pips:
            row = scan_cell(
                df=df,
                features=features,
                folds=folds,
                horizon=int(h),
                threshold_pips=int(t),
                long_th=float(args.long_th),
                short_th=float(args.short_th),
                model_params=model_params,
            )
            results.append(row)
            print(
                f"cell h={h} pips={t} trades={row.total_trades} "
                f"precision={row.trade_precision if row.trade_precision == row.trade_precision else float('nan'):.4f} "
                f"signed_bps={row.mean_signed_forward_return_bps if row.mean_signed_forward_return_bps == row.mean_signed_forward_return_bps else float('nan'):.3f} "
                f"cost_bps={row.est_round_trip_cost_bps if row.est_round_trip_cost_bps == row.est_round_trip_cost_bps else float('nan'):.3f} "
                f"net_bps={row.net_edge_bps if row.net_edge_bps == row.net_edge_bps else float('nan'):.3f}"
            )

    total_cells = len(results)
    positive_cells = sum(1 for r in results if r.net_edge_bps == r.net_edge_bps and r.net_edge_bps > 0.0)
    all_non_positive = positive_cells == 0
    recommendation = (
        "Stop direction prediction and pivot strategy type."
        if all_non_positive
        else "Some directional edge cells exist; refine around positive cells."
    )

    payload: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_csv": str(data_csv),
        "config": {
            "horizons": horizons,
            "label_threshold_pips": label_pips,
            "long_th": float(args.long_th),
            "short_th": float(args.short_th),
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
            "all_cells_non_positive": bool(all_non_positive),
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
    print(
        "summary "
        f"positive_cells={positive_cells}/{total_cells} all_non_positive={all_non_positive} "
        f"recommendation={recommendation}"
    )


if __name__ == "__main__":
    main()
