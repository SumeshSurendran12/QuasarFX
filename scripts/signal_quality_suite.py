from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FoldSpec:
    fold_id: int
    train_start: int
    test_start: int
    test_end: int


@dataclass
class ClassificationFoldResult:
    fold_id: int
    test_start: str
    test_end: str
    samples: int
    accuracy: float
    mean_forward_return: float


@dataclass
class RegressionFoldResult:
    fold_id: int
    test_start: str
    test_end: str
    samples: int
    trades_top10: int
    trades_top10_per_month: float
    hit_rate_top10: float
    mean_trade_return_top10: float
    mean_trade_return_top10_cost_adj: float
    trades_target_freq: int
    trades_target_per_month: float
    hit_rate_target_freq: float
    mean_trade_return_target_freq: float
    mean_trade_return_target_freq_cost_adj: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Signal-quality diagnostic suite: direction, expectancy, frequency, and structural features."
    )
    parser.add_argument("--data-csv", default="", help="CSV path. Defaults to latest in data/*.csv")
    parser.add_argument("--horizons", default="5,10,24,48", help="Comma-separated horizons in bars")
    parser.add_argument("--min-train-frac", type=float, default=0.60)
    parser.add_argument("--test-frac", type=float, default=0.04)
    parser.add_argument("--max-folds", type=int, default=10)
    parser.add_argument("--class-n-estimators", type=int, default=300)
    parser.add_argument("--reg-n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--num-leaves", type=int, default=64)
    parser.add_argument("--min-child-samples", type=int, default=200)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--subsample-freq", type=int, default=5)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-quantile", type=float, default=0.90, help="Top absolute prediction quantile for expectancy trades")
    parser.add_argument("--target-trades-per-month", type=float, default=10.0, help="Frequency target used in Step 3")
    parser.add_argument("--cost-spread", type=float, default=0.0002)
    parser.add_argument("--cost-slippage", type=float, default=0.00005)
    parser.add_argument("--cost-commission", type=float, default=0.0001)
    parser.add_argument("--out-prefix", default="signal_quality_suite")
    return parser.parse_args()


def parse_horizons(raw: str) -> List[int]:
    vals: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    vals = sorted(set(v for v in vals if v > 0))
    if not vals:
        raise ValueError("No valid horizons provided.")
    return vals


def find_data_csv(user_value: str) -> Path:
    if user_value:
        p = Path(user_value)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Data CSV not found: {p}")
        return p
    data_dir = ROOT / "data"
    csvs = sorted(data_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
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
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=needed)
    return df[needed].sort_index().copy()


def make_folds(df: pd.DataFrame, min_train_frac: float, test_frac: float, max_folds: int) -> List[FoldSpec]:
    n = len(df)
    if n < 2000:
        raise ValueError("Dataset too small.")
    min_train = int(n * min_train_frac)
    test_size = max(int(n * test_frac), 1000)
    if min_train + test_size > n:
        raise ValueError("Not enough rows for fold settings.")

    folds: List[FoldSpec] = []
    test_start = min_train
    fold_id = 1
    while test_start + test_size <= n and fold_id <= max_folds:
        folds.append(FoldSpec(fold_id=fold_id, train_start=0, test_start=test_start, test_end=test_start + test_size))
        test_start += test_size
        fold_id += 1
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


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
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
    feat["ma_fast_slow"] = (
        close.rolling(12, min_periods=1).mean() - close.rolling(48, min_periods=1).mean()
    ) / close_safe
    feat["atr_norm"] = (high - low).rolling(14, min_periods=1).mean() / close_safe
    feat["vol_z"] = (volume - vol_mean) / vol_std

    return feat.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)


def add_structural_features(base: pd.DataFrame, df: pd.DataFrame, bars_per_day: int) -> pd.DataFrame:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    week = max(5 * bars_per_day, 10)
    month = max(21 * bars_per_day, 30)
    qtr = max(63 * bars_per_day, 90)
    y52w = max(252 * bars_per_day, 250)

    ret1 = close.pct_change(1)
    out = base.copy()
    out["weekly_return"] = close.pct_change(week)
    out["monthly_return"] = close.pct_change(month)
    out["rolling_3m_vol"] = ret1.rolling(qtr, min_periods=10).std()

    range_short = high.rolling(week, min_periods=10).max() - low.rolling(week, min_periods=10).min()
    range_long = high.rolling(qtr, min_periods=20).max() - low.rolling(qtr, min_periods=20).min()
    out["range_compression"] = range_short / range_long.replace(0.0, np.nan)

    high_52w = high.rolling(y52w, min_periods=50).max()
    low_52w = low.rolling(y52w, min_periods=50).min()
    out["dist_52w_high"] = (close / high_52w) - 1.0
    out["dist_52w_low"] = (close / low_52w) - 1.0

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)


def make_forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return (close.shift(-horizon) - close) / close


def dynamic_costs(close_values: np.ndarray, step_idx: int, spread: float, slippage: float) -> Tuple[float, float]:
    if step_idx < 20:
        return spread, slippage
    lo = max(0, step_idx - 20)
    recent = close_values[lo:step_idx]
    if recent.shape[0] < 10:
        return spread, slippage
    prev = recent[:-1]
    if prev.size == 0:
        return spread, slippage
    changes = np.diff(recent) / np.clip(prev, 1e-12, None)
    vol = float(np.std(changes))
    mult = 1.0 + (vol * 100.0)
    return min(spread * mult, spread * 3.0), min(slippage * mult, slippage * 3.0)


def cost_adjusted_directional_return(
    entry_price: float,
    exit_price: float,
    direction: int,
    dyn_spread_open: float,
    dyn_slip_open: float,
    dyn_spread_close: float,
    dyn_slip_close: float,
    commission: float,
) -> float:
    notional = max(entry_price, 1e-12)
    raw = ((exit_price - entry_price) / notional) * float(direction)
    # Approximate round-trip cost in return space.
    cost = (
        (dyn_spread_open + dyn_slip_open) / notional
        + commission
        + (dyn_spread_close + dyn_slip_close) / max(exit_price, 1e-12)
        + commission
    )
    return float(raw - cost)


def build_lgbm_params(args: argparse.Namespace, task: str) -> Dict[str, float]:
    common = {
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
        "verbose": -1,
    }
    if task == "classification":
        common["n_estimators"] = int(args.class_n_estimators)
        common["class_weight"] = "balanced"
    else:
        common["n_estimators"] = int(args.reg_n_estimators)
    return common


def evaluate_classification(
    df: pd.DataFrame,
    features: pd.DataFrame,
    folds: List[FoldSpec],
    horizons: List[int],
    args: argparse.Namespace,
) -> Dict[str, object]:
    close = df["close"].astype(float)
    clf_params = build_lgbm_params(args, "classification")
    by_horizon: Dict[str, object] = {}

    for h in horizons:
        fwd = make_forward_return(close, horizon=h)
        fold_rows: List[ClassificationFoldResult] = []

        for fold in folds:
            train_end = max(fold.train_start, fold.test_start - h)
            test_last_entry = fold.test_end - h
            train_idx = np.arange(fold.train_start, train_end, dtype=int)
            test_idx = np.arange(fold.test_start, max(test_last_entry, fold.test_start), dtype=int)
            if train_idx.size < 500 or test_idx.size < 50:
                continue

            y_train = (fwd.iloc[train_idx] > 0.0).astype(int)
            X_train = features.iloc[train_idx]
            X_test = features.iloc[test_idx]
            fwd_test = fwd.iloc[test_idx].to_numpy(dtype=float)

            model = LGBMClassifier(**clf_params)
            model.fit(X_train, y_train)
            p_up = model.predict_proba(X_test)[:, 1]
            pred_sign = np.where(p_up >= 0.5, 1.0, -1.0)
            true_sign = np.where(fwd_test > 0.0, 1.0, -1.0)

            acc = float(np.mean(pred_sign == true_sign))
            mean_fwd = float(np.mean(pred_sign * fwd_test))

            fold_rows.append(
                ClassificationFoldResult(
                    fold_id=fold.fold_id,
                    test_start=str(df.index[fold.test_start]),
                    test_end=str(df.index[fold.test_end - 1]),
                    samples=int(test_idx.size),
                    accuracy=acc,
                    mean_forward_return=mean_fwd,
                )
            )

        if fold_rows:
            sample_weights = np.asarray([r.samples for r in fold_rows], dtype=float)
            accs = np.asarray([r.accuracy for r in fold_rows], dtype=float)
            rets = np.asarray([r.mean_forward_return for r in fold_rows], dtype=float)
            weighted_acc = float(np.average(accs, weights=sample_weights))
            weighted_ret = float(np.average(rets, weights=sample_weights))
            consistency = float(np.mean(accs >= 0.52))
        else:
            weighted_acc = float("nan")
            weighted_ret = float("nan")
            consistency = 0.0

        by_horizon[str(h)] = {
            "horizon": int(h),
            "folds": [asdict(r) for r in fold_rows],
            "summary": {
                "weighted_accuracy": weighted_acc,
                "weighted_mean_forward_return": weighted_ret,
                "weighted_mean_forward_return_bps": weighted_ret * 10000.0 if weighted_ret == weighted_ret else float("nan"),
                "fold_consistency_acc_ge_52": consistency,
                "fold_count": len(fold_rows),
            },
        }

    return by_horizon


def apply_signals(
    abs_indices: np.ndarray,
    preds: np.ndarray,
    close_values: np.ndarray,
    horizon: int,
    abs_threshold: float,
    cooldown_bars: int,
    spread: float,
    slippage: float,
    commission: float,
) -> Tuple[int, float, float, float]:
    trades = 0
    wins = 0
    returns_raw: List[float] = []
    returns_net: List[float] = []

    next_allowed = int(abs_indices[0]) if abs_indices.size else 0
    n = close_values.shape[0]

    for abs_idx, pred in zip(abs_indices, preds):
        i = int(abs_idx)
        if i < next_allowed:
            continue
        if abs(float(pred)) < abs_threshold or float(pred) == 0.0:
            continue
        exit_idx = i + horizon
        if exit_idx >= n:
            continue

        direction = 1 if pred > 0 else -1
        entry = float(close_values[i])
        exit_ = float(close_values[exit_idx])
        raw_ret = ((exit_ - entry) / max(entry, 1e-12)) * float(direction)

        dso, dlo = dynamic_costs(close_values, i, spread=spread, slippage=slippage)
        dsc, dlc = dynamic_costs(close_values, exit_idx, spread=spread, slippage=slippage)
        net_ret = cost_adjusted_directional_return(
            entry_price=entry,
            exit_price=exit_,
            direction=direction,
            dyn_spread_open=dso,
            dyn_slip_open=dlo,
            dyn_spread_close=dsc,
            dyn_slip_close=dlc,
            commission=commission,
        )

        trades += 1
        if raw_ret > 0.0:
            wins += 1
        returns_raw.append(float(raw_ret))
        returns_net.append(float(net_ret))
        next_allowed = i + max(int(cooldown_bars), 1)

    if trades == 0:
        return 0, 0.0, float("nan"), float("nan")
    return (
        int(trades),
        float(wins / trades),
        float(np.mean(returns_raw)),
        float(np.mean(returns_net)),
    )


def evaluate_regression_expectancy(
    df: pd.DataFrame,
    features: pd.DataFrame,
    folds: List[FoldSpec],
    horizons: List[int],
    bars_per_day: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    close = df["close"].astype(float)
    close_values = close.to_numpy(dtype=float)
    reg_params = build_lgbm_params(args, "regression")
    by_horizon: Dict[str, object] = {}

    bars_per_month = max(21 * bars_per_day, 1)
    target_rate = float(args.target_trades_per_month) / float(bars_per_month)
    target_rate = float(np.clip(target_rate, 1e-5, 0.5))

    for h in horizons:
        fwd = make_forward_return(close, horizon=h)
        fold_rows: List[RegressionFoldResult] = []

        for fold in folds:
            train_end = max(fold.train_start, fold.test_start - h)
            test_last_entry = fold.test_end - h
            train_idx = np.arange(fold.train_start, train_end, dtype=int)
            test_idx = np.arange(fold.test_start, max(test_last_entry, fold.test_start), dtype=int)
            if train_idx.size < 500 or test_idx.size < 50:
                continue

            X_train = features.iloc[train_idx]
            y_train = fwd.iloc[train_idx].to_numpy(dtype=float)
            X_test = features.iloc[test_idx]

            model = LGBMRegressor(**reg_params)
            model.fit(X_train, y_train)

            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)

            # Step 2: top 10% (absolute expected return) with horizon cooldown.
            thr_top = float(np.quantile(np.abs(pred_test), float(args.top_quantile)))
            t2_trades, t2_hit, t2_mean_ret, t2_mean_ret_net = apply_signals(
                abs_indices=test_idx,
                preds=pred_test,
                close_values=close_values,
                horizon=h,
                abs_threshold=thr_top,
                cooldown_bars=h,
                spread=float(args.cost_spread),
                slippage=float(args.cost_slippage),
                commission=float(args.cost_commission),
            )

            # Step 3: target low frequency using train-calibrated threshold.
            q_target = float(np.clip(1.0 - target_rate, 0.5, 0.9999))
            thr_target = float(np.quantile(np.abs(pred_train), q_target))
            t3_trades, t3_hit, t3_mean_ret, t3_mean_ret_net = apply_signals(
                abs_indices=test_idx,
                preds=pred_test,
                close_values=close_values,
                horizon=h,
                abs_threshold=thr_target,
                cooldown_bars=max(h, bars_per_day),
                spread=float(args.cost_spread),
                slippage=float(args.cost_slippage),
                commission=float(args.cost_commission),
            )

            period_months = float(test_idx.size) / float(bars_per_month)
            t2_pm = float(t2_trades / period_months) if period_months > 0 else 0.0
            t3_pm = float(t3_trades / period_months) if period_months > 0 else 0.0

            fold_rows.append(
                RegressionFoldResult(
                    fold_id=fold.fold_id,
                    test_start=str(df.index[fold.test_start]),
                    test_end=str(df.index[fold.test_end - 1]),
                    samples=int(test_idx.size),
                    trades_top10=int(t2_trades),
                    trades_top10_per_month=t2_pm,
                    hit_rate_top10=float(t2_hit),
                    mean_trade_return_top10=float(t2_mean_ret),
                    mean_trade_return_top10_cost_adj=float(t2_mean_ret_net),
                    trades_target_freq=int(t3_trades),
                    trades_target_per_month=t3_pm,
                    hit_rate_target_freq=float(t3_hit),
                    mean_trade_return_target_freq=float(t3_mean_ret),
                    mean_trade_return_target_freq_cost_adj=float(t3_mean_ret_net),
                )
            )

        if fold_rows:
            w = np.asarray([r.samples for r in fold_rows], dtype=float)
            t2_ret = np.asarray([r.mean_trade_return_top10 for r in fold_rows], dtype=float)
            t2_ret_net = np.asarray([r.mean_trade_return_top10_cost_adj for r in fold_rows], dtype=float)
            t2_hit = np.asarray([r.hit_rate_top10 for r in fold_rows], dtype=float)
            t2_pm = np.asarray([r.trades_top10_per_month for r in fold_rows], dtype=float)

            t3_ret = np.asarray([r.mean_trade_return_target_freq for r in fold_rows], dtype=float)
            t3_ret_net = np.asarray([r.mean_trade_return_target_freq_cost_adj for r in fold_rows], dtype=float)
            t3_hit = np.asarray([r.hit_rate_target_freq for r in fold_rows], dtype=float)
            t3_pm = np.asarray([r.trades_target_per_month for r in fold_rows], dtype=float)

            summary = {
                "fold_count": len(fold_rows),
                "top10_weighted_hit_rate": float(np.average(t2_hit, weights=w)),
                "top10_weighted_mean_trade_return": float(np.average(t2_ret, weights=w)),
                "top10_weighted_mean_trade_return_bps": float(np.average(t2_ret, weights=w) * 10000.0),
                "top10_weighted_mean_trade_return_cost_adj": float(np.average(t2_ret_net, weights=w)),
                "top10_weighted_mean_trade_return_cost_adj_bps": float(np.average(t2_ret_net, weights=w) * 10000.0),
                "top10_weighted_trades_per_month": float(np.average(t2_pm, weights=w)),
                "targetfreq_weighted_hit_rate": float(np.average(t3_hit, weights=w)),
                "targetfreq_weighted_mean_trade_return": float(np.average(t3_ret, weights=w)),
                "targetfreq_weighted_mean_trade_return_bps": float(np.average(t3_ret, weights=w) * 10000.0),
                "targetfreq_weighted_mean_trade_return_cost_adj": float(np.average(t3_ret_net, weights=w)),
                "targetfreq_weighted_mean_trade_return_cost_adj_bps": float(np.average(t3_ret_net, weights=w) * 10000.0),
                "targetfreq_weighted_trades_per_month": float(np.average(t3_pm, weights=w)),
            }
        else:
            summary = {"fold_count": 0}

        by_horizon[str(h)] = {
            "horizon": int(h),
            "folds": [asdict(r) for r in fold_rows],
            "summary": summary,
        }

    return by_horizon


def edge_flags(classification_results: Dict[str, object]) -> Dict[str, object]:
    checks: Dict[str, object] = {}
    strong_any = False
    for h, payload in classification_results.items():
        s = payload["summary"]
        acc = float(s.get("weighted_accuracy", float("nan")))
        consistency = float(s.get("fold_consistency_acc_ge_52", 0.0))
        has_edge = (acc == acc and acc >= 0.53 and consistency >= 0.6)
        if has_edge:
            strong_any = True
        checks[h] = {
            "weighted_accuracy": acc,
            "consistency_acc_ge_52": consistency,
            "edge_flag": bool(has_edge),
        }
    checks["overall_edge_present"] = bool(strong_any)
    checks["interpretation"] = (
        "At least one horizon met >=53% weighted accuracy with >=60% fold consistency."
        if strong_any
        else "No horizon met the 52-53% consistency edge threshold in current feature space."
    )
    return checks


def format_markdown(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Signal Quality Suite")
    lines.append("")
    lines.append(f"- Generated: {payload['generated_at']}")
    lines.append(f"- Data: `{payload['data_csv']}`")
    lines.append(f"- Bars/day inferred: {payload['bars_per_day']}")
    lines.append(f"- Folds: {payload['config']['folds_generated']}")
    lines.append(f"- Horizons: {payload['config']['horizons']}")
    lines.append("")

    for feature_set in ["base", "base_plus_structural"]:
        block = payload[feature_set]
        lines.append(f"## {feature_set}")
        lines.append("")

        lines.append("### Step 1: Pure Signal Quality (Classification)")
        lines.append("| Horizon | Weighted Accuracy | Fold Consistency >=52% | Mean Fwd Return (bps) |")
        lines.append("| --- | ---: | ---: | ---: |")
        for h, d in block["classification"].items():
            s = d["summary"]
            lines.append(
                f"| {h} | {s['weighted_accuracy']:.4f} | {s['fold_consistency_acc_ge_52']:.2f} | {s['weighted_mean_forward_return_bps']:.3f} |"
            )
        ef = block["edge_flags"]
        lines.append("")
        lines.append(f"- Edge interpretation: {ef['interpretation']}")
        lines.append("")

        lines.append("### Step 2/3: Expectancy Regression + Frequency Control")
        lines.append("| Horizon | Top10 Mean Return (bps) | Top10 Cost-Adj (bps) | Top10 Trades/Month | TargetFreq Mean Return (bps) | TargetFreq Cost-Adj (bps) | TargetFreq Trades/Month |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for h, d in block["regression"].items():
            s = d["summary"]
            lines.append(
                f"| {h} | {s.get('top10_weighted_mean_trade_return_bps', float('nan')):.3f} | "
                f"{s.get('top10_weighted_mean_trade_return_cost_adj_bps', float('nan')):.3f} | "
                f"{s.get('top10_weighted_trades_per_month', float('nan')):.2f} | "
                f"{s.get('targetfreq_weighted_mean_trade_return_bps', float('nan')):.3f} | "
                f"{s.get('targetfreq_weighted_mean_trade_return_cost_adj_bps', float('nan')):.3f} | "
                f"{s.get('targetfreq_weighted_trades_per_month', float('nan')):.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    horizons = parse_horizons(args.horizons)

    data_csv = find_data_csv(args.data_csv)
    df = load_ohlcv(data_csv)
    folds = make_folds(df, min_train_frac=float(args.min_train_frac), test_frac=float(args.test_frac), max_folds=int(args.max_folds))
    if not folds:
        raise SystemExit("No folds generated.")

    bars_per_day = infer_bars_per_day(df.index)

    base_features = build_base_features(df)
    structural_features = add_structural_features(base_features, df, bars_per_day=bars_per_day)

    base_class = evaluate_classification(df, base_features, folds, horizons, args)
    base_reg = evaluate_regression_expectancy(df, base_features, folds, horizons, bars_per_day, args)

    struct_class = evaluate_classification(df, structural_features, folds, horizons, args)
    struct_reg = evaluate_regression_expectancy(df, structural_features, folds, horizons, bars_per_day, args)

    payload: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_csv": str(data_csv),
        "bars_per_day": int(bars_per_day),
        "config": {
            "horizons": horizons,
            "folds_generated": len(folds),
            "min_train_frac": float(args.min_train_frac),
            "test_frac": float(args.test_frac),
            "max_folds": int(args.max_folds),
            "top_quantile": float(args.top_quantile),
            "target_trades_per_month": float(args.target_trades_per_month),
            "cost_spread": float(args.cost_spread),
            "cost_slippage": float(args.cost_slippage),
            "cost_commission": float(args.cost_commission),
        },
        "base": {
            "classification": base_class,
            "edge_flags": edge_flags(base_class),
            "regression": base_reg,
        },
        "base_plus_structural": {
            "classification": struct_class,
            "edge_flags": edge_flags(struct_class),
            "regression": struct_reg,
        },
    }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"{args.out_prefix}_{ts}.json"
    md_path = reports_dir / f"{args.out_prefix}_{ts}.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(format_markdown(payload), encoding="utf-8")

    print(f"report_json={json_path}")
    print(f"report_md={md_path}")

    base_interp = payload["base"]["edge_flags"]["interpretation"]
    struct_interp = payload["base_plus_structural"]["edge_flags"]["interpretation"]
    print(f"base_edge={base_interp}")
    print(f"structural_edge={struct_interp}")


if __name__ == "__main__":
    main()
