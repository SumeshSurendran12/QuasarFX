from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, roc_auc_score

import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.config import INITIAL_BALANCE, SPREAD, COMMISSION, SLIPPAGE, MIN_POSITION_SIZE


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
    return_pct: float
    max_drawdown_pct: float
    pass_flag: bool
    auc: float
    trade_f1: float
    trade_rate: float
    long_precision: float
    short_precision: float
    train_samples_used: int
    test_samples_scored: int
    leakage_purged_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward LightGBM benchmark with RL-style gate report and cost-adjusted PnL."
    )
    parser.add_argument("--data-csv", default="", help="Optional data CSV path. Defaults to latest in data/*.csv.")
    parser.add_argument("--min-train-frac", type=float, default=DEFAULT_MIN_TRAIN_FRAC)
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    parser.add_argument("--max-folds", type=int, default=DEFAULT_MAX_FOLDS)
    parser.add_argument("--min-trades-per-fold", type=int, default=DEFAULT_MIN_TRADES_PER_FOLD)
    parser.add_argument("--out-prefix", default="walk_forward_lgbm_eval")
    parser.add_argument("--horizon-bars", type=int, default=10)
    parser.add_argument("--label-threshold", type=float, default=0.0004)
    parser.add_argument("--long-th", type=float, default=0.58)
    parser.add_argument("--short-th", type=float, default=0.42)
    parser.add_argument("--trend-filter", action="store_true", default=True)
    parser.add_argument("--no-trend-filter", dest="trend_filter", action="store_false")
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--num-leaves", type=int, default=64)
    parser.add_argument("--min-child-samples", type=int, default=200)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--subsample-freq", type=int, default=5)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--lot-size", type=float, default=float(MIN_POSITION_SIZE))
    parser.add_argument("--initial-balance", type=float, default=float(INITIAL_BALANCE))
    parser.add_argument("--spread", type=float, default=float(SPREAD))
    parser.add_argument("--commission", type=float, default=float(COMMISSION))
    parser.add_argument("--slippage", type=float, default=float(SLIPPAGE))
    parser.add_argument("--compare-rl-report", default="", help="Optional existing RL walk-forward JSON report to compare.")
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
    df = df.sort_index()
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


def build_features_like_env(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(0.0, index=df.index)

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

    feat = feat.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return feat.astype(np.float32)


def make_direction_labels(df: pd.DataFrame, horizon_bars: int, threshold: float) -> pd.Series:
    close = df["close"].astype(float)
    fwd_close = close.shift(-horizon_bars)
    fwd_ret = (fwd_close - close) / close

    y = pd.Series(np.nan, index=df.index, dtype="float64")
    y[fwd_ret > threshold] = 1.0
    y[fwd_ret < -threshold] = 0.0
    return y


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
    volatility = float(np.std(changes))
    mult = 1.0 + (volatility * 100.0)
    dyn_spread = min(spread * mult, spread * 3.0)
    dyn_slippage = min(slippage * mult, slippage * 3.0)
    return dyn_spread, dyn_slippage


def trading_cost(price: float, lot_size: float, spread: float, slippage: float, commission: float) -> float:
    spread_cost = spread * lot_size * 100000.0
    commission_cost = price * lot_size * 100000.0 * commission
    slippage_cost = slippage * lot_size * 100000.0
    return float(spread_cost + commission_cost + slippage_cost)


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


def evaluate_fold(
    model: LGBMClassifier,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    fold_id: int,
    train_start: int,
    test_start: int,
    test_end: int,
    horizon_bars: int,
    long_th: float,
    short_th: float,
    trend_filter: bool,
    initial_balance: float,
    lot_size: float,
    spread: float,
    commission: float,
    slippage: float,
    min_trades_per_fold: int,
) -> FoldMetrics:
    n = len(df)
    close_values = df["close"].to_numpy(dtype=float)

    train_end_raw = test_start
    train_end_purged = max(train_start, test_start - horizon_bars)
    leakage_purged_rows = max(train_end_raw - train_end_purged, 0)

    train_idx = np.arange(train_start, train_end_purged, dtype=int)
    test_idx = np.arange(test_start, test_end, dtype=int)

    train_mask = y.iloc[train_idx].notna().to_numpy()
    train_idx_used = train_idx[train_mask]
    if train_idx_used.size < 200:
        raise ValueError(
            f"Fold {fold_id}: insufficient train samples after leakage purge/neutral drop ({train_idx_used.size})."
        )

    X_tr = X.iloc[train_idx_used]
    y_tr = y.iloc[train_idx_used].astype(int)
    model.fit(X_tr, y_tr)

    max_entry_idx = min(test_end - horizon_bars - 1, n - horizon_bars - 1)
    candidate_idx = test_idx[test_idx <= max_entry_idx]
    if candidate_idx.size == 0:
        empty_metrics = {
            "total_trades": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_profit": 0.0,
            "avg_trade_pnl": 0.0,
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }
        return FoldMetrics(
            fold_id=fold_id,
            train_rows=int(train_end_raw - train_start),
            test_rows=int(test_end - test_start),
            test_start=str(df.index[test_start]),
            test_end=str(df.index[test_end - 1]),
            total_reward=0.0,
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            net_profit=0.0,
            avg_trade_pnl=0.0,
            return_pct=0.0,
            max_drawdown_pct=0.0,
            pass_flag=fold_pass(empty_metrics, min_trades_per_fold),
            auc=float("nan"),
            trade_f1=float("nan"),
            trade_rate=0.0,
            long_precision=float("nan"),
            short_precision=float("nan"),
            train_samples_used=int(train_idx_used.size),
            test_samples_scored=0,
            leakage_purged_rows=int(leakage_purged_rows),
        )

    X_te = X.iloc[candidate_idx]
    p_raw = model.predict_proba(X_te)[:, 1]
    p_exec = p_raw.copy()

    if trend_filter:
        trend = X_te["ma_fast_slow"].to_numpy()
        p_exec = np.where(trend > 0, np.maximum(p_exec, short_th), p_exec)
        p_exec = np.where(trend < 0, np.minimum(p_exec, long_th), p_exec)

    y_te = y.iloc[candidate_idx]
    y_te_mask = y_te.notna().to_numpy()
    y_true = y_te.iloc[y_te_mask].astype(int).to_numpy()
    p_auc = p_raw[y_te_mask]
    auc = float("nan")
    if y_true.size > 0 and np.unique(y_true).size == 2:
        auc = float(roc_auc_score(y_true, p_auc))

    balance = float(initial_balance)
    equity_curve: List[float] = [balance]
    gross_profit = 0.0
    gross_loss = 0.0
    total_trades = 0
    wins = 0
    y_pred_traded: List[int] = []
    y_true_traded: List[int] = []
    next_entry_allowed_idx = test_start

    p_exec_by_abs = {int(idx): float(prob) for idx, prob in zip(candidate_idx, p_exec)}

    for abs_idx in candidate_idx:
        if abs_idx < next_entry_allowed_idx:
            equity_curve.append(balance)
            continue

        p = p_exec_by_abs[int(abs_idx)]
        action = -1
        if p > long_th:
            action = 1
        elif p < short_th:
            action = 0

        if action == -1:
            equity_curve.append(balance)
            continue

        exit_idx = abs_idx + horizon_bars
        if exit_idx >= test_end or exit_idx >= n:
            equity_curve.append(balance)
            continue

        direction = 1 if action == 1 else -1
        entry_close = float(close_values[abs_idx])
        exit_close = float(close_values[exit_idx])
        dyn_spread_open, dyn_slip_open = dynamic_costs(close_values, abs_idx, spread=spread, slippage=slippage)
        dyn_spread_close, dyn_slip_close = dynamic_costs(close_values, exit_idx, spread=spread, slippage=slippage)

        entry_price = entry_close * (1.0 + slippage if direction > 0 else 1.0 - slippage)
        exit_price = exit_close * (1.0 - slippage if direction > 0 else 1.0 + slippage)

        open_cost = trading_cost(entry_price, lot_size, dyn_spread_open, dyn_slip_open, commission)
        close_cost = trading_cost(exit_price, lot_size, dyn_spread_close, dyn_slip_close, commission)

        balance -= open_cost
        pnl = (exit_price - entry_price) * direction * lot_size * 100000.0
        pnl -= close_cost
        balance += pnl

        trade_pnl = pnl
        total_trades += 1
        if trade_pnl > 0:
            wins += 1
            gross_profit += float(trade_pnl)
        elif trade_pnl < 0:
            gross_loss += float(abs(trade_pnl))

        if not np.isnan(y.iloc[abs_idx]):
            y_true_traded.append(int(y.iloc[abs_idx]))
            y_pred_traded.append(int(action))

        equity_curve.append(balance)
        next_entry_allowed_idx = exit_idx + 1

    win_rate = (wins / total_trades) if total_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    net_profit = gross_profit - gross_loss
    avg_trade_pnl = (net_profit / total_trades) if total_trades > 0 else 0.0
    return_pct = ((balance - initial_balance) / max(initial_balance, 1e-9)) * 100.0
    max_dd = max_drawdown_pct(equity_curve)
    total_reward = net_profit / max(initial_balance, 1e-9)

    trade_rate = float(total_trades / max(candidate_idx.size, 1))
    trade_f1 = float("nan")
    long_precision = float("nan")
    short_precision = float("nan")
    if y_true_traded:
        yt = np.asarray(y_true_traded, dtype=int)
        yp = np.asarray(y_pred_traded, dtype=int)
        trade_f1 = float(f1_score(yt, yp, zero_division=0))
        long_mask = yp == 1
        short_mask = yp == 0
        if long_mask.any():
            long_precision = float((yt[long_mask] == 1).mean())
        if short_mask.any():
            short_precision = float((yt[short_mask] == 0).mean())

    metrics = {
        "total_trades": float(total_trades),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "net_profit": float(net_profit),
        "avg_trade_pnl": float(avg_trade_pnl),
        "return_pct": float(return_pct),
        "max_drawdown_pct": float(max_dd),
    }

    return FoldMetrics(
        fold_id=fold_id,
        train_rows=int(train_end_raw - train_start),
        test_rows=int(test_end - test_start),
        test_start=str(df.index[test_start]),
        test_end=str(df.index[test_end - 1]),
        total_reward=float(total_reward),
        total_trades=int(total_trades),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        gross_profit=float(gross_profit),
        gross_loss=float(gross_loss),
        net_profit=float(net_profit),
        avg_trade_pnl=float(avg_trade_pnl),
        return_pct=float(return_pct),
        max_drawdown_pct=float(max_dd),
        pass_flag=fold_pass(metrics, min_trades_per_fold=min_trades_per_fold),
        auc=float(auc),
        trade_f1=float(trade_f1),
        trade_rate=float(trade_rate),
        long_precision=float(long_precision),
        short_precision=float(short_precision),
        train_samples_used=int(train_idx_used.size),
        test_samples_scored=int(candidate_idx.size),
        leakage_purged_rows=int(leakage_purged_rows),
    )


def evaluate_lgbm_walk_forward(
    df: pd.DataFrame,
    folds: List[Tuple[int, int, int, int]],
    horizon_bars: int,
    label_threshold: float,
    long_th: float,
    short_th: float,
    trend_filter: bool,
    min_trades_per_fold: int,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    num_leaves: int,
    min_child_samples: int,
    subsample: float,
    subsample_freq: int,
    colsample_bytree: float,
    reg_lambda: float,
    random_state: int,
    initial_balance: float,
    lot_size: float,
    spread: float,
    commission: float,
    slippage: float,
) -> Dict[str, object]:
    X = build_features_like_env(df)
    y = make_direction_labels(df, horizon_bars=horizon_bars, threshold=label_threshold)

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        subsample_freq=subsample_freq,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced",
        verbose=-1,
    )

    results: List[FoldMetrics] = []
    gross_profit_total = 0.0
    gross_loss_total = 0.0
    net_profit_total = 0.0
    returns: List[float] = []
    rewards: List[float] = []
    drawdowns: List[float] = []
    trade_counts: List[int] = []

    for fold_id, train_start, test_start, test_end in folds:
        fold_metrics = evaluate_fold(
            model=model,
            df=df,
            X=X,
            y=y,
            fold_id=fold_id,
            train_start=train_start,
            test_start=test_start,
            test_end=test_end,
            horizon_bars=horizon_bars,
            long_th=long_th,
            short_th=short_th,
            trend_filter=trend_filter,
            initial_balance=initial_balance,
            lot_size=lot_size,
            spread=spread,
            commission=commission,
            slippage=slippage,
            min_trades_per_fold=min_trades_per_fold,
        )
        results.append(fold_metrics)
        gross_profit_total += fold_metrics.gross_profit
        gross_loss_total += fold_metrics.gross_loss
        net_profit_total += fold_metrics.net_profit
        returns.append(fold_metrics.return_pct)
        rewards.append(fold_metrics.total_reward)
        drawdowns.append(fold_metrics.max_drawdown_pct)
        trade_counts.append(fold_metrics.total_trades)

        print(
            f"fold={fold_metrics.fold_id} trades={fold_metrics.total_trades} "
            f"net={fold_metrics.net_profit:.2f} pf={fold_metrics.profit_factor:.3f} "
            f"dd={fold_metrics.max_drawdown_pct:.2f} pass={'Y' if fold_metrics.pass_flag else 'N'} "
            f"auc={fold_metrics.auc if fold_metrics.auc == fold_metrics.auc else float('nan'):.3f} "
            f"purged={fold_metrics.leakage_purged_rows}"
        )

    pass_count = sum(1 for r in results if r.pass_flag)
    pass_rate = (pass_count / len(results)) if results else 0.0
    overall_profit_factor = (
        gross_profit_total / gross_loss_total
        if gross_loss_total > 0
        else (float("inf") if gross_profit_total > 0 else 0.0)
    )

    summary = {
        "model_path": "lightgbm_env_features",
        "folds": len(results),
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "total_trades": int(sum(trade_counts)),
        "min_fold_trades": int(min(trade_counts)) if trade_counts else 0,
        "mean_fold_trades": float(np.mean(trade_counts)) if trade_counts else 0.0,
        "overall_net_profit": float(net_profit_total),
        "overall_profit_factor": float(overall_profit_factor),
        "mean_fold_return_pct": float(np.mean(returns)) if returns else 0.0,
        "median_fold_return_pct": float(np.median(returns)) if returns else 0.0,
        "mean_fold_reward": float(np.mean(rewards)) if rewards else 0.0,
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


def compare_with_rl_report(lgbm_report: Dict[str, object], rl_report_path: Path) -> Dict[str, object]:
    rl_payload = json.loads(rl_report_path.read_text(encoding="utf-8"))
    if not rl_payload.get("reports"):
        raise ValueError(f"RL report has no reports[]: {rl_report_path}")
    rl_summary = rl_payload["reports"][0]["summary"]
    lgbm_summary = lgbm_report["summary"]
    return {
        "rl_report_path": str(rl_report_path),
        "rl_summary": rl_summary,
        "lgbm_summary": lgbm_summary,
        "delta": {
            "pass_rate": float(lgbm_summary["pass_rate"]) - float(rl_summary.get("pass_rate", 0.0)),
            "overall_net_profit": float(lgbm_summary["overall_net_profit"]) - float(rl_summary.get("overall_net_profit", 0.0)),
            "overall_profit_factor": float(lgbm_summary["overall_profit_factor"]) - float(rl_summary.get("overall_profit_factor", 0.0)),
            "worst_fold_drawdown_pct": float(lgbm_summary["worst_fold_drawdown_pct"]) - float(rl_summary.get("worst_fold_drawdown_pct", 0.0)),
            "total_trades": int(lgbm_summary["total_trades"]) - int(rl_summary.get("total_trades", 0)),
        },
    }


def to_markdown(
    data_csv: Path,
    config: Dict[str, object],
    report: Dict[str, object],
    comparison: Dict[str, object] | None = None,
) -> str:
    summary = report["summary"]
    folds = report["folds"]

    lines: List[str] = []
    lines.append("# Walk-Forward LightGBM Evaluation Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Data: `{data_csv}`")
    lines.append(
        f"- Config: min_train_frac={config['min_train_frac']}, test_frac={config['test_frac']}, max_folds={config['max_folds']}, min_trades_per_fold={config['min_trades_per_fold']}"
    )
    lines.append(
        f"- LGBM: horizon={config['horizon_bars']}, label_threshold={config['label_threshold']}, long_th={config['long_th']}, short_th={config['short_th']}, trend_filter={config['trend_filter']}"
    )
    lines.append(
        f"- Costs: lot_size={config['lot_size']}, spread={config['spread']}, commission={config['commission']}, slippage={config['slippage']}"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Decision: **{summary['decision']}**")
    lines.append(f"- Retrain recommended: **{summary['retrain_recommended']}**")
    lines.append(f"- Fold pass rate: {summary['pass_count']}/{summary['folds']} ({summary['pass_rate']*100:.1f}%)")
    lines.append(f"- Total trades: {summary['total_trades']}")
    lines.append(f"- Min trades/fold: {summary['min_fold_trades']} (required: {summary['min_trades_per_fold_required']})")
    lines.append(f"- Collapse detected: **{summary['collapse_detected']}**")
    lines.append(f"- Overall net profit: {summary['overall_net_profit']:.2f}")
    lines.append(f"- Overall profit factor: {summary['overall_profit_factor']:.3f}")
    lines.append(f"- Mean fold return %: {summary['mean_fold_return_pct']:.2f}")
    lines.append(f"- Worst fold drawdown %: {summary['worst_fold_drawdown_pct']:.2f}")
    lines.append("")
    lines.append("| Fold | Period Start | Period End | Trades | Net Profit | PF | Win Rate | Max DD % | AUC | F1 | Purged | Pass |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |")
    for f in folds:
        lines.append(
            "| "
            f"{f['fold_id']} | {f['test_start']} | {f['test_end']} | {f['total_trades']} | "
            f"{f['net_profit']:.2f} | {f['profit_factor']:.3f} | {f['win_rate']*100:.1f}% | {f['max_drawdown_pct']:.2f} | "
            f"{f['auc'] if f['auc'] == f['auc'] else float('nan'):.3f} | {f['trade_f1'] if f['trade_f1'] == f['trade_f1'] else float('nan'):.3f} | "
            f"{f['leakage_purged_rows']} | {'Y' if f['pass_flag'] else 'N'} |"
        )
    lines.append("")

    if comparison is not None:
        rl = comparison["rl_summary"]
        delta = comparison["delta"]
        lines.append("## RL Comparison")
        lines.append(f"- RL source: `{comparison['rl_report_path']}`")
        lines.append(f"- RL decision: **{rl.get('decision', 'n/a')}** | LGBM decision: **{summary['decision']}**")
        lines.append(
            f"- Delta pass_rate: {delta['pass_rate']*100:.1f} pts | "
            f"Delta net_profit: {delta['overall_net_profit']:.2f} | "
            f"Delta PF: {delta['overall_profit_factor']:.3f} | "
            f"Delta worst_dd: {delta['worst_fold_drawdown_pct']:.2f} | "
            f"Delta trades: {delta['total_trades']}"
        )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    data_csv = find_data_csv(args.data_csv)
    df = load_ohlcv(data_csv)
    folds = make_folds(
        df=df,
        min_train_frac=float(args.min_train_frac),
        test_frac=float(args.test_frac),
        max_folds=int(args.max_folds),
    )
    if not folds:
        raise SystemExit("No folds generated.")

    min_trades_per_fold = max(int(args.min_trades_per_fold), 0)
    report = evaluate_lgbm_walk_forward(
        df=df,
        folds=folds,
        horizon_bars=int(args.horizon_bars),
        label_threshold=float(args.label_threshold),
        long_th=float(args.long_th),
        short_th=float(args.short_th),
        trend_filter=bool(args.trend_filter),
        min_trades_per_fold=min_trades_per_fold,
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_child_samples),
        subsample=float(args.subsample),
        subsample_freq=int(args.subsample_freq),
        colsample_bytree=float(args.colsample_bytree),
        reg_lambda=float(args.reg_lambda),
        random_state=int(args.random_state),
        initial_balance=float(args.initial_balance),
        lot_size=float(args.lot_size),
        spread=float(args.spread),
        commission=float(args.commission),
        slippage=float(args.slippage),
    )

    comparison = None
    if args.compare_rl_report:
        rl_path = Path(args.compare_rl_report)
        if not rl_path.is_absolute():
            rl_path = ROOT / rl_path
        if not rl_path.exists():
            raise FileNotFoundError(f"RL report not found: {rl_path}")
        comparison = compare_with_rl_report(report, rl_path)

    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{args.out_prefix}_{ts}.json"
    md_path = out_dir / f"{args.out_prefix}_{ts}.md"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "data_csv": str(data_csv),
        "config": {
            "min_train_frac": float(args.min_train_frac),
            "test_frac": float(args.test_frac),
            "max_folds": int(args.max_folds),
            "min_trades_per_fold": min_trades_per_fold,
            "folds_generated": len(folds),
            "horizon_bars": int(args.horizon_bars),
            "label_threshold": float(args.label_threshold),
            "long_th": float(args.long_th),
            "short_th": float(args.short_th),
            "trend_filter": bool(args.trend_filter),
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "max_depth": int(args.max_depth),
            "num_leaves": int(args.num_leaves),
            "min_child_samples": int(args.min_child_samples),
            "subsample": float(args.subsample),
            "subsample_freq": int(args.subsample_freq),
            "colsample_bytree": float(args.colsample_bytree),
            "reg_lambda": float(args.reg_lambda),
            "random_state": int(args.random_state),
            "initial_balance": float(args.initial_balance),
            "lot_size": float(args.lot_size),
            "spread": float(args.spread),
            "commission": float(args.commission),
            "slippage": float(args.slippage),
        },
        "reports": [report],
        "comparison": comparison,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(data_csv, payload["config"], report, comparison), encoding="utf-8")

    print(f"report_json={json_path}")
    print(f"report_md={md_path}")
    s = report["summary"]
    print(
        "summary "
        f"model={s['model_path']} decision={s['decision']} "
        f"pass_rate={s['pass_rate']*100:.1f}% net_profit={s['overall_net_profit']:.2f} "
        f"pf={s['overall_profit_factor']:.3f} worst_dd={s['worst_fold_drawdown_pct']:.2f} "
        f"min_fold_trades={s['min_fold_trades']} collapse={s['collapse_detected']}"
    )
    if comparison is not None:
        d = comparison["delta"]
        print(
            "compare_vs_rl "
            f"delta_pass_rate={d['pass_rate']*100:.1f}pts "
            f"delta_net={d['overall_net_profit']:.2f} "
            f"delta_pf={d['overall_profit_factor']:.3f} "
            f"delta_dd={d['worst_fold_drawdown_pct']:.2f} "
            f"delta_trades={d['total_trades']}"
        )


if __name__ == "__main__":
    main()
