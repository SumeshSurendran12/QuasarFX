from __future__ import annotations


import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

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
class FoldDirectionMetrics:
    fold_id: int
    test_start: str
    test_end: str
    samples: int
    accuracy: float
    auc: float
    mean_signed_forward_return_bps: float


@dataclass
class FoldTradeMetrics:
    fold_id: int
    test_start: str
    test_end: str
    total_trades: int
    win_rate: float
    profit_factor: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    avg_trade_pnl: float
    return_pct: float
    max_drawdown_pct: float
    avg_bars_in_trade: float
    avg_hours_in_trade: float
    trades_per_month: float
    pass_flag: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Directional LightGBM pipeline: confirm H1 edge first, then compare fixed vs adaptive exits."
    )
    parser.add_argument("--data-csv", default="", help="Optional data CSV. Defaults to latest in data/*.csv")
    parser.add_argument("--out-prefix", default="lgbm_directional_adaptive")

    parser.add_argument("--min-train-frac", type=float, default=DEFAULT_MIN_TRAIN_FRAC)
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    parser.add_argument("--max-folds", type=int, default=DEFAULT_MAX_FOLDS)
    parser.add_argument("--min-trades-per-fold", type=int, default=DEFAULT_MIN_TRADES_PER_FOLD)

    parser.add_argument("--horizon-bars", type=int, default=1, help="Prediction horizon in bars. H1 edge test uses 1 bar by default.")
    parser.add_argument("--label-threshold", type=float, default=0.0, help="Binary label threshold on forward return.")
    parser.add_argument("--long-th", type=float, default=0.55)
    parser.add_argument("--short-th", type=float, default=0.45)
    parser.add_argument("--use-structural-features", action="store_true", default=True)
    parser.add_argument("--no-structural-features", dest="use_structural_features", action="store_false")

    parser.add_argument("--edge-min-accuracy", type=float, default=0.52)
    parser.add_argument("--edge-min-consistency", type=float, default=0.60)

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

    parser.add_argument("--adaptive-stop-atr-mult", type=float, default=1.0)
    parser.add_argument("--adaptive-min-stop-pct", type=float, default=0.0015)
    parser.add_argument("--adaptive-rr-base", type=float, default=1.33)
    parser.add_argument("--adaptive-rr-confidence-mult", type=float, default=0.60)
    parser.add_argument("--adaptive-trail-atr-mult", type=float, default=0.75)
    parser.add_argument("--adaptive-min-trail-pct", type=float, default=0.0008)
    parser.add_argument("--adaptive-max-hold-bars", type=int, default=48)
    parser.add_argument("--adaptive-exit-on-opposite", action="store_true", default=True)
    parser.add_argument("--no-adaptive-exit-on-opposite", dest="adaptive_exit_on_opposite", action="store_false")

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
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=needed)
    return df.sort_index()[needed].copy()


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
        test_end = test_start + test_size
        folds.append((fold_id, 0, test_start, test_end))
        fold_id += 1
        test_start = test_end
    return folds

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


def make_labels(df: pd.DataFrame, horizon_bars: int, threshold: float) -> Tuple[pd.Series, pd.Series]:
    close = df["close"].astype(float)
    fwd_ret = (close.shift(-horizon_bars) - close) / close
    y = (fwd_ret > threshold).astype(int)
    return y, fwd_ret


def max_drawdown_pct(equity_curve: List[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        if peak > 0:
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)
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
    vol = float(np.std(changes))
    mult = 1.0 + (vol * 100.0)
    dyn_spread = min(spread * mult, spread * 3.0)
    dyn_slippage = min(slippage * mult, slippage * 3.0)
    return dyn_spread, dyn_slippage


def trading_cost(price: float, lot_size: float, spread: float, slippage: float, commission: float) -> float:
    spread_cost = spread * lot_size * 100000.0
    comm_cost = price * lot_size * 100000.0 * commission
    slip_cost = slippage * lot_size * 100000.0
    return float(spread_cost + comm_cost + slip_cost)


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


def _compute_trade_metrics(
    balance_start: float,
    balance_end: float,
    trade_pnls: List[float],
    hold_bars: List[int],
    equity_curve: List[float],
    bars_per_day: int,
    test_rows: int,
) -> Dict[str, float]:
    profits = [p for p in trade_pnls if p > 0.0]
    losses = [abs(p) for p in trade_pnls if p < 0.0]
    total_trades = len(trade_pnls)
    gross_profit = float(sum(profits))
    gross_loss = float(sum(losses))
    net_profit = gross_profit - gross_loss
    win_rate = float(len(profits) / total_trades) if total_trades > 0 else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    avg_trade_pnl = float(net_profit / total_trades) if total_trades > 0 else 0.0
    return_pct = float((balance_end - balance_start) / max(balance_start, 1e-9) * 100.0)
    avg_bars = float(np.mean(hold_bars)) if hold_bars else 0.0
    avg_hours = avg_bars * (24.0 / max(float(bars_per_day), 1.0))
    bars_per_month = max(21 * bars_per_day, 1)
    months = float(test_rows) / float(bars_per_month)
    tpm = float(total_trades / months) if months > 0 else 0.0
    return {
        "total_trades": float(total_trades),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "net_profit": float(net_profit),
        "avg_trade_pnl": float(avg_trade_pnl),
        "return_pct": float(return_pct),
        "max_drawdown_pct": float(max_drawdown_pct(equity_curve)),
        "avg_bars_in_trade": float(avg_bars),
        "avg_hours_in_trade": float(avg_hours),
        "trades_per_month": float(tpm),
    }

def simulate_fixed_exits(
    test_start: int,
    test_end: int,
    horizon_bars: int,
    p_map: Dict[int, float],
    close_values: np.ndarray,
    lot_size: float,
    spread: float,
    commission: float,
    slippage: float,
    long_th: float,
    short_th: float,
    initial_balance: float,
    bars_per_day: int,
) -> Dict[str, float]:
    balance = float(initial_balance)
    trade_pnls: List[float] = []
    hold_bars: List[int] = []
    equity_curve: List[float] = [balance]

    i = int(test_start)
    while i < int(test_end - 1):
        p = float(p_map.get(i, 0.5))
        direction = 1 if p > long_th else (-1 if p < short_th else 0)
        if direction == 0:
            equity_curve.append(balance)
            i += 1
            continue

        exit_idx = min(i + int(horizon_bars), int(test_end - 1))
        if exit_idx <= i:
            break

        entry_ref = float(close_values[i])
        exit_ref = float(close_values[exit_idx])
        entry_price = entry_ref * (1.0 + slippage if direction > 0 else 1.0 - slippage)
        exit_price = exit_ref * (1.0 - slippage if direction > 0 else 1.0 + slippage)

        sp_o, sl_o = dynamic_costs(close_values, i, spread=spread, slippage=slippage)
        sp_c, sl_c = dynamic_costs(close_values, exit_idx, spread=spread, slippage=slippage)
        open_cost = trading_cost(entry_price, lot_size, sp_o, sl_o, commission)
        close_cost = trading_cost(exit_price, lot_size, sp_c, sl_c, commission)

        move_pnl = (exit_price - entry_price) * direction * lot_size * 100000.0
        trade_net = move_pnl - open_cost - close_cost
        balance += trade_net

        trade_pnls.append(float(trade_net))
        hold_bars.append(int(exit_idx - i))
        equity_curve.append(balance)

        i = exit_idx + 1

    return _compute_trade_metrics(
        balance_start=float(initial_balance),
        balance_end=float(balance),
        trade_pnls=trade_pnls,
        hold_bars=hold_bars,
        equity_curve=equity_curve,
        bars_per_day=bars_per_day,
        test_rows=int(test_end - test_start),
    )


def simulate_adaptive_exits(
    test_start: int,
    test_end: int,
    p_map: Dict[int, float],
    close_values: np.ndarray,
    high_values: np.ndarray,
    low_values: np.ndarray,
    atr_values: np.ndarray,
    lot_size: float,
    spread: float,
    commission: float,
    slippage: float,
    long_th: float,
    short_th: float,
    initial_balance: float,
    bars_per_day: int,
    adaptive_stop_atr_mult: float,
    adaptive_min_stop_pct: float,
    adaptive_rr_base: float,
    adaptive_rr_confidence_mult: float,
    adaptive_trail_atr_mult: float,
    adaptive_min_trail_pct: float,
    adaptive_max_hold_bars: int,
    adaptive_exit_on_opposite: bool,
) -> Dict[str, float]:
    balance = float(initial_balance)
    trade_pnls: List[float] = []
    hold_bars: List[int] = []
    equity_curve: List[float] = [balance]

    open_trade: Optional[Dict[str, float]] = None
    i = int(test_start)

    while i < int(test_end):
        if open_trade is None:
            if i >= int(test_end - 1):
                break
            p = float(p_map.get(i, 0.5))
            direction = 1 if p > long_th else (-1 if p < short_th else 0)
            if direction == 0:
                equity_curve.append(balance)
                i += 1
                continue

            entry_ref = float(close_values[i])
            entry_price = entry_ref * (1.0 + slippage if direction > 0 else 1.0 - slippage)
            sp_o, sl_o = dynamic_costs(close_values, i, spread=spread, slippage=slippage)
            open_cost = trading_cost(entry_price, lot_size, sp_o, sl_o, commission)

            atr_entry = abs(float(atr_values[i]))
            stop_pct = max(float(adaptive_min_stop_pct), float(adaptive_stop_atr_mult) * atr_entry)
            confidence = max(min(abs(p - 0.5) * 2.0, 1.0), 0.0)
            rr = float(adaptive_rr_base) + float(adaptive_rr_confidence_mult) * confidence
            target_pct = max(stop_pct * rr, stop_pct * 1.05)

            if direction > 0:
                stop_price = entry_price * (1.0 - stop_pct)
                target_price = entry_price * (1.0 + target_pct)
            else:
                stop_price = entry_price * (1.0 + stop_pct)
                target_price = entry_price * (1.0 - target_pct)

            open_trade = {
                "direction": float(direction),
                "entry_idx": float(i),
                "entry_price": float(entry_price),
                "open_cost": float(open_cost),
                "stop_price": float(stop_price),
                "target_price": float(target_price),
                "high_water": float(high_values[i]),
                "low_water": float(low_values[i]),
            }
            i += 1
            continue

        direction = int(open_trade["direction"])
        entry_idx = int(open_trade["entry_idx"])
        hold = i - entry_idx

        atr_i = abs(float(atr_values[i]))
        trail_pct = max(float(adaptive_min_trail_pct), float(adaptive_trail_atr_mult) * atr_i)
        if direction > 0:
            open_trade["high_water"] = max(float(open_trade["high_water"]), float(high_values[i]))
            trail_stop = float(open_trade["high_water"]) * (1.0 - trail_pct)
            open_trade["stop_price"] = max(float(open_trade["stop_price"]), trail_stop)
        else:
            open_trade["low_water"] = min(float(open_trade["low_water"]), float(low_values[i]))
            trail_stop = float(open_trade["low_water"]) * (1.0 + trail_pct)
            open_trade["stop_price"] = min(float(open_trade["stop_price"]), trail_stop)

        if direction > 0:
            stop_hit = float(low_values[i]) <= float(open_trade["stop_price"])
            target_hit = float(high_values[i]) >= float(open_trade["target_price"])
        else:
            stop_hit = float(high_values[i]) >= float(open_trade["stop_price"])
            target_hit = float(low_values[i]) <= float(open_trade["target_price"])

        exit_ref: Optional[float] = None
        if stop_hit:
            exit_ref = float(open_trade["stop_price"])
        elif target_hit:
            exit_ref = float(open_trade["target_price"])
        elif hold >= int(adaptive_max_hold_bars):
            exit_ref = float(close_values[i])
        elif adaptive_exit_on_opposite:
            p_now = float(p_map.get(i, 0.5))
            if (direction > 0 and p_now < short_th) or (direction < 0 and p_now > long_th):
                exit_ref = float(close_values[i])

        if exit_ref is not None:
            exit_price = exit_ref * (1.0 - slippage if direction > 0 else 1.0 + slippage)
            sp_c, sl_c = dynamic_costs(close_values, i, spread=spread, slippage=slippage)
            close_cost = trading_cost(exit_price, lot_size, sp_c, sl_c, commission)

            move_pnl = (exit_price - float(open_trade["entry_price"])) * direction * lot_size * 100000.0
            trade_net = move_pnl - float(open_trade["open_cost"]) - close_cost
            balance += trade_net

            trade_pnls.append(float(trade_net))
            hold_bars.append(int(max(hold, 1)))
            equity_curve.append(balance)
            open_trade = None

        i += 1

    if open_trade is not None:
        i_exit = int(test_end - 1)
        direction = int(open_trade["direction"])
        entry_idx = int(open_trade["entry_idx"])
        exit_ref = float(close_values[i_exit])
        exit_price = exit_ref * (1.0 - slippage if direction > 0 else 1.0 + slippage)
        sp_c, sl_c = dynamic_costs(close_values, i_exit, spread=spread, slippage=slippage)
        close_cost = trading_cost(exit_price, lot_size, sp_c, sl_c, commission)

        move_pnl = (exit_price - float(open_trade["entry_price"])) * direction * lot_size * 100000.0
        trade_net = move_pnl - float(open_trade["open_cost"]) - close_cost
        balance += trade_net

        trade_pnls.append(float(trade_net))
        hold_bars.append(int(max(i_exit - entry_idx, 1)))
        equity_curve.append(balance)

    return _compute_trade_metrics(
        balance_start=float(initial_balance),
        balance_end=float(balance),
        trade_pnls=trade_pnls,
        hold_bars=hold_bars,
        equity_curve=equity_curve,
        bars_per_day=bars_per_day,
        test_rows=int(test_end - test_start),
    )

def aggregate_trade_folds(folds: List[FoldTradeMetrics], min_trades_per_fold: int) -> Dict[str, float]:
    gross_profit = sum(f.gross_profit for f in folds)
    gross_loss = sum(f.gross_loss for f in folds)
    total_trades = sum(f.total_trades for f in folds)
    pass_count = sum(1 for f in folds if f.pass_flag)

    summary = {
        "folds": len(folds),
        "pass_count": int(pass_count),
        "pass_rate": float(pass_count / len(folds)) if folds else 0.0,
        "total_trades": int(total_trades),
        "min_fold_trades": int(min((f.total_trades for f in folds), default=0)),
        "mean_fold_trades": float(np.mean([f.total_trades for f in folds])) if folds else 0.0,
        "overall_net_profit": float(sum(f.net_profit for f in folds)),
        "overall_profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "mean_fold_return_pct": float(np.mean([f.return_pct for f in folds])) if folds else 0.0,
        "median_fold_return_pct": float(np.median([f.return_pct for f in folds])) if folds else 0.0,
        "worst_fold_drawdown_pct": float(max((f.max_drawdown_pct for f in folds), default=0.0)),
        "mean_trade_holding_bars": float(np.mean([f.avg_bars_in_trade for f in folds])) if folds else 0.0,
        "mean_trade_holding_hours": float(np.mean([f.avg_hours_in_trade for f in folds])) if folds else 0.0,
        "mean_trades_per_month": float(np.mean([f.trades_per_month for f in folds])) if folds else 0.0,
        "min_trades_per_fold_required": int(min_trades_per_fold),
    }
    summary["decision"] = "PASS" if model_pass(summary, min_trades_per_fold=min_trades_per_fold) else "FAIL"
    summary["retrain_recommended"] = summary["decision"] == "FAIL"
    summary["collapse_detected"] = summary["min_fold_trades"] < int(min_trades_per_fold)
    return summary


def evaluate_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    data_csv = find_data_csv(args.data_csv)
    df = load_ohlcv(data_csv)
    folds = make_folds(df, min_train_frac=float(args.min_train_frac), test_frac=float(args.test_frac), max_folds=int(args.max_folds))
    if not folds:
        raise ValueError("No folds generated.")

    features = build_features(df, use_structural=bool(args.use_structural_features))
    y, fwd_ret = make_labels(df, horizon_bars=int(args.horizon_bars), threshold=float(args.label_threshold))

    close_values = df["close"].to_numpy(dtype=float)
    high_values = df["high"].to_numpy(dtype=float)
    low_values = df["low"].to_numpy(dtype=float)
    atr_values = features["atr_norm"].to_numpy(dtype=float)
    bars_per_day = infer_bars_per_day(df.index)

    model = LGBMClassifier(
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_child_samples),
        subsample=float(args.subsample),
        subsample_freq=int(args.subsample_freq),
        colsample_bytree=float(args.colsample_bytree),
        reg_lambda=float(args.reg_lambda),
        n_jobs=-1,
        random_state=int(args.random_state),
        class_weight="balanced",
        verbose=-1,
    )

    direction_folds: List[FoldDirectionMetrics] = []
    fixed_folds: List[FoldTradeMetrics] = []
    adaptive_folds: List[FoldTradeMetrics] = []

    horizon = int(args.horizon_bars)

    for fold_id, train_start, test_start, test_end in folds:
        train_end = max(train_start, test_start - horizon)
        train_idx = np.arange(train_start, train_end, dtype=int)
        test_idx_trade = np.arange(test_start, test_end - 1, dtype=int)
        test_idx_label = np.arange(test_start, max(test_start, test_end - horizon), dtype=int)

        if train_idx.size < 500 or test_idx_trade.size < 50:
            continue

        X_tr = features.iloc[train_idx]
        y_tr = y.iloc[train_idx].to_numpy(dtype=int)
        model.fit(X_tr, y_tr)

        X_te_trade = features.iloc[test_idx_trade]
        p_trade = model.predict_proba(X_te_trade)[:, 1]
        p_map = {int(i): float(p) for i, p in zip(test_idx_trade, p_trade)}

        X_te_label = features.iloc[test_idx_label]
        p_label = model.predict_proba(X_te_label)[:, 1]
        y_true = y.iloc[test_idx_label].to_numpy(dtype=int)
        sign_pred = np.where(p_label >= 0.5, 1.0, -1.0)
        fwd_vals = fwd_ret.iloc[test_idx_label].to_numpy(dtype=float)
        sign_true = np.where(fwd_vals > float(args.label_threshold), 1.0, -1.0)

        acc = float(np.mean(sign_pred == sign_true)) if sign_true.size else float("nan")
        auc = float("nan")
        if y_true.size > 0 and np.unique(y_true).size == 2:
            auc = float(roc_auc_score(y_true, p_label))
        mean_signed_bps = float(np.mean(sign_pred * fwd_vals) * 10000.0) if fwd_vals.size else float("nan")

        direction_folds.append(
            FoldDirectionMetrics(
                fold_id=int(fold_id),
                test_start=str(df.index[test_start]),
                test_end=str(df.index[test_end - 1]),
                samples=int(test_idx_label.size),
                accuracy=float(acc),
                auc=float(auc),
                mean_signed_forward_return_bps=float(mean_signed_bps),
            )
        )

        fixed_metrics = simulate_fixed_exits(
            test_start=int(test_start),
            test_end=int(test_end),
            horizon_bars=int(horizon),
            p_map=p_map,
            close_values=close_values,
            lot_size=float(args.lot_size),
            spread=float(args.spread),
            commission=float(args.commission),
            slippage=float(args.slippage),
            long_th=float(args.long_th),
            short_th=float(args.short_th),
            initial_balance=float(args.initial_balance),
            bars_per_day=bars_per_day,
        )
        fixed_folds.append(
            FoldTradeMetrics(
                fold_id=int(fold_id),
                test_start=str(df.index[test_start]),
                test_end=str(df.index[test_end - 1]),
                total_trades=int(fixed_metrics["total_trades"]),
                win_rate=float(fixed_metrics["win_rate"]),
                profit_factor=float(fixed_metrics["profit_factor"]),
                gross_profit=float(fixed_metrics["gross_profit"]),
                gross_loss=float(fixed_metrics["gross_loss"]),
                net_profit=float(fixed_metrics["net_profit"]),
                avg_trade_pnl=float(fixed_metrics["avg_trade_pnl"]),
                return_pct=float(fixed_metrics["return_pct"]),
                max_drawdown_pct=float(fixed_metrics["max_drawdown_pct"]),
                avg_bars_in_trade=float(fixed_metrics["avg_bars_in_trade"]),
                avg_hours_in_trade=float(fixed_metrics["avg_hours_in_trade"]),
                trades_per_month=float(fixed_metrics["trades_per_month"]),
                pass_flag=fold_pass(fixed_metrics, min_trades_per_fold=int(args.min_trades_per_fold)),
            )
        )

        adaptive_metrics = simulate_adaptive_exits(
            test_start=int(test_start),
            test_end=int(test_end),
            p_map=p_map,
            close_values=close_values,
            high_values=high_values,
            low_values=low_values,
            atr_values=atr_values,
            lot_size=float(args.lot_size),
            spread=float(args.spread),
            commission=float(args.commission),
            slippage=float(args.slippage),
            long_th=float(args.long_th),
            short_th=float(args.short_th),
            initial_balance=float(args.initial_balance),
            bars_per_day=bars_per_day,
            adaptive_stop_atr_mult=float(args.adaptive_stop_atr_mult),
            adaptive_min_stop_pct=float(args.adaptive_min_stop_pct),
            adaptive_rr_base=float(args.adaptive_rr_base),
            adaptive_rr_confidence_mult=float(args.adaptive_rr_confidence_mult),
            adaptive_trail_atr_mult=float(args.adaptive_trail_atr_mult),
            adaptive_min_trail_pct=float(args.adaptive_min_trail_pct),
            adaptive_max_hold_bars=int(args.adaptive_max_hold_bars),
            adaptive_exit_on_opposite=bool(args.adaptive_exit_on_opposite),
        )
        adaptive_folds.append(
            FoldTradeMetrics(
                fold_id=int(fold_id),
                test_start=str(df.index[test_start]),
                test_end=str(df.index[test_end - 1]),
                total_trades=int(adaptive_metrics["total_trades"]),
                win_rate=float(adaptive_metrics["win_rate"]),
                profit_factor=float(adaptive_metrics["profit_factor"]),
                gross_profit=float(adaptive_metrics["gross_profit"]),
                gross_loss=float(adaptive_metrics["gross_loss"]),
                net_profit=float(adaptive_metrics["net_profit"]),
                avg_trade_pnl=float(adaptive_metrics["avg_trade_pnl"]),
                return_pct=float(adaptive_metrics["return_pct"]),
                max_drawdown_pct=float(adaptive_metrics["max_drawdown_pct"]),
                avg_bars_in_trade=float(adaptive_metrics["avg_bars_in_trade"]),
                avg_hours_in_trade=float(adaptive_metrics["avg_hours_in_trade"]),
                trades_per_month=float(adaptive_metrics["trades_per_month"]),
                pass_flag=fold_pass(adaptive_metrics, min_trades_per_fold=int(args.min_trades_per_fold)),
            )
        )

        print(
            f"fold={fold_id} acc={acc:.4f} auc={auc if auc == auc else float('nan'):.3f} "
            f"fixed_net={fixed_metrics['net_profit']:.2f} adaptive_net={adaptive_metrics['net_profit']:.2f}"
        )

    if not direction_folds:
        raise ValueError("No valid folds evaluated.")

    w = np.asarray([f.samples for f in direction_folds], dtype=float)
    accs = np.asarray([f.accuracy for f in direction_folds], dtype=float)
    bps = np.asarray([f.mean_signed_forward_return_bps for f in direction_folds], dtype=float)
    weighted_acc = float(np.average(accs, weights=w))
    weighted_bps = float(np.average(bps, weights=w))
    consistency = float(np.mean(accs >= float(args.edge_min_accuracy)))
    edge_exists = bool(weighted_acc >= float(args.edge_min_accuracy) and consistency >= float(args.edge_min_consistency))

    fixed_summary = aggregate_trade_folds(fixed_folds, min_trades_per_fold=int(args.min_trades_per_fold))
    adaptive_summary = aggregate_trade_folds(adaptive_folds, min_trades_per_fold=int(args.min_trades_per_fold))

    payload: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_csv": str(data_csv),
        "config": {
            "min_train_frac": float(args.min_train_frac),
            "test_frac": float(args.test_frac),
            "max_folds": int(args.max_folds),
            "folds_generated": len(direction_folds),
            "horizon_bars": int(args.horizon_bars),
            "label_threshold": float(args.label_threshold),
            "long_th": float(args.long_th),
            "short_th": float(args.short_th),
            "use_structural_features": bool(args.use_structural_features),
            "edge_min_accuracy": float(args.edge_min_accuracy),
            "edge_min_consistency": float(args.edge_min_consistency),
            "lot_size": float(args.lot_size),
            "spread": float(args.spread),
            "commission": float(args.commission),
            "slippage": float(args.slippage),
            "adaptive_stop_atr_mult": float(args.adaptive_stop_atr_mult),
            "adaptive_min_stop_pct": float(args.adaptive_min_stop_pct),
            "adaptive_rr_base": float(args.adaptive_rr_base),
            "adaptive_rr_confidence_mult": float(args.adaptive_rr_confidence_mult),
            "adaptive_trail_atr_mult": float(args.adaptive_trail_atr_mult),
            "adaptive_min_trail_pct": float(args.adaptive_min_trail_pct),
            "adaptive_max_hold_bars": int(args.adaptive_max_hold_bars),
            "adaptive_exit_on_opposite": bool(args.adaptive_exit_on_opposite),
        },
        "edge_check": {
            "weighted_accuracy": weighted_acc,
            "weighted_mean_signed_forward_return_bps": weighted_bps,
            "fold_consistency": consistency,
            "edge_exists": edge_exists,
            "interpretation": (
                "H1 directional edge confirmed." if edge_exists else "H1 directional edge NOT confirmed under configured threshold."
            ),
        },
        "direction_folds": [asdict(f) for f in direction_folds],
        "fixed_exit": {"summary": fixed_summary, "folds": [asdict(f) for f in fixed_folds]},
        "adaptive_exit": {"summary": adaptive_summary, "folds": [asdict(f) for f in adaptive_folds]},
        "adaptive_vs_fixed": {
            "delta_pass_rate": float(adaptive_summary["pass_rate"] - fixed_summary["pass_rate"]),
            "delta_net_profit": float(adaptive_summary["overall_net_profit"] - fixed_summary["overall_net_profit"]),
            "delta_profit_factor": float(adaptive_summary["overall_profit_factor"] - fixed_summary["overall_profit_factor"]),
            "delta_worst_drawdown_pct": float(adaptive_summary["worst_fold_drawdown_pct"] - fixed_summary["worst_fold_drawdown_pct"]),
            "delta_mean_holding_bars": float(adaptive_summary["mean_trade_holding_bars"] - fixed_summary["mean_trade_holding_bars"]),
            "delta_mean_trades_per_month": float(adaptive_summary["mean_trades_per_month"] - fixed_summary["mean_trades_per_month"]),
        },
    }

    return payload

def to_markdown(payload: Dict[str, object]) -> str:
    edge = payload["edge_check"]
    fixed = payload["fixed_exit"]["summary"]
    adaptive = payload["adaptive_exit"]["summary"]
    delta = payload["adaptive_vs_fixed"]

    lines: List[str] = []
    lines.append("# LGBM Directional + Adaptive Exit Report")
    lines.append("")
    lines.append(f"- Generated: {payload['generated_at']}")
    lines.append(f"- Data: `{payload['data_csv']}`")
    lines.append("")

    lines.append("## Step 1: Directional Edge Check")
    lines.append(f"- Weighted accuracy: {edge['weighted_accuracy']:.4f}")
    lines.append(f"- Weighted mean signed forward return: {edge['weighted_mean_signed_forward_return_bps']:.3f} bps")
    lines.append(f"- Fold consistency (acc >= threshold): {edge['fold_consistency']:.2f}")
    lines.append(f"- Edge confirmed: **{edge['edge_exists']}**")
    lines.append(f"- Interpretation: {edge['interpretation']}")
    lines.append("")

    lines.append("## Step 2: Exit Layer Comparison")
    lines.append("| Mode | Decision | Pass Rate | Net Profit | PF | Worst DD % | Mean Hold Bars | Mean Trades/Month |")
    lines.append("| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| Fixed Horizon | {fixed['decision']} | {fixed['pass_rate']*100:.1f}% | {fixed['overall_net_profit']:.2f} | {fixed['overall_profit_factor']:.3f} | {fixed['worst_fold_drawdown_pct']:.2f} | {fixed['mean_trade_holding_bars']:.2f} | {fixed['mean_trades_per_month']:.2f} |"
    )
    lines.append(
        f"| Adaptive Exits | {adaptive['decision']} | {adaptive['pass_rate']*100:.1f}% | {adaptive['overall_net_profit']:.2f} | {adaptive['overall_profit_factor']:.3f} | {adaptive['worst_fold_drawdown_pct']:.2f} | {adaptive['mean_trade_holding_bars']:.2f} | {adaptive['mean_trades_per_month']:.2f} |"
    )
    lines.append("")
    lines.append(
        f"- Delta (adaptive - fixed): pass_rate={delta['delta_pass_rate']*100:.1f} pts, net={delta['delta_net_profit']:.2f}, "
        f"PF={delta['delta_profit_factor']:.3f}, worst_dd={delta['delta_worst_drawdown_pct']:.2f}, "
        f"hold_bars={delta['delta_mean_holding_bars']:.2f}, trades/month={delta['delta_mean_trades_per_month']:.2f}"
    )
    lines.append("")

    lines.append("## Direction Folds")
    lines.append("| Fold | Start | End | Samples | Accuracy | AUC | Mean Signed Return (bps) |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for f in payload["direction_folds"]:
        lines.append(
            f"| {f['fold_id']} | {f['test_start']} | {f['test_end']} | {f['samples']} | {f['accuracy']:.4f} | "
            f"{f['auc'] if f['auc'] == f['auc'] else float('nan'):.3f} | {f['mean_signed_forward_return_bps']:.3f} |"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    payload = evaluate_pipeline(args)

    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{args.out_prefix}_{ts}.json"
    md_path = out_dir / f"{args.out_prefix}_{ts}.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(payload), encoding="utf-8")

    print(f"report_json={json_path}")
    print(f"report_md={md_path}")
    edge = payload["edge_check"]
    print(
        "edge "
        f"exists={edge['edge_exists']} weighted_accuracy={edge['weighted_accuracy']:.4f} "
        f"consistency={edge['fold_consistency']:.2f} signed_bps={edge['weighted_mean_signed_forward_return_bps']:.3f}"
    )
    fixed = payload["fixed_exit"]["summary"]
    adaptive = payload["adaptive_exit"]["summary"]
    print(
        "fixed "
        f"decision={fixed['decision']} pass_rate={fixed['pass_rate']*100:.1f}% "
        f"net={fixed['overall_net_profit']:.2f} pf={fixed['overall_profit_factor']:.3f} dd={fixed['worst_fold_drawdown_pct']:.2f}"
    )
    print(
        "adaptive "
        f"decision={adaptive['decision']} pass_rate={adaptive['pass_rate']*100:.1f}% "
        f"net={adaptive['overall_net_profit']:.2f} pf={adaptive['overall_profit_factor']:.3f} dd={adaptive['worst_fold_drawdown_pct']:.2f}"
    )


if __name__ == "__main__":
    main()
