#!/usr/bin/env python
"""
Walk-forward volatility-event execution backtest.

Pipeline per fold:
1) Train LightGBM event model on train slice only:
      y=1 if |close[t+H]-close[t]| >= X_pips else 0
2) Predict event probability on test slice only.
3) Convert high-probability bars into execution signals and run:
      - conservative breakout, and/or
      - aggressive expansion
4) Aggregate fold metrics and fold-consistency scores.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


ROOT = Path(__file__).resolve().parents[1]
PIP = 0.0001
CONTRACT = 100_000


@dataclass
class Costs:
    spread: float
    slippage: float
    commission: float


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int
    entry: float
    exit: float
    lot: float
    pnl_usd: float
    bars_held: int
    reason: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_csv_strs(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def parse_hour_windows(raw: str) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    text = str(raw).strip()
    if not text:
        return windows
    for token in text.split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a.strip())
            end = int(b.strip())
        else:
            start = int(part)
            end = int(part)
        if not (0 <= start <= 23 and 0 <= end <= 23):
            raise ValueError(f"Invalid hour window '{part}'. Hours must be in 0..23.")
        windows.append((start, end))
    return windows


def matching_hour_window_index(ts: pd.Timestamp, windows: Sequence[Tuple[int, int]]) -> int:
    hour = int(ts.hour)
    for i, (start, end) in enumerate(windows):
        if start <= end and start <= hour <= end:
            return i
        if start > end and (hour >= start or hour <= end):
            return i
    return -1


def in_hour_windows(ts: pd.Timestamp, windows: Sequence[Tuple[int, int]]) -> bool:
    if not windows:
        return True
    return matching_hour_window_index(ts, windows) >= 0


def session_bucket_key(ts: pd.Timestamp, session_filter: str, hour_windows: Sequence[Tuple[int, int]]) -> str:
    day = ts.strftime("%Y-%m-%d")
    if hour_windows:
        win_idx = matching_hour_window_index(ts, hour_windows)
        return f"{day}|win{win_idx}"
    return f"{day}|{session_filter}"


def rolling_quantile_threshold(series: pd.Series, window: int, quantile: float) -> pd.Series:
    w = max(int(window), 10)
    q = float(quantile)
    min_periods = max(20, w // 4)
    # shift(1) ensures threshold at t only uses information available before t.
    return series.shift(1).rolling(window=w, min_periods=min_periods).quantile(q)


def load_ohlcv_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError("CSV must contain 'time' column.")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.sort_values("time").drop_duplicates(subset=["time"]).set_index("time")

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df[["open", "high", "low", "close", "volume"]].copy()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
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
    tr = (high - low).astype(float)
    atr14 = tr.rolling(14, min_periods=1).mean()
    atr56 = tr.rolling(56, min_periods=1).mean()
    feat["atr_norm"] = atr14 / close_safe
    feat["vol_z"] = (volume - vol_mean) / vol_std
    feat["range_compression"] = atr14 / atr56.replace(0.0, np.nan)

    feat = feat.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return feat.astype(np.float32)


def realized_abs_move_pips(close: pd.Series, horizon: int) -> pd.Series:
    return (close.shift(-horizon) - close).abs() / PIP


def make_event_label(close: pd.Series, horizon: int, threshold_pips: float) -> pd.Series:
    abs_move = realized_abs_move_pips(close, horizon)
    out = pd.Series(np.nan, index=close.index, dtype=float)
    valid = abs_move.notna()
    out.loc[valid] = (abs_move.loc[valid] >= float(threshold_pips)).astype(float)
    return out


def walk_forward_splits(
    index: pd.DatetimeIndex,
    train_years: int,
    test_months: int,
    step_months: int,
    max_folds: int = 10,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = pd.DatetimeIndex(index)
    start = idx.min()
    end = idx.max()

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    train_start = start
    train_end = train_start + pd.DateOffset(years=int(train_years))
    test_end = train_end + pd.DateOffset(months=int(test_months))

    while test_end <= end and len(splits) < int(max_folds):
        tr_mask = (idx >= train_start) & (idx < train_end)
        te_mask = (idx >= train_end) & (idx < test_end)
        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(te_mask)[0]
        if tr_idx.size > 0 and te_idx.size > 0:
            splits.append((tr_idx, te_idx))

        train_start = train_start + pd.DateOffset(months=int(step_months))
        train_end = train_start + pd.DateOffset(years=int(train_years))
        test_end = train_end + pd.DateOffset(months=int(test_months))

    return splits


def parse_utc_ts(raw: str) -> pd.Timestamp:
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def lockbox_split(
    index: pd.DatetimeIndex,
    train_end: str,
    test_start: str,
    test_end: str,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = pd.DatetimeIndex(index)
    tr_end = parse_utc_ts(train_end)
    te_start = parse_utc_ts(test_start)
    te_end = parse_utc_ts(test_end)
    if te_end < te_start:
        raise ValueError("lockbox test_end must be >= test_start")
    tr_mask = idx <= tr_end
    te_mask = (idx >= te_start) & (idx <= te_end)
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]
    if tr_idx.size == 0 or te_idx.size == 0:
        raise ValueError("Lockbox split produced empty train or test indices.")
    return [(tr_idx, te_idx)]


def in_session(ts: pd.Timestamp, mode: str) -> bool:
    if mode == "off":
        return True
    hour = int(ts.hour)
    if mode == "london_only":
        return 7 <= hour <= 16
    if mode == "ny_only":
        return 12 <= hour <= 21
    if mode == "london_ny":
        return 7 <= hour <= 21
    return True


def regime_ok(feat_row: pd.Series, mode: str, trend_min: float) -> bool:
    if mode == "off":
        return True
    trend_strength = abs(float(feat_row.get("ma_fast_slow", 0.0)))
    if mode == "trend_only":
        return trend_strength >= float(trend_min)
    if mode == "range_only":
        return trend_strength < float(trend_min)
    if mode == "trend_or_range":
        return True
    return True


def cost_pips_est(costs: Costs, price_ref: float) -> float:
    rt_price = costs.spread + 2.0 * costs.slippage + 2.0 * costs.commission * float(price_ref)
    return float(rt_price / PIP)


def pnl_usd_from_price_move(entry: float, exit_: float, side: int, lot: float) -> float:
    return (float(exit_) - float(entry)) * int(side) * float(lot) * CONTRACT


def apply_entry_price(side: int, raw_price: float, costs: Costs) -> float:
    half_spread = 0.5 * float(costs.spread)
    if int(side) == 1:
        return float(raw_price) + half_spread + float(costs.slippage)
    return float(raw_price) - half_spread - float(costs.slippage)


def apply_exit_price(side: int, raw_price: float, costs: Costs) -> float:
    half_spread = 0.5 * float(costs.spread)
    if int(side) == 1:
        return float(raw_price) - half_spread - float(costs.slippage)
    return float(raw_price) + half_spread + float(costs.slippage)


def commission_usd(price: float, lot: float, costs: Costs) -> float:
    notional = float(price) * float(lot) * CONTRACT
    return notional * float(costs.commission)


def train_predict_fold(
    X: pd.DataFrame,
    y: np.ndarray,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    lgbm_device: str,
    runtime_state: Dict[str, Any],
) -> Tuple[np.ndarray, str]:
    y_train = y[tr_idx]
    classes = np.unique(y_train)
    if classes.size < 2:
        base = float(classes[0]) if classes.size == 1 else 0.5
        return np.full(te_idx.shape[0], base, dtype=float), "cpu"

    if lgbm_device == "cpu":
        candidates = ["cpu"]
    elif runtime_state.get("gpu_disabled", False):
        candidates = ["cpu"]
    else:
        candidates = ["gpu", "cpu"]

    last_error: Exception | None = None
    for device in candidates:
        model = LGBMClassifier(
            n_estimators=900,
            learning_rate=0.04,
            max_depth=6,
            num_leaves=64,
            min_child_samples=200,
            subsample=0.85,
            subsample_freq=5,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
            verbose=-1,
            device_type=device,
        )
        try:
            model.fit(X.iloc[tr_idx], y_train)
            if device == "gpu":
                runtime_state["gpu_success_folds"] = int(runtime_state.get("gpu_success_folds", 0)) + 1
            else:
                runtime_state["cpu_used_folds"] = int(runtime_state.get("cpu_used_folds", 0)) + 1
                if str(lgbm_device) in {"auto", "gpu"}:
                    runtime_state["cpu_fallback_folds"] = int(runtime_state.get("cpu_fallback_folds", 0)) + 1
            return model.predict_proba(X.iloc[te_idx])[:, 1], device
        except Exception as exc:
            last_error = exc
            if device == "gpu":
                runtime_state["gpu_disabled"] = True
                runtime_state["gpu_error"] = str(exc)
                continue
            raise

    raise RuntimeError(f"LightGBM training failed on all device candidates: {last_error}")


def compute_event_quality(
    y_event: np.ndarray,
    p_event: np.ndarray,
    prob_th: float,
) -> Dict[str, float]:
    pred = p_event >= float(prob_th)
    valid = ~np.isnan(y_event)
    if int(np.sum(valid)) == 0:
        return {
            "event_prevalence": float("nan"),
            "predicted_event_rate": float("nan"),
            "event_precision": float("nan"),
            "event_recall": float("nan"),
        }

    yv = y_event[valid].astype(int)
    pv = pred[valid]
    prevalence = float(np.mean(yv == 1))
    pred_rate = float(np.mean(pv))
    tp = int(np.sum((pv == 1) & (yv == 1)))
    precision = float(tp / int(np.sum(pv))) if int(np.sum(pv)) > 0 else float("nan")
    recall = float(tp / int(np.sum(yv == 1))) if int(np.sum(yv == 1)) > 0 else float("nan")
    return {
        "event_prevalence": prevalence,
        "predicted_event_rate": pred_rate,
        "event_precision": precision,
        "event_recall": recall,
    }


def simulate_strategy_on_test(
    df: pd.DataFrame,
    feat: pd.DataFrame,
    p_event_full: np.ndarray,
    test_idx: np.ndarray,
    prob_th: float,
    mode: str,
    lookback: int,
    buffer_pips: float,
    sl_pips: float,
    tp_pips: float,
    sl_atr: float,
    tp_atr: float,
    time_stop_bars: int,
    session_filter: str,
    hour_windows: Sequence[Tuple[int, int]],
    max_trades_per_session: int,
    regime_filter: str,
    trend_min: float,
    require_close_confirm: bool,
    compression_threshold: np.ndarray | None,
    atr_norm_min_threshold: np.ndarray | None,
    atr_norm_max_threshold: np.ndarray | None,
    lot: float,
    costs: Costs,
) -> Dict[str, Any]:
    n = len(df)
    if p_event_full.shape[0] != n or len(feat) != n:
        raise ValueError("Length mismatch in simulate_strategy_on_test.")
    if test_idx.size == 0:
        return {
            "mode": mode,
            "trades": 0,
            "net_usd": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "avg_trade_usd": 0.0,
            "max_drawdown_usd": 0.0,
            "avg_bars_in_trade": 0.0,
            "trade_log_sample": [],
        }

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    idx = df.index
    atr = (df["high"] - df["low"]).rolling(14, min_periods=1).mean().to_numpy(dtype=float)
    test_set = set(int(x) for x in test_idx.tolist())
    test_start = int(test_idx[0])
    test_end = int(test_idx[-1])

    trades: List[Trade] = []
    in_pos = False
    side = 0
    entry_px = 0.0
    entry_i = -1
    entry_time = None
    buffer = float(buffer_pips) * PIP
    trades_by_session: Dict[str, int] = {}

    for t in range(max(int(lookback), test_start), test_end + 1):
        if t not in test_set:
            continue

        if in_pos:
            bars_held = int(t - entry_i)
            sl_dist = float(sl_pips) * PIP if float(sl_pips) > 0.0 else float(sl_atr) * float(atr[entry_i])
            tp_dist = float(tp_pips) * PIP if float(tp_pips) > 0.0 else float(tp_atr) * float(atr[entry_i])
            if sl_dist <= 0.0:
                sl_dist = 0.5 * buffer if buffer > 0 else 5.0 * PIP
            if tp_dist <= 0.0:
                tp_dist = 0.8 * buffer if buffer > 0 else 8.0 * PIP

            if side == 1:
                sl_level = entry_px - sl_dist
                tp_level = entry_px + tp_dist
                hit_sl = l[t] <= sl_level
                hit_tp = h[t] >= tp_level
            else:
                sl_level = entry_px + sl_dist
                tp_level = entry_px - tp_dist
                hit_sl = h[t] >= sl_level
                hit_tp = l[t] <= tp_level

            reason = None
            raw_exit = None
            if hit_sl and hit_tp:
                reason = "both_hit_worst"
                raw_exit = sl_level
            elif hit_tp:
                reason = "tp"
                raw_exit = tp_level
            elif hit_sl:
                reason = "sl"
                raw_exit = sl_level
            elif bars_held >= int(time_stop_bars):
                reason = "time"
                raw_exit = c[t]
            elif t == test_end:
                reason = "fold_eod"
                raw_exit = c[t]

            if reason is not None:
                exit_px = apply_exit_price(side, float(raw_exit), costs)
                pnl = pnl_usd_from_price_move(entry_px, exit_px, side, lot)
                pnl -= commission_usd(entry_px, lot, costs)
                pnl -= commission_usd(exit_px, lot, costs)
                trades.append(
                    Trade(
                        entry_time=entry_time,
                        exit_time=idx[t],
                        side=side,
                        entry=float(entry_px),
                        exit=float(exit_px),
                        lot=float(lot),
                        pnl_usd=float(pnl),
                        bars_held=bars_held,
                        reason=str(reason),
                    )
                )
                in_pos = False
                side = 0
                entry_px = 0.0
                entry_i = -1
                entry_time = None

        if in_pos:
            continue
        if t >= test_end:
            continue

        p = float(p_event_full[t]) if np.isfinite(p_event_full[t]) else float("nan")
        if not np.isfinite(p) or p < float(prob_th):
            continue
        if not in_session(idx[t], session_filter):
            continue
        if not in_hour_windows(idx[t], hour_windows):
            continue
        sess_key = session_bucket_key(idx[t], session_filter, hour_windows)
        if int(max_trades_per_session) > 0 and int(trades_by_session.get(sess_key, 0)) >= int(max_trades_per_session):
            continue
        if not regime_ok(feat.iloc[t], regime_filter, trend_min):
            continue
        if compression_threshold is not None:
            comp_val = float(feat.iloc[t].get("range_compression", np.nan))
            comp_th = float(compression_threshold[t])
            if not np.isfinite(comp_val) or not np.isfinite(comp_th) or comp_val > comp_th:
                continue
        if atr_norm_min_threshold is not None:
            atr_val = float(feat.iloc[t].get("atr_norm", np.nan))
            atr_min = float(atr_norm_min_threshold[t])
            if not np.isfinite(atr_val) or not np.isfinite(atr_min) or atr_val < atr_min:
                continue
        if atr_norm_max_threshold is not None:
            atr_val = float(feat.iloc[t].get("atr_norm", np.nan))
            atr_max = float(atr_norm_max_threshold[t])
            if not np.isfinite(atr_val) or not np.isfinite(atr_max) or atr_val > atr_max:
                continue

        recent_high = float(np.max(h[t - int(lookback) : t]))
        recent_low = float(np.min(l[t - int(lookback) : t]))
        buy_stop = recent_high + buffer
        sell_stop = recent_low - buffer
        t1 = t + 1
        if t1 not in test_set:
            continue
        if bool(require_close_confirm):
            buy_hit = c[t1] >= buy_stop
            sell_hit = c[t1] <= sell_stop
        else:
            buy_hit = h[t1] >= buy_stop
            sell_hit = l[t1] <= sell_stop

        if mode == "conservative":
            if buy_hit and sell_hit:
                continue
            if not buy_hit and not sell_hit:
                continue
            chosen_side = 1 if buy_hit else -1
            raw_entry = buy_stop if chosen_side == 1 else sell_stop
        elif mode == "aggressive":
            if not buy_hit and not sell_hit:
                continue
            if buy_hit and sell_hit:
                d_buy = abs(float(o[t1]) - buy_stop)
                d_sell = abs(float(o[t1]) - sell_stop)
                chosen_side = 1 if d_buy <= d_sell else -1
                raw_entry = buy_stop if chosen_side == 1 else sell_stop
            else:
                chosen_side = 1 if buy_hit else -1
                raw_entry = buy_stop if chosen_side == 1 else sell_stop
        else:
            raise ValueError("mode must be 'conservative' or 'aggressive'")

        if mode == "conservative":
            trend = float(feat.iloc[t].get("ma_fast_slow", 0.0))
            if abs(trend) >= float(trend_min):
                if trend > 0 and chosen_side == -1:
                    continue
                if trend < 0 and chosen_side == 1:
                    continue

        entry_px = apply_entry_price(chosen_side, float(raw_entry), costs)
        in_pos = True
        side = int(chosen_side)
        entry_i = int(t1)
        entry_time = idx[t1]
        trades_by_session[sess_key] = int(trades_by_session.get(sess_key, 0)) + 1

    pnl = np.asarray([tr.pnl_usd for tr in trades], dtype=float)
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    gross_profit = float(np.sum(wins)) if wins.size else 0.0
    gross_loss = float(-np.sum(losses)) if losses.size else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0.0 else (float("inf") if gross_profit > 0.0 else 0.0)
    equity = np.cumsum(pnl) if pnl.size else np.array([0.0], dtype=float)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(np.max(dd)) if dd.size else 0.0
    avg_bars = float(np.mean([tr.bars_held for tr in trades])) if trades else 0.0

    return {
        "mode": mode,
        "trades": int(len(trades)),
        "net_usd": float(np.sum(pnl)) if pnl.size else 0.0,
        "profit_factor": profit_factor,
        "win_rate": float(np.mean(pnl > 0.0)) if pnl.size else 0.0,
        "avg_trade_usd": float(np.mean(pnl)) if pnl.size else 0.0,
        "max_drawdown_usd": max_dd,
        "avg_bars_in_trade": avg_bars,
        "trade_log_sample": [
            {
                "entry": str(tr.entry_time),
                "exit": str(tr.exit_time),
                "side": "LONG" if tr.side == 1 else "SHORT",
                "pnl_usd": float(tr.pnl_usd),
                "bars": int(tr.bars_held),
                "reason": tr.reason,
            }
            for tr in trades[:10]
        ],
    }


def finite_stat_list(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def fold_consistency(values: Sequence[float], positive_if_gt: float = 0.0) -> Dict[str, float]:
    arr = finite_stat_list(values)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "worst": float("nan"),
            "positive_fold_rate": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "worst": float(np.min(arr)),
        "positive_fold_rate": float(np.mean(arr > float(positive_if_gt))),
    }


def to_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Volatility Breakout Walk-Forward Backtest")
    lines.append("")
    lines.append(f"- Generated (UTC): `{summary['generated_utc']}`")
    lines.append(f"- Data: `{summary['data_csv']}`")
    lines.append(f"- Range: `{summary['data_range'][0]}` -> `{summary['data_range'][1]}`")
    lines.append(
        f"- Event: H={summary['event']['horizon_bars']} bars, "
        f"X={summary['event']['move_threshold_pips']} pips, prob_th={summary['event']['prob_th']}"
    )
    lines.append(
        f"- Walk-forward: train {summary['walk_forward']['train_years']}y, "
        f"test {summary['walk_forward']['test_months']}m, "
        f"step {summary['walk_forward']['step_months']}m, "
        f"folds={summary['walk_forward']['folds']}"
    )
    if summary["walk_forward"]["mode"] == "lockbox":
        lines.append(
            f"- Lockbox: train_end={summary['walk_forward']['lockbox_train_end']}, "
            f"test_start={summary['walk_forward']['lockbox_test_start']}, "
            f"test_end={summary['walk_forward']['lockbox_test_end']}"
        )
    lines.append(
        f"- Filters: session={summary['filters']['session']}, "
        f"hours='{summary['filters']['hour_windows']}', "
        f"max_trades_per_session={summary['filters']['max_trades_per_session']}, "
        f"regime={summary['filters']['regime']}, trend_min={summary['filters']['trend_min']}"
    )
    lines.append(
        f"- Expansion gates: compression_q<={summary['filters']['compression_max_quantile']} "
        f"(window={summary['filters']['compression_window']}), "
        f"atr_norm_band=[{summary['filters']['atr_norm_min_quantile']}, {summary['filters']['atr_norm_max_quantile']}] "
        f"(window={summary['filters']['atr_window']}), "
        f"close_confirm={summary['filters']['require_close_confirm']}"
    )
    lines.append(
        f"- Execution: lookback={summary['execution']['lookback']}, "
        f"buffer_pips={summary['execution']['buffer_pips']}, "
        f"SL={summary['execution']['sl_pips']} pips (atr_fallback={summary['execution']['sl_atr']}), "
        f"TP={summary['execution']['tp_pips']} pips (atr_fallback={summary['execution']['tp_atr']}), "
        f"time_stop={summary['execution']['time_stop_bars']} bars"
    )
    lines.append(
        f"- Costs: spread={summary['costs']['spread']}, "
        f"slippage={summary['costs']['slippage']}, "
        f"commission={summary['costs']['commission']}, "
        f"est RT cost ~ {summary['costs']['rt_cost_est_pips']:.2f} pips"
    )
    lines.append(
        f"- Model runtime: requested_device={summary['model_runtime']['requested_device']}, "
        f"gpu_success_folds={summary['model_runtime']['gpu_success_folds']}, "
        f"cpu_used_folds={summary['model_runtime']['cpu_used_folds']}, "
        f"cpu_fallback_folds={summary['model_runtime']['cpu_fallback_folds']}"
    )
    if summary["model_runtime"]["gpu_error"]:
        lines.append(f"- GPU fallback reason: `{summary['model_runtime']['gpu_error']}`")
    lines.append("")

    for mode, block in summary["modes"].items():
        agg = block["agg"]
        lines.append(f"## Mode: {mode}")
        lines.append("")
        lines.append(f"- Net USD sum: `{agg['net_usd']['sum']:.2f}`")
        lines.append(f"- Net USD mean/fold: `{agg['net_usd']['mean']:.2f}`")
        lines.append(f"- Net USD std/fold: `{agg['net_usd']['std']:.2f}`")
        lines.append(f"- Positive fold rate (net>0): `{agg['net_usd']['positive_fold_rate']:.2f}`")
        lines.append(f"- Worst fold net USD: `{agg['net_usd']['worst']:.2f}`")
        lines.append(f"- Profit factor mean: `{agg['profit_factor']['mean']:.3f}`")
        lines.append(f"- Profit factor std: `{agg['profit_factor']['std']:.3f}`")
        lines.append(f"- Worst max DD USD: `{agg['max_drawdown_usd']['worst']:.2f}`")
        lines.append(f"- Trades total: `{agg['trades']['sum']}`")
        lines.append(f"- Trades mean/fold: `{agg['trades']['mean_per_fold']:.1f}`")
        lines.append("")
        lines.append("| Fold | Device | Test Start | Test End | Prev | Pred Rate | Precision | Recall | Trades | Net USD | PF | Max DD | Avg Hold Bars |")
        lines.append("| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in block["per_fold"]:
            lines.append(
                f"| {row['fold']} | {row['model_device']} | {row['test_start']} | {row['test_end']} | "
                f"{row['event_prevalence'] if row['event_prevalence'] == row['event_prevalence'] else float('nan'):.4f} | "
                f"{row['predicted_event_rate'] if row['predicted_event_rate'] == row['predicted_event_rate'] else float('nan'):.4f} | "
                f"{row['event_precision'] if row['event_precision'] == row['event_precision'] else float('nan'):.4f} | "
                f"{row['event_recall'] if row['event_recall'] == row['event_recall'] else float('nan'):.4f} | "
                f"{row['trades']} | {row['net_usd']:.2f} | {row['profit_factor']:.3f} | "
                f"{row['max_drawdown_usd']:.2f} | {row['avg_bars_in_trade']:.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--lockbox-train-end", default="", help="UTC timestamp/date inclusive train end for lockbox mode.")
    parser.add_argument("--lockbox-test-start", default="", help="UTC timestamp/date inclusive test start for lockbox mode.")
    parser.add_argument("--lockbox-test-end", default="", help="UTC timestamp/date inclusive test end for lockbox mode.")
    parser.add_argument("--horizon-bars", type=int, default=6)
    parser.add_argument("--move-threshold-pips", type=float, default=40.0)
    parser.add_argument("--prob-th", type=float, default=0.62)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--step-months", type=int, default=6)
    parser.add_argument("--max-folds", type=int, default=10)
    parser.add_argument("--modes", default="conservative,aggressive")
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--buffer-pips", type=float, default=1.0)
    parser.add_argument("--sl-pips", type=float, default=8.0, help="Fixed stop-loss distance in pips. Set <=0 to use ATR stop.")
    parser.add_argument("--tp-pips", type=float, default=12.0, help="Fixed take-profit distance in pips. Set <=0 to use ATR target.")
    parser.add_argument("--sl-atr", type=float, default=1.0)
    parser.add_argument("--tp-atr", type=float, default=1.6)
    parser.add_argument("--time-stop-bars", type=int, default=6)
    parser.add_argument("--session-filter", default="london_ny", choices=["off", "london_only", "ny_only", "london_ny"])
    parser.add_argument("--hour-windows", default="", help="Optional UTC windows, e.g. '7-10,12-16'. Applied in addition to session filter.")
    parser.add_argument(
        "--max-trades-per-session",
        type=int,
        default=0,
        help="Cap trades per session bucket. 0 disables cap. With hour windows, bucket=day+window.",
    )
    parser.add_argument("--regime-filter", default="trend_or_range", choices=["off", "trend_only", "range_only", "trend_or_range"])
    parser.add_argument("--trend-min", type=float, default=0.00010)
    parser.add_argument(
        "--compression-max-quantile",
        type=float,
        default=1.0,
        help="Only allow entries when range_compression <= rolling quantile threshold. Set <1.0 to enable (e.g., 0.35).",
    )
    parser.add_argument("--compression-window", type=int, default=720, help="Rolling window bars for range_compression quantile.")
    parser.add_argument(
        "--atr-norm-min-quantile",
        type=float,
        default=0.0,
        help="Minimum atr_norm rolling quantile threshold for ATR band. Set >0 to enable lower bound.",
    )
    parser.add_argument(
        "--atr-norm-max-quantile",
        type=float,
        default=1.0,
        help="Maximum atr_norm rolling quantile threshold for ATR band. Set <1.0 to enable upper bound.",
    )
    parser.add_argument("--atr-window", type=int, default=720, help="Rolling window bars for atr_norm quantile.")
    parser.add_argument(
        "--require-close-confirm",
        action="store_true",
        help="Require candle close beyond breakout level (instead of high/low touch).",
    )
    parser.add_argument("--lot", type=float, default=0.05)
    parser.add_argument("--spread", type=float, default=0.0002)
    parser.add_argument("--slippage", type=float, default=0.00005)
    parser.add_argument("--commission", type=float, default=0.0001)
    parser.add_argument(
        "--lgbm-device",
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="LightGBM device selection. auto/gpu try GPU first then fallback to CPU on error.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.data_csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    modes = parse_csv_strs(args.modes)
    for mode in modes:
        if mode not in {"conservative", "aggressive"}:
            raise ValueError("Invalid mode. Use conservative and/or aggressive.")
    hour_windows = parse_hour_windows(args.hour_windows)

    df = load_ohlcv_csv(csv_path)
    feat = build_features(df)
    compression_q = float(args.compression_max_quantile)
    atr_q_min = float(args.atr_norm_min_quantile)
    atr_q_max = float(args.atr_norm_max_quantile)
    if not (0.0 < compression_q <= 1.0):
        raise ValueError("--compression-max-quantile must be in (0, 1].")
    if not (0.0 <= atr_q_min <= 1.0):
        raise ValueError("--atr-norm-min-quantile must be in [0, 1].")
    if not (0.0 < atr_q_max <= 1.0):
        raise ValueError("--atr-norm-max-quantile must be in (0, 1].")
    if atr_q_min > atr_q_max:
        raise ValueError("--atr-norm-min-quantile cannot be greater than --atr-norm-max-quantile.")
    compression_threshold = (
        rolling_quantile_threshold(
            feat["range_compression"].astype(float),
            window=int(args.compression_window),
            quantile=compression_q,
        ).to_numpy(dtype=float)
        if compression_q < 1.0
        else None
    )
    atr_norm_min_threshold = (
        rolling_quantile_threshold(
            feat["atr_norm"].astype(float),
            window=int(args.atr_window),
            quantile=atr_q_min,
        ).to_numpy(dtype=float)
        if atr_q_min > 0.0
        else None
    )
    atr_norm_max_threshold = (
        rolling_quantile_threshold(
            feat["atr_norm"].astype(float),
            window=int(args.atr_window),
            quantile=atr_q_max,
        ).to_numpy(dtype=float)
        if atr_q_max < 1.0
        else None
    )
    horizon = int(args.horizon_bars)
    threshold_pips = float(args.move_threshold_pips)
    y_event = make_event_label(df["close"], horizon=horizon, threshold_pips=threshold_pips).to_numpy(dtype=float)
    y_event_int = np.where(np.isnan(y_event), 0, y_event).astype(int)

    has_lockbox = bool(args.lockbox_train_end or args.lockbox_test_start or args.lockbox_test_end)
    if has_lockbox:
        if not (args.lockbox_train_end and args.lockbox_test_start and args.lockbox_test_end):
            raise ValueError(
                "Lockbox mode requires all of --lockbox-train-end, --lockbox-test-start, --lockbox-test-end."
            )
        splits = lockbox_split(
            df.index,
            train_end=str(args.lockbox_train_end),
            test_start=str(args.lockbox_test_start),
            test_end=str(args.lockbox_test_end),
        )
    else:
        splits = walk_forward_splits(
            df.index,
            train_years=int(args.train_years),
            test_months=int(args.test_months),
            step_months=int(args.step_months),
            max_folds=int(args.max_folds),
        )
    if not splits:
        raise ValueError("No walk-forward splits produced. Adjust calendar args.")

    costs = Costs(
        spread=float(args.spread),
        slippage=float(args.slippage),
        commission=float(args.commission),
    )
    price_ref = float(np.median(df["close"].to_numpy(dtype=float)))
    rt_cost_est_pips = cost_pips_est(costs, price_ref)

    fold_results: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in modes}
    runtime_state: Dict[str, Any] = {
        "requested_device": str(args.lgbm_device),
        "gpu_disabled": False,
        "gpu_error": None,
        "gpu_success_folds": 0,
        "cpu_used_folds": 0,
        "cpu_fallback_folds": 0,
    }
    for fold_num, (tr_idx_raw, te_idx) in enumerate(splits, start=1):
        # Strict leakage guard:
        # training labels need close[t+horizon], so remove train rows
        # whose label window crosses into the test span.
        tr_cutoff = int(te_idx[0]) - int(horizon)
        tr_idx = tr_idx_raw[tr_idx_raw < tr_cutoff]
        tr_idx = tr_idx[~np.isnan(y_event[tr_idx])]
        if tr_idx.size < 1000:
            print(f"fold {fold_num:02d}/{len(splits)} skipped (insufficient strict-train rows)")
            continue

        p_te, fold_device = train_predict_fold(
            feat,
            y_event_int,
            tr_idx,
            te_idx,
            lgbm_device=str(args.lgbm_device),
            runtime_state=runtime_state,
        )
        p_event_full = np.full(len(df), np.nan, dtype=float)
        p_event_full[te_idx] = p_te
        y_te = y_event[te_idx]
        event_quality = compute_event_quality(y_te, p_te, prob_th=float(args.prob_th))

        for mode in modes:
            res = simulate_strategy_on_test(
                df=df,
                feat=feat,
                p_event_full=p_event_full,
                test_idx=te_idx,
                prob_th=float(args.prob_th),
                mode=mode,
                lookback=int(args.lookback),
                buffer_pips=float(args.buffer_pips),
                sl_pips=float(args.sl_pips),
                tp_pips=float(args.tp_pips),
                sl_atr=float(args.sl_atr),
                tp_atr=float(args.tp_atr),
                time_stop_bars=int(args.time_stop_bars),
                session_filter=str(args.session_filter),
                hour_windows=hour_windows,
                max_trades_per_session=int(args.max_trades_per_session),
                regime_filter=str(args.regime_filter),
                trend_min=float(args.trend_min),
                require_close_confirm=bool(args.require_close_confirm),
                compression_threshold=compression_threshold,
                atr_norm_min_threshold=atr_norm_min_threshold,
                atr_norm_max_threshold=atr_norm_max_threshold,
                lot=float(args.lot),
                costs=costs,
            )
            res.update(
                {
                    "fold": int(fold_num),
                    "test_start": str(df.index[int(te_idx[0])]),
                    "test_end": str(df.index[int(te_idx[-1])]),
                    "model_device": str(fold_device),
                    **event_quality,
                }
            )
            fold_results[mode].append(res)

        print(f"fold {fold_num:02d}/{len(splits)} done")

    summary: Dict[str, Any] = {
        "generated_utc": utc_now().isoformat(),
        "data_csv": str(csv_path),
        "data_range": [str(df.index.min()), str(df.index.max())],
        "event": {
            "horizon_bars": int(horizon),
            "move_threshold_pips": float(threshold_pips),
            "prob_th": float(args.prob_th),
        },
        "walk_forward": {
            "mode": "lockbox" if has_lockbox else "calendar_walk_forward",
            "train_years": int(args.train_years),
            "test_months": int(args.test_months),
            "step_months": int(args.step_months),
            "folds": int(len(splits)),
            "lockbox_train_end": str(args.lockbox_train_end) if has_lockbox else "",
            "lockbox_test_start": str(args.lockbox_test_start) if has_lockbox else "",
            "lockbox_test_end": str(args.lockbox_test_end) if has_lockbox else "",
        },
        "filters": {
            "session": str(args.session_filter),
            "hour_windows": str(args.hour_windows),
            "max_trades_per_session": int(args.max_trades_per_session),
            "regime": str(args.regime_filter),
            "trend_min": float(args.trend_min),
            "compression_max_quantile": float(args.compression_max_quantile),
            "compression_window": int(args.compression_window),
            "atr_norm_min_quantile": float(args.atr_norm_min_quantile),
            "atr_norm_max_quantile": float(args.atr_norm_max_quantile),
            "atr_window": int(args.atr_window),
            "require_close_confirm": bool(args.require_close_confirm),
        },
        "execution": {
            "lookback": int(args.lookback),
            "buffer_pips": float(args.buffer_pips),
            "sl_pips": float(args.sl_pips),
            "tp_pips": float(args.tp_pips),
            "sl_atr": float(args.sl_atr),
            "tp_atr": float(args.tp_atr),
            "time_stop_bars": int(args.time_stop_bars),
        },
        "costs": {
            "spread": float(costs.spread),
            "slippage": float(costs.slippage),
            "commission": float(costs.commission),
            "rt_cost_est_pips": float(rt_cost_est_pips),
        },
        "model_runtime": {
            "requested_device": str(args.lgbm_device),
            "gpu_success_folds": int(runtime_state.get("gpu_success_folds", 0)),
            "cpu_used_folds": int(runtime_state.get("cpu_used_folds", 0)),
            "cpu_fallback_folds": int(runtime_state.get("cpu_fallback_folds", 0)),
            "gpu_error": runtime_state.get("gpu_error"),
        },
        "modes": {},
    }

    for mode in modes:
        rows = fold_results[mode]
        net = [float(r["net_usd"]) for r in rows]
        pf = [float(r["profit_factor"]) if np.isfinite(float(r["profit_factor"])) else 10.0 for r in rows]
        dd = [float(r["max_drawdown_usd"]) for r in rows]
        trades = [int(r["trades"]) for r in rows]
        prev = [float(r["event_prevalence"]) for r in rows]
        precision = [float(r["event_precision"]) for r in rows]

        summary["modes"][mode] = {
            "per_fold": rows,
            "agg": {
                "net_usd": {
                    "sum": float(np.nansum(net)),
                    **fold_consistency(net, positive_if_gt=0.0),
                },
                "profit_factor": {
                    "mean": float(np.nanmean(pf)) if len(pf) else float("nan"),
                    "std": float(np.nanstd(pf)) if len(pf) else float("nan"),
                    "worst": float(np.nanmin(pf)) if len(pf) else float("nan"),
                },
                "max_drawdown_usd": {
                    "mean": float(np.nanmean(dd)) if len(dd) else float("nan"),
                    "worst": float(np.nanmax(dd)) if len(dd) else float("nan"),
                },
                "trades": {
                    "sum": int(np.nansum(trades)),
                    "mean_per_fold": float(np.nanmean(trades)) if len(trades) else float("nan"),
                },
                "event_quality": {
                    "prevalence_mean": float(np.nanmean(finite_stat_list(prev))) if len(prev) else float("nan"),
                    "precision_mean": float(np.nanmean(finite_stat_list(precision))) if len(precision) else float("nan"),
                },
            },
        }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = utc_now().strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"{args.out_prefix}_{ts}.json"
    md_path = reports_dir / f"{args.out_prefix}_{ts}.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(summary), encoding="utf-8")

    print("=== DONE ===")
    print(
        "model_runtime "
        f"requested_device={summary['model_runtime']['requested_device']} "
        f"gpu_success_folds={summary['model_runtime']['gpu_success_folds']} "
        f"cpu_used_folds={summary['model_runtime']['cpu_used_folds']} "
        f"cpu_fallback_folds={summary['model_runtime']['cpu_fallback_folds']}"
    )
    if summary["model_runtime"]["gpu_error"]:
        print(f"model_runtime gpu_error={summary['model_runtime']['gpu_error']}")
    print(f"report_json={json_path}")
    print(f"report_md={md_path}")


if __name__ == "__main__":
    main()
