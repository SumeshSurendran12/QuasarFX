#!/usr/bin/env python
"""
Strategy 2: NY Afternoon Volatility-Decay Mean Reversion

Design goals:
- Uncorrelated edge vs London breakout profile.
- Trade only during NY afternoon windows.
- Enter on stretched z-score in low/decaying volatility regime.
- Exit via fixed SL/TP, mean-reversion signal, or time stop.
- Support both calendar walk-forward and lockbox validation.
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
            raise ValueError(f"Invalid hour window '{part}'. Hours must be 0..23.")
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
        return f"{day}|win{matching_hour_window_index(ts, hour_windows)}"
    return f"{day}|{session_filter}"


def parse_utc_ts(raw: str) -> pd.Timestamp:
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def load_ohlcv_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError("CSV must contain 'time' column.")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"]).set_index("time")
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df[["open", "high", "low", "close", "volume"]].copy()


def rolling_quantile_threshold(series: pd.Series, window: int, quantile: float) -> pd.Series:
    w = max(int(window), 10)
    q = float(quantile)
    min_periods = max(20, w // 4)
    return series.shift(1).rolling(window=w, min_periods=min_periods).quantile(q)


def build_features(df: pd.DataFrame, z_lookback: int) -> pd.DataFrame:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    close_safe = close.replace(0.0, np.nan)
    tr = (high - low).astype(float)
    atr14 = tr.rolling(14, min_periods=1).mean()

    lb = max(int(z_lookback), 10)
    z_min = max(10, lb // 3)
    z_mean = close.rolling(lb, min_periods=z_min).mean()
    z_std = close.rolling(lb, min_periods=z_min).std().replace(0.0, np.nan)
    z = (close - z_mean) / z_std

    feat = pd.DataFrame(index=df.index)
    feat["ma_fast_slow"] = (close.rolling(12, min_periods=1).mean() - close.rolling(48, min_periods=1).mean()) / close_safe
    feat["atr_norm"] = atr14 / close_safe
    feat["zscore"] = z
    feat = feat.replace([np.inf, -np.inf], np.nan)
    return feat


def walk_forward_splits(
    index: pd.DatetimeIndex,
    train_years: int,
    test_months: int,
    step_months: int,
    max_folds: int,
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


def lockbox_split(index: pd.DatetimeIndex, train_end: str, test_start: str, test_end: str) -> List[Tuple[np.ndarray, np.ndarray]]:
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


def apply_entry_price(side: int, raw_price: float, costs: Costs) -> float:
    half = 0.5 * float(costs.spread)
    if int(side) == 1:
        return float(raw_price) + half + float(costs.slippage)
    return float(raw_price) - half - float(costs.slippage)


def apply_exit_price(side: int, raw_price: float, costs: Costs) -> float:
    half = 0.5 * float(costs.spread)
    if int(side) == 1:
        return float(raw_price) - half - float(costs.slippage)
    return float(raw_price) + half + float(costs.slippage)


def pnl_usd_from_price_move(entry: float, exit_: float, side: int, lot: float) -> float:
    return (float(exit_) - float(entry)) * int(side) * float(lot) * CONTRACT


def commission_usd(price: float, lot: float, costs: Costs) -> float:
    return float(price) * float(lot) * CONTRACT * float(costs.commission)


def fold_consistency(values: Sequence[float], positive_if_gt: float = 0.0) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
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


def simulate_fold(
    df: pd.DataFrame,
    feat: pd.DataFrame,
    test_idx: np.ndarray,
    session_filter: str,
    hour_windows: Sequence[Tuple[int, int]],
    max_trades_per_session: int,
    trend_abs_max: float,
    entry_z: float,
    exit_z: float,
    sl_pips: float,
    tp_pips: float,
    time_stop_bars: int,
    atr_min_threshold: np.ndarray | None,
    atr_max_threshold: np.ndarray | None,
    lot: float,
    costs: Costs,
    z_lookback: int,
) -> Dict[str, Any]:
    if test_idx.size == 0:
        return {
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

    test_set = set(int(x) for x in test_idx.tolist())
    test_start = int(test_idx[0])
    test_end = int(test_idx[-1])
    start_i = max(test_start, int(z_lookback))

    trades: List[Trade] = []
    trades_by_session: Dict[str, int] = {}
    in_pos = False
    side = 0
    entry_px = 0.0
    entry_i = -1
    entry_time = None
    sl_dist = float(sl_pips) * PIP
    tp_dist = float(tp_pips) * PIP

    for t in range(start_i, test_end + 1):
        if t not in test_set:
            continue

        z_t = float(feat["zscore"].iloc[t]) if np.isfinite(feat["zscore"].iloc[t]) else float("nan")

        if in_pos:
            bars_held = int(t - entry_i)
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
            elif np.isfinite(z_t):
                if side == 1 and z_t >= -float(exit_z):
                    reason = "mean_revert"
                    raw_exit = c[t]
                elif side == -1 and z_t <= float(exit_z):
                    reason = "mean_revert"
                    raw_exit = c[t]
            if reason is None and bars_held >= int(time_stop_bars):
                reason = "time"
                raw_exit = c[t]
            if reason is None and t == test_end:
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
        if not np.isfinite(z_t):
            continue
        if not in_session(idx[t], session_filter):
            continue
        if not in_hour_windows(idx[t], hour_windows):
            continue

        sess_key = session_bucket_key(idx[t], session_filter, hour_windows)
        if int(max_trades_per_session) > 0 and int(trades_by_session.get(sess_key, 0)) >= int(max_trades_per_session):
            continue

        trend_strength = abs(float(feat["ma_fast_slow"].iloc[t])) if np.isfinite(feat["ma_fast_slow"].iloc[t]) else float("nan")
        if not np.isfinite(trend_strength) or trend_strength > float(trend_abs_max):
            continue

        atr_t = float(feat["atr_norm"].iloc[t]) if np.isfinite(feat["atr_norm"].iloc[t]) else float("nan")
        if not np.isfinite(atr_t):
            continue
        if atr_min_threshold is not None:
            atr_min = float(atr_min_threshold[t])
            if not np.isfinite(atr_min) or atr_t < atr_min:
                continue
        if atr_max_threshold is not None:
            atr_max = float(atr_max_threshold[t])
            if not np.isfinite(atr_max) or atr_t > atr_max:
                continue

        signal_side = 0
        if z_t <= -float(entry_z):
            signal_side = 1
        elif z_t >= float(entry_z):
            signal_side = -1
        if signal_side == 0:
            continue

        t1 = t + 1
        if t1 not in test_set:
            continue
        raw_entry = float(o[t1])
        entry_px = apply_entry_price(signal_side, raw_entry, costs)
        in_pos = True
        side = int(signal_side)
        entry_i = int(t1)
        entry_time = idx[t1]
        trades_by_session[sess_key] = int(trades_by_session.get(sess_key, 0)) + 1

    pnl = np.asarray([tr.pnl_usd for tr in trades], dtype=float)
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    gross_profit = float(np.sum(wins)) if wins.size else 0.0
    gross_loss = float(-np.sum(losses)) if losses.size else 0.0
    pf = float(gross_profit / gross_loss) if gross_loss > 0.0 else (float("inf") if gross_profit > 0.0 else 0.0)
    eq = np.cumsum(pnl) if pnl.size else np.array([0.0], dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    max_dd = float(np.max(dd)) if dd.size else 0.0
    avg_bars = float(np.mean([tr.bars_held for tr in trades])) if trades else 0.0

    return {
        "trades": int(len(trades)),
        "net_usd": float(np.sum(pnl)) if pnl.size else 0.0,
        "profit_factor": pf,
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


def to_markdown(summary: Dict[str, Any]) -> str:
    agg = summary["agg"]
    lines: List[str] = []
    lines.append("# Strategy 2: NY Afternoon Volatility-Decay Mean Reversion")
    lines.append("")
    lines.append(f"- Generated (UTC): `{summary['generated_utc']}`")
    lines.append(f"- Data: `{summary['data_csv']}`")
    lines.append(f"- Range: `{summary['data_range'][0]}` -> `{summary['data_range'][1]}`")
    lines.append(
        f"- Split mode: `{summary['split']['mode']}` | folds={summary['split']['folds']} | "
        f"train={summary['split']['train_years']}y test={summary['split']['test_months']}m step={summary['split']['step_months']}m"
    )
    if summary["split"]["mode"] == "lockbox":
        lines.append(
            f"- Lockbox: train_end={summary['split']['lockbox_train_end']} | "
            f"test_start={summary['split']['lockbox_test_start']} | test_end={summary['split']['lockbox_test_end']}"
        )
    lines.append(
        f"- Entry: zscore >= {summary['params']['entry_z']} (short) or <= -{summary['params']['entry_z']} (long), "
        f"trend_abs_max={summary['params']['trend_abs_max']}"
    )
    lines.append(
        f"- Session: {summary['params']['session_filter']} | hours='{summary['params']['hour_windows']}' | "
        f"max_trades_per_session={summary['params']['max_trades_per_session']}"
    )
    lines.append(
        f"- Vol decay band: atr_norm_q in [{summary['params']['atr_min_quantile']}, {summary['params']['atr_max_quantile']}] "
        f"(window={summary['params']['atr_window']})"
    )
    lines.append(
        f"- Risk: SL={summary['params']['sl_pips']} pips | TP={summary['params']['tp_pips']} pips | "
        f"time_stop={summary['params']['time_stop_bars']} bars | exit_z={summary['params']['exit_z']}"
    )
    lines.append(
        f"- Costs: spread={summary['costs']['spread']} | slippage={summary['costs']['slippage']} | commission={summary['costs']['commission']}"
    )
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
    lines.append("| Fold | Test Start | Test End | Trades | Net USD | PF | Max DD | Avg Hold Bars |")
    lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in summary["per_fold"]:
        lines.append(
            f"| {row['fold']} | {row['test_start']} | {row['test_end']} | "
            f"{row['trades']} | {row['net_usd']:.2f} | {row['profit_factor']:.3f} | "
            f"{row['max_drawdown_usd']:.2f} | {row['avg_bars_in_trade']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-csv", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--mode", default="auto", choices=["auto", "wf", "lockbox"])
    p.add_argument("--lockbox-train-end", default="")
    p.add_argument("--lockbox-test-start", default="")
    p.add_argument("--lockbox-test-end", default="")
    p.add_argument("--train-years", type=int, default=3)
    p.add_argument("--test-months", type=int, default=6)
    p.add_argument("--step-months", type=int, default=6)
    p.add_argument("--max-folds", type=int, default=10)
    p.add_argument("--session-filter", default="ny_only", choices=["off", "london_only", "ny_only", "london_ny"])
    p.add_argument("--hour-windows", default="17-21")
    p.add_argument("--max-trades-per-session", type=int, default=1)
    p.add_argument("--z-lookback", type=int, default=24)
    p.add_argument("--entry-z", type=float, default=1.2)
    p.add_argument("--exit-z", type=float, default=0.2)
    p.add_argument("--trend-abs-max", type=float, default=0.00010)
    p.add_argument("--atr-min-quantile", type=float, default=0.10)
    p.add_argument("--atr-max-quantile", type=float, default=0.60)
    p.add_argument("--atr-window", type=int, default=720)
    p.add_argument("--sl-pips", type=float, default=12.0)
    p.add_argument("--tp-pips", type=float, default=14.0)
    p.add_argument("--time-stop-bars", type=int, default=8)
    p.add_argument("--lot", type=float, default=0.05)
    p.add_argument("--spread", type=float, default=0.0002)
    p.add_argument("--slippage", type=float, default=0.00005)
    p.add_argument("--commission", type=float, default=0.0001)
    p.add_argument("--lgbm-device", default="auto", choices=["auto", "gpu", "cpu"], help="Compatibility flag; not used by this strategy.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.data_csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    atr_q_min = float(args.atr_min_quantile)
    atr_q_max = float(args.atr_max_quantile)
    if not (0.0 <= atr_q_min <= 1.0 and 0.0 < atr_q_max <= 1.0 and atr_q_min <= atr_q_max):
        raise ValueError("ATR quantile band must satisfy 0 <= min <= max <= 1.")
    if float(args.entry_z) <= 0:
        raise ValueError("--entry-z must be > 0.")
    if float(args.exit_z) < 0:
        raise ValueError("--exit-z must be >= 0.")

    df = load_ohlcv_csv(csv_path)
    feat = build_features(df, z_lookback=int(args.z_lookback))
    hour_windows = parse_hour_windows(args.hour_windows)

    atr_min_threshold = (
        rolling_quantile_threshold(feat["atr_norm"].astype(float), window=int(args.atr_window), quantile=atr_q_min).to_numpy(dtype=float)
        if atr_q_min > 0.0
        else None
    )
    atr_max_threshold = (
        rolling_quantile_threshold(feat["atr_norm"].astype(float), window=int(args.atr_window), quantile=atr_q_max).to_numpy(dtype=float)
        if atr_q_max < 1.0
        else None
    )

    if args.mode == "lockbox":
        has_lockbox = True
    elif args.mode == "wf":
        has_lockbox = False
    else:
        has_lockbox = bool(args.lockbox_train_end or args.lockbox_test_start or args.lockbox_test_end)
    if has_lockbox:
        if not (args.lockbox_train_end and args.lockbox_test_start and args.lockbox_test_end):
            raise ValueError("Lockbox mode requires --lockbox-train-end, --lockbox-test-start, --lockbox-test-end.")
        splits = lockbox_split(df.index, args.lockbox_train_end, args.lockbox_test_start, args.lockbox_test_end)
    else:
        splits = walk_forward_splits(
            df.index,
            train_years=int(args.train_years),
            test_months=int(args.test_months),
            step_months=int(args.step_months),
            max_folds=int(args.max_folds),
        )
    if not splits:
        raise ValueError("No splits produced.")

    costs = Costs(spread=float(args.spread), slippage=float(args.slippage), commission=float(args.commission))
    fold_rows: List[Dict[str, Any]] = []
    for fold_num, (_, te_idx) in enumerate(splits, start=1):
        row = simulate_fold(
            df=df,
            feat=feat,
            test_idx=te_idx,
            session_filter=str(args.session_filter),
            hour_windows=hour_windows,
            max_trades_per_session=int(args.max_trades_per_session),
            trend_abs_max=float(args.trend_abs_max),
            entry_z=float(args.entry_z),
            exit_z=float(args.exit_z),
            sl_pips=float(args.sl_pips),
            tp_pips=float(args.tp_pips),
            time_stop_bars=int(args.time_stop_bars),
            atr_min_threshold=atr_min_threshold,
            atr_max_threshold=atr_max_threshold,
            lot=float(args.lot),
            costs=costs,
            z_lookback=int(args.z_lookback),
        )
        row.update(
            {
                "fold": int(fold_num),
                "test_start": str(df.index[int(te_idx[0])]),
                "test_end": str(df.index[int(te_idx[-1])]),
            }
        )
        fold_rows.append(row)
        print(f"fold {fold_num:02d}/{len(splits)} done")

    net = [float(r["net_usd"]) for r in fold_rows]
    pf = [float(r["profit_factor"]) if np.isfinite(float(r["profit_factor"])) else 10.0 for r in fold_rows]
    dd = [float(r["max_drawdown_usd"]) for r in fold_rows]
    trades = [int(r["trades"]) for r in fold_rows]
    agg = {
        "net_usd": {"sum": float(np.nansum(net)), **fold_consistency(net, positive_if_gt=0.0)},
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
    }

    summary: Dict[str, Any] = {
        "generated_utc": utc_now().isoformat(),
        "strategy_name": "NY Afternoon Volatility-Decay Mean Reversion",
        "data_csv": str(csv_path),
        "data_range": [str(df.index.min()), str(df.index.max())],
        "split": {
            "mode": "lockbox" if has_lockbox else "calendar_walk_forward",
            "train_years": int(args.train_years),
            "test_months": int(args.test_months),
            "step_months": int(args.step_months),
            "folds": int(len(splits)),
            "lockbox_train_end": str(args.lockbox_train_end) if has_lockbox else "",
            "lockbox_test_start": str(args.lockbox_test_start) if has_lockbox else "",
            "lockbox_test_end": str(args.lockbox_test_end) if has_lockbox else "",
        },
        "params": {
            "mode_arg": str(args.mode),
            "lgbm_device": str(args.lgbm_device),
            "session_filter": str(args.session_filter),
            "hour_windows": str(args.hour_windows),
            "max_trades_per_session": int(args.max_trades_per_session),
            "z_lookback": int(args.z_lookback),
            "entry_z": float(args.entry_z),
            "exit_z": float(args.exit_z),
            "trend_abs_max": float(args.trend_abs_max),
            "atr_min_quantile": float(args.atr_min_quantile),
            "atr_max_quantile": float(args.atr_max_quantile),
            "atr_window": int(args.atr_window),
            "sl_pips": float(args.sl_pips),
            "tp_pips": float(args.tp_pips),
            "time_stop_bars": int(args.time_stop_bars),
            "lot": float(args.lot),
        },
        "costs": {
            "spread": float(costs.spread),
            "slippage": float(costs.slippage),
            "commission": float(costs.commission),
        },
        "agg": agg,
        "per_fold": fold_rows,
    }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = utc_now().strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"{args.out_prefix}_{ts}.json"
    md_path = reports_dir / f"{args.out_prefix}_{ts}.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(summary), encoding="utf-8")

    print("=== DONE ===")
    print(f"report_json={json_path}")
    print(f"report_md={md_path}")


if __name__ == "__main__":
    main()
