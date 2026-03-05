#!/usr/bin/env python
"""
NY Pullback Trend Continuation - Walk-Forward + Lockbox Backtest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Trade:
    direction: int
    entry_i: int
    entry_time: str
    entry_price: float
    atr_entry: float
    sl_price: float
    tp_price: float
    trend_entry: float
    pnl_usd: float = 0.0
    bars_held: int = 0
    mfe_atr: float = 0.0
    mae_atr: float = 0.0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_bool(s: str) -> bool:
    v = str(s).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool: {s}")


def parse_hour_windows(raw: str) -> List[Tuple[int, int]]:
    text = (raw or "").strip()
    if not text:
        return []
    out: List[Tuple[int, int]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        a, b = token.split("-", 1)
        s = int(a.strip())
        e = int(b.strip())
        if not (0 <= s <= 23 and 0 <= e <= 24):
            raise ValueError(f"Invalid hour window: {token}")
        out.append((s, e))
    return out


def in_any_hour_window(hour: int, windows: Sequence[Tuple[int, int]]) -> bool:
    if not windows:
        return True
    for a, b in windows:
        if a <= hour < b:
            return True
    return False


def load_ohlcv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "time" in df.columns:
        tcol = "time"
    elif "date" in df.columns:
        tcol = "date"
    else:
        raise ValueError("CSV must contain time or date column.")
    df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.dropna(subset=[tcol]).sort_values(tcol).reset_index(drop=True)
    df = df.rename(columns={tcol: "time"})
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df


def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def add_indicators(df: pd.DataFrame, ma_fast: int, ma_slow: int, atr_window: int) -> pd.DataFrame:
    out = df.copy()
    out["ma_fast"] = out["close"].rolling(ma_fast, min_periods=ma_fast).mean()
    out["ma_slow"] = out["close"].rolling(ma_slow, min_periods=ma_slow).mean()
    out["atr"] = compute_atr(out, atr_window)
    out["trend_strength"] = (out["ma_fast"] - out["ma_slow"]) / out["close"].replace(0.0, np.nan)
    out["atr_norm"] = out["atr"] / out["close"].replace(0.0, np.nan)
    return out


def session_ok(hour: int, session_filter: str) -> bool:
    mode = (session_filter or "off").strip().lower()
    if mode == "off":
        return True
    london = 7 <= hour < 16
    ny = 12 <= hour < 21
    if mode == "london_only":
        return london
    if mode == "ny_only":
        return ny
    if mode == "london_ny":
        return london or ny
    raise ValueError(f"Invalid session filter: {session_filter}")


def wf_splits_time(
    times: pd.Series,
    train_years: int,
    test_months: int,
    step_months: int,
    max_folds: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    t0 = pd.Timestamp(times.iloc[0])
    t1 = pd.Timestamp(times.iloc[-1])
    out: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    test_start = t0 + pd.DateOffset(years=int(train_years))
    while len(out) < int(max_folds):
        train_end = test_start
        train_start = train_end - pd.DateOffset(years=int(train_years))
        test_end = test_start + pd.DateOffset(months=int(test_months))
        if train_start < t0:
            train_start = t0
        if test_end > t1:
            break
        out.append((train_start, train_end, test_start, test_end))
        test_start = test_start + pd.DateOffset(months=int(step_months))
    return out


def lockbox_split(
    times: pd.Series,
    train_end_utc: str,
    test_start_utc: str,
    test_end_utc: str,
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    train_start = pd.Timestamp(times.iloc[0])
    train_end = pd.Timestamp(train_end_utc)
    test_start = pd.Timestamp(test_start_utc)
    test_end = pd.Timestamp(test_end_utc)
    return train_start, train_end, test_start, test_end


def index_range(df: pd.DataFrame, t_start: pd.Timestamp, t_end: pd.Timestamp) -> Tuple[int, int]:
    mask = (df["time"] >= t_start) & (df["time"] < t_end)
    idx = np.where(mask.to_numpy())[0]
    if idx.size == 0:
        return 0, 0
    return int(idx[0]), int(idx[-1] + 1)


def commission_cost(price: float, lots: float, commission: float) -> float:
    return float(price) * float(lots) * 100000.0 * float(commission)


def profit_factor(pnls: Sequence[float]) -> float:
    gains = sum(x for x in pnls if x > 0.0)
    losses = -sum(x for x in pnls if x < 0.0)
    if losses <= 0 and gains > 0:
        return float("inf")
    if losses <= 0:
        return 0.0
    return float(gains / losses)


def drawdown_from_trades(pnls: Sequence[float]) -> float:
    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        eq += float(p)
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
    return float(max_dd)


def _forced_entry_backtest(
    slc: pd.DataFrame,
    entry_map: Dict[int, int],
    session_filter: str,
    hour_windows: Sequence[Tuple[int, int]],
    max_trades_per_session: int,
    sl_atr: float,
    tp_atr: float,
    time_stop_bars: int,
    lots: float,
    spread: float,
    slippage: float,
    commission: float,
) -> float:
    half = 0.5 * float(spread)
    session_count: Dict[str, int] = {}
    in_pos = False
    direction = 0
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    bars_held = 0
    eq = 0.0
    for i in range(len(slc) - 1):
        row = slc.iloc[i]
        hr = int(pd.Timestamp(row["time"]).hour)
        key = f"{pd.Timestamp(row['time']).date().isoformat()}:{session_filter}"
        if in_pos:
            bars_held += 1
            hi = float(row["high"])
            lo = float(row["low"])
            exit_raw = None
            if direction > 0:
                if lo <= sl_price:
                    exit_raw = sl_price
                elif hi >= tp_price:
                    exit_raw = tp_price
            else:
                if hi >= sl_price:
                    exit_raw = sl_price
                elif lo <= tp_price:
                    exit_raw = tp_price
            if exit_raw is None and bars_held >= int(time_stop_bars):
                exit_raw = float(row["close"])
            if exit_raw is not None:
                if direction > 0:
                    exit_px = float(exit_raw) - half - float(slippage)
                else:
                    exit_px = float(exit_raw) + half + float(slippage)
                pnl = (exit_px - entry_price) * direction * float(lots) * 100000.0
                pnl -= commission_cost(entry_price, lots, commission)
                pnl -= commission_cost(exit_px, lots, commission)
                eq += pnl
                in_pos = False
        if in_pos:
            continue
        if i not in entry_map:
            continue
        if not session_ok(hr, session_filter) or not in_any_hour_window(hr, list(hour_windows)):
            continue
        used = int(session_count.get(key, 0))
        if int(max_trades_per_session) > 0 and used >= int(max_trades_per_session):
            continue
        atr = float(row["atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue
        direction = int(entry_map[i])
        open_next = float(slc.iloc[i + 1]["open"])
        if direction > 0:
            entry_price = open_next + half + float(slippage)
            sl_price = entry_price - float(sl_atr) * atr
            tp_price = entry_price + float(tp_atr) * atr
        else:
            entry_price = open_next - half - float(slippage)
            sl_price = entry_price + float(sl_atr) * atr
            tp_price = entry_price - float(tp_atr) * atr
        bars_held = 0
        in_pos = True
        session_count[key] = used + 1
    return float(eq)


def permutation_test(
    slc: pd.DataFrame,
    obs_net: float,
    entry_count: int,
    long_ratio: float,
    session_filter: str,
    hour_windows: Sequence[Tuple[int, int]],
    max_trades_per_session: int,
    sl_atr: float,
    tp_atr: float,
    time_stop_bars: int,
    lots: float,
    spread: float,
    slippage: float,
    commission: float,
    perm_tests: int,
    seed: int = 7,
) -> Tuple[Optional[float], Optional[float]]:
    if int(perm_tests) <= 0 or int(entry_count) <= 5:
        return None, None
    allow = np.where((slc["allow"].to_numpy(dtype=bool))[:-1])[0]
    if allow.size < int(entry_count):
        return None, None
    rng = np.random.default_rng(seed)
    perm_nets: List[float] = []
    for _ in range(int(perm_tests)):
        picks = np.sort(rng.choice(allow, size=int(entry_count), replace=False))
        n_long = int(round(float(long_ratio) * int(entry_count)))
        dirs = np.array([1] * n_long + [-1] * (int(entry_count) - n_long), dtype=int)
        rng.shuffle(dirs)
        entry_map = {int(i): int(d) for i, d in zip(picks.tolist(), dirs.tolist())}
        net = _forced_entry_backtest(
            slc=slc,
            entry_map=entry_map,
            session_filter=session_filter,
            hour_windows=hour_windows,
            max_trades_per_session=max_trades_per_session,
            sl_atr=sl_atr,
            tp_atr=tp_atr,
            time_stop_bars=time_stop_bars,
            lots=lots,
            spread=spread,
            slippage=slippage,
            commission=commission,
        )
        perm_nets.append(net)
    arr = np.asarray(perm_nets, dtype=float)
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size > 2 else 0.0
    z = (float(obs_net) - mu) / sd if sd > 1e-9 else None
    p = float(np.mean(arr >= float(obs_net)))
    return p, (float(z) if z is not None else None)


def run_test_slice(
    slc: pd.DataFrame,
    trend_th: float,
    atr_min: float,
    atr_max: float,
    ma_fast: int,
    pullback_atr: float,
    reclaim_confirm: bool,
    sl_atr: float,
    tp_atr: float,
    time_stop_bars: int,
    max_trades_per_session: int,
    session_filter: str,
    hour_windows: Sequence[Tuple[int, int]],
    lots: float,
    spread: float,
    slippage: float,
    commission: float,
    drift_horizon_bars: int,
    perm_tests: int,
) -> Dict[str, float | int | None]:
    half = 0.5 * float(spread)
    slc = slc.copy()
    hrs = slc["time"].dt.hour.astype(int)
    slc["allow"] = (
        hrs.apply(lambda h: session_ok(int(h), session_filter)).astype(bool)
        & hrs.apply(lambda h: in_any_hour_window(int(h), list(hour_windows))).astype(bool)
        & (slc["trend_strength"].abs() >= float(trend_th))
        & (slc["atr_norm"] >= float(atr_min))
        & (slc["atr_norm"] <= float(atr_max))
    )

    trades: List[Trade] = []
    session_count: Dict[str, int] = {}
    cur: Optional[Trade] = None

    for i in range(len(slc) - 1):
        row = slc.iloc[i]
        ts = pd.Timestamp(row["time"])
        key = f"{ts.date().isoformat()}:{session_filter}"

        if cur is not None:
            cur.bars_held += 1
            hi = float(row["high"])
            lo = float(row["low"])
            if cur.direction > 0:
                cur.mae_atr = max(cur.mae_atr, (cur.entry_price - lo) / max(cur.atr_entry, 1e-12))
                cur.mfe_atr = max(cur.mfe_atr, (hi - cur.entry_price) / max(cur.atr_entry, 1e-12))
                hit_sl = lo <= cur.sl_price
                hit_tp = hi >= cur.tp_price
            else:
                cur.mae_atr = max(cur.mae_atr, (hi - cur.entry_price) / max(cur.atr_entry, 1e-12))
                cur.mfe_atr = max(cur.mfe_atr, (cur.entry_price - lo) / max(cur.atr_entry, 1e-12))
                hit_sl = hi >= cur.sl_price
                hit_tp = lo <= cur.tp_price
            exit_raw = None
            if hit_sl and hit_tp:
                exit_raw = cur.sl_price
            elif hit_sl:
                exit_raw = cur.sl_price
            elif hit_tp:
                exit_raw = cur.tp_price
            elif cur.bars_held >= int(time_stop_bars):
                exit_raw = float(row["close"])
            if exit_raw is not None:
                if cur.direction > 0:
                    exit_px = float(exit_raw) - half - float(slippage)
                else:
                    exit_px = float(exit_raw) + half + float(slippage)
                pnl = (exit_px - cur.entry_price) * cur.direction * float(lots) * 100000.0
                pnl -= commission_cost(cur.entry_price, lots, commission)
                pnl -= commission_cost(exit_px, lots, commission)
                cur.pnl_usd = float(pnl)
                trades.append(cur)
                cur = None
        if cur is not None:
            continue
        if not bool(slc["allow"].iloc[i]):
            continue
        used = int(session_count.get(key, 0))
        if int(max_trades_per_session) > 0 and used >= int(max_trades_per_session):
            continue
        atr = float(row["atr"])
        ma_f = float(row["ma_fast"])
        trend = float(row["trend_strength"])
        if not (np.isfinite(atr) and atr > 0 and np.isfinite(ma_f) and np.isfinite(trend)):
            continue
        long_pull = trend >= float(trend_th) and float(row["low"]) <= ma_f - float(pullback_atr) * atr
        short_pull = trend <= -float(trend_th) and float(row["high"]) >= ma_f + float(pullback_atr) * atr
        if bool(reclaim_confirm):
            long_ok = long_pull and float(row["close"]) > ma_f
            short_ok = short_pull and float(row["close"]) < ma_f
        else:
            long_ok = long_pull and float(row["close"]) >= ma_f - 0.10 * atr
            short_ok = short_pull and float(row["close"]) <= ma_f + 0.10 * atr
        direction = 0
        if long_ok and not short_ok:
            direction = 1
        elif short_ok and not long_ok:
            direction = -1
        if direction == 0:
            continue
        open_next = float(slc.iloc[i + 1]["open"])
        if direction > 0:
            entry_price = open_next + half + float(slippage)
            sl_price = entry_price - float(sl_atr) * atr
            tp_price = entry_price + float(tp_atr) * atr
        else:
            entry_price = open_next - half - float(slippage)
            sl_price = entry_price + float(sl_atr) * atr
            tp_price = entry_price - float(tp_atr) * atr
        cur = Trade(
            direction=direction,
            entry_i=i + 1,
            entry_time=str(slc.iloc[i + 1]["time"]),
            entry_price=float(entry_price),
            atr_entry=float(atr),
            sl_price=float(sl_price),
            tp_price=float(tp_price),
            trend_entry=float(trend),
        )
        session_count[key] = used + 1

    if cur is not None:
        last = slc.iloc[-1]
        if cur.direction > 0:
            exit_px = float(last["close"]) - half - float(slippage)
        else:
            exit_px = float(last["close"]) + half + float(slippage)
        pnl = (exit_px - cur.entry_price) * cur.direction * float(lots) * 100000.0
        pnl -= commission_cost(cur.entry_price, lots, commission)
        pnl -= commission_cost(exit_px, lots, commission)
        cur.pnl_usd = float(pnl)
        trades.append(cur)

    pnls = [t.pnl_usd for t in trades]
    align = [1.0 if np.sign(t.trend_entry) == float(t.direction) else 0.0 for t in trades]
    drift_vals: List[float] = []
    for t in trades:
        e = int(t.entry_i)
        k = max(1, min(int(drift_horizon_bars), int(time_stop_bars), len(slc) - 1 - e))
        if e + k < len(slc):
            p0 = float(slc.iloc[e]["close"])
            p1 = float(slc.iloc[e + k]["close"])
            drift_vals.append(((p1 - p0) / max(p0, 1e-12)) * 10000.0 * float(t.direction))

    long_ratio = float(np.mean([1.0 if t.direction > 0 else 0.0 for t in trades])) if trades else 0.5
    pval, z = permutation_test(
        slc=slc,
        obs_net=float(np.sum(pnls)) if pnls else 0.0,
        entry_count=len(trades),
        long_ratio=long_ratio,
        session_filter=session_filter,
        hour_windows=hour_windows,
        max_trades_per_session=max_trades_per_session,
        sl_atr=sl_atr,
        tp_atr=tp_atr,
        time_stop_bars=time_stop_bars,
        lots=lots,
        spread=spread,
        slippage=slippage,
        commission=commission,
        perm_tests=int(perm_tests),
    )

    return {
        "trades": int(len(trades)),
        "net_usd": float(np.sum(pnls)) if pnls else 0.0,
        "pf": float(profit_factor(pnls)),
        "max_dd_usd": float(drawdown_from_trades(pnls)),
        "avg_hold_bars": float(np.mean([t.bars_held for t in trades])) if trades else 0.0,
        "trend_align_rate": float(np.mean(align)) if align else 0.0,
        "mean_signed_drift_bps": float(np.mean(drift_vals)) if drift_vals else 0.0,
        "perm_p_value": pval,
        "perm_z": z,
        "mfe_atr_mean": float(np.mean([t.mfe_atr for t in trades])) if trades else 0.0,
        "mae_atr_mean": float(np.mean([t.mae_atr for t in trades])) if trades else 0.0,
    }


def summarize_profile(rows: pd.DataFrame) -> Dict[str, float | int | None]:
    fold_net = rows.groupby("fold")["net_usd"].sum()
    return {
        "folds": int(rows["fold"].nunique()),
        "net_sum": float(rows["net_usd"].sum()),
        "net_mean": float(rows["net_usd"].mean()),
        "net_std": float(rows["net_usd"].std(ddof=1)) if len(rows) > 1 else 0.0,
        "pos_fold_rate": float(np.mean(fold_net > 0.0)),
        "worst_fold_net": float(fold_net.min()) if not fold_net.empty else float("nan"),
        "pf_mean": float(rows["pf"].mean()),
        "dd_worst": float(rows["max_dd_usd"].max()),
        "trades_total": int(rows["trades"].sum()),
        "trend_align_mean": float(rows["trend_align_rate"].mean()),
        "signed_drift_bps_mean": float(rows["mean_signed_drift_bps"].mean()),
        "perm_p_median": float(rows["perm_p_value"].dropna().median()) if rows["perm_p_value"].notna().any() else None,
        "perm_z_median": float(rows["perm_z"].dropna().median()) if rows["perm_z"].notna().any() else None,
        "mfe_atr_mean": float(rows["mfe_atr_mean"].mean()),
        "mae_atr_mean": float(rows["mae_atr_mean"].mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--mode", default="wf", choices=["wf", "lockbox"])
    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--test-months", type=int, default=6)
    ap.add_argument("--step-months", type=int, default=6)
    ap.add_argument("--max-folds", type=int, default=10)
    ap.add_argument("--train-end-utc", default="")
    ap.add_argument("--test-start-utc", default="")
    ap.add_argument("--test-end-utc", default="")
    ap.add_argument("--session-filter", default="ny_only", choices=["off", "london_only", "ny_only", "london_ny"])
    ap.add_argument("--hour-windows", default="12-17")
    ap.add_argument("--max-trades-per-session", type=int, default=1)
    ap.add_argument("--ma-fast", type=int, default=20)
    ap.add_argument("--ma-slow", type=int, default=80)
    ap.add_argument("--atr-window", type=int, default=14)
    ap.add_argument("--pullback-atr", type=float, default=0.6)
    ap.add_argument("--reclaim-confirm", type=parse_bool, default=True)
    ap.add_argument("--trend-min", type=float, default=0.0)
    ap.add_argument("--trend-q", type=float, default=0.60)
    ap.add_argument("--atr-min-q", type=float, default=0.40)
    ap.add_argument("--atr-max-q", type=float, default=0.95)
    ap.add_argument("--sl-atr", type=float, default=1.0)
    ap.add_argument("--tp-atr", type=float, default=1.6)
    ap.add_argument("--time-stop-bars", type=int, default=10)
    ap.add_argument("--lot", type=float, default=0.10)
    ap.add_argument("--spread", type=float, default=0.00020)
    ap.add_argument("--slippage", type=float, default=0.00005)
    ap.add_argument("--commission", type=float, default=0.00010)
    ap.add_argument("--drift-horizon-bars", type=int, default=10)
    ap.add_argument("--perm-tests", type=int, default=200)
    ap.add_argument("--stress-profiles", default="base")
    ap.add_argument("--lgbm-device", default="auto", choices=["auto", "gpu", "cpu"], help="Compatibility flag; not used.")
    args = ap.parse_args()

    df = load_ohlcv(args.data_csv)
    df = add_indicators(df, args.ma_fast, args.ma_slow, args.atr_window)
    windows = parse_hour_windows(args.hour_windows)

    if args.mode == "wf":
        splits = wf_splits_time(df["time"], args.train_years, args.test_months, args.step_months, args.max_folds)
        if not splits:
            raise ValueError("No walk-forward folds produced.")
    else:
        if not (args.train_end_utc and args.test_start_utc and args.test_end_utc):
            raise ValueError("lockbox mode requires --train-end-utc --test-start-utc --test-end-utc")
        splits = [lockbox_split(df["time"], args.train_end_utc, args.test_start_utc, args.test_end_utc)]

    stress_names = [s.strip() for s in str(args.stress_profiles).split(",") if s.strip()]
    stress_map = {
        "base": {"spread": float(args.spread), "slippage": float(args.slippage), "commission": float(args.commission)},
        "spread25_slip2x": {"spread": 0.00025, "slippage": float(args.slippage) * 2.0, "commission": float(args.commission)},
        "spread30_slip2x": {"spread": 0.00030, "slippage": float(args.slippage) * 2.0, "commission": float(args.commission)},
    }
    for name in stress_names:
        if name not in stress_map:
            raise ValueError(f"Unknown stress profile: {name}")

    fold_rows: List[Dict[str, float | int | str | None]] = []
    fold_id = 0
    for tr_start, tr_end, te_start, te_end in splits:
        fold_id += 1
        tr0, tr1 = index_range(df, tr_start, tr_end)
        te0, te1 = index_range(df, te_start, te_end)
        train_df = df.iloc[tr0:tr1]
        test_df = df.iloc[te0:te1]
        if train_df.empty or test_df.empty:
            continue
        ts_abs = train_df["trend_strength"].abs().replace([np.inf, -np.inf], np.nan).dropna()
        atrn = train_df["atr_norm"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ts_abs) < max(int(args.ma_slow + args.atr_window), 100) or atrn.empty:
            continue
        trend_th = float(max(float(args.trend_min), ts_abs.quantile(float(args.trend_q))))
        atr_min = float(atrn.quantile(float(args.atr_min_q)))
        atr_max = float(atrn.quantile(float(args.atr_max_q)))
        if atr_max < atr_min:
            atr_min, atr_max = atr_max, atr_min

        for prof in stress_names:
            costs = stress_map[prof]
            res = run_test_slice(
                slc=test_df,
                trend_th=trend_th,
                atr_min=atr_min,
                atr_max=atr_max,
                ma_fast=int(args.ma_fast),
                pullback_atr=float(args.pullback_atr),
                reclaim_confirm=bool(args.reclaim_confirm),
                sl_atr=float(args.sl_atr),
                tp_atr=float(args.tp_atr),
                time_stop_bars=int(args.time_stop_bars),
                max_trades_per_session=int(args.max_trades_per_session),
                session_filter=str(args.session_filter),
                hour_windows=windows,
                lots=float(args.lot),
                spread=float(costs["spread"]),
                slippage=float(costs["slippage"]),
                commission=float(costs["commission"]),
                drift_horizon_bars=int(args.drift_horizon_bars),
                perm_tests=int(args.perm_tests),
            )
            fold_rows.append(
                {
                    "mode": args.mode,
                    "profile": prof,
                    "fold": int(fold_id),
                    "train_start": str(tr_start),
                    "train_end": str(tr_end),
                    "test_start": str(te_start),
                    "test_end": str(te_end),
                    "trend_threshold": float(trend_th),
                    "atr_min": float(atr_min),
                    "atr_max": float(atr_max),
                    **res,
                }
            )
        print(f"fold {fold_id:02d}/{len(splits)} done")

    rows_df = pd.DataFrame(fold_rows)
    summaries: Dict[str, Dict[str, float | int | None]] = {}
    for prof in stress_names:
        d = rows_df[rows_df["profile"] == prof].copy()
        if d.empty:
            continue
        summaries[prof] = summarize_profile(d)

    payload = {
        "generated_utc": utc_now_iso(),
        "data_csv": str(Path(args.data_csv).resolve()),
        "mode": args.mode,
        "params": vars(args),
        "fold_rows": fold_rows,
        "summaries": summaries,
    }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"{args.out_prefix}_{ts}.json"
    md_path = reports_dir / f"{args.out_prefix}_{ts}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# NY Pullback Trend Continuation - Walk-Forward / Lockbox Backtest",
        "",
        f"- Generated (UTC): `{payload['generated_utc']}`",
        f"- Data: `{payload['data_csv']}`",
        f"- Mode: `{args.mode}`",
        f"- Session: `{args.session_filter}` | Hour windows: `{args.hour_windows}` | Max trades/session: `{args.max_trades_per_session}`",
        f"- Params: ma_fast={args.ma_fast}, ma_slow={args.ma_slow}, atr_window={args.atr_window}, pullback_atr={args.pullback_atr}, reclaim_confirm={bool(args.reclaim_confirm)}",
        f"- Exits: SL={args.sl_atr}*ATR, TP={args.tp_atr}*ATR, time_stop={args.time_stop_bars} bars",
        f"- Base costs: spread={args.spread}, slippage={args.slippage}, commission={args.commission}, lot={args.lot}",
        "",
        "## Profile Summaries",
        "",
    ]
    for prof in stress_names:
        s = summaries.get(prof)
        if not s:
            continue
        md_lines.extend(
            [
                f"### {prof}",
                "",
                f"- folds: `{s['folds']}`",
                f"- net_sum: `{float(s['net_sum']):.2f}` | net_mean: `{float(s['net_mean']):.2f}` | net_std: `{float(s['net_std']):.2f}`",
                f"- pos_fold_rate: `{float(s['pos_fold_rate']):.2f}` | worst_fold_net: `{float(s['worst_fold_net']):.2f}`",
                f"- PF_mean: `{float(s['pf_mean']):.3f}` | worst_DD: `{float(s['dd_worst']):.2f}` | trades_total: `{int(s['trades_total'])}`",
                f"- trend_align_mean: `{float(s['trend_align_mean']):.3f}` | signed_drift_bps_mean: `{float(s['signed_drift_bps_mean']):.3f}`",
                f"- perm_p_median: `{s['perm_p_median']}` | perm_z_median: `{s['perm_z_median']}`",
                f"- MFE_ATR_mean: `{float(s['mfe_atr_mean']):.3f}` | MAE_ATR_mean: `{float(s['mae_atr_mean']):.3f}`",
                "",
            ]
        )
    md_lines.extend(
        [
            "## Fold Table (base profile rows)",
            "",
            "| Fold | Test Start | Test End | Trades | Net USD | PF | Max DD | Avg Hold | TrendAlign | Drift(bps) | Perm p | Perm z |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    base_rows = rows_df[rows_df["profile"] == "base"].sort_values(["fold"]).to_dict("records")
    for r in base_rows:
        md_lines.append(
            "| {fold} | {test_start} | {test_end} | {trades} | {net_usd:.2f} | {pf:.3f} | {max_dd_usd:.2f} | {avg_hold_bars:.2f} | {trend_align_rate:.3f} | {mean_signed_drift_bps:.3f} | {perm_p_value} | {perm_z} |".format(
                **r
            )
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("=== DONE ===")
    print(f"report_json={json_path}")
    print(f"report_md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
