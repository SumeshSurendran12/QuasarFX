#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.rlm_gate_env import RLMGateEnv, RLMRewardConfig
from scripts.vol_breakout_backtest_wf import (
    CONTRACT,
    PIP,
    Costs,
    apply_entry_price,
    apply_exit_price,
    build_features,
    commission_usd,
    cost_pips_est,
    in_hour_windows,
    in_session,
    load_ohlcv_csv,
    lockbox_split,
    make_event_label,
    parse_hour_windows,
    pnl_usd_from_price_move,
    regime_ok,
    rolling_quantile_threshold,
    session_bucket_key,
    walk_forward_splits,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def fit_event_model_predict(
    X: pd.DataFrame,
    y: np.ndarray,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    lgbm_device: str,
    runtime_state: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, str]:
    y_train = y[tr_idx]
    classes = np.unique(y_train)
    if classes.size < 2:
        base = float(classes[0]) if classes.size == 1 else 0.5
        return (
            np.full(tr_idx.shape[0], base, dtype=float),
            np.full(te_idx.shape[0], base, dtype=float),
            "cpu",
        )

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
            p_train = model.predict_proba(X.iloc[tr_idx])[:, 1]
            p_test = model.predict_proba(X.iloc[te_idx])[:, 1]
            if device == "gpu":
                runtime_state["gpu_success_folds"] = int(runtime_state.get("gpu_success_folds", 0)) + 1
            else:
                runtime_state["cpu_used_folds"] = int(runtime_state.get("cpu_used_folds", 0)) + 1
                if str(lgbm_device) in {"auto", "gpu"}:
                    runtime_state["cpu_fallback_folds"] = int(runtime_state.get("cpu_fallback_folds", 0)) + 1
            return p_train, p_test, device
        except Exception as exc:
            last_error = exc
            if device == "gpu":
                runtime_state["gpu_disabled"] = True
                runtime_state["gpu_error"] = str(exc)
                continue
            raise

    raise RuntimeError(f"LightGBM training failed on all device candidates: {last_error}")


def profit_factor_from_pnls(pnls: np.ndarray) -> float:
    if pnls.size == 0:
        return 0.0
    gross_profit = float(np.sum(pnls[pnls > 0.0])) if np.any(pnls > 0.0) else 0.0
    gross_loss = float(-np.sum(pnls[pnls < 0.0])) if np.any(pnls < 0.0) else 0.0
    if gross_loss <= 0.0:
        return float("inf") if gross_profit > 0.0 else 0.0
    return float(gross_profit / gross_loss)


def max_drawdown_usd_from_pnls(pnls: np.ndarray) -> float:
    if pnls.size == 0:
        return 0.0
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(np.max(dd)) if dd.size else 0.0


def collect_trade_events(
    df: pd.DataFrame,
    feat: pd.DataFrame,
    p_event_full: np.ndarray,
    active_idx: np.ndarray,
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
) -> List[Dict[str, Any]]:
    n = len(df)
    if p_event_full.shape[0] != n or len(feat) != n:
        raise ValueError("Length mismatch in collect_trade_events.")
    if active_idx.size == 0:
        return []

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    idx = df.index
    atr = (df["high"] - df["low"]).rolling(14, min_periods=1).mean().to_numpy(dtype=float)

    active_set = set(int(x) for x in active_idx.tolist())
    active_start = int(active_idx[0])
    active_end = int(active_idx[-1])
    buffer = float(buffer_pips) * PIP

    events: List[Dict[str, Any]] = []
    trades_by_session: Dict[str, int] = {}

    in_pos = False
    side = 0
    entry_px = 0.0
    entry_i = -1
    signal_i = -1
    signal_prob = 0.0

    for t in range(max(int(lookback), active_start), active_end + 1):
        if t not in active_set:
            continue
        if in_pos:
            bars_held = int(t - entry_i)
            sl_dist = float(sl_pips) * PIP if float(sl_pips) > 0.0 else float(sl_atr) * float(atr[entry_i])
            tp_dist = float(tp_pips) * PIP if float(tp_pips) > 0.0 else float(tp_atr) * float(atr[entry_i])
            if sl_dist <= 0.0:
                sl_dist = 5.0 * PIP
            if tp_dist <= 0.0:
                tp_dist = 8.0 * PIP

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
            elif t == active_end:
                reason = "fold_eod"
                raw_exit = c[t]

            if reason is not None:
                exit_px = apply_exit_price(side, float(raw_exit), costs)
                pnl = pnl_usd_from_price_move(entry_px, exit_px, side, lot)
                pnl -= commission_usd(entry_px, lot, costs)
                pnl -= commission_usd(exit_px, lot, costs)
                notional = abs(float(entry_px) * float(lot) * CONTRACT)
                pnl_bps = (float(pnl) / notional * 1e4) if notional > 0.0 else 0.0
                events.append(
                    {
                        "signal_index": int(signal_i),
                        "entry_index": int(entry_i),
                        "exit_index": int(t),
                        "signal_time": str(idx[int(signal_i)]),
                        "entry_time": str(idx[int(entry_i)]),
                        "exit_time": str(idx[int(t)]),
                        "side": int(side),
                        "p_event": float(signal_prob),
                        "market_obs": feat.iloc[int(signal_i)].to_numpy(dtype=np.float32),
                        "meta_obs": np.asarray(
                            [
                                float(side),
                                float(signal_prob),
                                float(cost_pips_est(costs, float(entry_px))),
                            ],
                            dtype=np.float32,
                        ),
                        "pnl_usd": float(pnl),
                        "pnl_bps": float(pnl_bps),
                        "bars_held": int(bars_held),
                        "reason": str(reason),
                    }
                )
                in_pos = False
                side = 0
                entry_px = 0.0
                entry_i = -1
                signal_i = -1
                signal_prob = 0.0

        if in_pos:
            continue
        if t >= active_end:
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
        if t1 not in active_set:
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
        signal_i = int(t)
        signal_prob = float(p)
        trades_by_session[sess_key] = int(trades_by_session.get(sess_key, 0)) + 1

    return events


def build_observation_matrix(events: List[Dict[str, Any]], market_features_only: bool) -> np.ndarray:
    if not events:
        return np.empty((0, 0), dtype=np.float32)
    market = np.stack([np.asarray(ev["market_obs"], dtype=np.float32) for ev in events], axis=0)
    if market_features_only:
        return market.astype(np.float32)
    meta = np.stack([np.asarray(ev["meta_obs"], dtype=np.float32) for ev in events], axis=0)
    return np.concatenate([market, meta], axis=1).astype(np.float32)


def normalize_train_test_obs(train_obs: np.ndarray, test_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if train_obs.size == 0:
        return train_obs, test_obs
    mean = np.mean(train_obs, axis=0)
    std = np.std(train_obs, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    train_n = (train_obs - mean) / std
    if test_obs.size == 0:
        return train_n.astype(np.float32), test_obs.astype(np.float32)
    test_n = (test_obs - mean) / std
    return train_n.astype(np.float32), test_n.astype(np.float32)


def train_gate_policy(
    observations: np.ndarray,
    rewards_bps: np.ndarray,
    algo: str,
    total_timesteps: int,
    seed: int,
    reward_cfg: RLMRewardConfig,
) -> PPO | A2C:
    env = DummyVecEnv([lambda: RLMGateEnv(observations, rewards_bps, reward_cfg)])
    if algo == "a2c":
        model: PPO | A2C = A2C(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            gamma=0.95,
            n_steps=64,
            vf_coef=0.5,
            ent_coef=0.0,
            verbose=0,
            seed=int(seed),
        )
    else:
        n_steps = int(max(32, min(512, observations.shape[0])))
        if n_steps % 8 != 0:
            n_steps = int(max(32, (n_steps // 8) * 8))
        batch_size = int(min(128, n_steps))
        while batch_size > 8 and (n_steps % batch_size) != 0:
            batch_size -= 1
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=0.95,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            seed=int(seed),
        )
    model.learn(total_timesteps=int(max(total_timesteps, observations.shape[0] * 8)))
    return model


def predict_take_actions(model: PPO | A2C | None, observations: np.ndarray) -> np.ndarray:
    if observations.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)
    if model is None:
        return np.ones((observations.shape[0],), dtype=np.int32)
    out = np.zeros((observations.shape[0],), dtype=np.int32)
    for i in range(observations.shape[0]):
        action, _ = model.predict(observations[i], deterministic=True)
        out[i] = 1 if int(np.asarray(action).reshape(-1)[0]) > 0 else 0
    return out


def metrics_from_actions(events: List[Dict[str, Any]], actions: np.ndarray) -> Dict[str, float]:
    if not events:
        return {
            "take_rate": 0.0,
            "trades": 0.0,
            "net_usd": 0.0,
            "pf": 0.0,
            "win_rate": 0.0,
            "avg_trade_usd": 0.0,
            "max_dd_usd": 0.0,
            "avg_hold_bars": 0.0,
        }
    if actions.shape[0] != len(events):
        raise ValueError("actions length mismatch")
    mask = actions.astype(bool)
    taken = [events[i] for i in range(len(events)) if bool(mask[i])]
    if not taken:
        return {
            "take_rate": float(np.mean(mask.astype(float))),
            "trades": 0.0,
            "net_usd": 0.0,
            "pf": 0.0,
            "win_rate": 0.0,
            "avg_trade_usd": 0.0,
            "max_dd_usd": 0.0,
            "avg_hold_bars": 0.0,
        }
    pnl = np.asarray([float(x["pnl_usd"]) for x in taken], dtype=float)
    bars = np.asarray([float(x["bars_held"]) for x in taken], dtype=float)
    pf = profit_factor_from_pnls(pnl)
    return {
        "take_rate": float(np.mean(mask.astype(float))),
        "trades": float(pnl.size),
        "net_usd": float(np.sum(pnl)),
        "pf": float(pf),
        "win_rate": float(np.mean(pnl > 0.0)),
        "avg_trade_usd": float(np.mean(pnl)),
        "max_dd_usd": float(max_drawdown_usd_from_pnls(pnl)),
        "avg_hold_bars": float(np.mean(bars)),
    }


def summarize_rows(rows: pd.DataFrame) -> Dict[str, float | int]:
    if rows.empty:
        return {
            "folds": 0,
            "net_sum": 0.0,
            "net_mean": 0.0,
            "net_std": 0.0,
            "pos_fold_rate": 0.0,
            "worst_fold_net": 0.0,
            "pf_mean": 0.0,
            "dd_worst": 0.0,
            "trades_total": 0,
            "take_rate_mean": 0.0,
            "base_net_sum": 0.0,
            "base_pf_mean": 0.0,
        }
    fold_net = rows.groupby("fold")["net_usd"].sum()
    pf_vals = rows["pf"].to_numpy(dtype=float)
    pf_vals = np.where(np.isfinite(pf_vals), pf_vals, 10.0)
    base_pf_vals = rows["base_pf"].to_numpy(dtype=float)
    base_pf_vals = np.where(np.isfinite(base_pf_vals), base_pf_vals, 10.0)
    return {
        "folds": int(rows["fold"].nunique()),
        "net_sum": float(rows["net_usd"].sum()),
        "net_mean": float(rows["net_usd"].mean()),
        "net_std": float(rows["net_usd"].std(ddof=1)) if len(rows) > 1 else 0.0,
        "pos_fold_rate": float(np.mean(fold_net > 0.0)),
        "worst_fold_net": float(fold_net.min()) if not fold_net.empty else 0.0,
        "pf_mean": float(np.mean(pf_vals)) if pf_vals.size else 0.0,
        "dd_worst": float(rows["max_dd_usd"].max()) if not rows.empty else 0.0,
        "trades_total": int(rows["trades"].sum()),
        "take_rate_mean": float(rows["take_rate"].mean()) if not rows.empty else 0.0,
        "base_net_sum": float(rows["base_net_usd"].sum()),
        "base_pf_mean": float(np.mean(base_pf_vals)) if base_pf_vals.size else 0.0,
    }


def to_markdown(payload: Dict[str, Any]) -> str:
    s = payload["summaries"].get("base", {})
    lines = [
        "# Strategy 1 + RLM Gate (Event-Level) Walk-Forward Report",
        "",
        f"- Generated (UTC): `{payload['generated_utc']}`",
        f"- Data: `{payload['data_csv']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Policy: `{payload['rl']['algo']}` | train_timesteps={payload['rl']['train_timesteps']} | min_train_events={payload['rl']['min_train_events']}",
        f"- Observation: `{'market_features_only' if payload['rl']['market_features_only'] else 'market_plus_trade_state'}`",
        f"- Costs: spread={payload['costs']['spread']}, slippage={payload['costs']['slippage']}, commission={payload['costs']['commission']}",
        "",
        "## Summary (base)",
        "",
        f"- net_sum: `{float(s.get('net_sum', 0.0)):.2f}` | net_mean: `{float(s.get('net_mean', 0.0)):.2f}` | net_std: `{float(s.get('net_std', 0.0)):.2f}`",
        f"- pos_fold_rate: `{float(s.get('pos_fold_rate', 0.0)):.2f}` | worst_fold_net: `{float(s.get('worst_fold_net', 0.0)):.2f}`",
        f"- PF_mean: `{float(s.get('pf_mean', 0.0)):.3f}` | worst_DD: `{float(s.get('dd_worst', 0.0)):.2f}` | trades_total: `{int(s.get('trades_total', 0))}`",
        "",
        "## Fold Table",
        "",
        "| Fold | Device | Test Start | Test End | Train Events | Test Events | Take Rate | Trades | Net USD | PF | Max DD | Base Trades | Base Net USD | Base PF |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in payload.get("fold_rows", []):
        lines.append(
            f"| {r['fold']} | {r['model_device']} | {r['test_start']} | {r['test_end']} | {r['train_events']} | {r['test_events']} | "
            f"{float(r['take_rate']):.3f} | {int(r['trades'])} | {float(r['net_usd']):.2f} | {float(r['pf']):.3f} | {float(r['max_dd_usd']):.2f} | "
            f"{int(r['base_trades'])} | {float(r['base_net_usd']):.2f} | {float(r['base_pf']):.3f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strategy 1 + RL gate walk-forward/lockbox evaluator.")
    p.add_argument("--data-csv", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--mode", default="auto", choices=["auto", "wf", "lockbox"])
    p.add_argument("--train-years", type=int, default=3)
    p.add_argument("--test-months", type=int, default=6)
    p.add_argument("--step-months", type=int, default=6)
    p.add_argument("--max-folds", type=int, default=10)
    p.add_argument("--lockbox-train-end", default="")
    p.add_argument("--lockbox-test-start", default="")
    p.add_argument("--lockbox-test-end", default="")

    p.add_argument("--horizon-bars", type=int, default=6)
    p.add_argument("--move-threshold-pips", type=float, default=40.0)
    p.add_argument("--prob-th", type=float, default=0.62)
    p.add_argument("--exec-mode", default="conservative", choices=["conservative", "aggressive"])
    p.add_argument("--lookback", type=int, default=24)
    p.add_argument("--buffer-pips", type=float, default=1.0)
    p.add_argument("--sl-pips", type=float, default=12.0)
    p.add_argument("--tp-pips", type=float, default=24.0)
    p.add_argument("--sl-atr", type=float, default=1.0)
    p.add_argument("--tp-atr", type=float, default=1.6)
    p.add_argument("--time-stop-bars", type=int, default=12)
    p.add_argument("--session-filter", default="london_ny", choices=["off", "london_only", "ny_only", "london_ny"])
    p.add_argument("--hour-windows", default="7-10")
    p.add_argument("--max-trades-per-session", type=int, default=1)
    p.add_argument("--regime-filter", default="trend_or_range", choices=["off", "trend_only", "range_only", "trend_or_range"])
    p.add_argument("--trend-min", type=float, default=0.00010)
    p.add_argument("--compression-max-quantile", type=float, default=0.35)
    p.add_argument("--compression-window", type=int, default=24)
    p.add_argument("--atr-norm-min-quantile", type=float, default=0.0)
    p.add_argument("--atr-norm-max-quantile", type=float, default=0.60)
    p.add_argument("--atr-window", type=int, default=24)
    p.add_argument("--require-close-confirm", action="store_true")
    p.add_argument("--lot", type=float, default=0.05)
    p.add_argument("--spread", type=float, default=0.00020)
    p.add_argument("--slippage", type=float, default=0.00005)
    p.add_argument("--commission", type=float, default=0.00010)
    p.add_argument("--lgbm-device", default="auto", choices=["auto", "gpu", "cpu"])

    p.add_argument("--rl-algo", default="ppo", choices=["ppo", "a2c"])
    p.add_argument("--rl-train-timesteps", type=int, default=30000)
    p.add_argument("--rl-min-train-events", type=int, default=30)
    p.add_argument("--rl-seed", type=int, default=42)
    p.add_argument("--rl-skip-penalty-bps", type=float, default=0.0)
    p.add_argument("--rl-trade-penalty-bps", type=float, default=0.05)
    p.add_argument("--rl-reward-clip-bps", type=float, default=500.0)
    p.add_argument("--market-features-only", dest="market_features_only", action="store_true")
    p.add_argument("--market-plus-trade-state", dest="market_features_only", action="store_false")
    p.set_defaults(market_features_only=True)
    p.add_argument("--save-fold-policies", dest="save_fold_policies", action="store_true")
    p.add_argument("--no-save-fold-policies", dest="save_fold_policies", action="store_false")
    p.set_defaults(save_fold_policies=False)
    p.add_argument("--policy-dir", default="models/rlm_gate")
    p.add_argument("--train-only", action="store_true", help="Train fold policies and report train-slice metrics only.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.data_csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    hour_windows = parse_hour_windows(args.hour_windows)
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
        raise ValueError("--atr-norm-min-quantile cannot exceed --atr-norm-max-quantile.")

    df = load_ohlcv_csv(csv_path)
    feat = build_features(df)
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

    has_lockbox = (
        args.mode == "lockbox"
        or (args.mode == "auto" and bool(args.lockbox_train_end or args.lockbox_test_start or args.lockbox_test_end))
    )
    if has_lockbox:
        if not (args.lockbox_train_end and args.lockbox_test_start and args.lockbox_test_end):
            raise ValueError("Lockbox mode requires --lockbox-train-end --lockbox-test-start --lockbox-test-end.")
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
        raise ValueError("No walk-forward splits produced.")

    costs = Costs(
        spread=float(args.spread),
        slippage=float(args.slippage),
        commission=float(args.commission),
    )
    price_ref = float(np.median(df["close"].to_numpy(dtype=float)))
    rt_cost_est_pips = cost_pips_est(costs, price_ref)
    reward_cfg = RLMRewardConfig(
        skip_penalty_bps=float(args.rl_skip_penalty_bps),
        take_trade_penalty_bps=float(args.rl_trade_penalty_bps),
        reward_clip_bps=float(args.rl_reward_clip_bps),
    )

    runtime_state: Dict[str, Any] = {
        "requested_device": str(args.lgbm_device),
        "gpu_disabled": False,
        "gpu_error": None,
        "gpu_success_folds": 0,
        "cpu_used_folds": 0,
        "cpu_fallback_folds": 0,
    }
    policy_dir = Path(args.policy_dir)
    if not policy_dir.is_absolute():
        policy_dir = ROOT / policy_dir
    if args.save_fold_policies:
        policy_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: List[Dict[str, Any]] = []
    for fold_num, (tr_idx_raw, te_idx) in enumerate(splits, start=1):
        tr_cutoff = int(te_idx[0]) - int(horizon)
        tr_idx = tr_idx_raw[tr_idx_raw < tr_cutoff]
        tr_idx = tr_idx[~np.isnan(y_event[tr_idx])]
        if tr_idx.size < 1000:
            print(f"fold {fold_num:02d}/{len(splits)} skipped (insufficient strict-train rows)")
            continue

        p_train, p_test, fold_device = fit_event_model_predict(
            feat,
            y_event_int,
            tr_idx,
            te_idx,
            lgbm_device=str(args.lgbm_device),
            runtime_state=runtime_state,
        )

        p_train_full = np.full(len(df), np.nan, dtype=float)
        p_test_full = np.full(len(df), np.nan, dtype=float)
        p_train_full[tr_idx] = p_train
        p_test_full[te_idx] = p_test

        train_events = collect_trade_events(
            df=df,
            feat=feat,
            p_event_full=p_train_full,
            active_idx=tr_idx,
            prob_th=float(args.prob_th),
            mode=str(args.exec_mode),
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
        test_events = collect_trade_events(
            df=df,
            feat=feat,
            p_event_full=p_test_full,
            active_idx=te_idx,
            prob_th=float(args.prob_th),
            mode=str(args.exec_mode),
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

        train_obs = build_observation_matrix(train_events, market_features_only=bool(args.market_features_only))
        test_obs = build_observation_matrix(test_events, market_features_only=bool(args.market_features_only))
        train_obs, test_obs = normalize_train_test_obs(train_obs, test_obs)
        train_rewards_bps = np.asarray([float(ev["pnl_bps"]) for ev in train_events], dtype=np.float32)

        model: PPO | A2C | None = None
        policy_path = ""
        if train_events and len(train_events) >= int(args.rl_min_train_events):
            model = train_gate_policy(
                observations=train_obs,
                rewards_bps=train_rewards_bps,
                algo=str(args.rl_algo),
                total_timesteps=int(args.rl_train_timesteps),
                seed=int(args.rl_seed) + int(fold_num),
                reward_cfg=reward_cfg,
            )
            if args.save_fold_policies:
                path = policy_dir / f"{args.out_prefix}_fold{fold_num:02d}.zip"
                model.save(str(path))
                policy_path = str(path)

        eval_events = train_events if bool(args.train_only) else test_events
        eval_obs = train_obs if bool(args.train_only) else test_obs
        actions = predict_take_actions(model, eval_obs)
        selected_metrics = metrics_from_actions(eval_events, actions)
        base_actions = np.ones((len(eval_events),), dtype=np.int32)
        base_metrics = metrics_from_actions(eval_events, base_actions)
        fold_rows.append(
            {
                "mode": "lockbox" if has_lockbox else "wf",
                "fold": int(fold_num),
                "train_start": str(df.index[int(tr_idx[0])]),
                "train_end": str(df.index[int(tr_idx[-1])]),
                "test_start": str(df.index[int(te_idx[0])]),
                "test_end": str(df.index[int(te_idx[-1])]),
                "model_device": str(fold_device),
                "train_events": int(len(train_events)),
                "test_events": int(len(test_events)),
                "policy_trained": bool(model is not None),
                "policy_path": policy_path,
                "eval_scope": "train" if bool(args.train_only) else "test",
                "take_rate": float(selected_metrics["take_rate"]),
                "trades": int(selected_metrics["trades"]),
                "net_usd": float(selected_metrics["net_usd"]),
                "pf": float(selected_metrics["pf"]),
                "win_rate": float(selected_metrics["win_rate"]),
                "avg_trade_usd": float(selected_metrics["avg_trade_usd"]),
                "max_dd_usd": float(selected_metrics["max_dd_usd"]),
                "avg_hold_bars": float(selected_metrics["avg_hold_bars"]),
                "base_take_rate": float(base_metrics["take_rate"]),
                "base_trades": int(base_metrics["trades"]),
                "base_net_usd": float(base_metrics["net_usd"]),
                "base_pf": float(base_metrics["pf"]),
                "base_max_dd_usd": float(base_metrics["max_dd_usd"]),
            }
        )
        print(f"fold {fold_num:02d}/{len(splits)} done")

    rows_df = pd.DataFrame(fold_rows)
    payload = {
        "generated_utc": utc_now().isoformat(),
        "data_csv": str(csv_path.resolve()),
        "mode": "lockbox" if has_lockbox else "wf",
        "event": {
            "horizon_bars": int(args.horizon_bars),
            "move_threshold_pips": float(args.move_threshold_pips),
            "prob_th": float(args.prob_th),
        },
        "walk_forward": {
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
            "exec_mode": str(args.exec_mode),
            "lookback": int(args.lookback),
            "buffer_pips": float(args.buffer_pips),
            "sl_pips": float(args.sl_pips),
            "tp_pips": float(args.tp_pips),
            "sl_atr": float(args.sl_atr),
            "tp_atr": float(args.tp_atr),
            "time_stop_bars": int(args.time_stop_bars),
        },
        "costs": {
            "spread": float(args.spread),
            "slippage": float(args.slippage),
            "commission": float(args.commission),
            "rt_cost_est_pips": float(rt_cost_est_pips),
        },
        "model_runtime": {
            "requested_device": str(args.lgbm_device),
            "gpu_success_folds": int(runtime_state.get("gpu_success_folds", 0)),
            "cpu_used_folds": int(runtime_state.get("cpu_used_folds", 0)),
            "cpu_fallback_folds": int(runtime_state.get("cpu_fallback_folds", 0)),
            "gpu_error": runtime_state.get("gpu_error"),
        },
        "rl": {
            "algo": str(args.rl_algo),
            "train_timesteps": int(args.rl_train_timesteps),
            "min_train_events": int(args.rl_min_train_events),
            "seed": int(args.rl_seed),
            "skip_penalty_bps": float(args.rl_skip_penalty_bps),
            "trade_penalty_bps": float(args.rl_trade_penalty_bps),
            "reward_clip_bps": float(args.rl_reward_clip_bps),
            "market_features_only": bool(args.market_features_only),
            "save_fold_policies": bool(args.save_fold_policies),
            "policy_dir": str(policy_dir.resolve()),
            "train_only": bool(args.train_only),
        },
        "fold_rows": fold_rows,
        "summaries": {"base": summarize_rows(rows_df)},
        "params": vars(args),
    }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = utc_now().strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"{args.out_prefix}_{ts}.json"
    md_path = reports_dir / f"{args.out_prefix}_{ts}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(payload), encoding="utf-8")

    print("=== DONE ===")
    print(
        "model_runtime "
        f"requested_device={payload['model_runtime']['requested_device']} "
        f"gpu_success_folds={payload['model_runtime']['gpu_success_folds']} "
        f"cpu_used_folds={payload['model_runtime']['cpu_used_folds']} "
        f"cpu_fallback_folds={payload['model_runtime']['cpu_fallback_folds']}"
    )
    if payload["model_runtime"]["gpu_error"]:
        print(f"model_runtime gpu_error={payload['model_runtime']['gpu_error']}")
    print(f"report_json={json_path}")
    print(f"report_md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
