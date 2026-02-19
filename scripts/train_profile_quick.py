from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick profile-aware PPO training run.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=0,
        help="Training timesteps for this quick run. Use <=0 to use profile default.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--data-csv",
        default="",
        help="Optional CSV path. If omitted, latest file in data/ is used (or downloaded).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output model path (relative to repo root or absolute).",
    )
    return parser.parse_args()


def _resolve_output(path_value: str) -> Path:
    out = Path(path_value)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _load_df(data_csv: str):
    from modules.data_fetcher import DataFetcher
    from modules.config import SYMBOL, TIMEFRAME

    if data_csv:
        path = Path(data_csv)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Data CSV not found: {path}")
        df = pd.read_csv(path)
    else:
        data_dir = ROOT / "data"
        csvs = sorted(data_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if csvs:
            path = csvs[0]
            df = pd.read_csv(path)
        else:
            fetcher = DataFetcher()
            df = fetcher.fetch_historical_data(download_if_missing=True, symbol=SYMBOL, timeframe=TIMEFRAME, add_indicators=False)
            path = None

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    needed = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[needed].copy(), path


def main() -> None:
    args = parse_args()
    out_path = _resolve_output(args.output)

    os.environ.setdefault("WANDB_MODE", "disabled")

    from modules.config import (
        TRAINING_PROFILE,
        NUM_ENVS,
        VEC_ENV_TYPE,
        N_STEPS,
        BATCH_SIZE,
        FEATURES_DIM,
        QUICK_TRAIN_TIMESTEPS,
        MAX_TRADES_PER_WEEK,
        ACTION_OPEN_REWARD,
        ACTION_NO_TRADE_PENALTY,
        ACTION_FORCE_CLOSE_DAYS,
        ENTROPY_SCHEDULE_ENABLED,
        ENTROPY_START_COEF,
        ENTROPY_END_COEF,
        ENTROPY_WARMUP_FRACTION,
        USE_CUSTOM_REWARD_WRAPPER,
        MIN_POSITION_SIZE,
        MAX_POSITION_SIZE,
        TRAIN_RANDOM_START,
        POSITION_BALANCE_SCALING,
        POSITION_BALANCE_FLOOR,
        EVAL_ACTION_SHAPING,
    )
    from modules.main import (
        REWARD_SHAPING,
        _build_model,
        create_vectorized_env,
        split_train_val_test,
        evaluate_model,
        EntropyScheduleCallback,
    )

    df, source_path = _load_df(args.data_csv)
    train_df, val_df, test_df = split_train_val_test(df, val_fraction=0.1, test_fraction=0.1)
    train_full_df = pd.concat([train_df, val_df]).copy()
    timesteps = int(args.timesteps) if int(args.timesteps) > 0 else int(QUICK_TRAIN_TIMESTEPS)

    print(
        f"profile={TRAINING_PROFILE} vec_env={VEC_ENV_TYPE} num_envs={NUM_ENVS} "
        f"n_steps={N_STEPS} batch_size={BATCH_SIZE} features_dim={FEATURES_DIM}"
    )
    print(
        f"timesteps={timesteps} max_trades_per_week={MAX_TRADES_PER_WEEK} "
        f"open_reward={ACTION_OPEN_REWARD} no_trade_penalty={ACTION_NO_TRADE_PENALTY} "
        f"force_close_days={ACTION_FORCE_CLOSE_DAYS}"
    )
    print(
        f"position_size_min={MIN_POSITION_SIZE} position_size_max={MAX_POSITION_SIZE} "
        f"train_random_start={TRAIN_RANDOM_START} balance_scaling={POSITION_BALANCE_SCALING} "
        f"balance_floor={POSITION_BALANCE_FLOOR}"
    )
    print(
        f"entropy_sched={ENTROPY_SCHEDULE_ENABLED} ent_start={ENTROPY_START_COEF} "
        f"ent_end={ENTROPY_END_COEF} ent_warmup={ENTROPY_WARMUP_FRACTION}"
    )
    print(f"use_custom_reward_wrapper={USE_CUSTOM_REWARD_WRAPPER}")
    print(f"eval_action_shaping={EVAL_ACTION_SHAPING}")
    print(f"rows_total={len(df)} rows_train_full={len(train_full_df)} rows_test={len(test_df)}")
    if source_path is not None:
        print(f"data_csv={source_path}")

    env = create_vectorized_env(train_full_df, seed=args.seed, reward_config=REWARD_SHAPING)
    model = _build_model(
        env,
        hyperparams={"n_steps": N_STEPS, "batch_size": BATCH_SIZE},
        verbose_override=0,
    )

    callbacks = []
    if ENTROPY_SCHEDULE_ENABLED:
        callbacks.append(
            EntropyScheduleCallback(
                total_timesteps=timesteps,
                start_coef=ENTROPY_START_COEF,
                end_coef=ENTROPY_END_COEF,
                warmup_fraction=ENTROPY_WARMUP_FRACTION,
            )
        )

    t0 = time.perf_counter()
    model.learn(total_timesteps=timesteps, callback=callbacks or None)
    elapsed = time.perf_counter() - t0

    model.save(str(out_path))
    metrics = evaluate_model(model, test_df, tag=f"quick_{TRAINING_PROFILE}")
    env.close()

    payload = {
        "generated_at": time.time(),
        "profile": TRAINING_PROFILE,
        "output_model": str(out_path),
        "timesteps": timesteps,
        "elapsed_sec": float(elapsed),
        "steps_per_sec": float(timesteps / elapsed) if elapsed > 0 else 0.0,
        "metrics": metrics,
    }
    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"output_model={out_path}")
    print(f"metrics_json={metrics_path}")
    print(f"elapsed_sec={elapsed:.2f}")
    print(f"steps_per_sec={payload['steps_per_sec']:.1f}")
    print(f"test_profit_factor={metrics.get('profit_factor', 0):.4f}")
    print(f"test_return_pct={metrics.get('return_pct', 0):.2f}")


if __name__ == "__main__":
    main()
