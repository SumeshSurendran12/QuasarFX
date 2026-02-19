"""
Quick GPU benchmark for training throughput (steps/sec).
Run from the modules/ directory:
  python benchmark.py
Optional env vars:
  BENCH_TIMESTEPS=20000
  BENCH_SEED=0
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

try:
    from .data_fetcher import DataFetcher
    from .config import (
        SYMBOL,
        TIMEFRAME,
        N_STEPS,
        BATCH_SIZE,
        TRAINING_PROFILE,
        NUM_ENVS,
        VEC_ENV_TYPE,
        FEATURES_DIM,
        POLICY_NET_ARCH,
    )
    from .main import create_vectorized_env, _build_model, REWARD_SHAPING
except ImportError:  # pragma: no cover - script mode fallback
    from data_fetcher import DataFetcher
    from config import (
        SYMBOL,
        TIMEFRAME,
        N_STEPS,
        BATCH_SIZE,
        TRAINING_PROFILE,
        NUM_ENVS,
        VEC_ENV_TYPE,
        FEATURES_DIM,
        POLICY_NET_ARCH,
    )
    from main import create_vectorized_env, _build_model, REWARD_SHAPING


def _load_df() -> pd.DataFrame:
    data_dir = Path("data")
    csvs = sorted(
        data_dir.glob("*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if csvs:
        df = pd.read_csv(csvs[0])
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df

    fetcher = DataFetcher()
    return fetcher.fetch_historical_data(download_if_missing=True, symbol=SYMBOL, timeframe=TIMEFRAME)


def main() -> None:
    bench_timesteps = int(os.getenv("BENCH_TIMESTEPS", "20000"))
    seed = int(os.getenv("BENCH_SEED", "0"))

    df = _load_df()
    if df is None or df.empty:
        raise ValueError("No data available for benchmark.")

    env = create_vectorized_env(df, seed=seed, reward_config=REWARD_SHAPING)
    model = _build_model(
        env,
        hyperparams={"n_steps": N_STEPS, "batch_size": BATCH_SIZE},
        verbose_override=0,
    )

    start = time.perf_counter()
    model.learn(total_timesteps=bench_timesteps)
    elapsed = time.perf_counter() - start

    steps_per_sec = bench_timesteps / elapsed if elapsed > 0 else 0.0
    print(f"profile={TRAINING_PROFILE}")
    print(f"vec_env={VEC_ENV_TYPE}")
    print(f"num_envs={NUM_ENVS}")
    print(f"n_steps={N_STEPS}")
    print(f"batch_size={BATCH_SIZE}")
    print(f"features_dim={FEATURES_DIM}")
    print(f"policy_net_arch={POLICY_NET_ARCH}")
    print(f"bench_timesteps={bench_timesteps}")
    print(f"elapsed_sec={elapsed:.2f}")
    print(f"steps_per_sec={steps_per_sec:.1f}")


if __name__ == "__main__":
    main()
