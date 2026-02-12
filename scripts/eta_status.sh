#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi

cd "$ROOT/modules"
ROOT="$ROOT" "$PYTHON" - <<'PY'
import os
from pathlib import Path

import pandas as pd

import contextlib
import io

import config
with contextlib.redirect_stdout(io.StringIO()):
    import main

root = Path(os.environ["ROOT"])
monitor_candidates = [
    root / "logs" / "monitor-0.monitor.csv",
    root / "modules" / "logs" / "monitor-0.monitor.csv",
]
monitor_path = next((p for p in monitor_candidates if p.exists()), None)

if monitor_path is None:
    print("ETA n/a")
    raise SystemExit(0)

df = pd.read_csv(monitor_path, comment="#")
if df.empty:
    print("ETA n/a")
    raise SystemExit(0)

done = int(df["l"].sum())
recent = df.tail(1000)
dt = recent["t"].iloc[-1] - recent["t"].iloc[0] if len(recent) > 1 else 0
sps = (recent["l"].sum() / dt) if dt > 0 else 0

selection_total = 0
if main.MODEL_SELECTION_ENABLED:
    selection_total = main.MODEL_SELECTION_TRIALS * len(main.MODEL_SELECTION_SEEDS) * main.MODEL_SELECTION_TIMESTEPS
total_all = selection_total + config.MAX_TIMESTEPS * (1 + len(main.FINAL_TRAIN_SEEDS))

done_main = max(done - selection_total, 0)
selection_remaining = max(selection_total - done, 0)
remaining_to_final = selection_remaining + max(config.MAX_TIMESTEPS - done_main, 0)
remaining_total = max(total_all - done, 0)

if sps > 0:
    eta_final = remaining_to_final / sps / 3600
    eta_full = remaining_total / sps / 3600
    print(f"ETA {eta_final:.2f}h/{eta_full:.2f}h SPS {sps:.0f}")
else:
    print("ETA n/a")
PY
