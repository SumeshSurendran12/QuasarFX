#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi

cd "$ROOT/modules"
"$PYTHON" - <<'PY'
import pandas as pd
from pathlib import Path
import config
import main

root = Path.cwd().parent
monitor_candidates = [
    root / "logs" / "monitor-0.monitor.csv",
    root / "modules" / "logs" / "monitor-0.monitor.csv",
]
df = None
for candidate in monitor_candidates:
    if candidate.exists():
        df = pd.read_csv(candidate, comment="#")
        break
if df is None:
    locations = ", ".join(str(p) for p in monitor_candidates)
    print(f"monitor file not found: {locations}")
    raise SystemExit(1)

done = int(df["l"].sum()) if len(df) else 0

recent = df.tail(1000)
dt = recent["t"].iloc[-1] - recent["t"].iloc[0] if len(recent) > 1 else 0
sps = (recent["l"].sum() / dt) if dt > 0 else 0

selection_total = main.MODEL_SELECTION_TRIALS * len(main.MODEL_SELECTION_SEEDS) * main.MODEL_SELECTION_TIMESTEPS
total_all = selection_total + config.MAX_TIMESTEPS * (1 + len(main.FINAL_TRAIN_SEEDS))

done_main = max(done - selection_total, 0)

selection_remaining = max(selection_total - done, 0)
remaining_to_final = selection_remaining + max(config.MAX_TIMESTEPS - done_main, 0)
remaining_total = max(total_all - done, 0)

print(f"steps/sec: {sps:.1f}")
print(f"main_done: {done_main:,} / {config.MAX_TIMESTEPS:,}")
print(f"total_done: {done:,} / {total_all:,}")

if sps > 0:
    print(f"eta_to_final_stage_hours: {remaining_to_final / sps / 3600:.2f}")
    print(f"eta_to_full_completion_hours: {remaining_total / sps / 3600:.2f}")
else:
    print("eta_to_final_stage_hours: n/a")
    print("eta_to_full_completion_hours: n/a")
PY
