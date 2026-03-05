#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "scripts" / "rlm_eval_wf.py"


def main() -> int:
    args = list(sys.argv[1:])
    if "--train-only" not in args:
        args.append("--train-only")
    if "--save-fold-policies" not in args and "--no-save-fold-policies" not in args:
        args.append("--save-fold-policies")
    cmd = [sys.executable, str(EVAL_SCRIPT), *args]
    proc = subprocess.run(cmd, cwd=str(ROOT), shell=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
