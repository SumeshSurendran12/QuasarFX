#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

gpu="GPU n/a"
if command -v nvidia-smi >/dev/null 2>&1; then
  info="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  if [ -n "$info" ]; then
    IFS=',' read -r util mem_used mem_total <<<"$info"
    gpu="GPU ${util}% ${mem_used}/${mem_total}MiB"
  fi
fi

eta="$("$ROOT/scripts/eta_status.sh" 2>/dev/null || true)"
if [ -z "$eta" ]; then
  eta="ETA n/a"
fi

printf "%s | %s" "$gpu" "$eta"
