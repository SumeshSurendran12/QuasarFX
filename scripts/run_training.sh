#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCK_FILE="${ROOT_DIR}/logs/train.lock"
SESSION_NAME="${SESSION_NAME:-train}"

if pgrep -fa "python.*modules/main.py" >/dev/null 2>&1; then
  echo "Training already running. Use: tmux attach -t ${SESSION_NAME}"
  exit 0
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' exists but no training process found."
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 0
fi

echo "Starting training in tmux session '${SESSION_NAME}'..."
tmux new-session -d -s "${SESSION_NAME}" "cd '${ROOT_DIR}' && source .venv/bin/activate && python modules/main.py | tee -a logs/train.log"
echo "Done. Attach with: tmux attach -t ${SESSION_NAME}"
echo "Logs: ${ROOT_DIR}/logs/train.log"
if [[ -f "${LOCK_FILE}" ]]; then
  echo "Lock file: ${LOCK_FILE}"
fi
