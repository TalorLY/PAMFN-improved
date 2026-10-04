#!/usr/bin/env bash
set -euo pipefail
V17_WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$V17_WORK"
test -d trainer/.git
test -d trainer/data
test -d trainer/pretrained_models
mkdir -p results/v17_rebuild_v29
# auto: choose only physical GPUs 0..5 with >=12000 MiB currently free.
# Set GPU_LIST="0 1 3 4" if you need to restrict to specific permitted cards.
GPU_LIST="${GPU_LIST:-auto}"
V17_LOG="results/v17_rebuild_v29/launcher_$(date +%Y%m%d_%H%M%S).log"
nohup python -u run.py --gpus "$GPU_LIST" \
  > "$V17_LOG" 2>&1 < /dev/null &
V17_PID=$!
printf 'Launcher PID: %s\nLog: %s\n' "$V17_PID" "$V17_LOG"
sleep 2
if kill -0 "$V17_PID" 2>/dev/null; then
  printf 'Launcher is alive; inspect progress with python run.py --stage status.\n'
else
  tail -n 60 "$V17_LOG"
  printf 'Launcher exited early. Inspect the log before retrying.\n' >&2
  exit 1
fi
