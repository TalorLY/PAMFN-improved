#!/usr/bin/env bash
set -euo pipefail
V30_WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$V30_WORK"
test -d trainer/.git
test -d trainer/data
test -d trainer/pretrained_models
mkdir -p results/v17_evidence_v30
# Only cards explicitly granted for this study. Change this line if availability changes.
GPU_LIST="${GPU_LIST:-0 1 3 4}"
V30_ORIGIN="${V30_ORIGIN:-/szztdata/ly/exp0_source/PAMFN-v17-rebuild-v29/results/v17_rebuild_v29}"
V30_LOG="results/v17_evidence_v30/launcher_$(date +%Y%m%d_%H%M%S).log"
nohup python -u run.py --stage all --gpus "$GPU_LIST" --origin "$V30_ORIGIN" \
  > "$V30_LOG" 2>&1 < /dev/null &
V30_PID=$!
printf 'Launcher PID: %s\nLog: %s\n' "$V30_PID" "$V30_LOG"
sleep 2
if kill -0 "$V30_PID" 2>/dev/null; then
  printf 'Launcher is alive; this does NOT yet mean preflight passed.\n'
else
  tail -n 60 "$V30_LOG"
  printf 'Launcher exited early. Inspect the log before retrying.\n' >&2
  exit 1
fi
