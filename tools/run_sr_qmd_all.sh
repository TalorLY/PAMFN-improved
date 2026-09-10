#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${GPU_LIST:?Set GPU_LIST to currently available GPUs}"
smoke_gpu="${SMOKE_GPU:-${GPU_LIST%% *}}"
smoke_root="${SMOKE_ROOT:-results/sr_qmd_ordered_v1_smoke}"
screen_root="${SCREEN_ROOT:-results/sr_qmd_ordered_v1_screen}"
mkdir -p "$smoke_root" "$screen_root"
GPU="$smoke_gpu" ROOT="$smoke_root" bash tools/run_sr_qmd_smoke.sh \
  2>&1 | tee -a "$smoke_root/launcher.log"
ROOT="$screen_root" SMOKE_ROOT="$smoke_root" bash tools/run_sr_qmd_screen.sh \
  2>&1 | tee -a "$screen_root/launcher.log"
echo 'SR_QMD_ALL_COMPLETE'
