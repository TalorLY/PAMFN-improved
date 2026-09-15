#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
QMD_ROOT=${QMD_ROOT:-results/qmd_light_v3}
BALL_ROOT=${BALL_ROOT:-results/qmd_light_v3_ball_mask}
JULY28_REPO=${JULY28_REPO:-../PAMFN-july28-qmd-light-v3}
REFERENCE_ROOT=${REFERENCE_ROOT:-../PAMFN-qmd-dual-v2/results/qmd_dual_v2}
GPU_LIST=${GPU_LIST:-0 1 3 4}
RUN_BALL_PROBE=${RUN_BALL_PROBE:-1}
PYTHON_BIN=${PYTHON_BIN:-python}
if [[ "$RUN_BALL_PROBE" != 0 && "$RUN_BALL_PROBE" != 1 ]]; then
  echo 'RUN_BALL_PROBE must be 0 or 1' >&2
  exit 2
fi
"$PYTHON_BIN" -u -m tools.run_qmd_light --stage all --study screen --root "$QMD_ROOT" --gpus "$GPU_LIST" --july28_repo "$JULY28_REPO" --reference_root "$REFERENCE_ROOT"
if [[ "$RUN_BALL_PROBE" == 1 ]]; then
  # Sequential studies: never oversubscribe the four GPUs with a second launcher.
  "$PYTHON_BIN" -u -m tools.run_qmd_light --stage all --study ball-mask --root "$BALL_ROOT" --gpus "$GPU_LIST" --july28_repo "$JULY28_REPO" --reference_root "$REFERENCE_ROOT"
fi
echo "QMD_LIGHT_ALL_COMPLETE: main 36/36; RUN_BALL_PROBE=$RUN_BALL_PROBE (3/3 when enabled)."
