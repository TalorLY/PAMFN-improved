#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
QMD_ROOT=${QMD_ROOT:-results/qmd_ema_v6}
REFERENCE_ROOT=${REFERENCE_ROOT:-../PAMFN-qmd-dual-v2/results/qmd_dual_v2}
GPU_LIST=${GPU_LIST:-0 1 3 4}
PYTHON_BIN=${PYTHON_BIN:-python}
"$PYTHON_BIN" -u -m tools.run_qmd_ema --stage all --root "$QMD_ROOT" --gpus "$GPU_LIST" --reference_root "$REFERENCE_ROOT"
