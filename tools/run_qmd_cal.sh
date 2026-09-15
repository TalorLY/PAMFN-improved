#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
QMD_ROOT=${QMD_ROOT:-results/qmd_cal_v4}
REFERENCE_ROOT=${REFERENCE_ROOT:-../PAMFN-qmd-dual-v2/results/qmd_dual_v2}
GPU_LIST=${GPU_LIST:-3 4 5}
PYTHON_BIN=${PYTHON_BIN:-python}
"$PYTHON_BIN" -u -m tools.run_qmd_cal --stage all --root "$QMD_ROOT" --gpus "$GPU_LIST" --reference_root "$REFERENCE_ROOT"
