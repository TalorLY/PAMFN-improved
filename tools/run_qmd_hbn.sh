#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u -m tools.run_qmd_hbn --stage "${STAGE:-all}" --gpus "${GPU_LIST:-0 1 4 5}" --root "${QMD_ROOT:-results/qmd_hbn_v7}" --reference_root "${REFERENCE_ROOT:-../PAMFN-qmd-dual-v2/results/qmd_dual_v2}" --july28_repo "${JULY28_REPO:-../PAMFN-july28-hbn-v7}"
