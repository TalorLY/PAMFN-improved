#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
: "${GPU_LIST:?Set GPU_LIST to available GPU indices}"
python -u tools/run_sr_qmd_experiments.py --stage screen \
  --root "${ROOT:-results/sr_qmd_ordered_v1_screen}" --gpus "$GPU_LIST" \
  --reference_root "${REFERENCE_ROOT:-../PAMFN-score-qmd-v1/results/score_qmd_v1_screen}" \
  --smoke_root "${SMOKE_ROOT:-results/sr_qmd_ordered_v1_smoke}" --resume "${RESUME:-1}"
