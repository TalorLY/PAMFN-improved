#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
: "${GPU:?Set GPU to an available GPU index}"
python -u tools/run_sr_qmd_experiments.py --stage smoke \
  --root "${ROOT:-results/sr_qmd_ordered_v1_smoke}" --gpus "$GPU" --resume "${RESUME:-1}"
