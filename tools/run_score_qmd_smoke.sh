#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
: "${GPU:?Set GPU to a currently available GPU}"
python -u tools/run_score_qmd_experiments.py --stage smoke \
  --root "${ROOT:-results/score_qmd_v1_smoke}" --gpus "$GPU" \
  --resume "${RESUME:-1}"
