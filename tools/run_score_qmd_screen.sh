#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
: "${GPU_LIST:?Set GPU_LIST to currently available GPUs}"
python -u tools/run_score_qmd_experiments.py --stage screen \
  --root "${ROOT:-results/score_qmd_v1_screen}" --gpus "$GPU_LIST" \
  --smoke_root "${SMOKE_ROOT:-results/score_qmd_v1_smoke}" \
  --base_root "${BASE_ROOT:-../PAMFN-coqm-all-action-screen-v1/results/coqm_all_action_formal_v1}" \
  --resume "${RESUME:-1}"
