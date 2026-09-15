#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -u -m tools.run_qmd_strength --stage "${STAGE:-all}" \
  --root "${QMD_ROOT:-results/qmd_strength_v11}" --gpus "${GPU_LIST:-0 1 3 4}" \
  --reference_root "${REFERENCE_ROOT:-../PAMFN-qmd-full-v10/results/qmd_full_v10}" \
  --july28_repo "${JULY28_REPO:-../PAMFN-july28-qmd-strength-v11}"
