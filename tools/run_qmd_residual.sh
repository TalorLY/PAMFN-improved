#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -u -m tools.run_qmd_residual --stage "${STAGE:-all}" \
  --root "${QMD_ROOT:-results/qmd_residual_v12}" --gpus "${GPU_LIST:-0 1 3 4}" \
  --reference_root "${REFERENCE_ROOT:-../PAMFN-qmd-strength-v11/results/qmd_strength_v11}" \
  --teacher_root "${TEACHER_ROOT:-../PAMFN-qmd-dual-v2/results/qmd_dual_v2}" \
  --july28_repo "${JULY28_REPO:-../PAMFN-july28-qmd-residual-v12}"
