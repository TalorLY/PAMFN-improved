#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -u -m tools.run_qmd_reskd --stage "${STAGE:-all}" \
  --root "${QMD_ROOT:-results/qmd_reskd_v14}" --gpus "${GPU_LIST:-0 1 3 4}" \
  --reference_root "${REFERENCE_ROOT:-../PAMFN-qmd-complement-v13/results/qmd_complement_v13}" \
  --teacher_root "${TEACHER_ROOT:-../PAMFN-qmd-dual-v2/results/qmd_dual_v2}" \
  --july28_repo "${JULY28_REPO:-../PAMFN-july28-qmd-reskd-v14}"
