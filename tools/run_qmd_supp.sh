#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -u -m tools.run_qmd_supp --stage "${STAGE:-all}" \
  --root "${QMD_ROOT:-results/qmd_supp_v17}" --gpus "${GPU_LIST:-0 1 3 4}" \
  --modal_root "${MODAL_ROOT:-../PAMFN-qmd-modal-v16/results/qmd_modal_v16}" \
  --teacher_root "${TEACHER_ROOT:-../PAMFN-qmd-dual-v2/results/qmd_dual_v2}" \
  --july28_repo "${JULY28_REPO:-../PAMFN-july28-qmd-supp-v17}"
