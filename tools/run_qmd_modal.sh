#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -u -m tools.run_qmd_modal --stage "${STAGE:-all}" \
  --root "${QMD_ROOT:-results/qmd_modal_v16}" --gpus "${GPU_LIST:-0 1 3 4}" \
  --reference_root "${REFERENCE_ROOT:-../PAMFN-qmd-complement-v13/results/qmd_complement_v13}" \
  --pair_root "${PAIR_ROOT:-../PAMFN-qmd-pair-v15/results/qmd_pair_v15}" \
  --teacher_root "${TEACHER_ROOT:-../PAMFN-qmd-dual-v2/results/qmd_dual_v2}" \
  --july28_repo "${JULY28_REPO:-../PAMFN-july28-qmd-modal-v16}"
