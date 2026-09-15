#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -u -m tools.run_qmd_full --stage "${STAGE:-all}" \
  --root "${QMD_ROOT:-results/qmd_full_v10}" --gpus "${GPU_LIST:-0 1 3 4}" \
  --reference_root "${REFERENCE_ROOT:-../PAMFN-qmd-dual-v2/results/qmd_dual_v2}" \
  --v9_root "${V9_ROOT:-../PAMFN-qmd-split-v9/results/qmd_split_v9}" \
  --july28_repo "${JULY28_REPO:-../PAMFN-july28-qmd-full-v10}"
