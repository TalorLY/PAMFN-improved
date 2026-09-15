#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -u -m tools.run_qmd_split --stage "${STAGE:-all}" \
  --root "${QMD_ROOT:-results/qmd_split_v9}" --gpus "${GPU_LIST:-4}" \
  --reference_root "${REFERENCE_ROOT:-../PAMFN-qmd-dual-v2/results/qmd_dual_v2}" \
  --v8_root "${V8_ROOT:-../PAMFN-qmd-grad-v8-fix1/results/qmd_grad_v8_fix1}" \
  --july28_repo "${JULY28_REPO:-../PAMFN-july28-qmd-split-v9}"
