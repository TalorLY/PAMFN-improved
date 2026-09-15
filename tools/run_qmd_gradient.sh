#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
reuse_args=()
if [ -n "${REUSE_ROOT:-}" ]; then reuse_args=(--reuse_root "$REUSE_ROOT"); fi
exec python -u -m tools.run_qmd_gradient --stage "${STAGE:-all}" --study "${STUDY:-screen}" \
  --root "${QMD_ROOT:-results/qmd_grad_v8_fix1}" --gpus "${GPU_LIST:-0 1 3 4}" \
  --reference_root "${REFERENCE_ROOT:-../PAMFN-qmd-dual-v2/results/qmd_dual_v2}" \
  --july28_repo "${JULY28_REPO:-../PAMFN-july28-qmd-grad-v8-fix1}" "${reuse_args[@]}"
