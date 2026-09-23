#!/usr/bin/env bash
set -euo pipefail
V30_WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V30_ORIGIN="${V30_ORIGIN:-/szztdata/ly/exp0_source/PAMFN-v17-rebuild-v29/results/v17_rebuild_v29}"
V30_DATA="${V30_DATA:-/szztdata/ly/exp0_source/PAMFN-main/data}"
V30_PRETRAINED="${V30_PRETRAINED:-/szztdata/ly/exp0_source/PAMFN-main/pretrained_models}"
V30_COMMIT=1437dd4b9ce3fc220bdf530946a83d4bac88c4f7
cd "$V30_WORK"
sha256sum -c SHA256SUMS
test -f "$V30_ORIGIN/completion.json"
test -d "$V30_DATA"
test -d "$V30_PRETRAINED/feats2"
if [ ! -e trainer ]; then
  git -c core.autocrlf=false clone --branch codex/july28-qmd-supp-v17 qmd_supp_v17_full.bundle trainer
  git -C trainer checkout --detach "$V30_COMMIT"
fi
test "$(git -C trainer rev-parse HEAD)" = "$V30_COMMIT"
test -z "$(git -C trainer status --porcelain --untracked-files=no)"
for V30_NAME in data pretrained_models; do
  if [ "$V30_NAME" = data ]; then V30_TARGET="$V30_DATA"; else V30_TARGET="$V30_PRETRAINED"; fi
  if [ ! -e "trainer/$V30_NAME" ] && [ ! -L "trainer/$V30_NAME" ]; then
    ln -s "$(readlink -f "$V30_TARGET")" "trainer/$V30_NAME"
  fi
  test "$(readlink -f "trainer/$V30_NAME")" = "$(readlink -f "$V30_TARGET")"
done
mkdir -p results/v17_evidence_v30
printf 'SETUP_OK: preserved trainer %s; v29 assets stay read-only.\n' "$V30_COMMIT"
