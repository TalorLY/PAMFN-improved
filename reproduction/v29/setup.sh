#!/usr/bin/env bash
set -euo pipefail
V17_WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V17_DATA="${V17_DATA:-/szztdata/ly/exp0_source/PAMFN-main/data}"
V17_PRETRAINED="${V17_PRETRAINED:-/szztdata/ly/exp0_source/PAMFN-main/pretrained_models}"
V17_COMMIT=1437dd4b9ce3fc220bdf530946a83d4bac88c4f7
cd "$V17_WORK"
test -f qmd_supp_v17_full.bundle
test -f v17_history.zip
test -d "$V17_DATA"
test -d "$V17_PRETRAINED/feats2"
if [ ! -e trainer ]; then
  git -c core.autocrlf=false clone --branch codex/july28-qmd-supp-v17 qmd_supp_v17_full.bundle trainer
  git -C trainer checkout --detach "$V17_COMMIT"
fi
test "$(git -C trainer rev-parse HEAD)" = "$V17_COMMIT"
test -z "$(git -C trainer status --porcelain --untracked-files=no)"
for V17_NAME in data pretrained_models; do
  if [ "$V17_NAME" = data ]; then V17_TARGET="$V17_DATA"; else V17_TARGET="$V17_PRETRAINED"; fi
  if [ ! -e "trainer/$V17_NAME" ] && [ ! -L "trainer/$V17_NAME" ]; then
    ln -s "$(readlink -f "$V17_TARGET")" "trainer/$V17_NAME"
  fi
  test "$(readlink -f "trainer/$V17_NAME")" = "$(readlink -f "$V17_TARGET")"
done
mkdir -p results/v17_rebuild_v29
printf 'SETUP_OK: original v17 trainer %s; source untouched.\n' "$V17_COMMIT"
