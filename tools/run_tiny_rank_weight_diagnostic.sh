#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-5}"
TRAIN_SEED="${TRAIN_SEED:-0}"
OUT_ROOT="${OUT_ROOT:-results/shared_rank_tiny_weight_0p001}"
BASE_ROOT_PRIMARY="${BASE_ROOT_PRIMARY:-results/shared_rank_official_protocol_v2}"
BASE_ROOT_SECONDARY="${BASE_ROOT_SECONDARY:-results/shared_rank_official_protocol_v2_remaining3}"
RESUME="${RESUME:-1}"

# This is a single pre-declared scale diagnostic, not a weight grid.
RANK_WEIGHT="0.001"
RANK_MARGIN="0.02"
RANK_TEMPERATURE="0.05"
ACTIONS=(TES PCS Ball Clubs Hoop Ribbon)
BASE_ROOTS=("$BASE_ROOT_PRIMARY" "$BASE_ROOT_SECONDARY")

if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 1
fi

find_base_log() {
  local action="$1"
  local name="${action}_base_seed${TRAIN_SEED}.log"
  local found=""
  local root
  for root in "${BASE_ROOTS[@]}"; do
    local candidate="$root/$action/$name"
    if [[ -f "$candidate" ]] \
        && grep -qF "best avg coef:" "$candidate"; then
      if [[ -n "$found" ]]; then
        echo "Duplicate completed Base logs for $action:" >&2
        echo "  $found" >&2
        echo "  $candidate" >&2
        return 1
      fi
      found="$candidate"
    fi
  done
  if [[ -z "$found" ]]; then
    echo "Completed Base log not found for $action under:" >&2
    printf '  %s\n' "${BASE_ROOTS[@]}" >&2
    return 1
  fi
  printf '%s\n' "$found"
}

echo "Tiny shared-rank diagnostic: weight=$RANK_WEIGHT seed=$TRAIN_SEED"
echo "Actions: ${ACTIONS[*]}"
for action in "${ACTIONS[@]}"; do
  base_log="$(find_base_log "$action")"
  echo "Reuse Base: $base_log"
done

mkdir -p "$OUT_ROOT"
python -m py_compile \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/verify_shared_pairwise_rank.py \
  tools/summarize_original_protocol.py
python tools/verify_shared_pairwise_rank.py \
  2>&1 | tee "$OUT_ROOT/shared_pairwise_verification.log"

for action in "${ACTIONS[@]}"; do
  out="$OUT_ROOT/$action"
  checkpoint="$out/${action}_pairwise_seed${TRAIN_SEED}.pth"
  log="$out/${action}_pairwise_seed${TRAIN_SEED}.log"
  mkdir -p "$out"

  if [[ "$RESUME" == 1 && -f "$checkpoint" && -f "$log" ]] \
      && grep -qF "best avg coef:" "$log" \
      && grep -qF "0.001*pairwise logistic" "$log"; then
    echo "Resume: reuse $action Pairwise weight=$RANK_WEIGHT"
    continue
  fi

  python main.py \
    --gpu "$GPU" \
    --feats 2 \
    --action "$action" \
    --seed "$TRAIN_SEED" \
    --sequence_sampling legacy \
    --multi_modality \
    --shared_rank_weight "$RANK_WEIGHT" \
    --shared_rank_margin "$RANK_MARGIN" \
    --shared_rank_temperature "$RANK_TEMPERATURE" \
    --save_final_model "$checkpoint" \
    2>&1 | tee "$log"
done

python tools/summarize_original_protocol.py \
  --base-roots "${BASE_ROOTS[@]}" \
  --pairwise-root "$OUT_ROOT" \
  --actions "${ACTIONS[@]}" \
  --seed "$TRAIN_SEED" \
  --output "$OUT_ROOT/original_protocol_summary.csv"

echo "Completed six-action tiny-weight diagnostic."
echo "Summary: $OUT_ROOT/original_protocol_summary.csv"
