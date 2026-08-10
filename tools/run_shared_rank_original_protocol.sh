#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-5}"
TRAIN_SEED="${TRAIN_SEED:-0}"
ACTIONS="${ACTIONS:-TES PCS Ball}"
OUT_ROOT="${OUT_ROOT:-results/shared_rank_official_protocol_v2}"
RANK_WEIGHT="${RANK_WEIGHT:-0.005}"
RANK_MARGIN="${RANK_MARGIN:-0.02}"
RANK_TEMPERATURE="${RANK_TEMPERATURE:-0.05}"
RESUME="${RESUME:-1}"

if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
python -m py_compile \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/verify_shared_pairwise_rank.py \
  tools/summarize_original_protocol.py
python tools/verify_shared_pairwise_rank.py \
  2>&1 | tee "$OUT_ROOT/shared_pairwise_verification.log"

run_model() {
  local action="$1"
  local mode="$2"
  local out="$OUT_ROOT/$action"
  local checkpoint="$out/${action}_${mode}_seed${TRAIN_SEED}.pth"
  local log="$out/${action}_${mode}_seed${TRAIN_SEED}.log"
  mkdir -p "$out"

  if [[ "$RESUME" == 1 && -f "$checkpoint" && -f "$log" ]] \
      && grep -qF "best avg coef:" "$log"; then
    echo "Resume: reuse $action mode=$mode"
    return
  fi

  local rank_args=()
  if [[ "$mode" == "pairwise" ]]; then
    rank_args=(
      --shared_rank_weight "$RANK_WEIGHT"
      --shared_rank_margin "$RANK_MARGIN"
      --shared_rank_temperature "$RANK_TEMPERATURE"
    )
  fi

  python main.py \
    --gpu "$GPU" \
    --feats 2 \
    --action "$action" \
    --seed "$TRAIN_SEED" \
    --sequence_sampling legacy \
    --multi_modality \
    "${rank_args[@]}" \
    --save_final_model "$checkpoint" \
    2>&1 | tee "$log"
}

for action in $ACTIONS; do
  run_model "$action" base
  run_model "$action" pairwise
done

python tools/summarize_original_protocol.py \
  --root "$OUT_ROOT" \
  --actions $ACTIONS \
  --seed "$TRAIN_SEED" \
  --output "$OUT_ROOT/original_protocol_summary.csv"

echo "Completed Base + Pairwise original-protocol screen."
echo "Summary: $OUT_ROOT/original_protocol_summary.csv"
