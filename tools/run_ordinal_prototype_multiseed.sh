#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-5}"
OUT_ROOT="${OUT_ROOT:-results/ordinal_prototype_multiseed}"
SEED0_ROOT="${SEED0_ROOT:-results/ordinal_prototype_seed0}"
RESUME="${RESUME:-1}"

# Freeze the seed-0 configuration for the formal paired validation.
PROTOTYPE_WEIGHT="0.0001"
PROTOTYPE_DIM="128"
PROTOTYPE_TEMPERATURE="0.1"
PROTOTYPE_ORDER_MARGIN="0.1"
PROTOTYPE_ORDER_SCALE="0.1"
SEEDS=(1 2)
ACTIONS=(TES PCS Ball Clubs Hoop Ribbon)

if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 1
fi
for required in \
    "$SEED0_ROOT/original_protocol_summary.csv" \
    "$SEED0_ROOT/ordinal_prototype_diagnostics.csv"; do
  if [[ ! -f "$required" ]]; then
    echo "Required seed-0 result not found: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"
python -m py_compile \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/verify_shared_pairwise_rank.py \
  tools/verify_shared_listwise_rank.py \
  tools/verify_ordinal_prototypes.py \
  tools/analyze_ordinal_prototypes.py \
  tools/summarize_original_protocol.py \
  tools/summarize_ordinal_multiseed.py
python tools/verify_shared_pairwise_rank.py \
  2>&1 | tee "$OUT_ROOT/shared_pairwise_verification.log"
python tools/verify_shared_listwise_rank.py \
  2>&1 | tee "$OUT_ROOT/shared_listwise_verification.log"
python tools/verify_ordinal_prototypes.py \
  2>&1 | tee "$OUT_ROOT/ordinal_prototype_verification.log"

run_model() {
  local seed="$1"
  local action="$2"
  local mode="$3"
  local seed_root="$OUT_ROOT/seed${seed}"
  local action_root="$seed_root/$action"
  local checkpoint="$action_root/${action}_${mode}_seed${seed}.pth"
  local log="$action_root/${action}_${mode}_seed${seed}.log"
  mkdir -p "$action_root"

  local completion_marker="Training objective: official PAMFN MSE"
  local prototype_args=()
  if [[ "$mode" == "prototype" ]]; then
    completion_marker="0.0001*ordinal prototypes"
    prototype_args=(
      --ordinal_prototype_weight "$PROTOTYPE_WEIGHT"
      --ordinal_prototype_dim "$PROTOTYPE_DIM"
      --ordinal_prototype_temperature "$PROTOTYPE_TEMPERATURE"
      --ordinal_prototype_order_margin "$PROTOTYPE_ORDER_MARGIN"
      --ordinal_prototype_order_scale "$PROTOTYPE_ORDER_SCALE"
    )
  fi

  if [[ "$RESUME" == "1" && -f "$checkpoint" && -f "$log" ]] \
      && grep -qF "best avg coef:" "$log" \
      && grep -qF "$completion_marker" "$log"; then
    echo "Resume: reuse action=$action seed=$seed mode=$mode"
    return
  fi

  python main.py \
    --gpu "$GPU" \
    --feats 2 \
    --action "$action" \
    --seed "$seed" \
    --sequence_sampling legacy \
    --multi_modality \
    "${prototype_args[@]}" \
    --save_final_model "$checkpoint" \
    2>&1 | tee "$log"
}

seed_results=("0=$SEED0_ROOT")
for seed in "${SEEDS[@]}"; do
  seed_root="$OUT_ROOT/seed${seed}"
  for action in "${ACTIONS[@]}"; do
    run_model "$seed" "$action" base
    run_model "$seed" "$action" prototype
  done

  python tools/summarize_original_protocol.py \
    --root "$seed_root" \
    --candidate-mode prototype \
    --actions "${ACTIONS[@]}" \
    --seed "$seed" \
    --output "$seed_root/original_protocol_summary.csv"

  python tools/analyze_ordinal_prototypes.py \
    --root "$seed_root" \
    --actions "${ACTIONS[@]}" \
    --seed "$seed" \
    --output "$seed_root/ordinal_prototype_diagnostics.csv"

  seed_results+=("$seed=$seed_root")
done

python tools/summarize_ordinal_multiseed.py \
  --seed-results "${seed_results[@]}" \
  --actions "${ACTIONS[@]}" \
  --output-runs "$OUT_ROOT/three_seed_runs.csv" \
  --output-summary "$OUT_ROOT/three_seed_summary.csv" \
  --output-diagnostics \
    "$OUT_ROOT/three_seed_prototype_diagnostics.csv"

echo "Completed paired Base/Prototype seeds 1 and 2."
echo "Three-seed summary: $OUT_ROOT/three_seed_summary.csv"
