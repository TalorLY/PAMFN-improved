#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-5}"
FEAT_DIR="${FEAT_DIR:-./data/features}"
SEED0_ABLATION_ROOT="${SEED0_ABLATION_ROOT:-results/ordinal_prototype_ablation_seed0}"
ORDERED_MULTI_ROOT="${ORDERED_MULTI_ROOT:-results/ordinal_prototype_multiseed}"
OUT_ROOT="${OUT_ROOT:-results/ordinal_prototype_ablation_multiseed}"
RESUME="${RESUME:-1}"

AUX_WEIGHT="0.0001"
PROTOTYPE_DIM="128"
PROTOTYPE_TEMPERATURE="0.1"
PROTOTYPE_ORDER_MARGIN="0.1"
SEEDS=(1 2)
ACTIONS=(TES PCS Ball Clubs Hoop Ribbon)

if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 1
fi
for required in \
    "$SEED0_ABLATION_ROOT/ablation_runs.csv" \
    "$SEED0_ABLATION_ROOT/feature_evidence_summary.csv"; do
  if [[ ! -f "$required" ]]; then
    echo "Required seed-0 ablation not found: $required" >&2
    exit 1
  fi
done
for seed in "${SEEDS[@]}"; do
  reference_root="$ORDERED_MULTI_ROOT/seed${seed}"
  if [[ ! -f "$reference_root/original_protocol_summary.csv" ]]; then
    echo "Required Base/Ordered summary not found: $reference_root" >&2
    exit 1
  fi
  for action in "${ACTIONS[@]}"; do
    checkpoint="$reference_root/$action/${action}_prototype_seed${seed}.pth"
    if [[ ! -f "$checkpoint" ]]; then
      echo "Required Ordered checkpoint not found: $checkpoint" >&2
      exit 1
    fi
  done
done

mkdir -p "$OUT_ROOT"
python -m py_compile \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/verify_ordinal_ablation.py \
  tools/verify_ordinal_ablation_multiseed.py \
  tools/analyze_ordinal_prototypes.py \
  tools/analyze_ordinal_feature_evidence.py \
  tools/summarize_ordinal_ablation.py \
  tools/summarize_ordinal_ablation_multiseed.py
python tools/verify_ordinal_ablation.py \
  2>&1 | tee "$OUT_ROOT/ordinal_ablation_verification.log"
python tools/verify_ordinal_ablation_multiseed.py \
  2>&1 | tee "$OUT_ROOT/multiseed_ablation_verification.log"

run_mode() {
  local seed="$1"
  local action="$2"
  local mode="$3"
  local seed_root="$OUT_ROOT/seed${seed}"
  local action_root="$seed_root/$action"
  local checkpoint="$action_root/${action}_${mode}_seed${seed}.pth"
  local log="$action_root/${action}_${mode}_seed${seed}.log"
  local marker=""
  local args=()
  mkdir -p "$action_root"

  if [[ "$mode" == "grade" ]]; then
    marker="${AUX_WEIGHT}*four-grade head"
    args=(
      --ordinal_grade_weight "$AUX_WEIGHT"
      --ordinal_prototype_dim "$PROTOTYPE_DIM"
    )
  elif [[ "$mode" == "unordered" ]]; then
    marker="order_scale=0"
    args=(
      --ordinal_prototype_weight "$AUX_WEIGHT"
      --ordinal_prototype_dim "$PROTOTYPE_DIM"
      --ordinal_prototype_temperature "$PROTOTYPE_TEMPERATURE"
      --ordinal_prototype_order_margin "$PROTOTYPE_ORDER_MARGIN"
      --ordinal_prototype_order_scale 0
    )
  else
    echo "Unsupported mode: $mode" >&2
    return 1
  fi

  if [[ "$RESUME" == "1" && -f "$checkpoint" && -f "$log" ]] \
      && grep -qF "best avg coef:" "$log" \
      && grep -qF "$marker" "$log"; then
    echo "Resume: reuse action=$action seed=$seed mode=$mode"
    return
  fi

  python main.py \
    --gpu "$GPU" \
    --feats 2 \
    --action "$action" \
    --seed "$seed" \
    --feat_dir "$FEAT_DIR" \
    --sequence_sampling legacy \
    --multi_modality \
    "${args[@]}" \
    --save_final_model "$checkpoint" \
    2>&1 | tee "$log"
}

seed_results=("0=$SEED0_ABLATION_ROOT")
for seed in "${SEEDS[@]}"; do
  seed_root="$OUT_ROOT/seed${seed}"
  reference_root="$ORDERED_MULTI_ROOT/seed${seed}"

  for action in "${ACTIONS[@]}"; do
    run_mode "$seed" "$action" grade
    run_mode "$seed" "$action" unordered
  done

  python tools/summarize_ordinal_ablation.py \
    --reference-summary \
      "$reference_root/original_protocol_summary.csv" \
    --candidate-root "$seed_root" \
    --actions "${ACTIONS[@]}" \
    --seed "$seed" \
    --output-runs "$seed_root/ablation_runs.csv" \
    --output-summary "$seed_root/ablation_summary.csv" \
    --output-advantages "$seed_root/ordered_advantages.csv"

  python tools/analyze_ordinal_prototypes.py \
    --root "$reference_root" \
    --actions "${ACTIONS[@]}" \
    --seed "$seed" \
    --mode prototype \
    --output "$seed_root/ordered_prototype_diagnostics.csv"

  python tools/analyze_ordinal_prototypes.py \
    --root "$seed_root" \
    --actions "${ACTIONS[@]}" \
    --seed "$seed" \
    --mode unordered \
    --allow-unordered \
    --output "$seed_root/unordered_prototype_diagnostics.csv"

  python tools/analyze_ordinal_feature_evidence.py \
    --mode-roots \
      "grade=$seed_root" \
      "unordered=$seed_root" \
      "prototype=$reference_root" \
    --actions "${ACTIONS[@]}" \
    --seed "$seed" \
    --feat-dir "$FEAT_DIR" \
    --gpu "$GPU" \
    --output-summary "$seed_root/feature_evidence_summary.csv" \
    --output-samples "$seed_root/feature_evidence_samples.csv"

  seed_results+=("$seed=$seed_root")
done

python tools/summarize_ordinal_ablation_multiseed.py \
  --seed-results "${seed_results[@]}" \
  --actions "${ACTIONS[@]}" \
  --output-runs "$OUT_ROOT/three_seed_four_method_runs.csv" \
  --output-action-table "$OUT_ROOT/three_seed_action_table.csv" \
  --output-source-table "$OUT_ROOT/three_seed_source_table.csv" \
  --output-feature-runs "$OUT_ROOT/three_seed_feature_runs.csv" \
  --output-feature-table "$OUT_ROOT/three_seed_feature_table.csv" \
  --output-report "$OUT_ROOT/mentor_report.md"

echo "Completed Grade/Unordered seeds 1 and 2."
echo "Mentor report: $OUT_ROOT/mentor_report.md"
echo "Action table: $OUT_ROOT/three_seed_action_table.csv"
echo "Source table: $OUT_ROOT/three_seed_source_table.csv"
echo "Feature table: $OUT_ROOT/three_seed_feature_table.csv"
