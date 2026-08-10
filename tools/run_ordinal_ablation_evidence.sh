#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-5}"
TRAIN_SEED="${TRAIN_SEED:-0}"
REFERENCE_ROOT="${REFERENCE_ROOT:-results/ordinal_prototype_seed0}"
OUT_ROOT="${OUT_ROOT:-results/ordinal_prototype_ablation_seed0}"
FEAT_DIR="${FEAT_DIR:-./data/features}"
RESUME="${RESUME:-1}"

AUX_WEIGHT="0.0001"
PROTOTYPE_DIM="128"
PROTOTYPE_TEMPERATURE="0.1"
PROTOTYPE_ORDER_MARGIN="0.1"
ACTIONS=(TES PCS Ball Clubs Hoop Ribbon)

if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 1
fi

for required in \
    "$REFERENCE_ROOT/original_protocol_summary.csv" \
    "$REFERENCE_ROOT/ordinal_prototype_diagnostics.csv"; do
  if [[ ! -f "$required" ]]; then
    echo "Required reference result not found: $required" >&2
    exit 1
  fi
done
for action in "${ACTIONS[@]}"; do
  checkpoint="$REFERENCE_ROOT/$action/${action}_prototype_seed${TRAIN_SEED}.pth"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Required ordered checkpoint not found: $checkpoint" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"
python -m py_compile \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/verify_ordinal_prototypes.py \
  tools/verify_ordinal_ablation.py \
  tools/analyze_ordinal_prototypes.py \
  tools/analyze_ordinal_feature_evidence.py \
  tools/summarize_original_protocol.py \
  tools/summarize_ordinal_ablation.py
python tools/verify_ordinal_prototypes.py \
  2>&1 | tee "$OUT_ROOT/ordinal_prototype_verification.log"
python tools/verify_ordinal_ablation.py \
  2>&1 | tee "$OUT_ROOT/ordinal_ablation_verification.log"

run_mode() {
  local action="$1"
  local mode="$2"
  local out="$OUT_ROOT/$action"
  local checkpoint="$out/${action}_${mode}_seed${TRAIN_SEED}.pth"
  local log="$out/${action}_${mode}_seed${TRAIN_SEED}.log"
  local marker=""
  local args=()
  mkdir -p "$out"

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
    echo "Resume: reuse action=$action mode=$mode"
    return
  fi

  python main.py \
    --gpu "$GPU" \
    --feats 2 \
    --action "$action" \
    --seed "$TRAIN_SEED" \
    --feat_dir "$FEAT_DIR" \
    --sequence_sampling legacy \
    --multi_modality \
    "${args[@]}" \
    --save_final_model "$checkpoint" \
    2>&1 | tee "$log"
}

for action in "${ACTIONS[@]}"; do
  run_mode "$action" grade
  run_mode "$action" unordered
done

python tools/summarize_ordinal_ablation.py \
  --reference-summary \
    "$REFERENCE_ROOT/original_protocol_summary.csv" \
  --candidate-root "$OUT_ROOT" \
  --actions "${ACTIONS[@]}" \
  --seed "$TRAIN_SEED" \
  --output-runs "$OUT_ROOT/ablation_runs.csv" \
  --output-summary "$OUT_ROOT/ablation_summary.csv" \
  --output-advantages "$OUT_ROOT/ordered_advantages.csv"

python tools/analyze_ordinal_prototypes.py \
  --root "$REFERENCE_ROOT" \
  --actions "${ACTIONS[@]}" \
  --seed "$TRAIN_SEED" \
  --mode prototype \
  --output "$OUT_ROOT/ordered_prototype_diagnostics.csv"

python tools/analyze_ordinal_prototypes.py \
  --root "$OUT_ROOT" \
  --actions "${ACTIONS[@]}" \
  --seed "$TRAIN_SEED" \
  --mode unordered \
  --allow-unordered \
  --output "$OUT_ROOT/unordered_prototype_diagnostics.csv"

python tools/analyze_ordinal_feature_evidence.py \
  --mode-roots \
    "grade=$OUT_ROOT" \
    "unordered=$OUT_ROOT" \
    "prototype=$REFERENCE_ROOT" \
  --actions "${ACTIONS[@]}" \
  --seed "$TRAIN_SEED" \
  --feat-dir "$FEAT_DIR" \
  --gpu "$GPU" \
  --output-summary "$OUT_ROOT/feature_evidence_summary.csv" \
  --output-samples "$OUT_ROOT/feature_evidence_samples.csv"

echo "Completed six-action ordinal ablation and feature evidence."
echo "Ablation: $OUT_ROOT/ablation_summary.csv"
echo "Ordered advantages: $OUT_ROOT/ordered_advantages.csv"
echo "Feature evidence: $OUT_ROOT/feature_evidence_summary.csv"
