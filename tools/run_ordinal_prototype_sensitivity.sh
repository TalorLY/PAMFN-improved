#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-5}"
TRAIN_SEED="${TRAIN_SEED:-0}"
REFERENCE_ROOT="${REFERENCE_ROOT:-results/ordinal_prototype_seed0}"
BASE_ROOT_PRIMARY="${BASE_ROOT_PRIMARY:-results/shared_rank_official_protocol_v2}"
BASE_ROOT_SECONDARY="${BASE_ROOT_SECONDARY:-results/shared_rank_official_protocol_v2_remaining3}"
OUT_ROOT="${OUT_ROOT:-results/ordinal_prototype_sensitivity_seed0}"
FEAT_DIR="${FEAT_DIR:-./data/features}"
RESUME="${RESUME:-1}"

PROTOTYPE_DIM="128"
PROTOTYPE_TEMPERATURE="0.1"
PROTOTYPE_ORDER_MARGIN="0.1"
PROTOTYPE_ORDER_SCALE="0.1"
ACTIONS=(TES PCS Ball Clubs Hoop Ribbon)
BASE_ROOTS=("$BASE_ROOT_PRIMARY" "$BASE_ROOT_SECONDARY")

# One-factor-at-a-time screen. K=4,w=1e-4 is reused from REFERENCE_ROOT,
# therefore only these four configurations (24 action runs) are trained.
SETTINGS=(
  "k3 3 0.0001"
  "k5 5 0.0001"
  "w5e5 4 0.00005"
  "w2e4 4 0.0002"
)

if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 1
fi
for required in \
    "$REFERENCE_ROOT/original_protocol_summary.csv" \
    "$REFERENCE_ROOT/ordinal_prototype_diagnostics.csv"; do
  if [[ ! -f "$required" ]]; then
    echo "Required K=4,w=1e-4 reference not found: $required" >&2
    exit 1
  fi
done

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

for action in "${ACTIONS[@]}"; do
  base_log="$(find_base_log "$action")"
  echo "Reuse Base: $base_log"
  reference_log="$REFERENCE_ROOT/$action/${action}_prototype_seed${TRAIN_SEED}.log"
  if [[ ! -f "$reference_log" ]]; then
    echo "Required reference log not found: $reference_log" >&2
    exit 1
  fi
  checkpoint="$REFERENCE_ROOT/$action/${action}_prototype_seed${TRAIN_SEED}.pth"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Required reference checkpoint not found: $checkpoint" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"
python -m py_compile \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/verify_ordinal_prototypes.py \
  tools/verify_ordinal_sensitivity.py \
  tools/analyze_ordinal_prototypes.py \
  tools/analyze_ordinal_feature_evidence.py \
  tools/summarize_original_protocol.py \
  tools/summarize_ordinal_sensitivity.py
python tools/verify_ordinal_prototypes.py \
  2>&1 | tee "$OUT_ROOT/ordinal_prototype_verification.log"
python tools/verify_ordinal_sensitivity.py \
  2>&1 | tee "$OUT_ROOT/ordinal_sensitivity_verification.log"

run_setting() {
  local setting="$1"
  local num_grades="$2"
  local weight="$3"
  local setting_root="$OUT_ROOT/$setting"
  local weight_marker
  printf -v weight_marker "%.12g" "$weight"
  mkdir -p "$setting_root"

  for action in "${ACTIONS[@]}"; do
    local action_root="$setting_root/$action"
    local checkpoint="$action_root/${action}_prototype_seed${TRAIN_SEED}.pth"
    local log="$action_root/${action}_prototype_seed${TRAIN_SEED}.log"
    mkdir -p "$action_root"

    if [[ "$RESUME" == "1" && -f "$checkpoint" && -f "$log" ]] \
        && grep -qF "best avg coef:" "$log" \
        && grep -qF "${weight_marker}*ordinal prototypes" "$log" \
        && grep -qF "grades=${num_grades};" "$log"; then
      echo "Resume: reuse setting=$setting action=$action"
      continue
    fi

    python main.py \
      --gpu "$GPU" \
      --feats 2 \
      --action "$action" \
      --seed "$TRAIN_SEED" \
      --feat_dir "$FEAT_DIR" \
      --sequence_sampling legacy \
      --multi_modality \
      --ordinal_num_grades "$num_grades" \
      --ordinal_prototype_weight "$weight" \
      --ordinal_prototype_dim "$PROTOTYPE_DIM" \
      --ordinal_prototype_temperature "$PROTOTYPE_TEMPERATURE" \
      --ordinal_prototype_order_margin "$PROTOTYPE_ORDER_MARGIN" \
      --ordinal_prototype_order_scale "$PROTOTYPE_ORDER_SCALE" \
      --save_final_model "$checkpoint" \
      2>&1 | tee "$log"
  done

  python tools/summarize_original_protocol.py \
    --base-roots "${BASE_ROOTS[@]}" \
    --candidate-root "$setting_root" \
    --candidate-mode prototype \
    --actions "${ACTIONS[@]}" \
    --seed "$TRAIN_SEED" \
    --output "$setting_root/original_protocol_summary.csv"

  python tools/analyze_ordinal_prototypes.py \
    --root "$setting_root" \
    --actions "${ACTIONS[@]}" \
    --seed "$TRAIN_SEED" \
    --mode prototype \
    --order-margin "$PROTOTYPE_ORDER_MARGIN" \
    --output "$setting_root/ordinal_prototype_diagnostics.csv"

  python tools/analyze_ordinal_feature_evidence.py \
    --mode-roots "prototype=$setting_root" \
    --actions "${ACTIONS[@]}" \
    --seed "$TRAIN_SEED" \
    --feat-dir "$FEAT_DIR" \
    --gpu "$GPU" \
    --ordinal-num-grades "$num_grades" \
    --ordinal-prototype-weight "$weight" \
    --ordinal-prototype-order-scale "$PROTOTYPE_ORDER_SCALE" \
    --output-summary "$setting_root/feature_evidence_summary.csv" \
    --output-samples "$setting_root/feature_evidence_samples.csv"
}

for setting_spec in "${SETTINGS[@]}"; do
  read -r setting num_grades weight <<< "$setting_spec"
  run_setting "$setting" "$num_grades" "$weight"
done

# Produce a report-compatible reference bundle without retraining.
REFERENCE_BUNDLE="$OUT_ROOT/k4_reference"
mkdir -p "$REFERENCE_BUNDLE"
cp "$REFERENCE_ROOT/original_protocol_summary.csv" \
  "$REFERENCE_BUNDLE/original_protocol_summary.csv"
cp "$REFERENCE_ROOT/ordinal_prototype_diagnostics.csv" \
  "$REFERENCE_BUNDLE/ordinal_prototype_diagnostics.csv"
python tools/analyze_ordinal_feature_evidence.py \
  --mode-roots "prototype=$REFERENCE_ROOT" \
  --actions "${ACTIONS[@]}" \
  --seed "$TRAIN_SEED" \
  --feat-dir "$FEAT_DIR" \
  --gpu "$GPU" \
  --ordinal-num-grades 4 \
  --ordinal-prototype-weight 0.0001 \
  --ordinal-prototype-order-scale "$PROTOTYPE_ORDER_SCALE" \
  --output-summary "$REFERENCE_BUNDLE/feature_evidence_summary.csv" \
  --output-samples "$REFERENCE_BUNDLE/feature_evidence_samples.csv"

python tools/summarize_ordinal_sensitivity.py \
  --setting-roots \
    "k3=$OUT_ROOT/k3" \
    "k4=$REFERENCE_BUNDLE" \
    "k5=$OUT_ROOT/k5" \
    "w5e5=$OUT_ROOT/w5e5" \
    "w1e4=$REFERENCE_BUNDLE" \
    "w2e4=$OUT_ROOT/w2e4" \
  --actions "${ACTIONS[@]}" \
  --output-runs "$OUT_ROOT/sensitivity_runs.csv" \
  --output-k "$OUT_ROOT/k_sensitivity.csv" \
  --output-weight "$OUT_ROOT/weight_sensitivity.csv" \
  --output-features "$OUT_ROOT/feature_sensitivity.csv" \
  --output-report "$OUT_ROOT/mentor_sensitivity_report.md"

echo "Completed 24 new one-factor sensitivity runs."
echo "Mentor report: $OUT_ROOT/mentor_sensitivity_report.md"
