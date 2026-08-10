#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-5}"
SOURCE_ROOT="${SOURCE_ROOT:-results/quality_guardrail_v2_cross_action}"
OUT_ROOT="${OUT_ROOT:-results/clean_gate_oracle_cross_action}"
BASE_DIR="${BASE_DIR:-pretrained_models/feats2}"
FISV_SPLIT="${FISV_SPLIT:-results/calibration_probe/FISV_seed0_split.json}"
RG_SPLIT="${RG_SPLIT:-$SOURCE_ROOT/RG_seed0_split.json}"
GRID_INTERVALS="${GRID_INTERVALS:-10}"
BOOTSTRAP="${BOOTSTRAP:-2000}"

[[ "$GRID_INTERVALS" == "10" ]] || {
  echo "This locked oracle protocol requires GRID_INTERVALS=10" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

mkdir -p "$OUT_ROOT"
python -m py_compile \
  main.py pipeline.py models/pamfn.py \
  tools/analyze_clean_gate_oracle.py
python tools/analyze_clean_gate_oracle.py --self_test \
  2>&1 | tee "$OUT_ROOT/oracle_analyzer_self_test.log"
python tools/verify_quality_guardrail_v2.py \
  2>&1 | tee "$OUT_ROOT/oracle_v2_verification.log"

run_action() {
  local action="$1"
  local expected_samples="$2"
  local split="$3"
  local out="$OUT_ROOT/$action"
  local model="$SOURCE_ROOT/$action/${action}_guardrail_v2_seed0.pth"
  local guardrail_grid="$out/${action}_clean_guardrail_grid.csv"
  local prior_grid="$out/${action}_clean_prior_only_grid.csv"
  local report="$out/${action}_clean_oracle_report.json"

  require_file "$model"
  require_file "$split"
  require_file "$BASE_DIR/${action}_rgb_UMT.pth"
  require_file "$BASE_DIR/${action}_flow_I3D.pth"
  require_file "$BASE_DIR/${action}_audio_MAST.pth"
  mkdir -p "$out"

  local common=(
    --gpu "$GPU"
    --feats 2
    --action "$action"
    --multi_modality
    --seed 0
    --split_seed 0
    --branch_ckpt_dir "$BASE_DIR"
    --calibration_ratio 0.2
    --split_file "$split"
    --use_skeleton
    --skeleton_root .
    --skeleton_gate_max 0.02
    --motion_saliency_proxy
    --motion_saliency_alpha 0.015
    --motion_saliency_source delta
    --motion_saliency_positive_only
    --anchored_residual
    --hierarchical_reliability_gate
    --quality_guardrail_v2
    --residual_adapter_dim 16
    --autogate_expert_weight 1
    --autogate_zero_mean_weight 1
    --reliability_quality_weight 1
    --calibration_probe
    --gate_grid_split calibration
    --gate_grid_intervals "$GRID_INTERVALS"
    --ckpt_path "$model"
  )

  python main.py "${common[@]}" \
    --gate_grid_output "$guardrail_grid" \
    2>&1 | tee "$out/${action}_clean_guardrail_grid.log"

  python main.py "${common[@]}" \
    --gate_grid_disable_reliability \
    --gate_grid_output "$prior_grid" \
    2>&1 | tee "$out/${action}_clean_prior_only_grid.log"

  python tools/analyze_clean_gate_oracle.py \
    --action "$action" \
    --guardrail_grid "$guardrail_grid" \
    --prior_only_grid "$prior_grid" \
    --expected_samples "$expected_samples" \
    --expected_grid_intervals "$GRID_INTERVALS" \
    --bootstrap "$BOOTSTRAP" \
    --seed 0 \
    --signal_threshold 0.5 \
    --mse_guardrail 0.02 \
    --output "$report" \
    2>&1 | tee "$out/${action}_clean_oracle_report.log"
}

run_action PCS 80 "$FISV_SPLIT"
run_action Ball 40 "$RG_SPLIT"

python tools/analyze_clean_gate_oracle.py \
  --combine_reports \
    "$OUT_ROOT/PCS/PCS_clean_oracle_report.json" \
    "$OUT_ROOT/Ball/Ball_clean_oracle_report.json" \
  --expected_grid_intervals "$GRID_INTERVALS" \
  --output "$OUT_ROOT/clean_oracle_cross_action_report.json" \
  2>&1 | tee "$OUT_ROOT/clean_oracle_cross_action_report.log"

sha256sum \
  main.py pipeline.py models/pamfn.py \
  tools/analyze_clean_gate_oracle.py \
  tools/run_clean_gate_oracle_cross_action.sh \
  "$SOURCE_ROOT/PCS/PCS_guardrail_v2_seed0.pth" \
  "$SOURCE_ROOT/Ball/Ball_guardrail_v2_seed0.pth" \
  "$FISV_SPLIT" "$RG_SPLIT" \
  "$OUT_ROOT"/PCS/*.csv "$OUT_ROOT"/PCS/*.json "$OUT_ROOT"/PCS/*.log \
  "$OUT_ROOT"/Ball/*.csv "$OUT_ROOT"/Ball/*.json "$OUT_ROOT"/Ball/*.log \
  "$OUT_ROOT"/*.json "$OUT_ROOT"/*.log \
  > "$OUT_ROOT/clean_oracle_sha256.txt"

echo
echo "Final clean oracle diagnosis complete: $OUT_ROOT"
echo "Read decision.status in clean_oracle_cross_action_report.json."
echo "Oracle results are diagnostic only and never authorize a full run."
