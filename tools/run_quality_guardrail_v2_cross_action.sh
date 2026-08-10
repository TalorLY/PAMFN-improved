#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-5}"
OUT_ROOT="${OUT_ROOT:-results/quality_guardrail_v2_cross_action}"
BASE_DIR="${BASE_DIR:-pretrained_models/feats2}"
TES_REPORT="${TES_REPORT:-results/quality_guardrail_v2/TES/TES_guardrail_v2_report.json}"
FISV_SPLIT="${FISV_SPLIT:-results/calibration_probe/FISV_seed0_split.json}"
RG_SPLIT="${RG_SPLIT:-$OUT_ROOT/RG_seed0_split.json}"
PCS_EPOCHS="${PCS_EPOCHS:-100}"
BALL_EPOCHS="${BALL_EPOCHS:-100}"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

mkdir -p "$OUT_ROOT"
require_file "$TES_REPORT"
python -m py_compile \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/verify_hierarchical_reliability.py \
  tools/verify_quality_guardrail_v2.py \
  tools/analyze_quality_guardrail_v2.py \
  tools/analyze_quality_guardrail_cross_action.py
python tools/verify_hierarchical_reliability.py \
  2>&1 | tee "$OUT_ROOT/cross_action_legacy_verification.log"
python tools/verify_quality_guardrail_v2.py \
  2>&1 | tee "$OUT_ROOT/cross_action_v2_verification.log"
python tools/analyze_quality_guardrail_v2.py --self_test \
  2>&1 | tee "$OUT_ROOT/cross_action_action_analyzer_self_test.log"
python tools/analyze_quality_guardrail_cross_action.py --self_test \
  2>&1 | tee "$OUT_ROOT/cross_action_summary_analyzer_self_test.log"

run_action() {
  local action="$1"
  local epochs="$2"
  local expected_train="$3"
  local expected_cal="$4"
  local feature_family="$5"
  local split="$6"
  local out="$OUT_ROOT/$action"
  local base="$BASE_DIR/${action}_multimodal.pth"
  local model="$out/${action}_guardrail_v2_seed0.pth"

  for file in \
    "$base" \
    "$BASE_DIR/${action}_rgb_UMT.pth" \
    "$BASE_DIR/${action}_flow_I3D.pth" \
    "$BASE_DIR/${action}_audio_MAST.pth" \
    "data/features/${feature_family}_rgb_UMT.npy" \
    "data/features/${feature_family}_flow_I3D.npy" \
    "data/features/${feature_family}_audio_MAST.npy"; do
    require_file "$file"
  done
  if [[ "$action" == "Ball" ]]; then
    if [[ ! -d RG_skeleton_npy ]] || [[ -z "$(
      find RG_skeleton_npy -maxdepth 1 -type f -name 'Ball_*.npy' -print -quit
    )" ]]; then
      echo "Missing Ball Skeleton files: RG_skeleton_npy/Ball_*.npy" >&2
      exit 1
    fi
  else
    if [[ ! -d skeleton_npy ]] || [[ -z "$(
      find skeleton_npy -maxdepth 1 -type f -name '*.npy' -print -quit
    )" ]]; then
      echo "Missing FISV Skeleton files: skeleton_npy/*.npy" >&2
      exit 1
    fi
  fi

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
  )
  local guardrail=(
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
  )

  python main.py \
    "${common[@]}" "${guardrail[@]}" \
    --epochs "$epochs" \
    --calibration_final_only \
    --blind_train \
    --init_ckpt "$base" \
    --save_final_model "$model" \
    2>&1 | tee "$out/${action}_guardrail_v2_seed0.log"
  require_file "$model"
  require_file "$split"
  grep -qF "Frozen Base invariant: passed" \
    "$out/${action}_guardrail_v2_seed0.log"
  grep -qF "Train-only quality references:" \
    "$out/${action}_guardrail_v2_seed0.log"
  grep -qF "Train-only full-task priors:" \
    "$out/${action}_guardrail_v2_seed0.log"

  export_prediction() {
    local split_name="$1"
    local variant="$2"
    local output="$3"
    local options=()
    case "$variant" in
      off)
        options=(--prediction_motion_gate 0 --prediction_skeleton_gate 0
          --prediction_disable_reliability) ;;
      motion)
        options=(--prediction_motion_gate 1 --prediction_skeleton_gate 0
          --prediction_disable_reliability) ;;
      skeleton)
        options=(--prediction_motion_gate 0 --prediction_skeleton_gate 1
          --prediction_disable_reliability) ;;
      universal)
        options=(--prediction_motion_gate 1 --prediction_skeleton_gate 1
          --prediction_disable_reliability) ;;
      prior_only) options=(--prediction_disable_reliability) ;;
      guardrail) ;;
      *) echo "Unknown export variant: $variant" >&2; exit 1 ;;
    esac
    python main.py \
      "${common[@]}" "${guardrail[@]}" \
      --prediction_probe \
      --prediction_split "$split_name" \
      --ckpt_path "$model" \
      --prediction_output "$output" \
      "${options[@]}"
  }

  local split_name suffix variant
  for split_name in train calibration; do
    suffix="train"
    [[ "$split_name" == "calibration" ]] && suffix="cal"
    for variant in off motion skeleton universal prior_only guardrail; do
      export_prediction "$split_name" "$variant" \
        "$out/${action}_guardrail_v2_${variant}_${suffix}.csv"
    done
  done

  local policy cue percent severity
  for policy in guardrail prior_only; do
    local reliability_option=()
    [[ "$policy" == "prior_only" ]] && \
      reliability_option=(--prediction_disable_reliability)
    for cue in motion skeleton; do
      for percent in 25 50 75 100; do
        severity="0.$percent"
        [[ "$percent" == "100" ]] && severity="1"
        python main.py \
          "${common[@]}" "${guardrail[@]}" \
          --prediction_probe \
          --prediction_split calibration \
          --ckpt_path "$model" \
          --prediction_output \
            "$out/${action}_guardrail_v2_${policy}_cal_${cue}${percent}.csv" \
          --prediction_corruption_cue "$cue" \
          --prediction_corruption_severity "$severity" \
          "${reliability_option[@]}"
      done
    done
  done

  python tools/analyze_quality_guardrail_v2.py \
    --action "$action" \
    --directory "$out" \
    --output "$out/${action}_guardrail_v2_report.json" \
    --expected_train_samples "$expected_train" \
    --expected_cal_samples "$expected_cal" \
    --bootstrap 2000 \
    --seed 0 \
    2>&1 | tee "$out/${action}_guardrail_v2_report.log"

  sha256sum \
    main.py pipeline.py models/pamfn.py datasets.py \
    tools/verify_hierarchical_reliability.py \
    tools/verify_quality_guardrail_v2.py \
    tools/analyze_quality_guardrail_v2.py \
    tools/analyze_quality_guardrail_cross_action.py \
    tools/run_quality_guardrail_v2_cross_action.sh \
    "$base" "$split" \
    "$BASE_DIR/${action}_rgb_UMT.pth" \
    "$BASE_DIR/${action}_flow_I3D.pth" \
    "$BASE_DIR/${action}_audio_MAST.pth" \
    "$model" \
    "$out"/*.csv "$out"/*.json "$out"/*.log \
    > "$out/${action}_guardrail_v2_sha256.txt"
}

run_action PCS "$PCS_EPOCHS" 320 80 FISV "$FISV_SPLIT"
run_action Ball "$BALL_EPOCHS" 160 40 Ball "$RG_SPLIT"

python tools/analyze_quality_guardrail_cross_action.py \
  --pcs_report "$OUT_ROOT/PCS/PCS_guardrail_v2_report.json" \
  --ball_report "$OUT_ROOT/Ball/Ball_guardrail_v2_report.json" \
  --tes_report "$TES_REPORT" \
  --output "$OUT_ROOT/cross_action_report.json" \
  --base_no_regret -0.10 \
  --signal_threshold 0.50 \
  2>&1 | tee "$OUT_ROOT/cross_action_report.log"

sha256sum \
  "$TES_REPORT" \
  "$OUT_ROOT/PCS/PCS_guardrail_v2_report.json" \
  "$OUT_ROOT/Ball/Ball_guardrail_v2_report.json" \
  "$OUT_ROOT/cross_action_report.json" \
  "$OUT_ROOT"/*.log \
  > "$OUT_ROOT/cross_action_sha256.txt"

echo
echo "PCS/Ball quality-guardrail v2 screening complete: $OUT_ROOT"
echo "Read decision.status in cross_action_report.json."
echo "Only GO_FULL_EXPERIMENT permits the six-action multi-seed run."
