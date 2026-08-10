#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-5}"
OUT="${OUT:-results/quality_guardrail_v2/TES}"
BASE="${BASE:-results/independent_experts/TES/TES_base_seed0.pth}"
SPLIT="${SPLIT:-results/calibration_probe/FISV_seed0_split.json}"
BRANCH="${BRANCH:-results/calibration_probe/TES_branches}"
MODEL="$OUT/TES_guardrail_v2_seed0.pth"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

for file in \
  "$BASE" "$SPLIT" \
  "$BRANCH/TES_rgb_UMT.pth" \
  "$BRANCH/TES_flow_I3D.pth" \
  "$BRANCH/TES_audio_MAST.pth" \
  data/features/FISV_rgb_UMT.npy \
  data/features/FISV_flow_I3D.npy \
  data/features/FISV_audio_MAST.npy; do
  require_file "$file"
done
if [[ ! -d skeleton_npy ]] || [[ -z "$(
    find skeleton_npy -maxdepth 1 -type f -name '*.npy' -print -quit
  )" ]]; then
  echo "Missing Skeleton files: skeleton_npy/*.npy" >&2
  exit 1
fi

mkdir -p "$OUT"
python -m py_compile \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/verify_hierarchical_reliability.py \
  tools/verify_quality_guardrail_v2.py \
  tools/analyze_quality_guardrail_v2.py
python tools/verify_hierarchical_reliability.py \
  2>&1 | tee "$OUT/TES_guardrail_v2_legacy_verification.log"
python tools/verify_quality_guardrail_v2.py \
  2>&1 | tee "$OUT/TES_guardrail_v2_code_verification.log"
python tools/analyze_quality_guardrail_v2.py --self_test \
  2>&1 | tee "$OUT/TES_guardrail_v2_analyzer_self_test.log"

COMMON=(
  --gpu "$GPU"
  --feats 2
  --action TES
  --multi_modality
  --seed 0
  --split_seed 0
  --branch_ckpt_dir "$BRANCH"
  --calibration_ratio 0.2
  --split_file "$SPLIT"
)
GUARDRAIL=(
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
  "${COMMON[@]}" "${GUARDRAIL[@]}" \
  --epochs 100 \
  --calibration_final_only \
  --blind_train \
  --init_ckpt "$BASE" \
  --save_final_model "$MODEL" \
  2>&1 | tee "$OUT/TES_guardrail_v2_seed0.log"
require_file "$MODEL"
grep -qF "Frozen Base invariant: passed" \
  "$OUT/TES_guardrail_v2_seed0.log"
grep -qF "Train-only quality references:" \
  "$OUT/TES_guardrail_v2_seed0.log"
grep -qF "Train-only full-task priors:" \
  "$OUT/TES_guardrail_v2_seed0.log"

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
    "${COMMON[@]}" "${GUARDRAIL[@]}" \
    --prediction_probe \
    --prediction_split "$split_name" \
    --ckpt_path "$MODEL" \
    --prediction_output "$output" \
    "${options[@]}"
}

for split_name in train calibration; do
  suffix="train"
  [[ "$split_name" == "calibration" ]] && suffix="cal"
  for variant in off motion skeleton universal prior_only guardrail; do
    export_prediction "$split_name" "$variant" \
      "$OUT/TES_guardrail_v2_${variant}_${suffix}.csv"
  done
done

for policy in guardrail prior_only; do
  reliability_option=()
  [[ "$policy" == "prior_only" ]] && \
    reliability_option=(--prediction_disable_reliability)
  for cue in motion skeleton; do
    for percent in 25 50 75 100; do
      severity="0.$percent"
      [[ "$percent" == "100" ]] && severity="1"
      python main.py \
        "${COMMON[@]}" "${GUARDRAIL[@]}" \
        --prediction_probe \
        --prediction_split calibration \
        --ckpt_path "$MODEL" \
        --prediction_output \
          "$OUT/TES_guardrail_v2_${policy}_cal_${cue}${percent}.csv" \
        --prediction_corruption_cue "$cue" \
        --prediction_corruption_severity "$severity" \
        "${reliability_option[@]}"
    done
  done
done

python tools/analyze_quality_guardrail_v2.py \
  --directory "$OUT" \
  --output "$OUT/TES_guardrail_v2_report.json" \
  --expected_train_samples 320 \
  --expected_cal_samples 80 \
  --bootstrap 2000 \
  --seed 0 \
  2>&1 | tee "$OUT/TES_guardrail_v2_report.log"

sha256sum \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/verify_hierarchical_reliability.py \
  tools/verify_quality_guardrail_v2.py \
  tools/analyze_quality_guardrail_v2.py \
  tools/run_quality_guardrail_v2_tes.sh \
  "$BASE" "$SPLIT" \
  "$BRANCH/TES_rgb_UMT.pth" \
  "$BRANCH/TES_flow_I3D.pth" \
  "$BRANCH/TES_audio_MAST.pth" \
  "$MODEL" \
  "$OUT"/*.csv "$OUT"/*.json "$OUT"/*.log \
  > "$OUT/TES_guardrail_v2_sha256.txt"

echo
echo "TES quality-guardrail v2 probe completed: $OUT"
echo "Read decision.status in TES_guardrail_v2_report.json."
echo "GO_NEXT_TASKS permits PCS/Ball seed-0 checks, not final OOF claims."
