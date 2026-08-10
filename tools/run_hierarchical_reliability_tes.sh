#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-5}"
OUT="${OUT:-results/hierarchical_reliability/TES}"
BASE="${BASE:-results/independent_experts/TES/TES_base_seed0.pth}"
SPLIT="${SPLIT:-results/calibration_probe/FISV_seed0_split.json}"
BRANCH="${BRANCH:-results/calibration_probe/TES_branches}"
MODEL="$OUT/TES_hierarchical_seed0.pth"

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
  tools/analyze_hierarchical_reliability.py
python tools/verify_hierarchical_reliability.py \
  2>&1 | tee "$OUT/TES_hierarchical_code_verification.log"
python tools/analyze_hierarchical_reliability.py --self_test \
  2>&1 | tee "$OUT/TES_hierarchical_analyzer_self_test.log"

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
HIERARCHICAL=(
  --use_skeleton
  --skeleton_root .
  --skeleton_gate_max 0.02
  --motion_saliency_proxy
  --motion_saliency_alpha 0.015
  --motion_saliency_source delta
  --motion_saliency_positive_only
  --anchored_residual
  --hierarchical_reliability_gate
  --residual_adapter_dim 16
  --autogate_expert_weight 1
  --autogate_zero_mean_weight 1
  --reliability_quality_weight 1
)

python main.py \
  "${COMMON[@]}" "${HIERARCHICAL[@]}" \
  --epochs 100 \
  --calibration_final_only \
  --blind_train \
  --init_ckpt "$BASE" \
  --save_final_model "$MODEL" \
  2>&1 | tee "$OUT/TES_hierarchical_seed0.log"
require_file "$MODEL"
grep -qF "Frozen Base invariant: passed" \
  "$OUT/TES_hierarchical_seed0.log"
grep -qF "Train-only full-task priors:" \
  "$OUT/TES_hierarchical_seed0.log"

export_prediction() {
  local split_name="$1"
  local variant="$2"
  local output="$3"
  local gates=()
  case "$variant" in
    off) gates=(--prediction_motion_gate 0 --prediction_skeleton_gate 0) ;;
    motion) gates=(--prediction_motion_gate 1 --prediction_skeleton_gate 0) ;;
    skeleton) gates=(--prediction_motion_gate 0 --prediction_skeleton_gate 1) ;;
    universal) gates=(--prediction_motion_gate 1 --prediction_skeleton_gate 1) ;;
    adaptive) ;;
    *) echo "Unknown export variant: $variant" >&2; exit 1 ;;
  esac
  python main.py \
    "${COMMON[@]}" "${HIERARCHICAL[@]}" \
    --prediction_probe \
    --prediction_split "$split_name" \
    --ckpt_path "$MODEL" \
    --prediction_output "$output" \
    "${gates[@]}"
}

for split_name in train calibration; do
  suffix="train"
  [[ "$split_name" == "calibration" ]] && suffix="cal"
  for variant in off motion skeleton universal adaptive; do
    export_prediction "$split_name" "$variant" \
      "$OUT/TES_hierarchical_${variant}_${suffix}.csv"
  done
done

for cue in motion skeleton; do
  for percent in 25 50 75 100; do
    severity="0.$percent"
    [[ "$percent" == "100" ]] && severity="1"
    python main.py \
      "${COMMON[@]}" "${HIERARCHICAL[@]}" \
      --prediction_probe \
      --prediction_split calibration \
      --ckpt_path "$MODEL" \
      --prediction_output \
        "$OUT/TES_hierarchical_adaptive_cal_${cue}${percent}.csv" \
      --prediction_corruption_cue "$cue" \
      --prediction_corruption_severity "$severity"
  done
done

python tools/analyze_hierarchical_reliability.py \
  --directory "$OUT" \
  --output "$OUT/TES_hierarchical_report.json" \
  --expected_train_samples 320 \
  --expected_cal_samples 80 \
  --bootstrap 2000 \
  --seed 0 \
  2>&1 | tee "$OUT/TES_hierarchical_report.log"

sha256sum \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/verify_hierarchical_reliability.py \
  tools/analyze_hierarchical_reliability.py \
  tools/run_hierarchical_reliability_tes.sh \
  "$BASE" "$SPLIT" \
  "$BRANCH/TES_rgb_UMT.pth" \
  "$BRANCH/TES_flow_I3D.pth" \
  "$BRANCH/TES_audio_MAST.pth" \
  "$MODEL" \
  "$OUT"/*.csv "$OUT"/*.json "$OUT"/*.log \
  > "$OUT/TES_hierarchical_sha256.txt"

echo
echo "TES hierarchical reliability probe completed: $OUT"
echo "Read decision.status in TES_hierarchical_report.json."
echo "GO_NEXT_TASKS permits PCS/Ball development checks, not final OOF claims."
