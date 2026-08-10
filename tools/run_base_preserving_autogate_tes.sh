#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-5}"
OUT="${OUT:-results/base_preserving_autogate/TES}"
BASE="${BASE:-results/independent_experts/TES/TES_base_seed0.pth}"
SPLIT="${SPLIT:-results/calibration_probe/FISV_seed0_split.json}"
BRANCH="${BRANCH:-results/calibration_probe/TES_branches}"
MODEL="$OUT/TES_base_preserving_autogate_seed0.pth"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_file "$BASE"
require_file "$SPLIT"
require_file "$BRANCH/TES_rgb_UMT.pth"
require_file "$BRANCH/TES_flow_I3D.pth"
require_file "$BRANCH/TES_audio_MAST.pth"
require_file data/features/FISV_rgb_UMT.npy
require_file data/features/FISV_flow_I3D.npy
require_file data/features/FISV_audio_MAST.npy
if [[ ! -d skeleton_npy ]] || [[ -z "$(
    find skeleton_npy -maxdepth 1 -type f -name '*.npy' -print -quit
  )" ]]; then
  echo "Missing Skeleton files: skeleton_npy/*.npy" >&2
  exit 1
fi

mkdir -p "$OUT"
python -m py_compile \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/verify_anchored_residual.py \
  tools/analyze_base_preserving_autogate.py
python tools/verify_anchored_residual.py \
  2>&1 | tee "$OUT/TES_autogate_code_verification.log"
python tools/analyze_base_preserving_autogate.py --self_test \
  2>&1 | tee "$OUT/TES_autogate_analyzer_self_test.log"

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
AUTOGATE=(
  --use_skeleton
  --skeleton_root .
  --skeleton_gate_max 0.02
  --motion_saliency_proxy
  --motion_saliency_alpha 0.015
  --motion_saliency_source delta
  --motion_saliency_positive_only
  --anchored_residual
  --base_preserving_autogate
  --residual_adapter_dim 16
  --autogate_hidden_dim 8
  --autogate_no_regret_weight 1
)

# Cheap real-data smoke check before the 100-epoch development run.
python main.py \
  "${COMMON[@]}" "${AUTOGATE[@]}" \
  --epochs 1 \
  --calibration_final_only \
  --init_ckpt "$BASE" \
  --save_final_model "$OUT/TES_autogate_smoke_seed0.pth" \
  2>&1 | tee "$OUT/TES_autogate_smoke_seed0.log"
grep -qF "Frozen Base invariant: passed" \
  "$OUT/TES_autogate_smoke_seed0.log"

python main.py \
  "${COMMON[@]}" "${AUTOGATE[@]}" \
  --epochs 100 \
  --calibration_final_only \
  --init_ckpt "$BASE" \
  --save_final_model "$MODEL" \
  2>&1 | tee "$OUT/TES_autogate_seed0.log"
require_file "$MODEL"
grep -qF "Frozen Base invariant: passed" "$OUT/TES_autogate_seed0.log"

export_base() {
  local split_name="$1"
  local output="$2"
  python main.py \
    "${COMMON[@]}" \
    --prediction_probe \
    --prediction_split "$split_name" \
    --ckpt_path "$BASE" \
    --prediction_output "$output"
}

export_autogate() {
  local split_name="$1"
  local variant="$2"
  local output="$3"
  local gates=()
  case "$variant" in
    off) gates=(--prediction_motion_gate 0 --prediction_skeleton_gate 0) ;;
    motion) gates=(--prediction_motion_gate 1 --prediction_skeleton_gate 0) ;;
    skeleton) gates=(--prediction_motion_gate 0 --prediction_skeleton_gate 1) ;;
    auto) ;;
    *) echo "Unknown export variant: $variant" >&2; exit 1 ;;
  esac
  python main.py \
    "${COMMON[@]}" "${AUTOGATE[@]}" \
    --prediction_probe \
    --prediction_split "$split_name" \
    --ckpt_path "$MODEL" \
    --prediction_output "$output" \
    "${gates[@]}"
}

for split_name in train calibration; do
  suffix="train"
  [[ "$split_name" == "calibration" ]] && suffix="cal"
  export_base "$split_name" "$OUT/TES_base_${suffix}.csv"
  for variant in off motion skeleton auto; do
    export_autogate "$split_name" "$variant" \
      "$OUT/TES_autogate_${variant}_${suffix}.csv"
  done
done

python tools/analyze_base_preserving_autogate.py \
  --base_train "$OUT/TES_base_train.csv" \
  --off_train "$OUT/TES_autogate_off_train.csv" \
  --motion_train "$OUT/TES_autogate_motion_train.csv" \
  --skeleton_train "$OUT/TES_autogate_skeleton_train.csv" \
  --auto_train "$OUT/TES_autogate_auto_train.csv" \
  --base_cal "$OUT/TES_base_cal.csv" \
  --off_cal "$OUT/TES_autogate_off_cal.csv" \
  --motion_cal "$OUT/TES_autogate_motion_cal.csv" \
  --skeleton_cal "$OUT/TES_autogate_skeleton_cal.csv" \
  --auto_cal "$OUT/TES_autogate_auto_cal.csv" \
  --output "$OUT/TES_base_preserving_autogate_report.json" \
  --expected_train_samples 320 \
  --expected_cal_samples 80 \
  --bootstrap 2000 \
  --seed 0 \
  2>&1 | tee "$OUT/TES_base_preserving_autogate_report.log"

sha256sum \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/verify_anchored_residual.py \
  tools/analyze_base_preserving_autogate.py \
  tools/run_base_preserving_autogate_tes.sh \
  "$BASE" "$SPLIT" \
  "$BRANCH/TES_rgb_UMT.pth" \
  "$BRANCH/TES_flow_I3D.pth" \
  "$BRANCH/TES_audio_MAST.pth" \
  "$MODEL" \
  > "$OUT/TES_base_preserving_autogate_sha256.txt"

echo
echo "TES Base-preserving AutoGate development probe completed: $OUT"
echo "Read status in TES_base_preserving_autogate_report.json before seeds 1/2 or any fresh64/test access."
