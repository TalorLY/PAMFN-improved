#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-results/bias_separated/TES}"
BASE="${BASE:-results/independent_experts/TES/TES_base_seed0.pth}"
SPLIT="${SPLIT:-results/calibration_probe/FISV_seed0_split.json}"
BRANCH="${BRANCH:-results/calibration_probe/TES_branches}"

GPU_SKEL001="${GPU_SKEL001:-5}"
GPU_SKEL0="${GPU_SKEL0:-5}"
GPU_MOTION="${GPU_MOTION:-5}"
GPU_EXPORT="${GPU_EXPORT:-5}"

mkdir -p "$OUT"

COMMON=(
  --feats 2
  --action TES
  --multi_modality
  --seed 0
  --split_seed 0
  --branch_ckpt_dir "$BRANCH"
  --calibration_ratio 0.2
  --split_file "$SPLIT"
)

BIAS=(
  --anchored_residual
  --bias_separated_residual
  --residual_adapter_dim 64
  --residual_zero_mean_weight 1
)

MOTION=(
  --motion_saliency_proxy
  --motion_saliency_alpha 0.015
  --motion_saliency_source delta
  --motion_saliency_positive_only
)

SKELETON=(
  --use_skeleton
  --skeleton_root .
  --skeleton_gate_max 0.02
)

test -f "$BASE"
test -f "$SPLIT"
test -f "$BRANCH/TES_rgb_UMT.pth"
test -f "$BRANCH/TES_flow_I3D.pth"
test -f "$BRANCH/TES_audio_MAST.pth"
test -f pretrained_models/feats2/TES_multimodal.pth
test -f data/features/FISV_rgb_UMT.npy
test -f data/features/FISV_flow_I3D.npy
test -f data/features/FISV_audio_MAST.npy
test -d skeleton_npy
test -n "$(find skeleton_npy -maxdepth 1 -type f -name '*.npy' -print -quit)"

python -m py_compile \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/analyze_direct_cue_effect.py \
  tools/verify_anchored_residual.py \
  tools/verify_calibration_probe.py \
  tools/verify_independent_expert_export.py
python tools/verify_anchored_residual.py
python tools/analyze_direct_cue_effect.py --self_test
python tools/verify_calibration_probe.py
python tools/verify_independent_expert_export.py

train_bias_model() {
  local gpu="$1"
  local rank_weight="$2"
  local epochs="$3"
  local checkpoint="$4"
  local log="$5"
  shift 5

  python main.py \
    --gpu "$gpu" \
    "${COMMON[@]}" \
    "$@" \
    "${BIAS[@]}" \
    --residual_rank_weight "$rank_weight" \
    --epochs "$epochs" \
    --calibration_final_only \
    --init_ckpt "$BASE" \
    --save_final_model "$checkpoint" \
    2>&1 | tee "$log"
}

# Smoke checks use the two primary method settings.
train_bias_model \
  "$GPU_SKEL001" 0.001 1 \
  "$OUT/TES_skeleton_rank001_smoke_seed0.pth" \
  "$OUT/TES_skeleton_rank001_smoke_seed0.log" \
  "${SKELETON[@]}"
train_bias_model \
  "$GPU_MOTION" 0 1 \
  "$OUT/TES_motion_rank0_smoke_seed0.pth" \
  "$OUT/TES_motion_rank0_smoke_seed0.log" \
  "${MOTION[@]}"

for log in \
  "$OUT/TES_skeleton_rank001_smoke_seed0.log" \
  "$OUT/TES_motion_rank0_smoke_seed0.log"; do
  grep -qF "Shared train-only score bias:" "$log"
  grep -qF "Frozen Base invariant: passed" "$log"
done

# Three preregistered full runs: Skeleton main, Skeleton rank ablation,
# and the minimum Motion bias-separation test.
train_bias_model \
  "$GPU_SKEL001" 0.001 100 \
  "$OUT/TES_skeleton_rank001_seed0.pth" \
  "$OUT/TES_skeleton_rank001_seed0.log" \
  "${SKELETON[@]}"
train_bias_model \
  "$GPU_SKEL0" 0 100 \
  "$OUT/TES_skeleton_rank0_seed0.pth" \
  "$OUT/TES_skeleton_rank0_seed0.log" \
  "${SKELETON[@]}"
train_bias_model \
  "$GPU_MOTION" 0 100 \
  "$OUT/TES_motion_rank0_seed0.pth" \
  "$OUT/TES_motion_rank0_seed0.log" \
  "${MOTION[@]}"

for log in \
  "$OUT/TES_skeleton_rank001_seed0.log" \
  "$OUT/TES_skeleton_rank0_seed0.log" \
  "$OUT/TES_motion_rank0_seed0.log"; do
  grep -qF "Frozen Base invariant: passed" "$log"
done

python - \
  "$OUT/TES_skeleton_rank001_seed0.pth" \
  "$OUT/TES_skeleton_rank0_seed0.pth" \
  "$OUT/TES_motion_rank0_seed0.pth" <<'PY'
import math
import sys

import torch


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


specs = (
    (sys.argv[1], 0.001, "skeleton"),
    (sys.argv[2], 0.0, "skeleton"),
    (sys.argv[3], 0.0, "motion"),
)
biases = []
for path, expected_rank, cue in specs:
    checkpoint = load(path)
    metadata = checkpoint["model_metadata"]
    state = checkpoint["state_dict"]
    assert metadata["schema_version"] == 2
    assert metadata["bias_separated_residual"] is True
    assert metadata["residual_adapter_dim"] == 64
    assert metadata["residual_zero_mean_weight"] == 1.0
    assert metadata["residual_rank_weight"] == expected_rank
    bias = float(state["shared_score_bias"].item())
    assert math.isfinite(bias)
    biases.append(bias)
    assert state[f"{cue}_encoder.0.weight"].shape[0] == 64
    assert f"{cue}_score_residual_head.1.bias" not in state
    print(path, "rank=", expected_rank, "shared_bias=", bias)
assert max(biases) - min(biases) <= 1e-7
print("schema2/shared-bias checkpoint checks: passed")
PY

export_base() {
  local split="$1"
  local output="$2"
  python main.py \
    --gpu "$GPU_EXPORT" \
    "${COMMON[@]}" \
    --prediction_probe \
    --prediction_split "$split" \
    --ckpt_path "$BASE" \
    --prediction_output "$output"
}

export_motion_rank0() {
  local split="$1"
  local gate="$2"
  local output="$3"
  python main.py \
    --gpu "$GPU_EXPORT" \
    "${COMMON[@]}" \
    "${MOTION[@]}" \
    "${BIAS[@]}" \
    --residual_rank_weight 0 \
    --prediction_probe \
    --prediction_split "$split" \
    --prediction_motion_gate "$gate" \
    --prediction_skeleton_gate 0 \
    --ckpt_path "$OUT/TES_motion_rank0_seed0.pth" \
    --prediction_output "$output"
}

export_skeleton() {
  local rank="$1"
  local checkpoint="$2"
  local split="$3"
  local gate="$4"
  local output="$5"
  python main.py \
    --gpu "$GPU_EXPORT" \
    "${COMMON[@]}" \
    "${SKELETON[@]}" \
    "${BIAS[@]}" \
    --residual_rank_weight "$rank" \
    --prediction_probe \
    --prediction_split "$split" \
    --prediction_motion_gate 0 \
    --prediction_skeleton_gate "$gate" \
    --ckpt_path "$checkpoint" \
    --prediction_output "$output"
}

export_base train "$OUT/TES_base_train.csv"
export_base calibration "$OUT/TES_base_cal.csv"

export_motion_rank0 train 1 "$OUT/TES_motion_rank0_on_train.csv"
export_motion_rank0 train 0 "$OUT/TES_motion_rank0_off_train.csv"
export_motion_rank0 calibration 1 "$OUT/TES_motion_rank0_on_cal.csv"
export_motion_rank0 calibration 0 "$OUT/TES_motion_rank0_off_cal.csv"

export_skeleton 0.001 \
  "$OUT/TES_skeleton_rank001_seed0.pth" \
  train 1 "$OUT/TES_skeleton_rank001_on_train.csv"
export_skeleton 0.001 \
  "$OUT/TES_skeleton_rank001_seed0.pth" \
  train 0 "$OUT/TES_skeleton_rank001_off_train.csv"
export_skeleton 0.001 \
  "$OUT/TES_skeleton_rank001_seed0.pth" \
  calibration 1 "$OUT/TES_skeleton_rank001_on_cal.csv"
export_skeleton 0.001 \
  "$OUT/TES_skeleton_rank001_seed0.pth" \
  calibration 0 "$OUT/TES_skeleton_rank001_off_cal.csv"

export_skeleton 0 \
  "$OUT/TES_skeleton_rank0_seed0.pth" \
  train 1 "$OUT/TES_skeleton_rank0_on_train.csv"
export_skeleton 0 \
  "$OUT/TES_skeleton_rank0_seed0.pth" \
  train 0 "$OUT/TES_skeleton_rank0_off_train.csv"
export_skeleton 0 \
  "$OUT/TES_skeleton_rank0_seed0.pth" \
  calibration 1 "$OUT/TES_skeleton_rank0_on_cal.csv"
export_skeleton 0 \
  "$OUT/TES_skeleton_rank0_seed0.pth" \
  calibration 0 "$OUT/TES_skeleton_rank0_off_cal.csv"

CSV_FILES=(
  "$OUT/TES_base_train.csv"
  "$OUT/TES_base_cal.csv"
  "$OUT/TES_motion_rank0_on_train.csv"
  "$OUT/TES_motion_rank0_off_train.csv"
  "$OUT/TES_motion_rank0_on_cal.csv"
  "$OUT/TES_motion_rank0_off_cal.csv"
  "$OUT/TES_skeleton_rank001_on_train.csv"
  "$OUT/TES_skeleton_rank001_off_train.csv"
  "$OUT/TES_skeleton_rank001_on_cal.csv"
  "$OUT/TES_skeleton_rank001_off_cal.csv"
  "$OUT/TES_skeleton_rank0_on_train.csv"
  "$OUT/TES_skeleton_rank0_off_train.csv"
  "$OUT/TES_skeleton_rank0_on_cal.csv"
  "$OUT/TES_skeleton_rank0_off_cal.csv"
)
for file in "${CSV_FILES[@]}"; do
  test -f "$file"
done
echo "Prediction CSV count: ${#CSV_FILES[@]}"

analyze_variant() {
  local skeleton_tag="$1"
  local report_tag="$2"
  python tools/analyze_direct_cue_effect.py \
    --dataset TES \
    --base_train "$OUT/TES_base_train.csv" \
    --motion_on_train "$OUT/TES_motion_rank0_on_train.csv" \
    --motion_off_train "$OUT/TES_motion_rank0_off_train.csv" \
    --skeleton_on_train "$OUT/TES_${skeleton_tag}_on_train.csv" \
    --skeleton_off_train "$OUT/TES_${skeleton_tag}_off_train.csv" \
    --base_cal "$OUT/TES_base_cal.csv" \
    --motion_on_cal "$OUT/TES_motion_rank0_on_cal.csv" \
    --motion_off_cal "$OUT/TES_motion_rank0_off_cal.csv" \
    --skeleton_on_cal "$OUT/TES_${skeleton_tag}_on_cal.csv" \
    --skeleton_off_cal "$OUT/TES_${skeleton_tag}_off_cal.csv" \
    --output "$OUT/TES_${report_tag}_direct_effect.json" \
    --expected_train_samples 320 \
    --expected_calibration_samples 80 \
    --bootstrap 2000 \
    --gate_step 0.01 \
    --seed 0 \
    --require_bias_separated \
    --motion_bound 0.015 \
    --skeleton_bound 0.02 \
    2>&1 | tee "$OUT/TES_${report_tag}_direct_effect.log"
}

analyze_variant skeleton_rank001 biassep_skeleton_rank001
analyze_variant skeleton_rank0 biassep_skeleton_rank0

python - \
  "$OUT/TES_biassep_skeleton_rank001_direct_effect.json" \
  "$OUT/TES_biassep_skeleton_rank0_direct_effect.json" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    print("\nREPORT:", path)
    print(
        "anchor_passed=", report["anchor_parity"]["passed"],
        "shared_bias=", report["anchor_parity"]["shared_bias"],
    )
    for cue in ("motion", "skeleton"):
        effect = report["cue_effects"][cue]
        direct = effect["cue_on_vs_off"]["calibration"]
        decision = effect["preliminary_decision"]
        print(
            cue,
            "delta_coef=", direct["delta_coef"],
            "mse_gain=", direct["relative_mse_gain"],
            "fixed_supported=", decision["fixed_cue_supported"],
            "dynamic_supported=", decision["dynamic_gate_supported"],
            "authorize_oof=", decision["authorize_oof_gate"],
        )
PY

sha256sum \
  "${CSV_FILES[@]}" \
  "$OUT"/TES_biassep_*_direct_effect.json \
  "$OUT"/TES_biassep_*_direct_effect.log \
  "$OUT/TES_skeleton_rank001_seed0.log" \
  "$OUT/TES_skeleton_rank0_seed0.log" \
  "$OUT/TES_motion_rank0_seed0.log" \
  > "$OUT/TES_biassep_seed0_sha256.txt"

echo
echo "Bias-separated TES seed-0 probe completed: $OUT"
echo "Do not run PCS, test, or OOF AutoGate before reviewing both reports."
