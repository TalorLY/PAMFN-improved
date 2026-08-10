#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-5}"
OUT="${OUT:-results/residual_v2/TES}"
V1_OUT="${V1_OUT:-results/bias_separated/TES}"
BASE="${BASE:-results/independent_experts/TES/TES_base_seed0.pth}"
SPLIT="${SPLIT:-results/calibration_probe/FISV_seed0_split.json}"
BRANCH="${BRANCH:-results/calibration_probe/TES_branches}"

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

V1_RESIDUAL=(
  --anchored_residual
  --bias_separated_residual
  --residual_adapter_dim 64
  --residual_zero_mean_weight 1
  --residual_rank_weight 0
)

V2_RESIDUAL=(
  --anchored_residual
  --bias_separated_residual
  --residual_v2
  --residual_adapter_dim 16
  --residual_zero_mean_weight 1
  --residual_rank_weight 0
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

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_file "$BASE"
require_file "$SPLIT"
require_file "$V1_OUT/TES_motion_rank0_seed0.pth"
require_file "$V1_OUT/TES_skeleton_rank0_seed0.pth"
require_file "$BRANCH/TES_rgb_UMT.pth"
require_file "$BRANCH/TES_flow_I3D.pth"
require_file "$BRANCH/TES_audio_MAST.pth"
require_file pretrained_models/feats2/TES_multimodal.pth
require_file data/features/FISV_rgb_UMT.npy
require_file data/features/FISV_flow_I3D.npy
require_file data/features/FISV_audio_MAST.npy
if [[ ! -d skeleton_npy ]] || [[ -z "$(
    find skeleton_npy -maxdepth 1 -type f -name '*.npy' -print -quit
  )" ]]; then
  echo "Missing Skeleton files: skeleton_npy/*.npy" >&2
  exit 1
fi

python -m py_compile \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/analyze_direct_cue_effect.py \
  tools/verify_anchored_residual.py
python tools/verify_anchored_residual.py
python tools/analyze_direct_cue_effect.py --self_test

run_v1_parity() {
  local checkpoint="$1"
  local log="$2"
  shift 2

  python main.py \
    --gpu "$GPU" \
    "${COMMON[@]}" \
    "$@" \
    "${V1_RESIDUAL[@]}" \
    --epochs 1 \
    --residual_parity_probe \
    --ckpt_path "$checkpoint" \
    2>&1 | tee "$log"
}

# Read-only diagnosis of the old v1 adapters on their real training batches.
run_v1_parity \
  "$V1_OUT/TES_motion_rank0_seed0.pth" \
  "$OUT/TES_motion_v1_parity_seed0.log" \
  "${MOTION[@]}"
run_v1_parity \
  "$V1_OUT/TES_skeleton_rank0_seed0.pth" \
  "$OUT/TES_skeleton_v1_parity_seed0.log" \
  "${SKELETON[@]}"

for log in \
  "$OUT/TES_motion_v1_parity_seed0.log" \
  "$OUT/TES_skeleton_v1_parity_seed0.log"; do
  grep -qF "Residual train/eval parity:" "$log"
done

train_v2() {
  local epochs="$1"
  local checkpoint="$2"
  local log="$3"
  shift 3

  python main.py \
    --gpu "$GPU" \
    "${COMMON[@]}" \
    "$@" \
    "${V2_RESIDUAL[@]}" \
    --epochs "$epochs" \
    --calibration_final_only \
    --init_ckpt "$BASE" \
    --save_final_model "$checkpoint" \
    2>&1 | tee "$log"
}

# One-epoch checks catch wiring, centering, and checkpoint errors cheaply.
train_v2 1 \
  "$OUT/TES_motion_v2_smoke_seed0.pth" \
  "$OUT/TES_motion_v2_smoke_seed0.log" \
  "${MOTION[@]}"
train_v2 1 \
  "$OUT/TES_skeleton_v2_smoke_seed0.pth" \
  "$OUT/TES_skeleton_v2_smoke_seed0.log" \
  "${SKELETON[@]}"

grep -qF "Anchored adapters trainable: 7 tensors / 5696 parameters" \
  "$OUT/TES_motion_v2_smoke_seed0.log"
grep -qF "Anchored adapters trainable: 7 tensors / 3712 parameters" \
  "$OUT/TES_skeleton_v2_smoke_seed0.log"
for log in \
  "$OUT/TES_motion_v2_smoke_seed0.log" \
  "$OUT/TES_skeleton_v2_smoke_seed0.log"; do
  grep -qF "Residual-v2 logit center:" "$log"
  grep -qF "Residual train/eval parity:" "$log"
  grep -qF "passed=True" "$log"
  grep -qF "Frozen Base invariant: passed" "$log"
done

# Preregistered seed-0 comparison: architecture is the only method change.
train_v2 100 \
  "$OUT/TES_motion_v2_seed0.pth" \
  "$OUT/TES_motion_v2_seed0.log" \
  "${MOTION[@]}"
train_v2 100 \
  "$OUT/TES_skeleton_v2_seed0.pth" \
  "$OUT/TES_skeleton_v2_seed0.log" \
  "${SKELETON[@]}"

for log in \
  "$OUT/TES_motion_v2_seed0.log" \
  "$OUT/TES_skeleton_v2_seed0.log"; do
  grep -qF "Residual-v2 logit center:" "$log"
  grep -qF "Residual train/eval parity:" "$log"
  grep -qF "passed=True" "$log"
  grep -qF "Frozen Base invariant: passed" "$log"
done

python - \
  "$OUT/TES_motion_v2_seed0.pth" \
  "$OUT/TES_skeleton_v2_seed0.pth" <<'PY'
import math
import sys

import torch


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


specs = (
    (sys.argv[1], "motion", 5696),
    (sys.argv[2], "skeleton", 3712),
)
biases = []
for path, cue, expected_parameters in specs:
    checkpoint = load(path)
    metadata = checkpoint["model_metadata"]
    state = checkpoint["state_dict"]
    assert metadata["schema_version"] == 3
    assert metadata["bias_separated_residual"] is True
    assert metadata["residual_v2"] is True
    assert metadata["residual_adapter_dim"] == 16
    assert metadata["residual_zero_mean_weight"] == 1.0
    assert metadata["residual_rank_weight"] == 0.0
    assert metadata["residual_architecture"] == "centered_no_norm_v2"
    assert (
        metadata["residual_center_method"]
        == "train_zero_mean_tanh_root"
    )
    assert metadata["motion_saliency_proxy"] is (cue == "motion")
    assert metadata["use_skeleton"] is (cue == "skeleton")

    bias = float(state["shared_score_bias"].item())
    center_key = f"{cue}_score_residual_head.2.logit_center"
    center = float(state[center_key].item())
    assert math.isfinite(bias)
    assert math.isfinite(center)
    biases.append(bias)
    assert state[f"{cue}_encoder.0.weight"].shape[0] == 16
    assert f"{cue}_score_residual_head.1.bias" not in state
    active_keys = [
        key for key in state
        if key.startswith((f"{cue}_encoder.", f"{cue}_score_residual_head."))
    ]
    assert not any(
        token in key
        for key in active_keys
        for token in ("running_mean", "running_var", "num_batches_tracked")
    )
    parameter_count = sum(
        value.numel() for key, value in state.items()
        if key.startswith((f"{cue}_encoder.", f"{cue}_score_residual_head."))
        and not key.endswith("logit_center")
    )
    assert parameter_count == expected_parameters
    print(path, "shared_bias=", bias, "logit_center=", center)

assert max(biases) - min(biases) <= 1e-7
print("schema3/center/no-state checkpoint checks: passed")
PY

export_base() {
  local split="$1"
  local output="$2"
  python main.py \
    --gpu "$GPU" \
    "${COMMON[@]}" \
    --prediction_probe \
    --prediction_split "$split" \
    --ckpt_path "$BASE" \
    --prediction_output "$output"
}

export_motion_v2() {
  local split="$1"
  local gate="$2"
  local output="$3"
  python main.py \
    --gpu "$GPU" \
    "${COMMON[@]}" \
    "${MOTION[@]}" \
    "${V2_RESIDUAL[@]}" \
    --prediction_probe \
    --prediction_split "$split" \
    --prediction_motion_gate "$gate" \
    --prediction_skeleton_gate 0 \
    --ckpt_path "$OUT/TES_motion_v2_seed0.pth" \
    --prediction_output "$output"
}

export_skeleton_v2() {
  local split="$1"
  local gate="$2"
  local output="$3"
  python main.py \
    --gpu "$GPU" \
    "${COMMON[@]}" \
    "${SKELETON[@]}" \
    "${V2_RESIDUAL[@]}" \
    --prediction_probe \
    --prediction_split "$split" \
    --prediction_motion_gate 0 \
    --prediction_skeleton_gate "$gate" \
    --ckpt_path "$OUT/TES_skeleton_v2_seed0.pth" \
    --prediction_output "$output"
}

export_base train "$OUT/TES_base_train.csv"
export_base calibration "$OUT/TES_base_cal.csv"
export_motion_v2 train 1 "$OUT/TES_motion_v2_on_train.csv"
export_motion_v2 train 0 "$OUT/TES_motion_v2_off_train.csv"
export_motion_v2 calibration 1 "$OUT/TES_motion_v2_on_cal.csv"
export_motion_v2 calibration 0 "$OUT/TES_motion_v2_off_cal.csv"
export_skeleton_v2 train 1 "$OUT/TES_skeleton_v2_on_train.csv"
export_skeleton_v2 train 0 "$OUT/TES_skeleton_v2_off_train.csv"
export_skeleton_v2 calibration 1 "$OUT/TES_skeleton_v2_on_cal.csv"
export_skeleton_v2 calibration 0 "$OUT/TES_skeleton_v2_off_cal.csv"

CSV_FILES=(
  "$OUT/TES_base_train.csv"
  "$OUT/TES_base_cal.csv"
  "$OUT/TES_motion_v2_on_train.csv"
  "$OUT/TES_motion_v2_off_train.csv"
  "$OUT/TES_motion_v2_on_cal.csv"
  "$OUT/TES_motion_v2_off_cal.csv"
  "$OUT/TES_skeleton_v2_on_train.csv"
  "$OUT/TES_skeleton_v2_off_train.csv"
  "$OUT/TES_skeleton_v2_on_cal.csv"
  "$OUT/TES_skeleton_v2_off_cal.csv"
)
for file in "${CSV_FILES[@]}"; do
  test -f "$file"
done
test "${#CSV_FILES[@]}" -eq 10
echo "Prediction CSV count: ${#CSV_FILES[@]}"

python tools/analyze_direct_cue_effect.py \
  --dataset TES \
  --base_train "$OUT/TES_base_train.csv" \
  --motion_on_train "$OUT/TES_motion_v2_on_train.csv" \
  --motion_off_train "$OUT/TES_motion_v2_off_train.csv" \
  --skeleton_on_train "$OUT/TES_skeleton_v2_on_train.csv" \
  --skeleton_off_train "$OUT/TES_skeleton_v2_off_train.csv" \
  --base_cal "$OUT/TES_base_cal.csv" \
  --motion_on_cal "$OUT/TES_motion_v2_on_cal.csv" \
  --motion_off_cal "$OUT/TES_motion_v2_off_cal.csv" \
  --skeleton_on_cal "$OUT/TES_skeleton_v2_on_cal.csv" \
  --skeleton_off_cal "$OUT/TES_skeleton_v2_off_cal.csv" \
  --output "$OUT/TES_residual_v2_direct_effect.json" \
  --expected_train_samples 320 \
  --expected_calibration_samples 80 \
  --bootstrap 2000 \
  --gate_step 0.01 \
  --seed 0 \
  --require_bias_separated \
  --motion_bound 0.015 \
  --skeleton_bound 0.02 \
  2>&1 | tee "$OUT/TES_residual_v2_direct_effect.log"

python - "$OUT/TES_residual_v2_direct_effect.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    report = json.load(handle)

assert report["anchor_parity"]["passed"]
print("anchor_passed=", report["anchor_parity"]["passed"])
print("shared_bias=", report["anchor_parity"]["shared_bias"])
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

ARTIFACTS=(
  "$OUT/TES_motion_v1_parity_seed0.log"
  "$OUT/TES_skeleton_v1_parity_seed0.log"
  "$OUT/TES_motion_v2_smoke_seed0.pth"
  "$OUT/TES_motion_v2_smoke_seed0.log"
  "$OUT/TES_skeleton_v2_smoke_seed0.pth"
  "$OUT/TES_skeleton_v2_smoke_seed0.log"
  "$OUT/TES_motion_v2_seed0.pth"
  "$OUT/TES_motion_v2_seed0.log"
  "$OUT/TES_skeleton_v2_seed0.pth"
  "$OUT/TES_skeleton_v2_seed0.log"
  "${CSV_FILES[@]}"
  "$OUT/TES_residual_v2_direct_effect.json"
  "$OUT/TES_residual_v2_direct_effect.log"
)
sha256sum "${ARTIFACTS[@]}" \
  > "$OUT/TES_residual_v2_seed0_sha256.txt"

echo
echo "Residual-v2 TES seed-0 probe completed: $OUT"
echo "Do not run PCS, test, OOF, or a full action sweep before reviewing the report."
