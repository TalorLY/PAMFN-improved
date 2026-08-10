#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-5}"
OUT="${OUT:-results/stage_motion_direct/TES}"
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

STAGE_MOTION=(
  --motion_saliency_proxy
  --motion_saliency_alpha 0.015
  --motion_saliency_source delta
  --motion_saliency_positive_only
)

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

python -m py_compile \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/analyze_direct_cue_effect.py \
  tools/verify_stage_motion_probe.py
python tools/verify_stage_motion_probe.py \
  --checkpoint "$BASE" \
  --branch_ckpt_dir "$BRANCH" \
  2>&1 | tee "$OUT/TES_stage_motion_verification.log"
python tools/analyze_direct_cue_effect.py --self_test

export_base() {
  local split="$1"
  local output="$2"
  python main.py \
    --gpu "$GPU" \
    "${COMMON[@]}" \
    --prediction_probe \
    --prediction_split "$split" \
    --prediction_motion_gate 0 \
    --prediction_skeleton_gate 0 \
    --ckpt_path "$BASE" \
    --prediction_output "$output"
}

export_stage_motion() {
  local split="$1"
  local gate="$2"
  local output="$3"
  python main.py \
    --gpu "$GPU" \
    "${COMMON[@]}" \
    "${STAGE_MOTION[@]}" \
    --prediction_probe \
    --prediction_split "$split" \
    --prediction_motion_gate "$gate" \
    --prediction_skeleton_gate 0 \
    --ckpt_path "$BASE" \
    --prediction_output "$output"
}

# This is inference only: every export strict-loads the same cue-free Base.
export_base train "$OUT/TES_base_train.csv"
export_base calibration "$OUT/TES_base_cal.csv"
export_stage_motion train 0 "$OUT/TES_stage_motion_off_train.csv"
export_stage_motion train 1 "$OUT/TES_stage_motion_on_train.csv"
export_stage_motion \
  calibration 0 "$OUT/TES_stage_motion_off_cal.csv"
export_stage_motion \
  calibration 1 "$OUT/TES_stage_motion_on_cal.csv"

if ! cmp -s \
    "$OUT/TES_base_train.csv" \
    "$OUT/TES_stage_motion_off_train.csv"; then
  echo "Stage Motion train gate0 is not byte-identical to Base" >&2
  exit 1
fi
if ! cmp -s \
    "$OUT/TES_base_cal.csv" \
    "$OUT/TES_stage_motion_off_cal.csv"; then
  echo "Stage Motion calibration gate0 is not byte-identical to Base" >&2
  exit 1
fi
echo "Stage Motion gate0 and Base anchors: byte-identical"

# The shared analyzer expects two cues. The unused Skeleton inputs deliberately
# point to Base as an exact null control; only the Motion decision is actionable.
python tools/analyze_direct_cue_effect.py \
  --dataset TES \
  --base_train "$OUT/TES_base_train.csv" \
  --motion_on_train "$OUT/TES_stage_motion_on_train.csv" \
  --motion_off_train "$OUT/TES_stage_motion_off_train.csv" \
  --skeleton_on_train "$OUT/TES_base_train.csv" \
  --skeleton_off_train "$OUT/TES_base_train.csv" \
  --base_cal "$OUT/TES_base_cal.csv" \
  --motion_on_cal "$OUT/TES_stage_motion_on_cal.csv" \
  --motion_off_cal "$OUT/TES_stage_motion_off_cal.csv" \
  --skeleton_on_cal "$OUT/TES_base_cal.csv" \
  --skeleton_off_cal "$OUT/TES_base_cal.csv" \
  --output "$OUT/TES_stage_motion_direct_effect.json" \
  --expected_train_samples 320 \
  --expected_calibration_samples 80 \
  --bootstrap 2000 \
  --gate_step 1 \
  --seed 0 \
  --require_anchored \
  --motion_bound 1 \
  --skeleton_bound 1 \
  2>&1 | tee "$OUT/TES_stage_motion_direct_effect.log"

python - \
  "$OUT/TES_stage_motion_direct_effect.json" \
  "$BASE" "$SPLIT" <<'PY'
import hashlib
import json
import os
import sys


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


report_path, base_path, split_path = sys.argv[1:]
with open(report_path, "r", encoding="utf-8") as handle:
    report = json.load(handle)

assert report["anchor_parity"]["passed"]
skeleton = report["cue_effects"]["skeleton"]
assert skeleton["decomposition"]["calibration_effects"]["direct"][
    "rms"
] == 0.0
assert not skeleton["preliminary_decision"]["authorize_oof_gate"]

motion = report["cue_effects"]["motion"]
decision = motion["preliminary_decision"]
seed0_oof_screen_passed = bool(decision["authorize_oof_gate"])
if seed0_oof_screen_passed:
    status = "GO_SEED_CONFIRMATION"
    next_step = (
        "Confirm the unchanged stage mechanism on seeds 1/2, then run "
        "the group-OOF Motion AutoGate probe."
    )
elif decision["average_effect_supported"]:
    status = "FIXED_MOTION_ONLY"
    next_step = (
        "Keep stage Motion fixed; sample-level AutoGate is not supported."
    )
else:
    status = "NO_GO"
    next_step = (
        "Stop this frozen-Base post-hoc TES stage-Motion route; do not "
        "sweep alpha, epochs, adapter width, or rank weight on this "
        "calibration split."
    )

# The shared analyzer's generic OOF authorization assumes a score-residual
# protocol. For stage Motion it is only an endpoint-oracle screen. Preserve
# the screen result explicitly, then remove every actionable OOF instruction.
decision["authorize_oof_gate"] = False
decision["interpretation"] = (
    "Seed0 endpoint-headroom screening only; confirm seeds 1/2 and add a "
    "train-fitted scalar control before any OOF gate can be authorized."
)
report["preliminary_decisions"]["next_step"] = next_step
report["preliminary_decisions"]["oof_authorized"] = False
report["preliminary_decisions"][
    "motion_seed0_endpoint_screen_passed"
] = seed0_oof_screen_passed
report["decision_authority"] = "stage_motion_decision_only"
report["warnings"] = [
    warning for warning in report["warnings"]
    if "positive decision only authorizes" not in warning.lower()
]
report["warnings"].insert(
    0,
    "Generic direct-effect OOF authorization is superseded for stage "
    "Motion. Only stage_motion_decision is actionable.",
)
report["warnings"].insert(
    1,
    "The same 80-sample calibration split has been inspected by prior "
    "development probes. This result is exploratory evidence, not an "
    "independent confirmation.",
)

report["stage_motion_protocol"] = {
    "method": "parameter_free_stage_motion",
    "injection_stages": ["stage1", "stage2"],
    "motion_saliency_alpha": 0.015,
    "motion_saliency_source": "delta",
    "motion_saliency_positive_only": True,
    "motion_specific_parameters": 0,
    "training_performed": False,
    "gate_candidates": [0.0, 1.0],
    "checkpoint_loading": "strict same cue-free Base for all exports",
    "base_checkpoint": {
        "path": os.path.abspath(base_path),
        "sha256": sha256(base_path),
    },
    "split_manifest": {
        "path": os.path.abspath(split_path),
        "sha256": sha256(split_path),
    },
    "score_difference_bound": 1.0,
    "score_bound_note": (
        "alpha=0.015 is a feature modulation strength, not a final-score "
        "residual bound"
    ),
    "unused_skeleton_is_exact_null_control": True,
}
report["stage_motion_decision"] = {
    "status": status,
    "average_effect_supported": decision["average_effect_supported"],
    "endpoint_headroom_screen_passed": (
        decision["dynamic_gate_supported"]
    ),
    "seed0_analyzer_oof_screen_passed": (
        seed0_oof_screen_passed
    ),
    "eligible_for_seed_confirmation": (
        seed0_oof_screen_passed
    ),
    "authorize_oof_gate": False,
    "authorization_note": (
        "Seed0 endpoint-oracle evidence is only a screening result. OOF "
        "remains unauthorized until seeds 1/2 confirm the fixed effect "
        "and a train-fitted scalar control is included."
    ),
    "next_step": next_step,
}

with open(report_path, "w", encoding="utf-8") as handle:
    json.dump(
        report, handle, indent=2, ensure_ascii=False, allow_nan=False
    )

fixed = motion["fixed_gate"]
effect = fixed["calibration_comparison_to_off"]
bootstrap = motion["bootstrap"]["fixed_gate_delta_coef_over_off"]
print("stage_motion_status=", status)
print("fixed_gate=", fixed["gate"])
print("calibration_delta_coef=", effect["delta_coef"])
print("calibration_relative_mse_gain=", effect["relative_mse_gain"])
print("bootstrap_p_delta_positive=", bootstrap["positive_probability"])
print("bootstrap_delta_coef_95ci=", bootstrap["95ci"])
print("average_effect_supported=", decision["average_effect_supported"])
print("endpoint_headroom_screen_passed=", decision["dynamic_gate_supported"])
print("eligible_for_seed_confirmation=", seed0_oof_screen_passed)
print("authorize_oof_gate=", False)
print("next_step=", next_step)
PY

CSV_FILES=(
  "$OUT/TES_base_train.csv"
  "$OUT/TES_base_cal.csv"
  "$OUT/TES_stage_motion_off_train.csv"
  "$OUT/TES_stage_motion_on_train.csv"
  "$OUT/TES_stage_motion_off_cal.csv"
  "$OUT/TES_stage_motion_on_cal.csv"
)
for file in "${CSV_FILES[@]}"; do
  test -f "$file"
done
test "${#CSV_FILES[@]}" -eq 6
sha256sum "$BASE" "$SPLIT" \
  > "$OUT/TES_stage_motion_reference_sha256.txt"
ARTIFACTS=(
  "$OUT/TES_stage_motion_verification.log"
  "${CSV_FILES[@]}"
  "$OUT/TES_stage_motion_direct_effect.json"
  "$OUT/TES_stage_motion_direct_effect.log"
  "$OUT/TES_stage_motion_reference_sha256.txt"
)
sha256sum "${ARTIFACTS[@]}" \
  > "$OUT/TES_stage_motion_seed0_sha256.txt"

echo
echo "TES stage-level Motion direct-effect probe completed: $OUT"
echo "Review stage_motion_decision before running seeds, OOF, or any test."
