#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-5}"
OUT="${OUT:-results/paired_motion_exposure/TES}"
PARENT_SPLIT="${PARENT_SPLIT:-results/calibration_probe/FISV_seed0_split.json}"
PARENT_BRANCH="${PARENT_BRANCH:-results/calibration_probe/TES_branches}"
SPLIT_SEED=20260719
SPLIT="$OUT/FISV_refresh64_seed20260719_split.json"
SPLIT_AUDIT="$OUT/FISV_refresh64_seed20260719_audit.json"
BRANCH="$OUT/branches"
CONTROL="$OUT/TES_paired_control_seed0.pth"
PAIRED="$OUT/TES_paired_off_on_seed0.pth"
PROTOCOL_LOCK="$OUT/TES_paired_protocol_lock.json"
LOCKBOX_OPEN_MARKER="$OUT/TES_lockbox_opened_once.json"
LOCKBOX_REGISTRY="results/paired_motion_exposure/lockbox_registry"
LOCKBOX_RESERVATION="$LOCKBOX_REGISTRY/TES_refresh64_seed20260719.reserved.json"
CONTROL_TRAIN_MANIFEST="$OUT/TES_paired_control_train.manifest.json"
G0_TRAIN_MANIFEST="$OUT/TES_paired_g0_train.manifest.json"
G1_TRAIN_MANIFEST="$OUT/TES_paired_g1_train.manifest.json"
TRAIN_FUTILITY="$OUT/TES_paired_train_futility.json"
CONTROL_LOCKBOX_MANIFEST="$OUT/TES_paired_control_lockbox64.manifest.json"
G0_LOCKBOX_MANIFEST="$OUT/TES_paired_g0_lockbox64.manifest.json"
G1_LOCKBOX_MANIFEST="$OUT/TES_paired_g1_lockbox64.manifest.json"

mkdir -p "$OUT" "$BRANCH" "$LOCKBOX_REGISTRY"
if [[ -e "$PROTOCOL_LOCK" ]] || \
   [[ -e "$LOCKBOX_OPEN_MARKER" ]] || \
   [[ -e "$LOCKBOX_RESERVATION" ]] || \
   [[ -e "$OUT/TES_paired_motion_effect.json" ]] || \
   [[ -e "$OUT/TES_paired_control_lockbox64.csv" ]] || \
   [[ -e "$OUT/TES_paired_g0_lockbox64.csv" ]] || \
   [[ -e "$OUT/TES_paired_g1_lockbox64.csv" ]] || \
   [[ -e "$CONTROL_TRAIN_MANIFEST" ]] || \
   [[ -e "$G0_TRAIN_MANIFEST" ]] || \
   [[ -e "$G1_TRAIN_MANIFEST" ]] || \
   [[ -e "$CONTROL_LOCKBOX_MANIFEST" ]] || \
   [[ -e "$G0_LOCKBOX_MANIFEST" ]] || \
   [[ -e "$G1_LOCKBOX_MANIFEST" ]]; then
  echo "This refresh64 split has already been locked, reserved, or opened." >&2
  echo "Refusing to retrain or access it again, even through another OUT." >&2
  exit 1
fi
exec > >(tee "$OUT/TES_paired_all_seed0.log") 2>&1

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_file "$PARENT_SPLIT"
require_file "$PARENT_BRANCH/TES_rgb_UMT.pth"
require_file "$PARENT_BRANCH/TES_flow_I3D.pth"
require_file "$PARENT_BRANCH/TES_audio_MAST.pth"
require_file data/FISV/train.txt
require_file data/FISV/test.txt
require_file data/features/FISV_rgb_UMT.npy
require_file data/features/FISV_flow_I3D.npy
require_file data/features/FISV_audio_MAST.npy

python -m py_compile \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/make_refresh_split.py \
  tools/analyze_direct_cue_effect.py \
  tools/analyze_paired_train_futility.py \
  tools/verify_paired_motion_training.py \
  tools/lock_paired_motion_protocol.py \
  tools/finalize_paired_motion_report.py
python tools/verify_paired_motion_training.py \
  2>&1 | tee "$OUT/TES_paired_code_verification.log"
python tools/analyze_direct_cue_effect.py --self_test \
  2>&1 | tee "$OUT/TES_analyzer_self_test.log"

# Exercise the real PAMFN/data/checkpoint path on the retired 320/80 protocol.
# The fresh 64-sample lockbox does not exist yet and cannot be touched here.
SMOKE_COMMON=(
  --gpu "$GPU"
  --feats 2
  --action TES
  --multi_modality
  --seed 0
  --split_seed 0
  --branch_ckpt_dir "$PARENT_BRANCH"
  --calibration_ratio 0.2
  --split_file "$PARENT_SPLIT"
  --calibration_final_only
  --blind_train
  --batch_size 16
  --epochs 1
)

python main.py \
  "${SMOKE_COMMON[@]}" \
  --paired_motion_training off_off \
  --save_final_model "$OUT/TES_paired_control_smoke_seed0.pth" \
  2>&1 | tee "$OUT/TES_paired_control_smoke_seed0.log"

python main.py \
  "${SMOKE_COMMON[@]}" \
  --motion_saliency_proxy \
  --motion_saliency_alpha 0.015 \
  --motion_saliency_source delta \
  --motion_saliency_positive_only \
  --paired_motion_training off_on \
  --save_final_model "$OUT/TES_paired_off_on_smoke_seed0.pth" \
  2>&1 | tee "$OUT/TES_paired_off_on_smoke_seed0.log"

CONTROL_SMOKE_DIGEST="$(
  sed -n 's/^Initial paired model digest: //p' \
    "$OUT/TES_paired_control_smoke_seed0.log"
)"
PAIRED_SMOKE_DIGEST="$(
  sed -n 's/^Initial paired model digest: //p' \
    "$OUT/TES_paired_off_on_smoke_seed0.log"
)"
if [[ -z "$CONTROL_SMOKE_DIGEST" ]] || \
   [[ "$CONTROL_SMOKE_DIGEST" != "$PAIRED_SMOKE_DIGEST" ]]; then
  echo "Smoke controls did not start from an identical model state" >&2
  exit 1
fi
grep -qF "Blind training complete" \
  "$OUT/TES_paired_control_smoke_seed0.log"
grep -qF "Blind training complete" \
  "$OUT/TES_paired_off_on_smoke_seed0.log"
echo "Real-model paired smoke test: passed"

python tools/make_refresh_split.py \
  --action TES \
  --label_file data/FISV/train.txt \
  --parent_split "$PARENT_SPLIT" \
  --selection_seed "$SPLIT_SEED" \
  --strata 8 \
  --samples_per_stratum 8 \
  --output "$SPLIT" \
  --audit_output "$SPLIT_AUDIT" \
  2>&1 | tee "$OUT/TES_refresh_split.log"

TRAIN_SPLIT=(
  --gpu "$GPU"
  --feats 2
  --action TES
  --seed 0
  --split_seed "$SPLIT_SEED"
  --calibration_ratio 0.16
  --split_file "$SPLIT"
  --calibration_final_only
  --blind_train
)

python main.py \
  "${TRAIN_SPLIT[@]}" \
  --modality V \
  --batch_size 32 \
  --epochs 250 \
  --save_base_model "$BRANCH/TES_rgb_UMT.pth" \
  2>&1 | tee "$OUT/TES_rgb_seed0.log"

python main.py \
  "${TRAIN_SPLIT[@]}" \
  --modality F \
  --batch_size 32 \
  --epochs 250 \
  --save_base_model "$BRANCH/TES_flow_I3D.pth" \
  2>&1 | tee "$OUT/TES_flow_seed0.log"

python main.py \
  "${TRAIN_SPLIT[@]}" \
  --modality A \
  --batch_size 32 \
  --epochs 250 \
  --save_base_model "$BRANCH/TES_audio_MAST.pth" \
  2>&1 | tee "$OUT/TES_audio_seed0.log"

SCORER_COMMON=(
  "${TRAIN_SPLIT[@]}"
  --multi_modality
  --branch_ckpt_dir "$BRANCH"
  --batch_size 16
  --epochs 100
)

python main.py \
  "${SCORER_COMMON[@]}" \
  --paired_motion_training off_off \
  --save_final_model "$CONTROL" \
  2>&1 | tee "$OUT/TES_paired_control_seed0.log"

python main.py \
  "${SCORER_COMMON[@]}" \
  --motion_saliency_proxy \
  --motion_saliency_alpha 0.015 \
  --motion_saliency_source delta \
  --motion_saliency_positive_only \
  --paired_motion_training off_on \
  --save_final_model "$PAIRED" \
  2>&1 | tee "$OUT/TES_paired_off_on_seed0.log"

require_file "$BRANCH/TES_rgb_UMT.pth"
require_file "$BRANCH/TES_flow_I3D.pth"
require_file "$BRANCH/TES_audio_MAST.pth"
require_file "$CONTROL"
require_file "$PAIRED"

PREDICTION_COMMON=(
  --gpu "$GPU"
  --feats 2
  --action TES
  --multi_modality
  --seed 0
  --split_seed "$SPLIT_SEED"
  --branch_ckpt_dir "$BRANCH"
  --calibration_ratio 0.16
  --split_file "$SPLIT"
  --prediction_probe
  --prediction_motion_gate 0
  --prediction_skeleton_gate 0
)

export_control() {
  local split_name="$1"
  local output="$2"
  local role="$3"
  local manifest="$4"
  python main.py \
    "${PREDICTION_COMMON[@]}" \
    --prediction_split "$split_name" \
    --ckpt_path "$CONTROL" \
    --prediction_output "$output" \
    --prediction_role "$role" \
    --prediction_manifest_output "$manifest"
}

export_paired() {
  local split_name="$1"
  local gate="$2"
  local output="$3"
  local role="$4"
  local manifest="$5"
  python main.py \
    --gpu "$GPU" \
    --feats 2 \
    --action TES \
    --multi_modality \
    --seed 0 \
    --split_seed "$SPLIT_SEED" \
    --branch_ckpt_dir "$BRANCH" \
    --calibration_ratio 0.16 \
    --split_file "$SPLIT" \
    --motion_saliency_proxy \
    --motion_saliency_alpha 0.015 \
    --motion_saliency_source delta \
    --motion_saliency_positive_only \
    --prediction_probe \
    --prediction_split "$split_name" \
    --prediction_motion_gate "$gate" \
    --prediction_skeleton_gate 0 \
    --ckpt_path "$PAIRED" \
    --prediction_output "$output" \
    --prediction_role "$role" \
    --prediction_manifest_output "$manifest"
}

# Lock all train-derived predictions before reserving the fresh lockbox.
export_control train "$OUT/TES_paired_control_train.csv" \
  control_train "$CONTROL_TRAIN_MANIFEST"
export_paired train 0 "$OUT/TES_paired_g0_train.csv" \
  paired_g0_train "$G0_TRAIN_MANIFEST"
export_paired train 1 "$OUT/TES_paired_g1_train.csv" \
  paired_g1_train "$G1_TRAIN_MANIFEST"

python tools/analyze_paired_train_futility.py \
  --control "$OUT/TES_paired_control_train.csv" \
  --g0 "$OUT/TES_paired_g0_train.csv" \
  --g1 "$OUT/TES_paired_g1_train.csv" \
  --output "$TRAIN_FUTILITY" \
  --expected_samples 336
if grep -q '"status": "EARLY_NO_GO_TRAIN"' "$TRAIN_FUTILITY"; then
  echo
  echo "Paired Motion stopped before protocol lock: train selected gate 0."
  echo "The fresh64 lockbox remains unopened. Report: $TRAIN_FUTILITY"
  exit 0
fi

python tools/lock_paired_motion_protocol.py \
  --output "$PROTOCOL_LOCK" \
  --reservation "$LOCKBOX_RESERVATION" \
  --split "$SPLIT" \
  --split_audit "$SPLIT_AUDIT" \
  --parent_split "$PARENT_SPLIT" \
  --parent_branch_dir "$PARENT_BRANCH" \
  --branch_dir "$BRANCH" \
  --control_checkpoint "$CONTROL" \
  --paired_checkpoint "$PAIRED" \
  --rgb_log "$OUT/TES_rgb_seed0.log" \
  --flow_log "$OUT/TES_flow_seed0.log" \
  --audio_log "$OUT/TES_audio_seed0.log" \
  --control_log "$OUT/TES_paired_control_seed0.log" \
  --paired_log "$OUT/TES_paired_off_on_seed0.log" \
  --control_train_manifest "$CONTROL_TRAIN_MANIFEST" \
  --g0_train_manifest "$G0_TRAIN_MANIFEST" \
  --g1_train_manifest "$G1_TRAIN_MANIFEST" \
  --control_lockbox_csv "$OUT/TES_paired_control_lockbox64.csv" \
  --g0_lockbox_csv "$OUT/TES_paired_g0_lockbox64.csv" \
  --g1_lockbox_csv "$OUT/TES_paired_g1_lockbox64.csv" \
  --control_lockbox_manifest "$CONTROL_LOCKBOX_MANIFEST" \
  --g0_lockbox_manifest "$G0_LOCKBOX_MANIFEST" \
  --g1_lockbox_manifest "$G1_LOCKBOX_MANIFEST" \
  2>&1 | tee "$OUT/TES_paired_protocol_lock.log"
require_file "$PROTOCOL_LOCK"
require_file "$LOCKBOX_RESERVATION"

# The global reservation makes this the only allowed opening of this split.
export_control calibration "$OUT/TES_paired_control_lockbox64.csv" \
  control_lockbox "$CONTROL_LOCKBOX_MANIFEST"
export_paired calibration 0 "$OUT/TES_paired_g0_lockbox64.csv" \
  paired_g0_lockbox "$G0_LOCKBOX_MANIFEST"
export_paired calibration 1 "$OUT/TES_paired_g1_lockbox64.csv" \
  paired_g1_lockbox "$G1_LOCKBOX_MANIFEST"

python tools/analyze_direct_cue_effect.py \
  --dataset TES \
  --base_train "$OUT/TES_paired_control_train.csv" \
  --motion_on_train "$OUT/TES_paired_g1_train.csv" \
  --motion_off_train "$OUT/TES_paired_g0_train.csv" \
  --skeleton_on_train "$OUT/TES_paired_control_train.csv" \
  --skeleton_off_train "$OUT/TES_paired_control_train.csv" \
  --base_cal "$OUT/TES_paired_control_lockbox64.csv" \
  --motion_on_cal "$OUT/TES_paired_g1_lockbox64.csv" \
  --motion_off_cal "$OUT/TES_paired_g0_lockbox64.csv" \
  --skeleton_on_cal "$OUT/TES_paired_control_lockbox64.csv" \
  --skeleton_off_cal "$OUT/TES_paired_control_lockbox64.csv" \
  --output "$OUT/TES_paired_motion_effect.json" \
  --expected_train_samples 336 \
  --expected_calibration_samples 64 \
  --bootstrap 2000 \
  --gate_step 1 \
  --seed 0 \
  --paired_motion_exposure \
  --motion_bound 1 \
  --skeleton_bound 1 \
  2>&1 | tee "$OUT/TES_paired_motion_effect.log"

python tools/finalize_paired_motion_report.py \
  --report "$OUT/TES_paired_motion_effect.json" \
  --control_checkpoint "$CONTROL" \
  --paired_checkpoint "$PAIRED" \
  --control_log "$OUT/TES_paired_control_seed0.log" \
  --paired_log "$OUT/TES_paired_off_on_seed0.log" \
  --split "$SPLIT" \
  --split_audit "$SPLIT_AUDIT" \
  --branch_dir "$BRANCH" \
  --protocol_lock "$PROTOCOL_LOCK" \
  --lockbox_reservation "$LOCKBOX_RESERVATION" \
  --lockbox_open_marker "$LOCKBOX_OPEN_MARKER" \
  --control_train_manifest "$CONTROL_TRAIN_MANIFEST" \
  --g0_train_manifest "$G0_TRAIN_MANIFEST" \
  --g1_train_manifest "$G1_TRAIN_MANIFEST" \
  --control_lockbox_manifest "$CONTROL_LOCKBOX_MANIFEST" \
  --g0_lockbox_manifest "$G0_LOCKBOX_MANIFEST" \
  --g1_lockbox_manifest "$G1_LOCKBOX_MANIFEST" \
  2>&1 | tee "$OUT/TES_paired_finalize.log"

sha256sum \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/make_refresh_split.py \
  tools/analyze_direct_cue_effect.py \
  tools/analyze_paired_train_futility.py \
  tools/verify_paired_motion_training.py \
  tools/lock_paired_motion_protocol.py \
  tools/finalize_paired_motion_report.py \
  tools/run_paired_motion_tes.sh \
  configs/FISV_feats2.py \
  utils/lr_scheduler.py utils/utils.py \
  data/FISV/train.txt data/FISV/test.txt \
  data/features/FISV_rgb_UMT.npy \
  data/features/FISV_flow_I3D.npy \
  data/features/FISV_audio_MAST.npy \
  "$PARENT_SPLIT" \
  "$PARENT_BRANCH/TES_rgb_UMT.pth" \
  "$PARENT_BRANCH/TES_flow_I3D.pth" \
  "$PARENT_BRANCH/TES_audio_MAST.pth" \
  "$SPLIT" "$SPLIT_AUDIT" \
  "$BRANCH/TES_rgb_UMT.pth" \
  "$BRANCH/TES_flow_I3D.pth" \
  "$BRANCH/TES_audio_MAST.pth" \
  "$CONTROL" "$PAIRED" \
  "$TRAIN_FUTILITY" \
  "$PROTOCOL_LOCK" "$LOCKBOX_RESERVATION" \
  "$LOCKBOX_OPEN_MARKER" \
  > "$OUT/TES_paired_reference_sha256.txt"

sha256sum \
  "$OUT/TES_paired_code_verification.log" \
  "$OUT/TES_analyzer_self_test.log" \
  "$OUT/TES_paired_control_smoke_seed0.pth" \
  "$OUT/TES_paired_control_smoke_seed0.log" \
  "$OUT/TES_paired_off_on_smoke_seed0.pth" \
  "$OUT/TES_paired_off_on_smoke_seed0.log" \
  "$OUT/TES_refresh_split.log" \
  "$OUT/TES_rgb_seed0.log" \
  "$OUT/TES_flow_seed0.log" \
  "$OUT/TES_audio_seed0.log" \
  "$OUT/TES_paired_control_seed0.log" \
  "$OUT/TES_paired_off_on_seed0.log" \
  "$OUT/TES_paired_protocol_lock.log" \
  "$TRAIN_FUTILITY" \
  "$PROTOCOL_LOCK" \
  "$LOCKBOX_RESERVATION" \
  "$LOCKBOX_OPEN_MARKER" \
  "$OUT/TES_paired_control_train.csv" \
  "$CONTROL_TRAIN_MANIFEST" \
  "$OUT/TES_paired_control_lockbox64.csv" \
  "$CONTROL_LOCKBOX_MANIFEST" \
  "$OUT/TES_paired_g0_train.csv" \
  "$G0_TRAIN_MANIFEST" \
  "$OUT/TES_paired_g1_train.csv" \
  "$G1_TRAIN_MANIFEST" \
  "$OUT/TES_paired_g0_lockbox64.csv" \
  "$G0_LOCKBOX_MANIFEST" \
  "$OUT/TES_paired_g1_lockbox64.csv" \
  "$G1_LOCKBOX_MANIFEST" \
  "$OUT/TES_paired_motion_effect.json" \
  "$OUT/TES_paired_motion_effect.log" \
  "$OUT/TES_paired_finalize.log" \
  "$OUT/TES_paired_reference_sha256.txt" \
  > "$OUT/TES_paired_seed0_sha256.txt"

echo
echo "TES fresh-lockbox paired Motion experiment completed: $OUT"
echo "Do not run test, tune this mechanism, or implement AutoGate before reviewing paired_motion_decision."
