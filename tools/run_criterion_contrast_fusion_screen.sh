#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-5}"
SPLIT_SEED="${SPLIT_SEED:-20260725}"
TRAIN_SEED="${TRAIN_SEED:-0}"
OUT_ROOT="${OUT_ROOT:-results/criterion_contrast_seed${SPLIT_SEED}}"
BOOTSTRAP="${BOOTSTRAP:-2000}"
RESUME="${RESUME:-1}"

FISV_BRANCH_EPOCHS="${FISV_BRANCH_EPOCHS:-250}"
BALL_BRANCH_EPOCHS="${BALL_BRANCH_EPOCHS:-250}"
FISV_MODEL_EPOCHS="${FISV_MODEL_EPOCHS:-100}"
BALL_MODEL_EPOCHS="${BALL_MODEL_EPOCHS:-500}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-0.1}"
RANK_WEIGHT="${RANK_WEIGHT:-0.1}"
CONTRAST_WEIGHT="${CONTRAST_WEIGHT:-1.0}"
PRIVATE_WEIGHT="${PRIVATE_WEIGHT:-0.01}"
PENALTY_WEIGHT="${PENALTY_WEIGHT:-0.1}"
PENALTY_POSITIVE_WEIGHT="${PENALTY_POSITIVE_WEIGHT:-4.0}"

if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 1
fi

require_file() {
  [[ -f "$1" ]] || {
    echo "Missing required file: $1" >&2
    exit 1
  }
}

mkdir -p "$OUT_ROOT/splits"

python -m py_compile \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/make_phase_screen_split.py \
  tools/verify_criterion_contrast_fusion.py \
  tools/analyze_criterion_contrast_fusion.py
python tools/verify_criterion_contrast_fusion.py \
  2>&1 | tee "$OUT_ROOT/criterion_contrast_verification.log"
python tools/analyze_criterion_contrast_fusion.py --self_test \
  2>&1 | tee "$OUT_ROOT/criterion_contrast_analyzer_self_test.log"

FISV_SPLIT="$OUT_ROOT/splits/FISV_seed${SPLIT_SEED}_split.json"
FISV_AUDIT="$OUT_ROOT/splits/FISV_seed${SPLIT_SEED}_audit.json"
BALL_SPLIT="$OUT_ROOT/splits/Ball_seed${SPLIT_SEED}_split.json"
BALL_AUDIT="$OUT_ROOT/splits/Ball_seed${SPLIT_SEED}_audit.json"

python tools/make_phase_screen_split.py \
  --action TES \
  --seed "$SPLIT_SEED" \
  --calibration_ratio 0.2 \
  --output "$FISV_SPLIT" \
  --audit_output "$FISV_AUDIT"
python tools/make_phase_screen_split.py \
  --action Ball \
  --seed "$SPLIT_SEED" \
  --calibration_ratio 0.2 \
  --output "$BALL_SPLIT" \
  --audit_output "$BALL_AUDIT"

common_args() {
  local action="$1"
  local split="$2"
  printf '%s\n' \
    --gpu "$GPU" \
    --feats 2 \
    --action "$action" \
    --seed "$TRAIN_SEED" \
    --split_seed "$SPLIT_SEED" \
    --sequence_sampling normalized_full \
    --calibration_ratio 0.2 \
    --split_file "$split"
}

criterion_args() {
  local mode="$1"
  if [[ "$mode" == "single" ]]; then
    return
  fi
  printf '%s\n' \
    --criterion_screen_mode "$mode" \
    --criterion_consistency_weight "$CONSISTENCY_WEIGHT" \
    --criterion_rank_weight "$RANK_WEIGHT" \
    --criterion_contrast_weight "$CONTRAST_WEIGHT" \
    --criterion_private_weight "$PRIVATE_WEIGHT" \
    --criterion_penalty_weight "$PENALTY_WEIGHT" \
    --criterion_penalty_positive_weight "$PENALTY_POSITIVE_WEIGHT"
}

train_branch() {
  local action="$1"
  local split="$2"
  local modality="$3"
  local name="$4"
  local epochs="$5"
  local out="$6"
  local log="$7"
  if [[ "$RESUME" == 1 && -f "$out" && -f "$log" ]] \
      && grep -qF "Blind training complete;" "$log" \
      && grep -qF "Saved the final blind-training branch model;" "$log"; then
    echo "Resume: reuse $action $name branch"
    return
  fi
  mapfile -t common < <(common_args "$action" "$split")
  python main.py \
    "${common[@]}" \
    --modality "$modality" \
    --epochs "$epochs" \
    --calibration_final_only \
    --blind_train \
    --save_base_model "$out" \
    2>&1 | tee "$log"
  require_file "$out"
  grep -qF "Blind training complete;" "$log"
  grep -qF "Saved the final blind-training branch model;" "$log"
}

train_model() {
  local action="$1"
  local split="$2"
  local branches="$3"
  local mode="$4"
  local epochs="$5"
  local out="$6"
  local log="$7"
  if [[ "$RESUME" == 1 && -f "$out" && -f "$log" ]] \
      && grep -qF "Blind training complete;" "$log" \
      && grep -qF "Saved the final blind-training checkpoint;" "$log"; then
    if [[ "$mode" == "single" ]] \
        || grep -qF "Criterion contrast screen: mode=$mode" "$log"; then
      echo "Resume: reuse $action mode=$mode"
      return
    fi
  fi
  mapfile -t common < <(common_args "$action" "$split")
  mapfile -t criterion < <(criterion_args "$mode")
  python main.py \
    "${common[@]}" "${criterion[@]}" \
    --multi_modality \
    --branch_ckpt_dir "$branches" \
    --epochs "$epochs" \
    --calibration_final_only \
    --blind_train \
    --save_final_model "$out" \
    2>&1 | tee "$log"
  require_file "$out"
  grep -qF "Blind training complete;" "$log"
  grep -qF "Saved the final blind-training checkpoint;" "$log"
  if [[ "$mode" != "single" ]]; then
    grep -qF "Criterion contrast screen: mode=$mode" "$log"
  fi
}

export_model() {
  local action="$1"
  local split="$2"
  local branches="$3"
  local mode="$4"
  local checkpoint="$5"
  local split_name="$6"
  local output="$7"
  local control="${8:-}"
  mapfile -t common < <(common_args "$action" "$split")
  mapfile -t criterion < <(criterion_args "$mode")
  local control_args=()
  if [[ -n "$control" ]]; then
    control_args=(--prediction_criterion_control "$control")
  fi
  python main.py \
    "${common[@]}" "${criterion[@]}" "${control_args[@]}" \
    --multi_modality \
    --branch_ckpt_dir "$branches" \
    --prediction_probe \
    --prediction_split "$split_name" \
    --ckpt_path "$checkpoint" \
    --prediction_output "$output"
}

run_action() {
  local action="$1"
  local split="$2"
  local feature_family="$3"
  local branch_epochs="$4"
  local model_epochs="$5"
  local out="$OUT_ROOT/$action"
  local branches="$out/branches"
  mkdir -p "$out" "$branches"

  local feature
  for feature in \
    "data/features/${feature_family}_rgb_UMT.npy" \
    "data/features/${feature_family}_flow_I3D.npy" \
    "data/features/${feature_family}_audio_MAST.npy"; do
    require_file "$feature"
  done

  train_branch \
    "$action" "$split" V rgb "$branch_epochs" \
    "$branches/${action}_rgb_UMT.pth" \
    "$out/${action}_rgb_seed${TRAIN_SEED}.log"
  train_branch \
    "$action" "$split" F flow "$branch_epochs" \
    "$branches/${action}_flow_I3D.pth" \
    "$out/${action}_flow_seed${TRAIN_SEED}.log"
  train_branch \
    "$action" "$split" A audio "$branch_epochs" \
    "$branches/${action}_audio_MAST.pth" \
    "$out/${action}_audio_seed${TRAIN_SEED}.log"

  local mode checkpoint
  for mode in single rank_shared contrast_fusion; do
    checkpoint="$out/${action}_${mode}_seed${TRAIN_SEED}.pth"
    train_model \
      "$action" "$split" "$branches" "$mode" "$model_epochs" \
      "$checkpoint" "$out/${action}_${mode}_seed${TRAIN_SEED}.log"
  done

  for mode in single rank_shared contrast_fusion; do
    checkpoint="$out/${action}_${mode}_seed${TRAIN_SEED}.pth"
    export_model \
      "$action" "$split" "$branches" "$mode" "$checkpoint" train \
      "$out/${action}_${mode}_train.csv"
    export_model \
      "$action" "$split" "$branches" "$mode" "$checkpoint" calibration \
      "$out/${action}_${mode}_cal.csv"
  done
  checkpoint="$out/${action}_contrast_fusion_seed${TRAIN_SEED}.pth"
  export_model \
    "$action" "$split" "$branches" contrast_fusion "$checkpoint" \
    calibration "$out/${action}_contrast_fusion_uniform_cal.csv" uniform

  sha256sum \
    "$split" "$branches"/*.pth "$out"/*.pth \
    "$out"/*.csv "$out"/*.log \
    > "$out/${action}_criterion_contrast_sha256.txt"
}

run_action \
  TES "$FISV_SPLIT" FISV \
  "$FISV_BRANCH_EPOCHS" "$FISV_MODEL_EPOCHS"
run_action \
  PCS "$FISV_SPLIT" FISV \
  "$FISV_BRANCH_EPOCHS" "$FISV_MODEL_EPOCHS"
run_action \
  Ball "$BALL_SPLIT" Ball \
  "$BALL_BRANCH_EPOCHS" "$BALL_MODEL_EPOCHS"

python tools/analyze_criterion_contrast_fusion.py \
  --root "$OUT_ROOT" \
  --output "$OUT_ROOT/criterion_contrast_report.json" \
  --bootstrap "$BOOTSTRAP" \
  --seed "$TRAIN_SEED" \
  --low_margin 0.05 \
  --min_macro_delta 0.5 \
  --min_action_delta 0.5 \
  --min_vs_shared 0.3 \
  --min_uniform_drop 0.2 \
  --min_low_margin_gain 0.0 \
  --max_affine_mse_drop_pct 2.0 \
  --min_modality_weight_js 0.005 \
  --min_contrast_coef 10.0 \
  --min_penalty_auc 0.55 \
  --max_consistency_gap 0.02 \
  2>&1 | tee "$OUT_ROOT/criterion_contrast_report.log"

sha256sum \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/make_phase_screen_split.py \
  tools/verify_criterion_contrast_fusion.py \
  tools/analyze_criterion_contrast_fusion.py \
  tools/run_criterion_contrast_fusion_screen.sh \
  "$OUT_ROOT"/splits/*.json \
  "$OUT_ROOT"/TES/*.csv "$OUT_ROOT"/PCS/*.csv "$OUT_ROOT"/Ball/*.csv \
  "$OUT_ROOT/criterion_contrast_report.json" \
  > "$OUT_ROOT/criterion_contrast_sha256.txt"

echo
echo "Criterion-contrast fusion screen complete: $OUT_ROOT"
echo "Read decision.status in criterion_contrast_report.json."
echo "Only GO_TO_THREE_SEED_SIX_ACTION_CRITERION_CONTRAST authorizes the full run."
