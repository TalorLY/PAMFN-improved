#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-5}"
SPLIT_SEED="${SPLIT_SEED:-20260723}"
TRAIN_SEED="${TRAIN_SEED:-0}"
OUT_ROOT="${OUT_ROOT:-results/c_oprr_screen_seed${SPLIT_SEED}}"
SCREEN_EPOCHS="${SCREEN_EPOCHS:-100}"
BOOTSTRAP="${BOOTSTRAP:-2000}"
RESUME="${RESUME:-1}"

FISV_BRANCH_EPOCHS="${FISV_BRANCH_EPOCHS:-250}"
BALL_BRANCH_EPOCHS="${BALL_BRANCH_EPOCHS:-250}"
FISV_BASE_EPOCHS="${FISV_BASE_EPOCHS:-100}"
BALL_BASE_EPOCHS="${BALL_BASE_EPOCHS:-500}"

PHASE_COUNT="${PHASE_COUNT:-3}"
ADAPTER_DIM="${ADAPTER_DIM:-16}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.05}"
RANK_WEIGHT="${RANK_WEIGHT:-0.1}"
GUARD_WEIGHT="${GUARD_WEIGHT:-1.0}"
ZERO_MEAN_WEIGHT="${ZERO_MEAN_WEIGHT:-1.0}"
LOW_MARGIN="${LOW_MARGIN:-0.02}"
GUARD_MARGIN="${GUARD_MARGIN:-0.05}"
RANK_TEMPERATURE="${RANK_TEMPERATURE:-0.05}"
MIN_PHASE_MASS="${MIN_PHASE_MASS:-0.1}"
PHASE_SOFTNESS="${PHASE_SOFTNESS:-0.05}"

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

mkdir -p "$OUT_ROOT/splits" "$OUT_ROOT/pair_audits"

python -m py_compile \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/make_phase_screen_split.py \
  tools/verify_phase_rank_adapter.py \
  tools/analyze_phase_rank_screen.py
python tools/verify_phase_rank_adapter.py \
  2>&1 | tee "$OUT_ROOT/phase_rank_verification.log"
python tools/analyze_phase_rank_screen.py --self_test \
  2>&1 | tee "$OUT_ROOT/phase_rank_analyzer_self_test.log"

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

train_base() {
  local action="$1"
  local split="$2"
  local branches="$3"
  local epochs="$4"
  local out="$5"
  local log="$6"
  if [[ "$RESUME" == 1 && -f "$out" && -f "$log" ]] \
      && grep -qF "Blind training complete;" "$log" \
      && grep -qF "Saved the final blind-training checkpoint;" "$log"; then
    echo "Resume: reuse clean-inner Base $action"
    return
  fi
  mapfile -t common < <(common_args "$action" "$split")
  python main.py \
    "${common[@]}" \
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
}

phase_args() {
  local mode="$1"
  printf '%s\n' \
    --anchored_residual \
    --phase_rank_mode "$mode" \
    --phase_count "$PHASE_COUNT" \
    --residual_adapter_dim "$ADAPTER_DIM" \
    --phase_residual_scale "$RESIDUAL_SCALE" \
    --phase_rank_weight "$RANK_WEIGHT" \
    --phase_guard_weight "$GUARD_WEIGHT" \
    --phase_zero_mean_weight "$ZERO_MEAN_WEIGHT" \
    --phase_low_margin "$LOW_MARGIN" \
    --phase_guard_margin "$GUARD_MARGIN" \
    --phase_rank_temperature "$RANK_TEMPERATURE" \
    --phase_min_mass "$MIN_PHASE_MASS" \
    --phase_softness "$PHASE_SOFTNESS"
}

train_phase() {
  local action="$1"
  local split="$2"
  local branches="$3"
  local base="$4"
  local mode="$5"
  local out="$6"
  local log="$7"
  if [[ "$RESUME" == 1 && -f "$out" && -f "$log" ]] \
      && grep -qF "Phase rank screen: mode=$mode" "$log" \
      && grep -qF "Frozen Base invariant: passed" "$log" \
      && grep -qF "Blind training complete;" "$log"; then
    echo "Resume: reuse $action phase mode=$mode"
    return
  fi
  mapfile -t common < <(common_args "$action" "$split")
  mapfile -t phase < <(phase_args "$mode")
  python main.py \
    "${common[@]}" "${phase[@]}" \
    --multi_modality \
    --branch_ckpt_dir "$branches" \
    --epochs "$SCREEN_EPOCHS" \
    --calibration_final_only \
    --blind_train \
    --init_ckpt "$base" \
    --save_final_model "$out" \
    2>&1 | tee "$log"
  require_file "$out"
  grep -qF "Phase rank screen: mode=$mode" "$log"
  grep -qF "Frozen Base invariant: passed" "$log"
  grep -qF "Blind training complete;" "$log"
  grep -qF "Saved the final blind-training checkpoint;" "$log"
}

export_base() {
  local action="$1"
  local split="$2"
  local branches="$3"
  local base="$4"
  local split_name="$5"
  local suffix="$6"
  local output="$7"
  mapfile -t common < <(common_args "$action" "$split")
  python main.py \
    "${common[@]}" \
    --multi_modality \
    --branch_ckpt_dir "$branches" \
    --prediction_probe \
    --prediction_split "$split_name" \
    --ckpt_path "$base" \
    --prediction_output "$output"
}

export_phase() {
  local action="$1"
  local split="$2"
  local branches="$3"
  local mode="$4"
  local checkpoint="$5"
  local split_name="$6"
  local output="$7"
  local control="${8:-}"
  mapfile -t common < <(common_args "$action" "$split")
  mapfile -t phase < <(phase_args "$mode")
  local control_args=()
  if [[ -n "$control" ]]; then
    control_args=(--prediction_phase_control "$control")
  fi
  python main.py \
    "${common[@]}" "${phase[@]}" \
    "${control_args[@]}" \
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
  local base_epochs="$5"
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

  local base="$out/${action}_inner_base_seed${TRAIN_SEED}.pth"
  train_base \
    "$action" "$split" "$branches" "$base_epochs" "$base" \
    "$out/${action}_inner_base_seed${TRAIN_SEED}.log"

  local mode checkpoint
  for mode in global fixed ordered; do
    checkpoint="$out/${action}_${mode}_seed${TRAIN_SEED}.pth"
    train_phase \
      "$action" "$split" "$branches" "$base" "$mode" \
      "$checkpoint" "$out/${action}_${mode}_seed${TRAIN_SEED}.log"
  done

  export_base \
    "$action" "$split" "$branches" "$base" train train \
    "$out/${action}_base_train.csv"
  export_base \
    "$action" "$split" "$branches" "$base" calibration cal \
    "$out/${action}_base_cal.csv"
  for mode in global fixed ordered; do
    checkpoint="$out/${action}_${mode}_seed${TRAIN_SEED}.pth"
    export_phase \
      "$action" "$split" "$branches" "$mode" "$checkpoint" train \
      "$out/${action}_${mode}_train.csv"
    export_phase \
      "$action" "$split" "$branches" "$mode" "$checkpoint" calibration \
      "$out/${action}_${mode}_cal.csv"
  done
  checkpoint="$out/${action}_ordered_seed${TRAIN_SEED}.pth"
  export_phase \
    "$action" "$split" "$branches" ordered "$checkpoint" calibration \
    "$out/${action}_ordered_reverse_cal.csv" reverse
  export_phase \
    "$action" "$split" "$branches" ordered "$checkpoint" calibration \
    "$out/${action}_ordered_shuffle_cal.csv" shuffle

  sha256sum \
    "$split" "$branches"/*.pth "$base" \
    "$out"/*_global_seed*.pth \
    "$out"/*_fixed_seed*.pth \
    "$out"/*_ordered_seed*.pth \
    "$out"/*.csv "$out"/*.log \
    > "$out/${action}_phase_screen_sha256.txt"
}

run_action \
  TES "$FISV_SPLIT" FISV \
  "$FISV_BRANCH_EPOCHS" "$FISV_BASE_EPOCHS"
run_action \
  PCS "$FISV_SPLIT" FISV \
  "$FISV_BRANCH_EPOCHS" "$FISV_BASE_EPOCHS"
run_action \
  Ball "$BALL_SPLIT" Ball \
  "$BALL_BRANCH_EPOCHS" "$BALL_BASE_EPOCHS"

python tools/analyze_phase_rank_screen.py \
  --root "$OUT_ROOT" \
  --output "$OUT_ROOT/phase_rank_screen_report.json" \
  --pair_audit_dir "$OUT_ROOT/pair_audits" \
  --actions TES PCS Ball \
  --bootstrap "$BOOTSTRAP" \
  --seed "$TRAIN_SEED" \
  --min_macro_delta 0.5 \
  --min_action_delta 0.5 \
  --min_vs_control 0.3 \
  --max_affine_mse_drop_pct 2.0 \
  --min_phase_mass "$MIN_PHASE_MASS" \
  --min_center_gap 0.1 \
  --min_order_control_drop 0.1 \
  2>&1 | tee "$OUT_ROOT/phase_rank_screen_report.log"

sha256sum \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/make_phase_screen_split.py \
  tools/verify_phase_rank_adapter.py \
  tools/analyze_phase_rank_screen.py \
  tools/run_phase_rank_screen.sh \
  "$OUT_ROOT"/splits/*.json \
  "$OUT_ROOT"/TES/*.csv "$OUT_ROOT"/PCS/*.csv "$OUT_ROOT"/Ball/*.csv \
  "$OUT_ROOT"/pair_audits/*.csv \
  "$OUT_ROOT/phase_rank_screen_report.json" \
  > "$OUT_ROOT/phase_rank_screen_sha256.txt"

echo
echo "C-OPRR screen complete: $OUT_ROOT"
echo "Read decision.status in phase_rank_screen_report.json."
echo "Only GO_TO_THREE_SEED_SIX_ACTION authorizes the full experiment."
