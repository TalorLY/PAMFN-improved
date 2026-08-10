#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-5}"
OUT_ROOT="${OUT_ROOT:-results/clean_inner_base_ranking_rescue}"
FISV_SPLIT="${FISV_SPLIT:-results/calibration_probe/FISV_seed0_split.json}"
RG_SPLIT="${RG_SPLIT:-$OUT_ROOT/RG_seed0_split.json}"
EPOCHS="${EPOCHS:-100}"
BOOTSTRAP="${BOOTSTRAP:-2000}"
RESUME="${RESUME:-0}"

if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 1
fi

# Protocol-locked clean-inner-Base ranking screen.  Every modality branch and
# the multimodal Base is trained only on the manifest's inner-train IDs.  The
# Base then stays frozen for the MSE-only control and two ranking weights.
RANK_WEIGHTS=(0 0.01 0.05)

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

weight_tag() {
  printf '%s' "${1/./p}"
}

mkdir -p "$OUT_ROOT"
require_file "$FISV_SPLIT"

python -m py_compile \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/verify_anchored_residual.py \
  tools/analyze_direct_cue_effect.py \
  tools/analyze_ranking_expert_rescue.py
python tools/verify_anchored_residual.py \
  2>&1 | tee "$OUT_ROOT/anchored_residual_verification.log"
python tools/analyze_direct_cue_effect.py --self_test \
  2>&1 | tee "$OUT_ROOT/direct_effect_analyzer_self_test.log"
python tools/analyze_ranking_expert_rescue.py --self_test \
  2>&1 | tee "$OUT_ROOT/ranking_analyzer_self_test.log"

run_action() {
  local action="$1"
  local expected_train="$2"
  local expected_cal="$3"
  local feature_family="$4"
  local split="$5"
  local out="$OUT_ROOT/$action"
  local branch="$out/branches"
  local base="$out/${action}_inner_base_seed0.pth"
  local base_log="$out/${action}_inner_base_seed0.log"
  local file

  mkdir -p "$out" "$branch"
  for file in \
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

  local common=(
    --gpu "$GPU"
    --feats 2
    --action "$action"
    --multi_modality
    --seed 0
    --split_seed 0
    --branch_ckpt_dir "$branch"
    --calibration_ratio 0.2
    --split_file "$split"
  )
  local residual=(
    --anchored_residual
    --bias_separated_residual
    --residual_v2
    --residual_adapter_dim 16
    --residual_zero_mean_weight 1
  )
  local motion=(
    --motion_saliency_proxy
    --motion_saliency_alpha 0.015
    --motion_saliency_source delta
    --motion_saliency_positive_only
  )
  local skeleton=(
    --use_skeleton
    --skeleton_root .
    --skeleton_gate_max 0.02
  )

  train_branch() {
    local modality="$1"
    local descriptor="$2"
    local checkpoint="$3"
    local log="$4"
    if [[ "$RESUME" == 1 && -f "$checkpoint" && -f "$log" ]] \
        && grep -qF "Blind training complete;" "$log" \
        && grep -qF "Saved the final blind-training branch model;" "$log" \
        && ! grep -qF "Initialized from " "$log"; then
      echo "Resume: reuse clean inner branch $action $descriptor"
      return
    fi
    python main.py \
      --gpu "$GPU" \
      --feats 2 \
      --action "$action" \
      --modality "$modality" \
      --seed 0 \
      --split_seed 0 \
      --calibration_ratio 0.2 \
      --split_file "$split" \
      --calibration_final_only \
      --blind_train \
      --save_base_model "$checkpoint" \
      2>&1 | tee "$log"
    require_file "$checkpoint"
    grep -qF "Blind training complete;" "$log"
    grep -qF "Saved the final blind-training branch model;" "$log"
    if grep -qF "Initialized from " "$log"; then
      echo "Forbidden init checkpoint in clean branch: $log" >&2
      exit 1
    fi
    echo "Clean inner branch complete: $action $descriptor"
  }

  train_branch V rgb \
    "$branch/${action}_rgb_UMT.pth" \
    "$out/${action}_inner_rgb_seed0.log"
  train_branch F flow \
    "$branch/${action}_flow_I3D.pth" \
    "$out/${action}_inner_flow_seed0.log"
  train_branch A audio \
    "$branch/${action}_audio_MAST.pth" \
    "$out/${action}_inner_audio_seed0.log"

  if [[ "$RESUME" == 1 && -f "$base" && -f "$base_log" ]] \
      && grep -qF "Blind training complete;" "$base_log" \
      && grep -qF "Saved the final blind-training checkpoint;" "$base_log" \
      && ! grep -qF "Initialized from " "$base_log"; then
    echo "Resume: reuse clean inner Base $action"
  else
    python main.py \
      "${common[@]}" \
      --calibration_final_only \
      --blind_train \
      --save_final_model "$base" \
      2>&1 | tee "$base_log"
  fi
  require_file "$base"
  require_file "$split"
  grep -qF "Blind training complete;" "$base_log"
  grep -qF "Saved the final blind-training checkpoint;" "$base_log"
  if grep -qF "Initialized from " "$base_log"; then
    echo "Forbidden init checkpoint in clean inner Base: $base_log" >&2
    exit 1
  fi

  train_expert() {
    local cue="$1"
    local rank_weight="$2"
    local tag="$3"
    local model="$out/${action}_rank${tag}_${cue}_seed0.pth"
    local log="$out/${action}_rank${tag}_${cue}_seed0.log"
    local cue_options=()
    case "$cue" in
      motion) cue_options=("${motion[@]}") ;;
      skeleton) cue_options=("${skeleton[@]}") ;;
      *) echo "Unknown cue: $cue" >&2; exit 1 ;;
    esac

    if [[ "$RESUME" == 1 && -f "$model" && -f "$log" ]] \
        && grep -qF "Residual-v2 logit center:" "$log" \
        && grep -qF "Residual train/eval parity:" "$log" \
        && grep -qF "passed=True" "$log" \
        && grep -qF "Frozen Base invariant: passed" "$log"; then
      echo "Resume: reuse expert $action $cue rank=$rank_weight"
      return
    fi

    python main.py \
      "${common[@]}" "${cue_options[@]}" "${residual[@]}" \
      --residual_rank_weight "$rank_weight" \
      --epochs "$EPOCHS" \
      --calibration_final_only \
      --init_ckpt "$base" \
      --save_final_model "$model" \
      2>&1 | tee "$log"
    require_file "$model"
    grep -qF "Residual-v2 logit center:" "$log"
    grep -qF "Residual train/eval parity:" "$log"
    grep -qF "passed=True" "$log"
    grep -qF "Frozen Base invariant: passed" "$log"
  }

  export_base() {
    local split_name="$1"
    local suffix="$2"
    python main.py \
      "${common[@]}" \
      --prediction_probe \
      --prediction_split "$split_name" \
      --ckpt_path "$base" \
      --prediction_output "$out/${action}_base_${suffix}.csv"
  }

  export_expert() {
    local cue="$1"
    local rank_weight="$2"
    local tag="$3"
    local split_name="$4"
    local suffix="$5"
    local gate="$6"
    local state="$7"
    local cue_options=()
    local motion_gate=0
    local skeleton_gate=0
    case "$cue" in
      motion)
        cue_options=("${motion[@]}")
        motion_gate="$gate"
        ;;
      skeleton)
        cue_options=("${skeleton[@]}")
        skeleton_gate="$gate"
        ;;
      *) echo "Unknown cue: $cue" >&2; exit 1 ;;
    esac
    python main.py \
      "${common[@]}" "${cue_options[@]}" "${residual[@]}" \
      --residual_rank_weight "$rank_weight" \
      --prediction_probe \
      --prediction_split "$split_name" \
      --prediction_motion_gate "$motion_gate" \
      --prediction_skeleton_gate "$skeleton_gate" \
      --ckpt_path "$out/${action}_rank${tag}_${cue}_seed0.pth" \
      --prediction_output \
        "$out/${action}_rank${tag}_${cue}_${state}_${suffix}.csv"
  }

  local rank_weight tag cue
  for rank_weight in "${RANK_WEIGHTS[@]}"; do
    tag="$(weight_tag "$rank_weight")"
    for cue in motion skeleton; do
      train_expert "$cue" "$rank_weight" "$tag"
    done
  done

  # Ball creates its deterministic split during its first branch run.
  require_file "$split"
  export_base train train
  export_base calibration cal

  for rank_weight in "${RANK_WEIGHTS[@]}"; do
    tag="$(weight_tag "$rank_weight")"
    for cue in motion skeleton; do
      export_expert "$cue" "$rank_weight" "$tag" train train 1 on
      export_expert "$cue" "$rank_weight" "$tag" train train 0 off
      export_expert "$cue" "$rank_weight" "$tag" calibration cal 1 on
      export_expert "$cue" "$rank_weight" "$tag" calibration cal 0 off
    done

    python tools/analyze_direct_cue_effect.py \
      --dataset "$action" \
      --base_train "$out/${action}_base_train.csv" \
      --motion_on_train "$out/${action}_rank${tag}_motion_on_train.csv" \
      --motion_off_train "$out/${action}_rank${tag}_motion_off_train.csv" \
      --skeleton_on_train "$out/${action}_rank${tag}_skeleton_on_train.csv" \
      --skeleton_off_train "$out/${action}_rank${tag}_skeleton_off_train.csv" \
      --base_cal "$out/${action}_base_cal.csv" \
      --motion_on_cal "$out/${action}_rank${tag}_motion_on_cal.csv" \
      --motion_off_cal "$out/${action}_rank${tag}_motion_off_cal.csv" \
      --skeleton_on_cal "$out/${action}_rank${tag}_skeleton_on_cal.csv" \
      --skeleton_off_cal "$out/${action}_rank${tag}_skeleton_off_cal.csv" \
      --output "$out/${action}_rank${tag}_direct_effect.json" \
      --expected_train_samples "$expected_train" \
      --expected_calibration_samples "$expected_cal" \
      --bootstrap "$BOOTSTRAP" \
      --gate_step 0.01 \
      --seed 0 \
      --require_bias_separated \
      --motion_bound 0.015 \
      --skeleton_bound 0.02 \
      2>&1 | tee "$out/${action}_rank${tag}_direct_effect.log"
  done

  sha256sum \
    main.py pipeline.py models/pamfn.py datasets.py \
    tools/analyze_direct_cue_effect.py \
    tools/analyze_ranking_expert_rescue.py \
    tools/run_ranking_expert_rescue.sh \
    "$base" "$split" \
    "$branch/${action}_rgb_UMT.pth" \
    "$branch/${action}_flow_I3D.pth" \
    "$branch/${action}_audio_MAST.pth" \
    "$out"/*.pth "$out"/*.csv "$out"/*.json "$out"/*.log \
    > "$out/${action}_ranking_expert_rescue_sha256.txt"
}

run_action TES 320 80 FISV "$FISV_SPLIT"
run_action PCS 320 80 FISV "$FISV_SPLIT"
run_action Ball 160 40 Ball "$RG_SPLIT"

python tools/analyze_ranking_expert_rescue.py \
  --root "$OUT_ROOT" \
  --output "$OUT_ROOT/clean_inner_base_ranking_rescue_report.json" \
  --actions TES PCS Ball \
  --weights 0 0.01 0.05 \
  --bootstrap "$BOOTSTRAP" \
  --seed 0 \
  --require_clean_inner_base \
  2>&1 | tee "$OUT_ROOT/clean_inner_base_ranking_rescue_report.log"

sha256sum \
  tools/analyze_ranking_expert_rescue.py \
  tools/run_ranking_expert_rescue.sh \
  "$OUT_ROOT/TES/TES_rank"*_direct_effect.json \
  "$OUT_ROOT/PCS/PCS_rank"*_direct_effect.json \
  "$OUT_ROOT/Ball/Ball_rank"*_direct_effect.json \
  "$OUT_ROOT/clean_inner_base_ranking_rescue_report.json" \
  "$OUT_ROOT"/*.log \
  > "$OUT_ROOT/clean_inner_base_ranking_rescue_sha256.txt"

echo
echo "Clean-inner-Base ranking rescue screen complete: $OUT_ROOT"
echo "Read decision.status in clean_inner_base_ranking_rescue_report.json."
echo "This screen never authorizes the six-action multi-seed experiment directly."
