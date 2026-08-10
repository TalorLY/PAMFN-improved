#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-5}"
SOURCE_ROOT="${SOURCE_ROOT:-results/clean_inner_base_ranking_rescue}"
OUT_ROOT="${OUT_ROOT:-results/temporal_feature_expert_screen}"
FISV_SPLIT="${FISV_SPLIT:-results/calibration_probe/FISV_seed0_split.json}"
RG_SPLIT="${RG_SPLIT:-$SOURCE_ROOT/RG_seed0_split.json}"
EPOCHS="${EPOCHS:-100}"
BOOTSTRAP="${BOOTSTRAP:-2000}"
RESUME="${RESUME:-0}"

if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 1
fi

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

mkdir -p "$OUT_ROOT"
require_file "$FISV_SPLIT"
require_file "$RG_SPLIT"

python -m py_compile \
  main.py pipeline.py models/pamfn.py datasets.py \
  tools/verify_temporal_feature_adapter.py \
  tools/analyze_direct_cue_effect.py \
  tools/analyze_temporal_feature_experts.py
python tools/verify_temporal_feature_adapter.py \
  2>&1 | tee "$OUT_ROOT/temporal_feature_verification.log"
python tools/analyze_temporal_feature_experts.py --self_test \
  2>&1 | tee "$OUT_ROOT/temporal_feature_analyzer_self_test.log"

run_action() {
  local action="$1"
  local feature_family="$2"
  local split="$3"
  local out="$OUT_ROOT/$action"
  local source="$SOURCE_ROOT/$action"
  local branch="$source/branches"
  local base="$source/${action}_inner_base_seed0.pth"
  local file

  mkdir -p "$out"
  require_file "$base"
  require_file "$split"
  for file in \
    "$branch/${action}_rgb_UMT.pth" \
    "$branch/${action}_flow_I3D.pth" \
    "$branch/${action}_audio_MAST.pth" \
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
  local temporal=(
    --anchored_residual
    --temporal_feature_adapter
    --temporal_feature_scale 0.1
    --temporal_feature_reg_weight 0.01
    --residual_adapter_dim 16
    --residual_rank_weight 0.01
  )
  local motion=(
    --motion_saliency_proxy
    --motion_saliency_source delta
    --motion_saliency_positive_only
  )
  local skeleton=(
    --use_skeleton
    --skeleton_root .
  )

  train_expert() {
    local cue="$1"
    local model="$out/${action}_${cue}_seed0.pth"
    local log="$out/${action}_${cue}_seed0.log"
    local cue_options=()
    case "$cue" in
      motion) cue_options=("${motion[@]}") ;;
      skeleton) cue_options=("${skeleton[@]}") ;;
      *) echo "Unknown cue: $cue" >&2; exit 1 ;;
    esac
    if [[ "$RESUME" == 1 && -f "$model" && -f "$log" ]] \
        && grep -qF "Temporal feature screen:" "$log" \
        && grep -qF "Frozen Base invariant: passed" "$log" \
        && grep -qF "Blind training complete;" "$log"; then
      echo "Resume: reuse temporal expert $action $cue"
      return
    fi
    python main.py \
      "${common[@]}" "${cue_options[@]}" "${temporal[@]}" \
      --epochs "$EPOCHS" \
      --calibration_final_only \
      --blind_train \
      --init_ckpt "$base" \
      --save_final_model "$model" \
      2>&1 | tee "$log"
    require_file "$model"
    grep -qF "Temporal feature screen:" "$log"
    grep -qF "Frozen Base invariant: passed" "$log"
    grep -qF "Blind training complete;" "$log"
    grep -qF "Saved the final blind-training checkpoint;" "$log"
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
    local split_name="$2"
    local suffix="$3"
    local gate="$4"
    local state="$5"
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
      "${common[@]}" "${cue_options[@]}" "${temporal[@]}" \
      --prediction_probe \
      --prediction_split "$split_name" \
      --prediction_motion_gate "$motion_gate" \
      --prediction_skeleton_gate "$skeleton_gate" \
      --ckpt_path "$out/${action}_${cue}_seed0.pth" \
      --prediction_output \
        "$out/${action}_${cue}_${state}_${suffix}.csv"
  }

  local cue
  for cue in motion skeleton; do
    train_expert "$cue"
  done
  export_base train train
  export_base calibration cal
  for cue in motion skeleton; do
    export_expert "$cue" train train 0 off
    export_expert "$cue" train train 1 on
    export_expert "$cue" calibration cal 0 off
    export_expert "$cue" calibration cal 1 on
  done

  sha256sum \
    main.py pipeline.py models/pamfn.py datasets.py \
    tools/verify_temporal_feature_adapter.py \
    tools/analyze_temporal_feature_experts.py \
    tools/run_temporal_feature_expert_screen.sh \
    "$base" "$split" \
    "$branch/${action}_rgb_UMT.pth" \
    "$branch/${action}_flow_I3D.pth" \
    "$branch/${action}_audio_MAST.pth" \
    "$out"/*.pth "$out"/*.csv "$out"/*.log \
    > "$out/${action}_temporal_feature_sha256.txt"
}

run_action TES FISV "$FISV_SPLIT"
run_action PCS FISV "$FISV_SPLIT"
run_action Ball Ball "$RG_SPLIT"

python tools/analyze_temporal_feature_experts.py \
  --root "$OUT_ROOT" \
  --output "$OUT_ROOT/temporal_feature_expert_report.json" \
  --actions TES PCS Ball \
  --bootstrap "$BOOTSTRAP" \
  --seed 0 \
  2>&1 | tee "$OUT_ROOT/temporal_feature_expert_report.log"

sha256sum \
  tools/analyze_temporal_feature_experts.py \
  tools/run_temporal_feature_expert_screen.sh \
  "$OUT_ROOT"/TES/*.csv "$OUT_ROOT"/PCS/*.csv "$OUT_ROOT"/Ball/*.csv \
  "$OUT_ROOT/temporal_feature_expert_report.json" \
  "$OUT_ROOT"/*.log \
  > "$OUT_ROOT/temporal_feature_expert_screen_sha256.txt"

echo
echo "Temporal feature expert screen complete: $OUT_ROOT"
echo "Read decision.status in temporal_feature_expert_report.json."
echo "This screen does not authorize the full six-action experiment directly."
