#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-0}"
ACTION="${ACTION:-Ribbon}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-1}"
COQM_WEIGHT="${COQM_WEIGHT:-0.0001}"
ROOT="${ROOT:-results/coqm_stage_a_smoke}"

test -d data/features
test -d pretrained_models/feats2
python -m py_compile \
  main.py pipeline.py datasets.py models/pamfn.py models/coqm.py \
  tools/calibrate_coqm_gradient.py tools/verify_coqm_smoke.py
python -m unittest discover -s tests -v
python tools/calibrate_coqm_gradient.py \
  --gpu "$GPU" --action TES --seed 0 --weight "$COQM_WEIGHT" \
  --output "$ROOT/gradient_calibration.json"

run_mode() {
  local mode="$1"
  local out="$ROOT/$mode"
  local checkpoint="$out/model.pth"
  local extra=()
  mkdir -p "$out"
  if [[ "$mode" == "coqm" ]]; then
    extra=(
      --coqm_weight "$COQM_WEIGHT"
      --coqm_dim 128
      --coqm_temperature 0.1
      --coqm_theta_max 1.0471975511965976
      --coqm_coord_scale 1.0
      --quality_aux_rng_isolated
    )
  fi
  python -u main.py \
    --gpu "$GPU" --feats 2 --action "$ACTION" --seed "$SEED" \
    --epochs "$EPOCHS" --sequence_sampling legacy --multi_modality \
    --run_tag "$mode" "${extra[@]}" \
    --save_final_model "$checkpoint" \
    --result_json "$out/train_result.json" \
    > "$out/train.log" 2>&1
  python -u main.py \
    --gpu "$GPU" --feats 2 --action "$ACTION" --seed "$SEED" \
    --sequence_sampling legacy --multi_modality \
    --run_tag "$mode" "${extra[@]}" \
    --test --ckpt_path "$checkpoint" \
    --result_json "$out/test_result.json" \
    > "$out/test.log" 2>&1
}

run_mode base
run_mode coqm
python tools/verify_coqm_smoke.py \
  --root "$ROOT" --output "$ROOT/smoke_gate.json"
git rev-parse HEAD > "$ROOT/git_commit.txt"
echo "COQM_STAGE_A_SMOKE_FINISHED root=$ROOT"
