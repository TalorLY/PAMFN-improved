#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU_LIST_STRING="${GPU_LIST:-}"
ROOT="${ROOT:-results/coqm_all_action_screen_v1}"
COQM_WEIGHT="${COQM_WEIGHT:-0.006}"
RESUME="${RESUME:-1}"

read -r -a GPUS <<< "$GPU_LIST_STRING"
if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "Set GPU_LIST to exactly four currently idle GPUs." >&2
  exit 2
fi
if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
  echo "RESUME must be 0 or 1." >&2
  exit 2
fi

python -m unittest \
  tests.test_coqm_geometry \
  tests.test_coqm_targets \
  tests.test_coqm_rng_isolation \
  tests.test_eval_policy

mkdir -p "$ROOT/worker_launchers" "$ROOT/report"
git rev-parse HEAD > "$ROOT/git_commit.txt"
git branch --show-current > "$ROOT/git_branch.txt"
printf '%s\n' "$COQM_WEIGHT" > "$ROOT/coqm_weight.txt"

run_method() {
  local action="$1"
  local gpu="$2"
  local mode="$3"
  local out="$ROOT/$action/seed0/$mode"
  local checkpoint="$out/model.pth"
  local anchor_mode
  if [[ "$mode" == "coqm_uniform" ]]; then
    anchor_mode="uniform"
  else
    anchor_mode="train_centers"
  fi
  mkdir -p "$out"

  if [[ "$RESUME" == 1 \
        && -s "$out/train_result.json" \
        && -s "$out/test_result.json" \
        && -s "$checkpoint" ]]; then
    echo "SKIP action=$action seed=0 mode=$mode"
    return 0
  fi

  echo "START action=$action seed=0 mode=$mode gpu=$gpu"
  python -u main.py \
    --gpu "$gpu" --feats 2 --action "$action" --seed 0 \
    --sequence_sampling legacy --multi_modality \
    --eval_policy argmax --run_tag "$mode" \
    --coqm_weight "$COQM_WEIGHT" --coqm_coord_scale 0 \
    --coqm_anchor_mode "$anchor_mode" --quality_aux_rng_isolated \
    --save_final_model "$checkpoint" \
    --result_json "$out/train_result.json" \
    > "$out/train.log" 2>&1
  grep -qF "best avg coef:" "$out/train.log"
  python -u main.py \
    --gpu "$gpu" --feats 2 --action "$action" --seed 0 \
    --sequence_sampling legacy --multi_modality \
    --eval_policy argmax --run_tag "$mode" \
    --coqm_weight "$COQM_WEIGHT" --coqm_coord_scale 0 \
    --coqm_anchor_mode "$anchor_mode" --quality_aux_rng_isolated \
    --test --ckpt_path "$checkpoint" \
    --result_json "$out/test_result.json" \
    > "$out/test.log" 2>&1
  grep -qF "Coef:" "$out/test.log"
  echo "FINISH action=$action seed=0 mode=$mode gpu=$gpu"
}

run_worker() {
  local worker_index="$1"
  local gpu="$2"
  local cells=() cell action mode
  case "$worker_index" in
    0) cells=(
      "Ball coqm_uniform"
      "Clubs coqm_uniform"
      "Ribbon coqm_uniform"
    ) ;;
    1) cells=(
      "Ball coqm_train_center"
      "Clubs coqm_train_center"
      "Ribbon coqm_train_center"
    ) ;;
    2) cells=(
      "Hoop coqm_uniform"
      "TES coqm_uniform"
      "PCS coqm_uniform"
    ) ;;
    3) cells=(
      "Hoop coqm_train_center"
      "TES coqm_train_center"
      "PCS coqm_train_center"
    ) ;;
  esac
  for cell in "${cells[@]}"; do
    read -r action mode <<< "$cell"
    run_method "$action" "$gpu" "$mode"
  done
}

pids=()
for worker_index in "${!GPUS[@]}"; do
  gpu="${GPUS[$worker_index]}"
  launcher="$ROOT/worker_launchers/worker${worker_index}_gpu${gpu}.log"
  run_worker "$worker_index" "$gpu" > "$launcher" 2>&1 &
  pids+=("$!")
  echo "LAUNCHED worker=$worker_index gpu=$gpu pid=$!"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" != 0 ]]; then
  echo "COQM_ALL_ACTION_SCREEN_INCOMPLETE; inspect $ROOT/worker_launchers" >&2
  exit 1
fi

python tools/summarize_coqm_all_action.py \
  --stage screen --root "$ROOT" \
  --output_csv "$ROOT/report/runs.csv" \
  --output_json "$ROOT/report/decision.json" \
  --output_md "$ROOT/report/summary.md"
python tools/package_coqm_results.py \
  --root "$ROOT" --output "${ROOT}_return.zip"
echo "COQM_ALL_ACTION_SCREEN_COMPLETE root=$ROOT archive=${ROOT}_return.zip"
