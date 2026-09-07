#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU_LIST_STRING="${GPU_LIST:-}"
ACTIONS_STRING="${ACTIONS:-TES Clubs Ribbon}"
SEEDS_STRING="${SEEDS:-0 1 2}"
ORDERED_WEIGHT="${ORDERED_WEIGHT:-0.0001}"
ROOT="${ROOT:-results/coqm_stage_a_v1}"
SMOKE_ROOT="${SMOKE_ROOT:-results/coqm_stage_a_smoke}"
RESUME="${RESUME:-1}"

if [[ -z "${COQM_WEIGHT:-}" ]]; then
  COQM_WEIGHT="$(python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["weight"])' \
    "$SMOKE_ROOT/gradient_calibration.json")"
fi

read -r -a GPUS <<< "$GPU_LIST_STRING"
read -r -a TASKS <<< "$ACTIONS_STRING"
read -r -a SEED_VALUES <<< "$SEEDS_STRING"
METHODS=(
  base ordered_discrete ordered_continuous coqm_no_coord coqm
)

if [[ ${#GPUS[@]} -ne 3 ]]; then
  echo "Set GPU_LIST to exactly three currently idle GPUs." >&2
  exit 2
fi
if [[ "${TASKS[*]}" != "TES Clubs Ribbon" ]]; then
  echo "Stage A is pre-registered for ACTIONS='TES Clubs Ribbon'." >&2
  exit 2
fi
if [[ "${SEED_VALUES[*]}" != "0 1 2" ]]; then
  echo "Stage A is pre-registered for SEEDS='0 1 2'." >&2
  exit 2
fi
if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
  echo "RESUME must be 0 or 1." >&2
  exit 2
fi

python tools/verify_coqm_smoke.py --root "$SMOKE_ROOT"
mkdir -p "$ROOT/task_launchers" "$ROOT/report"
git rev-parse HEAD > "$ROOT/git_commit.txt"
git branch --show-current > "$ROOT/git_branch.txt"

run_method() {
  local action="$1"
  local gpu="$2"
  local seed="$3"
  local mode="$4"
  local out="$ROOT/$action/seed$seed/$mode"
  local checkpoint="$out/model.pth"
  local extra=()
  mkdir -p "$out"

  case "$mode" in
    base)
      ;;
    ordered_discrete)
      extra=(
        --ordinal_prototype_weight "$ORDERED_WEIGHT"
        --ordinal_prototype_target_mode discrete
        --quality_aux_rng_isolated
      )
      ;;
    ordered_continuous)
      extra=(
        --ordinal_prototype_weight "$ORDERED_WEIGHT"
        --ordinal_prototype_target_mode continuous
        --quality_aux_rng_isolated
      )
      ;;
    coqm_no_coord)
      extra=(
        --coqm_weight "$COQM_WEIGHT"
        --coqm_coord_scale 0.0
        --quality_aux_rng_isolated
      )
      ;;
    coqm)
      extra=(
        --coqm_weight "$COQM_WEIGHT"
        --coqm_coord_scale 1.0
        --quality_aux_rng_isolated
      )
      ;;
    *)
      echo "Unknown mode: $mode" >&2
      return 2
      ;;
  esac

  if [[ "$RESUME" == 1 \
        && -s "$out/train_result.json" \
        && -s "$out/test_result.json" \
        && -s "$checkpoint" ]]; then
    echo "SKIP action=$action seed=$seed mode=$mode"
    return 0
  fi

  echo "START action=$action seed=$seed mode=$mode gpu=$gpu"
  python -u main.py \
    --gpu "$gpu" --feats 2 --action "$action" --seed "$seed" \
    --sequence_sampling legacy --multi_modality \
    --run_tag "$mode" "${extra[@]}" \
    --save_final_model "$checkpoint" \
    --result_json "$out/train_result.json" \
    > "$out/train.log" 2>&1
  grep -qF "best avg coef:" "$out/train.log"
  python -u main.py \
    --gpu "$gpu" --feats 2 --action "$action" --seed "$seed" \
    --sequence_sampling legacy --multi_modality \
    --run_tag "$mode" "${extra[@]}" \
    --test --ckpt_path "$checkpoint" \
    --result_json "$out/test_result.json" \
    > "$out/test.log" 2>&1
  grep -qF "Coef:" "$out/test.log"
  echo "FINISH action=$action seed=$seed mode=$mode gpu=$gpu"
}

run_task() {
  local action="$1"
  local gpu="$2"
  local seed mode
  for seed in "${SEED_VALUES[@]}"; do
    for mode in "${METHODS[@]}"; do
      run_method "$action" "$gpu" "$seed" "$mode"
    done
  done
}

pids=()
for index in "${!TASKS[@]}"; do
  action="${TASKS[$index]}"
  gpu="${GPUS[$index]}"
  launcher="$ROOT/task_launchers/${action}_gpu${gpu}.log"
  run_task "$action" "$gpu" > "$launcher" 2>&1 &
  pids+=("$!")
  echo "LAUNCHED action=$action gpu=$gpu pid=$!"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" != 0 ]]; then
  echo "COQM_STAGE_A_INCOMPLETE; inspect $ROOT/task_launchers" >&2
  exit 1
fi

python tools/summarize_coqm_stage_a.py \
  --root "$ROOT" \
  --output_csv "$ROOT/report/runs.csv" \
  --output_json "$ROOT/report/gate.json" \
  --output_md "$ROOT/report/summary.md"
python tools/package_coqm_results.py \
  --root "$ROOT" --output "${ROOT}_return.zip"
echo "COQM_STAGE_A_FINISHED root=$ROOT"
