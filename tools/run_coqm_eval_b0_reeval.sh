#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU_LIST_STRING="${GPU_LIST:-}"
ACTIONS_STRING="${ACTIONS:-TES Clubs Ribbon}"
SEEDS_STRING="${SEEDS:-0 1 2}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-results/coqm_stage_a_v1_calibrated}"
ROOT="${ROOT:-results/coqm_eval_b0_v1}"
SMOKE_ROOT="${SMOKE_ROOT:-results/coqm_eval_b0_smoke}"
ORDERED_WEIGHT="${ORDERED_WEIGHT:-0.0001}"
COQM_WEIGHT="${COQM_WEIGHT:-0.006}"
EVAL_SEED="${EVAL_SEED:-20260908}"
MC_SAMPLES="${MC_SAMPLES:-20}"
RESUME="${RESUME:-1}"

read -r -a GPUS <<< "$GPU_LIST_STRING"
read -r -a TASKS <<< "$ACTIONS_STRING"
read -r -a SEED_VALUES <<< "$SEEDS_STRING"
METHODS=(
  base ordered_discrete ordered_continuous coqm_no_coord coqm
)
POLICIES=(argmax mc_gumbel)

if [[ ${#GPUS[@]} -ne 3 ]]; then
  echo "Set GPU_LIST to exactly three currently idle GPUs." >&2
  exit 2
fi
if [[ "${TASKS[*]}" != "TES Clubs Ribbon" ]]; then
  echo "B0 is pre-registered for ACTIONS='TES Clubs Ribbon'." >&2
  exit 2
fi
if [[ "${SEED_VALUES[*]}" != "0 1 2" ]]; then
  echo "B0 re-evaluates the existing SEEDS='0 1 2' checkpoints." >&2
  exit 2
fi
if [[ "$MC_SAMPLES" -lt 2 ]]; then
  echo "MC_SAMPLES must be at least two." >&2
  exit 2
fi
if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
  echo "RESUME must be 0 or 1." >&2
  exit 2
fi

python tools/verify_coqm_eval_b0_smoke.py \
  --root "$SMOKE_ROOT" --eval_seed "$EVAL_SEED" \
  --mc_samples "$MC_SAMPLES"
mkdir -p "$ROOT/task_launchers" "$ROOT/report"
git rev-parse HEAD > "$ROOT/git_commit.txt"
git branch --show-current > "$ROOT/git_branch.txt"
printf '%s\n' "$CHECKPOINT_ROOT" > "$ROOT/source_checkpoint_root.txt"

method_args() {
  local method="$1"
  case "$method" in
    base)
      ;;
    ordered_discrete)
      printf '%s\n' \
        --ordinal_prototype_weight "$ORDERED_WEIGHT" \
        --ordinal_prototype_target_mode discrete \
        --quality_aux_rng_isolated
      ;;
    ordered_continuous)
      printf '%s\n' \
        --ordinal_prototype_weight "$ORDERED_WEIGHT" \
        --ordinal_prototype_target_mode continuous \
        --quality_aux_rng_isolated
      ;;
    coqm_no_coord)
      printf '%s\n' \
        --coqm_weight "$COQM_WEIGHT" \
        --coqm_coord_scale 0.0 \
        --quality_aux_rng_isolated
      ;;
    coqm)
      printf '%s\n' \
        --coqm_weight "$COQM_WEIGHT" \
        --coqm_coord_scale 1.0 \
        --quality_aux_rng_isolated
      ;;
    *)
      echo "Unknown method: $method" >&2
      return 2
      ;;
  esac
}

run_eval() {
  local action="$1"
  local gpu="$2"
  local seed="$3"
  local method="$4"
  local policy="$5"
  local checkpoint="$CHECKPOINT_ROOT/$action/seed$seed/$method/model.pth"
  local old_result="$CHECKPOINT_ROOT/$action/seed$seed/$method/test_result.json"
  local out="$ROOT/$action/seed$seed/$method/$policy"
  local extra=()
  test -s "$checkpoint"
  test -s "$old_result"
  if [[ "$RESUME" == 1 && -s "$out/test_result.json" ]]; then
    echo "SKIP action=$action seed=$seed method=$method policy=$policy"
    return 0
  fi
  mapfile -t extra < <(method_args "$method")
  mkdir -p "$out"
  echo "START action=$action seed=$seed method=$method policy=$policy gpu=$gpu"
  python -u main.py \
    --gpu "$gpu" --feats 2 --action "$action" --seed "$seed" \
    --sequence_sampling legacy --multi_modality \
    --run_tag "$method" "${extra[@]}" \
    --eval_policy "$policy" --eval_seed "$EVAL_SEED" \
    --eval_mc_samples "$MC_SAMPLES" \
    --test --ckpt_path "$checkpoint" \
    --result_json "$out/test_result.json" \
    > "$out/test.log" 2>&1
  grep -qF "Coef:" "$out/test.log"
  echo "FINISH action=$action seed=$seed method=$method policy=$policy gpu=$gpu"
}

run_task() {
  local action="$1"
  local gpu="$2"
  local seed method policy
  for seed in "${SEED_VALUES[@]}"; do
    for method in "${METHODS[@]}"; do
      for policy in "${POLICIES[@]}"; do
        run_eval "$action" "$gpu" "$seed" "$method" "$policy"
      done
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
  echo "COQM_EVAL_B0_INCOMPLETE; inspect $ROOT/task_launchers" >&2
  exit 1
fi

manifest="$ROOT/report/source_checkpoints.sha256"
: > "$manifest"
for action in "${TASKS[@]}"; do
  for seed in "${SEED_VALUES[@]}"; do
    for method in "${METHODS[@]}"; do
      sha256sum \
        "$CHECKPOINT_ROOT/$action/seed$seed/$method/model.pth" \
        >> "$manifest"
    done
  done
done

python tools/summarize_coqm_eval_b0.py \
  --checkpoint_root "$CHECKPOINT_ROOT" --root "$ROOT" \
  --eval_seed "$EVAL_SEED" --mc_samples "$MC_SAMPLES" \
  --output_csv "$ROOT/report/runs.csv" \
  --output_json "$ROOT/report/gate.json" \
  --output_md "$ROOT/report/summary.md"
python tools/package_coqm_results.py \
  --root "$ROOT" --output "${ROOT}_return.zip"
echo "COQM_EVAL_B0_FINISHED root=$ROOT"
