#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU_LIST_STRING="${GPU_LIST:-}"
SCREEN_ROOT="${SCREEN_ROOT:-results/coqm_all_action_screen_v1}"
ROOT="${ROOT:-results/coqm_all_action_formal_v1}"
COQM_WEIGHT="${COQM_WEIGHT:-0.006}"
RESUME="${RESUME:-1}"
ACTIONS=(TES PCS Ball Clubs Hoop Ribbon)
SEEDS=(0 1 2 42 3407)
METHODS=(base coqm_geom)

read -r -a GPUS <<< "$GPU_LIST_STRING"
if [[ ${#GPUS[@]} -lt 1 ]]; then
  echo "Set GPU_LIST to one or more currently idle GPUs." >&2
  exit 2
fi
if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
  echo "RESUME must be 0 or 1." >&2
  exit 2
fi
if [[ ! -s "$SCREEN_ROOT/report/decision.json" ]]; then
  echo "Missing screen decision: $SCREEN_ROOT/report/decision.json" >&2
  exit 2
fi
CANDIDATE="$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["selected_anchor_mode"])' \
  "$SCREEN_ROOT/report/decision.json")"
if [[ "$CANDIDATE" != "uniform" && "$CANDIDATE" != "train_centers" ]]; then
  echo "Invalid selected anchor mode: $CANDIDATE" >&2
  exit 2
fi

mkdir -p "$ROOT/worker_launchers" "$ROOT/report"
git rev-parse HEAD > "$ROOT/git_commit.txt"
git branch --show-current > "$ROOT/git_branch.txt"
printf '%s\n' "$COQM_WEIGHT" > "$ROOT/coqm_weight.txt"
printf '%s\n' "$CANDIDATE" > "$ROOT/selected_anchor_mode.txt"

run_method() {
  local action="$1"
  local gpu="$2"
  local seed="$3"
  local mode="$4"
  local out="$ROOT/$action/seed$seed/$mode"
  local checkpoint="$out/model.pth"
  local extra=()
  mkdir -p "$out"
  if [[ "$mode" == "coqm_geom" ]]; then
    extra=(
      --coqm_weight "$COQM_WEIGHT"
      --coqm_coord_scale 0
      --coqm_anchor_mode "$CANDIDATE"
      --quality_aux_rng_isolated
    )
  fi

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
    --eval_policy argmax --run_tag "$mode" "${extra[@]}" \
    --save_final_model "$checkpoint" \
    --result_json "$out/train_result.json" \
    > "$out/train.log" 2>&1
  grep -qF "best avg coef:" "$out/train.log"
  python -u main.py \
    --gpu "$gpu" --feats 2 --action "$action" --seed "$seed" \
    --sequence_sampling legacy --multi_modality \
    --eval_policy argmax --run_tag "$mode" "${extra[@]}" \
    --test --ckpt_path "$checkpoint" \
    --result_json "$out/test_result.json" \
    > "$out/test.log" 2>&1
  grep -qF "Coef:" "$out/test.log"
  echo "FINISH action=$action seed=$seed mode=$mode gpu=$gpu"
}

run_worker() {
  local worker_index="$1"
  local gpu="$2"
  local cell_index=0 action seed mode
  for action in "${ACTIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      if (( cell_index % ${#GPUS[@]} == worker_index )); then
        for mode in "${METHODS[@]}"; do
          run_method "$action" "$gpu" "$seed" "$mode"
        done
      fi
      ((cell_index+=1))
    done
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
  echo "COQM_ALL_ACTION_FORMAL_INCOMPLETE; inspect $ROOT/worker_launchers" >&2
  exit 1
fi

python tools/summarize_coqm_all_action.py \
  --stage formal --candidate "$CANDIDATE" --root "$ROOT" \
  --output_csv "$ROOT/report/runs.csv" \
  --output_json "$ROOT/report/decision.json" \
  --output_md "$ROOT/report/summary.md"
python tools/package_coqm_results.py \
  --root "$ROOT" --output "${ROOT}_return.zip"
echo "COQM_ALL_ACTION_FORMAL_COMPLETE root=$ROOT archive=${ROOT}_return.zip"
