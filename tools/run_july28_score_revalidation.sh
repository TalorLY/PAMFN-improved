#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU_LIST_STRING="${GPU_LIST:-0 1 2 3 4 5}"
ACTIONS_STRING="${ACTIONS:-TES PCS Ball Clubs Hoop Ribbon}"
SEEDS_STRING="${SEEDS:-0 1 2}"
ROOT="${ROOT:-results/july28_score_revalidation}"
REPORT_ROOT="${REPORT_ROOT:-reports/july28_score_revalidation}"
HISTORICAL_ROOT="${HISTORICAL_ROOT:-reports/july28_recovery/reference_logs}"
RESUME="${RESUME:-1}"

PROTOTYPE_WEIGHT="0.0001"
PROTOTYPE_DIM="128"
PROTOTYPE_TEMPERATURE="0.1"
PROTOTYPE_ORDER_MARGIN="0.1"
PROTOTYPE_ORDER_SCALE="0.1"

read -r -a GPUS <<< "$GPU_LIST_STRING"
read -r -a TASKS <<< "$ACTIONS_STRING"
read -r -a SEED_VALUES <<< "$SEEDS_STRING"

if [[ ${#TASKS[@]} -ne 6 ]]; then
  echo "JULY28_REVALIDATION_REQUIRES_SIX_ACTIONS" >&2
  exit 2
fi
if [[ ${#GPUS[@]} -lt ${#TASKS[@]} ]]; then
  echo "JULY28_REVALIDATION_REQUIRES_SIX_GPUS" >&2
  exit 2
fi
if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$ROOT/task_launchers" "$REPORT_ROOT"
python -m py_compile \
  main.py pipeline.py datasets.py models/pamfn.py \
  tools/verify_july28_recovery.py \
  tools/compare_july28_trajectory.py \
  tools/analyze_ordinal_prototypes.py \
  tools/summarize_original_protocol.py \
  tools/summarize_ordinal_multiseed.py
python tools/verify_july28_recovery.py \
  --output "$REPORT_ROOT/source_manifest.json"

git rev-parse HEAD > "$REPORT_ROOT/git_commit.txt"
git branch --show-current > "$REPORT_ROOT/git_branch.txt"

run_method() {
  local action="$1"
  local gpu="$2"
  local seed="$3"
  local mode="$4"
  local out="$ROOT/seed${seed}/$action"
  local log="$out/${action}_${mode}_seed${seed}.log"
  local checkpoint="$out/${action}_${mode}_seed${seed}.pth"
  local completion_marker="Training objective: official PAMFN MSE"
  local prototype_args=()

  mkdir -p "$out"
  if [[ "$mode" == "prototype" ]]; then
    completion_marker="0.0001*ordinal prototypes"
    prototype_args=(
      --ordinal_prototype_weight "$PROTOTYPE_WEIGHT"
      --ordinal_prototype_dim "$PROTOTYPE_DIM"
      --ordinal_prototype_temperature "$PROTOTYPE_TEMPERATURE"
      --ordinal_prototype_order_margin "$PROTOTYPE_ORDER_MARGIN"
      --ordinal_prototype_order_scale "$PROTOTYPE_ORDER_SCALE"
    )
  fi

  if [[ "$RESUME" == 1 && -f "$checkpoint" && -f "$log" ]] \
      && grep -qF "best avg coef:" "$log" \
      && grep -qF "$completion_marker" "$log"; then
    echo "SKIP complete action=$action seed=$seed mode=$mode"
    return 0
  fi

  echo "START action=$action seed=$seed mode=$mode gpu=$gpu"
  python -u main.py \
    --gpu "$gpu" \
    --feats 2 \
    --action "$action" \
    --seed "$seed" \
    --sequence_sampling legacy \
    --multi_modality \
    "${prototype_args[@]}" \
    --save_final_model "$checkpoint" \
    > "$log" 2>&1
  grep -qF "best avg coef:" "$log"
  grep -qF "$completion_marker" "$log"
  echo "FINISH action=$action seed=$seed mode=$mode gpu=$gpu"
}

run_pair() {
  local action="$1"
  local gpu="$2"
  local seed="$3"
  run_method "$action" "$gpu" "$seed" base
  run_method "$action" "$gpu" "$seed" prototype

  if [[ "$seed" == 0 ]]; then
    python tools/compare_july28_trajectory.py \
      --historical-log \
        "$HISTORICAL_ROOT/$action/${action}_prototype_seed0.txt" \
      --reproduced-log \
        "$ROOT/seed0/$action/${action}_prototype_seed0.log" \
      --epochs 3 \
      --output-dir "$ROOT/seed0/$action/trajectory"
  fi
}

seed_results=()
for seed in "${SEED_VALUES[@]}"; do
  pids=()
  for index in "${!TASKS[@]}"; do
    action="${TASKS[$index]}"
    gpu="${GPUS[$index]}"
    launcher="$ROOT/task_launchers/${action}_seed${seed}_gpu${gpu}.log"
    run_pair "$action" "$gpu" "$seed" > "$launcher" 2>&1 &
    pids+=("$!")
    echo "LAUNCHED action=$action seed=$seed gpu=$gpu pid=$!"
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != 0 ]]; then
    echo "JULY28_SCORE_REVALIDATION_INCOMPLETE seed=$seed" >&2
    exit 1
  fi

  seed_root="$ROOT/seed${seed}"
  python tools/summarize_original_protocol.py \
    --root "$seed_root" \
    --candidate-mode prototype \
    --actions "${TASKS[@]}" \
    --seed "$seed" \
    --output "$seed_root/original_protocol_summary.csv"
  python tools/analyze_ordinal_prototypes.py \
    --root "$seed_root" \
    --actions "${TASKS[@]}" \
    --seed "$seed" \
    --output "$seed_root/ordinal_prototype_diagnostics.csv"
  seed_results+=("$seed=$seed_root")
  echo "SEED_FINISHED seed=$seed"
done

python tools/summarize_ordinal_multiseed.py \
  --seed-results "${seed_results[@]}" \
  --actions "${TASKS[@]}" \
  --output-runs "$REPORT_ROOT/three_seed_runs.csv" \
  --output-summary "$REPORT_ROOT/three_seed_summary.csv" \
  --output-diagnostics "$REPORT_ROOT/three_seed_diagnostics.csv"

echo "JULY28_SCORE_REVALIDATION_FINISHED"
echo "Results: $ROOT"
echo "Summary: $REPORT_ROOT/three_seed_summary.csv"
