#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU_LIST="${GPU_LIST:-5}"
ACTIONS_STRING="${ACTIONS:-Ribbon}"
ROOT="${ROOT:-results/july28_ordered_reproduction_seed0}"
HISTORICAL_ROOT="${HISTORICAL_ROOT:-reports/july28_recovery/reference_logs}"
RESUME="${RESUME:-1}"

read -r -a GPUS <<< "$GPU_LIST"
read -r -a TASKS <<< "$ACTIONS_STRING"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "GPU_LIST must contain at least one GPU" >&2
  exit 2
fi
if [[ ${#TASKS[@]} -gt ${#GPUS[@]} ]]; then
  echo "JULY28_REPRODUCTION_GPU_OVERSUBSCRIPTION: actions=${#TASKS[@]} gpus=${#GPUS[@]}" >&2
  echo "Run at most one action per GPU, or split the actions into multiple phases." >&2
  exit 2
fi

mkdir -p "$ROOT/task_launchers" reports/july28_recovery
python -m py_compile main.py pipeline.py datasets.py models/pamfn.py \
  tools/verify_july28_recovery.py \
  tools/compare_july28_trajectory.py
python tools/verify_july28_recovery.py \
  --output reports/july28_recovery/source_manifest.json

run_task() {
  local action="$1"
  local gpu="$2"
  local out="$ROOT/$action"
  local log="$out/${action}_prototype_seed0.log"
  local checkpoint="$out/${action}_prototype_seed0.pth"
  mkdir -p "$out"
  if [[ "$RESUME" == 1 && -f "$checkpoint" && -f "$log" ]] \
      && grep -qF "best avg coef:" "$log"; then
    echo "SKIP complete action=$action"
    return 0
  fi
  echo "START action=$action gpu=$gpu"
  python -u main.py \
    --gpu "$gpu" \
    --feats 2 \
    --action "$action" \
    --seed 0 \
    --sequence_sampling legacy \
    --multi_modality \
    --ordinal_prototype_weight 0.0001 \
    --ordinal_prototype_dim 128 \
    --ordinal_prototype_temperature 0.1 \
    --ordinal_prototype_order_margin 0.1 \
    --ordinal_prototype_order_scale 0.1 \
    --save_final_model "$checkpoint" \
    2>&1 | tee "$log"
  python tools/compare_july28_trajectory.py \
    --historical-log "$HISTORICAL_ROOT/$action/${action}_prototype_seed0.txt" \
    --reproduced-log "$log" \
    --epochs 3 \
    --output-dir "$out/trajectory"
  echo "FINISH action=$action gpu=$gpu"
}

pids=()
for index in "${!TASKS[@]}"; do
  action="${TASKS[$index]}"
  gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
  launcher="$ROOT/task_launchers/${action}_gpu${gpu}.log"
  run_task "$action" "$gpu" > "$launcher" 2>&1 &
  pid="$!"
  pids+=("$pid")
  echo "LAUNCHED action=$action gpu=$gpu pid=$pid"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" != 0 ]]; then
  echo "JULY28_REPRODUCTION_INCOMPLETE" >&2
  exit 1
fi
echo "JULY28_REPRODUCTION_FINISHED"
