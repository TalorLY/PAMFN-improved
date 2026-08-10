#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

SOURCE="${SOURCE:-results/decoupled_autogate/TES}"
OUT="${OUT:-results/action_level_gate/TES}"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

for split in train cal; do
  for variant in off motion skeleton; do
    require_file "$SOURCE/TES_decoupled_${variant}_${split}.csv"
  done
done

mkdir -p "$OUT"
python -m py_compile tools/analyze_action_level_gate.py
python tools/analyze_action_level_gate.py --self_test \
  2>&1 | tee "$OUT/TES_action_gate_self_test.log"

python tools/analyze_action_level_gate.py \
  --off_train "$SOURCE/TES_decoupled_off_train.csv" \
  --motion_train "$SOURCE/TES_decoupled_motion_train.csv" \
  --skeleton_train "$SOURCE/TES_decoupled_skeleton_train.csv" \
  --off_cal "$SOURCE/TES_decoupled_off_cal.csv" \
  --motion_cal "$SOURCE/TES_decoupled_motion_cal.csv" \
  --skeleton_cal "$SOURCE/TES_decoupled_skeleton_cal.csv" \
  --train_output "$OUT/TES_action_gate_train.csv" \
  --cal_output "$OUT/TES_action_gate_cal.csv" \
  --output "$OUT/TES_action_gate_report.json" \
  --expected_train_samples 320 \
  --expected_cal_samples 80 \
  --gate_step 0.01 \
  --mse_guardrail 0.02 \
  --bootstrap 2000 \
  --seed 0 \
  2>&1 | tee "$OUT/TES_action_gate_report.log"

sha256sum \
  tools/analyze_action_level_gate.py \
  tools/run_action_level_gate_tes.sh \
  "$SOURCE"/TES_decoupled_{off,motion,skeleton}_{train,cal}.csv \
  "$OUT"/*.csv "$OUT"/*.json "$OUT"/*.log \
  > "$OUT/TES_action_gate_sha256.txt"

echo
echo "TES action-level Gate probe completed: $OUT"
echo "This is a train-fitted development probe; a GO result still requires OOF."
