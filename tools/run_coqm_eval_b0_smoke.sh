#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU="${GPU:-0}"
ACTION="${ACTION:-Ribbon}"
SEED="${SEED:-0}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-results/coqm_stage_a_v1_calibrated}"
ROOT="${ROOT:-results/coqm_eval_b0_smoke}"
COQM_WEIGHT="${COQM_WEIGHT:-0.006}"
EVAL_SEED="${EVAL_SEED:-20260908}"
MC_SAMPLES="${MC_SAMPLES:-20}"

test -d data/features
test -d pretrained_models/feats2
mkdir -p "$ROOT"
python -m py_compile \
  main.py pipeline.py models/pamfn.py models/coqm.py \
  tools/verify_coqm_eval_b0_smoke.py \
  tools/summarize_coqm_eval_b0.py
python -m unittest discover -s tests -p 'test_*.py' -v \
  > "$ROOT/unit_tests.log" 2>&1

method_args() {
  local method="$1"
  case "$method" in
    base)
      ;;
    coqm_no_coord)
      printf '%s\n' \
        --coqm_weight "$COQM_WEIGHT" \
        --coqm_coord_scale 0.0 \
        --quality_aux_rng_isolated
      ;;
    *)
      echo "Unknown smoke method: $method" >&2
      return 2
      ;;
  esac
}

run_eval() {
  local method="$1"
  local output_name="$2"
  local policy="$3"
  local checkpoint="$CHECKPOINT_ROOT/$ACTION/seed$SEED/$method/model.pth"
  local out="$ROOT/$ACTION/seed$SEED/$method/$output_name"
  local extra=()
  test -s "$checkpoint"
  mapfile -t extra < <(method_args "$method")
  mkdir -p "$out"
  python -u main.py \
    --gpu "$GPU" --feats 2 --action "$ACTION" --seed "$SEED" \
    --sequence_sampling legacy --multi_modality \
    --run_tag "$method" "${extra[@]}" \
    --eval_policy "$policy" --eval_seed "$EVAL_SEED" \
    --eval_mc_samples "$MC_SAMPLES" \
    --test --ckpt_path "$checkpoint" \
    --result_json "$out/test_result.json" \
    > "$out/test.log" 2>&1
  grep -qF "Coef:" "$out/test.log"
}

for method in base coqm_no_coord; do
  run_eval "$method" argmax_a argmax
  run_eval "$method" argmax_b argmax
  run_eval "$method" mc_gumbel mc_gumbel
done

python tools/verify_coqm_eval_b0_smoke.py \
  --root "$ROOT" --action "$ACTION" --seed "$SEED" \
  --eval_seed "$EVAL_SEED" --mc_samples "$MC_SAMPLES" \
  --output "$ROOT/smoke_gate.json"
git rev-parse HEAD > "$ROOT/git_commit.txt"
git branch --show-current > "$ROOT/git_branch.txt"
printf '%s\n' "$CHECKPOINT_ROOT" > "$ROOT/source_checkpoint_root.txt"
echo "COQM_EVAL_B0_SMOKE_FINISHED root=$ROOT"
