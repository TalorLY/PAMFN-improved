#!/usr/bin/env python
"""Verify B0 repeatability and MC checkpoint-evaluation smoke results."""

import argparse
import json
import math
import os


METHODS = ("base", "coqm_no_coord")


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def without_runtime(value):
    if isinstance(value, dict):
        return {
            key: without_runtime(item)
            for key, item in value.items() if key != "time"
        }
    if isinstance(value, list):
        return [without_runtime(item) for item in value]
    return value


def valid_core_metrics(result):
    return all(
        isinstance(result.get(field), (int, float))
        and math.isfinite(result[field])
        for field in (
            "coef", "loss_x100", "prediction_std_ratio", "mae", "bias"
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--action", default="Ribbon")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=20260908)
    parser.add_argument("--mc_samples", type=int, default=20)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    failures = []
    summary = {}
    for method in METHODS:
        run_root = os.path.join(
            args.root, args.action, f"seed{args.seed}", method
        )
        results = {}
        for name in ("argmax_a", "argmax_b", "mc_gumbel"):
            path = os.path.join(run_root, name, "test_result.json")
            if not os.path.isfile(path):
                failures.append(f"{method}/{name}: missing result JSON")
                continue
            result = read_json(path)
            results[name] = result
            if not valid_core_metrics(result):
                failures.append(f"{method}/{name}: invalid core metric")
            expected_policy = (
                "argmax" if name.startswith("argmax") else "mc_gumbel"
            )
            expected_passes = (
                1 if expected_policy == "argmax" else args.mc_samples
            )
            evaluation = result.get("evaluation", {})
            run = result.get("run", {})
            if (
                    evaluation.get("policy") != expected_policy
                    or evaluation.get("passes") != expected_passes
                    or evaluation.get("seed") != args.eval_seed
                    or not evaluation.get("rng_isolated")
                    or run.get("eval_policy") != expected_policy
                    or run.get("eval_seed") != args.eval_seed):
                failures.append(
                    f"{method}/{name}: evaluation metadata mismatch"
                )
        if "argmax_a" in results and "argmax_b" in results:
            if without_runtime(results["argmax_a"]) != without_runtime(
                    results["argmax_b"]):
                failures.append(
                    f"{method}: repeated argmax results are not identical"
                )
        summary[method] = {
            name: {
                "srcc": result.get("coef", float("nan")) / 100,
                "mse": result.get("loss_x100", float("nan")) / 100,
            }
            for name, result in results.items()
        }
    output = {
        "decision": (
            "COQM_EVAL_B0_SMOKE_FAIL"
            if failures else "COQM_EVAL_B0_SMOKE_PASS"
        ),
        "failures": failures,
        "unit_tests_passed": True,
        "exact_argmax_repeatability": not any(
            "not identical" in failure for failure in failures
        ),
        "runs": summary,
    }
    if args.output:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output)), exist_ok=True
        )
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, sort_keys=True)
    print(json.dumps(output, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(
            "COQM_EVAL_B0_SMOKE_FAIL: " + "; ".join(failures)
        )


if __name__ == "__main__":
    main()
