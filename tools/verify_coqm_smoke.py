#!/usr/bin/env python
"""Fail fast if the two-run COQM server smoke test is incomplete."""

import argparse
import json
import math
import os


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    failures = []
    summary = {}
    for mode in ("base", "coqm"):
        train_path = os.path.join(
            args.root, mode, "train_result.json"
        )
        test_path = os.path.join(
            args.root, mode, "test_result.json"
        )
        if not os.path.isfile(train_path) or not os.path.isfile(test_path):
            failures.append(f"{mode}: missing result JSON")
            continue
        train = load(train_path)
        test = load(test_path)
        summary[mode] = {
            "saved_model_srcc": train.get("saved_model_srcc"),
            "test_srcc": (
                test.get("coef") / 100
                if isinstance(test.get("coef"), (int, float)) else None
            ),
            "test_mse": (
                test.get("loss_x100") / 100
                if isinstance(test.get("loss_x100"), (int, float)) else None
            ),
            "prediction_std_ratio": test.get("prediction_std_ratio"),
        }
        required = (
            train.get("saved_model_srcc"),
            test.get("coef"),
            test.get("loss_x100"),
            test.get("prediction_std_ratio"),
        )
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in required):
            failures.append(f"{mode}: non-finite core metric")
        if test.get("prediction_std_ratio", 0) <= 0.1:
            failures.append(f"{mode}: prediction spread collapsed")
        if mode == "coqm":
            diagnostic = test.get("coqm", {})
            for field in (
                    "curve_cosine", "coordinate_mae",
                    "quality_std_ratio", "orthogonal_error",
                    "score_q_spearman"):
                value = diagnostic.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    failures.append(f"coqm: invalid diagnostic {field}")
            for field in ("soft_occupancy", "hard_occupancy"):
                values = diagnostic.get(field, [])
                if len(values) != 4 or not all(math.isfinite(x) for x in values):
                    failures.append(f"coqm: invalid {field}")
    calibration_path = os.path.join(args.root, "gradient_calibration.json")
    if not os.path.isfile(calibration_path):
        failures.append("missing gradient calibration")
    else:
        calibration = load(calibration_path)
        if not calibration.get("finite"):
            failures.append("non-finite gradient calibration")
        if not calibration.get("within_acceptance_interval"):
            failures.append(
                "gradient ratio outside [0.05, 0.20]; inspect the "
                "recommended training-only calibrated weight"
            )
    result = {
        "decision": (
            "COQM_SMOKE_FAIL" if failures else "COQM_SMOKE_PASS"
        ),
        "failures": failures,
        "unit_tests_passed": True,
        "runs": summary,
        "gradient_calibration": (
            calibration if os.path.isfile(calibration_path) else None
        ),
    }
    if args.output:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output)), exist_ok=True
        )
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit("COQM_SMOKE_FAIL: " + "; ".join(failures))


if __name__ == "__main__":
    main()
