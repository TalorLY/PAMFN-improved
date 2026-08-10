import argparse
import csv
import hashlib
import json
import os

import numpy as np
from scipy.stats import spearmanr


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stop a paired endpoint experiment before lockbox access when "
            "train MSE locks the cue gate to zero"
        )
    )
    parser.add_argument("--control", required=True)
    parser.add_argument("--g0", required=True)
    parser.add_argument("--g1", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected_samples", type=int, default=336)
    return parser.parse_args()


def file_record(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": os.path.abspath(path),
        "sha256": digest.hexdigest(),
        "bytes": os.path.getsize(path),
    }


def read_predictions(path, expected_samples):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "label", "prediction"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} is missing prediction columns")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != expected_samples or len(set(sample_ids)) != len(
            sample_ids):
        raise ValueError(
            f"{path} must contain {expected_samples} unique samples"
        )
    labels = np.asarray([float(row["label"]) for row in rows])
    predictions = np.asarray([float(row["prediction"]) for row in rows])
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError(f"{path} contains non-finite values")
    return sample_ids, labels, predictions


def metrics(predictions, labels):
    coefficient, _ = spearmanr(predictions, labels)
    if not np.isfinite(coefficient):
        coefficient = 0.0
    return {
        "coef": float(100.0 * coefficient),
        "mse_raw": float(np.mean((predictions - labels) ** 2)),
    }


def main():
    args = parse_args()
    tables = {
        name: read_predictions(path, args.expected_samples)
        for name, path in (
            ("control", args.control),
            ("g0", args.g0),
            ("g1", args.g1),
        )
    }
    reference_ids, reference_labels, _ = tables["control"]
    for name, (sample_ids, labels, _) in tables.items():
        if sample_ids != reference_ids or not np.array_equal(
                labels, reference_labels):
            raise ValueError(f"{name} predictions are not sample-aligned")

    calculated = {
        name: metrics(predictions, labels)
        for name, (_, labels, predictions) in tables.items()
    }
    fixed_gate = (
        1 if calculated["g1"]["mse_raw"] < calculated["g0"]["mse_raw"]
        else 0
    )
    status = (
        "PROCEED_TO_LOCKBOX" if fixed_gate == 1
        else "EARLY_NO_GO_TRAIN"
    )
    g0_predictions = tables["g0"][2]
    g1_predictions = tables["g1"][2]
    direct_effect = g1_predictions - g0_predictions
    payload = {
        "schema_version": 1,
        "status": status,
        "decision_scope": "train_only_before_protocol_lock",
        "lockbox_accessed": False,
        "sample_count": len(reference_ids),
        "fixed_gate_selection": {
            "metric": "train MSE; ties prefer gate 0",
            "gate": fixed_gate,
        },
        "metrics": calculated,
        "comparisons": {
            "g0_minus_control_delta_coef": (
                calculated["g0"]["coef"]
                - calculated["control"]["coef"]
            ),
            "g1_minus_g0_delta_coef": (
                calculated["g1"]["coef"] - calculated["g0"]["coef"]
            ),
            "g1_vs_g0_relative_mse_gain": (
                (calculated["g0"]["mse_raw"]
                 - calculated["g1"]["mse_raw"])
                / calculated["g0"]["mse_raw"]
                if calculated["g0"]["mse_raw"] > 0 else 0.0
            ),
        },
        "direct_effect": {
            "mean": float(direct_effect.mean()),
            "std": float(direct_effect.std()),
            "min": float(direct_effect.min()),
            "max": float(direct_effect.max()),
        },
        "reason": (
            "The pre-registered endpoint is disabled because train MSE "
            "selected gate 0; the fresh lockbox must remain unopened."
            if fixed_gate == 0 else
            "Train MSE selected gate 1; protocol locking may continue."
        ),
        "inputs": {
            "control": file_record(args.control),
            "g0": file_record(args.g0),
            "g1": file_record(args.g1),
        },
    }
    parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(parent, exist_ok=True)
    with open(args.output, "x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"train_futility_status={status}")
    print(f"train_locked_fixed_gate={fixed_gate}")
    print(f"report={os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
