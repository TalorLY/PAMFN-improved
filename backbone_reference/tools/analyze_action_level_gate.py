import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import spearmanr


VARIANTS = ("off", "motion", "skeleton")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit one train-only Base/Motion/Skeleton gate for an entire "
            "task and evaluate it on a held-out development split"
        )
    )
    for split in ("train", "cal"):
        for variant in VARIANTS:
            parser.add_argument(f"--{variant}_{split}")
    parser.add_argument("--output")
    parser.add_argument("--train_output")
    parser.add_argument("--cal_output")
    parser.add_argument("--expected_train_samples", type=int, default=320)
    parser.add_argument("--expected_cal_samples", type=int, default=80)
    parser.add_argument("--gate_step", type=float, default=0.01)
    parser.add_argument("--mse_guardrail", type=float, default=0.02)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        required = (
            "output", "train_output", "cal_output",
            *(f"{variant}_{split}" for split in ("train", "cal")
              for variant in VARIANTS),
        )
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            parser.error(f"missing required arguments: {missing}")
    if not 0 < args.gate_step <= 1:
        parser.error("--gate_step must be in (0, 1]")
    inverse = 1.0 / args.gate_step
    if not np.isclose(inverse, round(inverse), atol=1e-9, rtol=0):
        parser.error("--gate_step must divide 1 exactly")
    if not np.isfinite(args.mse_guardrail) or args.mse_guardrail < 0:
        parser.error("--mse_guardrail must be finite and non-negative")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    return args


def read_table(path, expected_samples):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "label", "prediction"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} is missing columns: {sorted(required)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != expected_samples:
        raise ValueError(
            f"{path} contains {len(sample_ids)} samples; "
            f"expected {expected_samples}"
        )
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{path} contains duplicate sample IDs")
    labels = np.asarray([float(row["label"]) for row in rows])
    predictions = np.asarray([float(row["prediction"]) for row in rows])
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError(f"{path} contains non-finite values")
    return {
        "sample_ids": sample_ids,
        "labels": labels,
        "predictions": predictions,
    }


def aligned_tables(paths, expected_samples):
    tables = {
        name: read_table(path, expected_samples)
        for name, path in paths.items()
    }
    reference = tables["off"]
    for name, table in tables.items():
        if table["sample_ids"] != reference["sample_ids"] or not (
                np.array_equal(table["labels"], reference["labels"])):
            raise ValueError(f"{name} table is not aligned with off")
    return tables


def metric(predictions, labels):
    if predictions.size < 2 or np.all(predictions == predictions[0]) or (
            np.all(labels == labels[0])):
        coefficient = 0.0
    else:
        coefficient, _ = spearmanr(predictions, labels)
        if not np.isfinite(coefficient):
            coefficient = 0.0
    return {
        "coef": float(100.0 * coefficient),
        "mse_raw": float(np.mean((predictions - labels) ** 2)),
    }


def blend(tables, weights):
    off = tables["off"]["predictions"]
    motion_delta = tables["motion"]["predictions"] - off
    skeleton_delta = tables["skeleton"]["predictions"] - off
    return (
        off
        + weights[1] * motion_delta
        + weights[2] * skeleton_delta
    )


def grid_weights(step):
    count = int(round(1.0 / step))
    for motion_index in range(count + 1):
        for skeleton_index in range(count - motion_index + 1):
            motion = motion_index / count
            skeleton = skeleton_index / count
            yield (1.0 - motion - skeleton, motion, skeleton)


def fit_gate(tables, step, mse_guardrail):
    labels = tables["off"]["labels"]
    endpoint_metrics = {
        name: metric(table["predictions"], labels)
        for name, table in tables.items()
    }
    best_endpoint_mse = min(
        value["mse_raw"] for value in endpoint_metrics.values()
    )
    mse_limit = best_endpoint_mse * (1.0 + mse_guardrail)
    candidates = []
    for weights in grid_weights(step):
        predictions = blend(tables, weights)
        values = metric(predictions, labels)
        if values["mse_raw"] <= mse_limit + 1e-15:
            candidates.append((
                values["coef"],
                -values["mse_raw"],
                weights[0],
                weights,
                values,
            ))
    if not candidates:
        raise RuntimeError("no action-gate candidate passed the MSE guardrail")
    _, _, _, weights, values = max(candidates)
    return {
        "weights": weights,
        "metrics": values,
        "endpoint_metrics": endpoint_metrics,
        "best_endpoint_mse": best_endpoint_mse,
        "mse_limit": mse_limit,
        "admissible_candidates": len(candidates),
    }


def comparison(candidate, reference, labels):
    candidate_metric = metric(candidate, labels)
    reference_metric = metric(reference, labels)
    reference_mse = reference_metric["mse_raw"]
    return {
        "candidate": candidate_metric,
        "reference": reference_metric,
        "delta_coef": candidate_metric["coef"] - reference_metric["coef"],
        "relative_mse_gain": (
            (reference_mse - candidate_metric["mse_raw"]) / reference_mse
            if reference_mse > 0 else 0.0
        ),
    }


def bootstrap(candidate, reference, labels, repeats, seed):
    rng = np.random.RandomState(seed)
    delta_coef = []
    relative_mse_gain = []
    for _ in range(repeats):
        indices = rng.randint(0, len(labels), len(labels))
        result = comparison(
            candidate[indices], reference[indices], labels[indices]
        )
        delta_coef.append(result["delta_coef"])
        relative_mse_gain.append(result["relative_mse_gain"])
    delta_coef = np.asarray(delta_coef)
    relative_mse_gain = np.asarray(relative_mse_gain)
    return {
        "delta_coef_95ci": np.percentile(
            delta_coef, (2.5, 97.5)
        ).tolist(),
        "relative_mse_gain_95ci": np.percentile(
            relative_mse_gain, (2.5, 97.5)
        ).tolist(),
        "delta_coef_positive_probability": float(np.mean(delta_coef > 0)),
        "mse_guardrail_probability": float(
            np.mean(relative_mse_gain >= -0.02)
        ),
    }


def analyze(train, calibration, step, mse_guardrail, repeats, seed):
    fitted = fit_gate(train, step, mse_guardrail)
    weights = fitted["weights"]
    train_prediction = blend(train, weights)
    cal_prediction = blend(calibration, weights)
    train_labels = train["off"]["labels"]
    cal_labels = calibration["off"]["labels"]
    train_comparison = comparison(
        train_prediction, train["off"]["predictions"], train_labels
    )
    cal_comparison = comparison(
        cal_prediction, calibration["off"]["predictions"], cal_labels
    )
    cal_bootstrap = bootstrap(
        cal_prediction, calibration["off"]["predictions"],
        cal_labels, repeats, seed
    )
    endpoint_metrics = {
        split: {
            name: metric(table["predictions"], tables["off"]["labels"])
            for name, table in tables.items()
        }
        for split, tables in (
            ("train", train), ("calibration", calibration)
        )
    }
    go = (
        cal_comparison["delta_coef"] >= 0.5
        and cal_comparison["relative_mse_gain"] >= -0.02
        and cal_bootstrap["delta_coef_positive_probability"] >= 0.9
    )
    report = {
        "schema_version": 1,
        "state": (
            "GO_ACTION_LEVEL_GATE_OOF_REQUIRED"
            if go else "STOP_CURRENT_ACTION_LEVEL_GATE"
        ),
        "scope": (
            "necessary-condition development probe; fitted on in-sample "
            "training predictions, not final paper evidence"
        ),
        "selection": {
            "source": "train_only_full_task",
            "objective": "maximize_full_task_spearman",
            "mse_reference": "best_train_endpoint",
            "mse_guardrail": mse_guardrail,
            "gate_step": step,
            "weights": {
                "base": weights[0],
                "motion": weights[1],
                "skeleton": weights[2],
            },
            "train_selected_metrics": fitted["metrics"],
            "best_endpoint_mse": fitted["best_endpoint_mse"],
            "mse_limit": fitted["mse_limit"],
            "admissible_candidates": fitted["admissible_candidates"],
        },
        "endpoint_metrics": endpoint_metrics,
        "comparison_to_off": {
            "train": train_comparison,
            "calibration": cal_comparison,
        },
        "calibration_bootstrap": cal_bootstrap,
        "go_rule": {
            "minimum_delta_coef": 0.5,
            "minimum_relative_mse_gain": -0.02,
            "minimum_delta_coef_positive_probability": 0.9,
        },
    }
    return report, train_prediction, cal_prediction


def write_predictions(path, tables, predictions, weights):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "sample_id", "group_id", "label", "prediction",
            "base_gate", "motion_gate", "skeleton_gate",
        ))
        writer.writeheader()
        for sample_id, label, prediction in zip(
                tables["off"]["sample_ids"],
                tables["off"]["labels"], predictions):
            writer.writerow({
                "sample_id": sample_id,
                "group_id": sample_id,
                "label": float(label),
                "prediction": float(prediction),
                "base_gate": weights[0],
                "motion_gate": weights[1],
                "skeleton_gate": weights[2],
            })


def self_test():
    labels = np.linspace(0.0, 1.0, 30)
    off = labels[::-1] * 0.1 + 0.4
    motion = labels.copy()
    skeleton = np.full_like(labels, 0.5)

    def tables(prefix):
        return {
            name: {
                "sample_ids": [f"{prefix}_{index}" for index in range(30)],
                "labels": labels,
                "predictions": values,
            }
            for name, values in (
                ("off", off), ("motion", motion),
                ("skeleton", skeleton)
            )
        }

    train = tables("train")
    calibration = tables("cal")
    report, train_prediction, cal_prediction = analyze(
        train, calibration, 0.1, 0.02, 50, 0
    )
    weights = report["selection"]["weights"]
    assert np.isclose(sum(weights.values()), 1.0)
    assert weights["motion"] == 1.0
    assert np.allclose(train_prediction, motion)
    assert np.allclose(cal_prediction, motion)
    assert report["state"] == "GO_ACTION_LEVEL_GATE_OOF_REQUIRED"
    print("action-level gate analyzer self-test passed")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    train = aligned_tables({
        name: getattr(args, f"{name}_train") for name in VARIANTS
    }, args.expected_train_samples)
    calibration = aligned_tables({
        name: getattr(args, f"{name}_cal") for name in VARIANTS
    }, args.expected_cal_samples)
    if set(train["off"]["sample_ids"]) & set(
            calibration["off"]["sample_ids"]):
        raise ValueError("train and calibration sample IDs overlap")
    report, train_prediction, cal_prediction = analyze(
        train, calibration, args.gate_step, args.mse_guardrail,
        args.bootstrap, args.seed
    )
    weights = tuple(
        report["selection"]["weights"][name]
        for name in ("base", "motion", "skeleton")
    )
    write_predictions(
        args.train_output, train, train_prediction, weights
    )
    write_predictions(
        args.cal_output, calibration, cal_prediction, weights
    )
    parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "state": report["state"],
        "weights": report["selection"]["weights"],
        "calibration": report["comparison_to_off"]["calibration"],
        "bootstrap": report["calibration_bootstrap"],
    }, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
