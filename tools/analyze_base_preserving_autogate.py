import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import spearmanr


VARIANTS = ("base", "off", "motion", "skeleton", "auto")
GATE_NAMES = ("base", "motion", "skeleton")
ENDPOINT_BOUNDS = {"motion": 0.015, "skeleton": 0.02}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze the decoupled competitive AutoGate probe"
    )
    for split in ("train", "cal"):
        for variant in VARIANTS:
            parser.add_argument(f"--{variant}_{split}")
    parser.add_argument("--output")
    parser.add_argument("--expected_train_samples", type=int, default=320)
    parser.add_argument("--expected_cal_samples", type=int, default=80)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        required = (
            "output",
            *(f"{variant}_{split}" for split in ("train", "cal")
              for variant in VARIANTS),
        )
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            parser.error(f"missing required arguments: {missing}")
    return args


def read_table(path, expected_samples, require_gates=False):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "label", "prediction"}
    if require_gates:
        required.update(f"{name}_gate" for name in GATE_NAMES)
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} is missing columns: {sorted(required)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != expected_samples or len(set(sample_ids)) != len(
            sample_ids):
        raise ValueError(
            f"{path} must contain {expected_samples} unique samples"
        )
    result = {
        "sample_ids": sample_ids,
        "labels": np.asarray([float(row["label"]) for row in rows]),
        "predictions": np.asarray([
            float(row["prediction"]) for row in rows
        ]),
    }
    if require_gates:
        result["gates"] = np.asarray([
            [float(row[f"{name}_gate"]) for name in GATE_NAMES]
            for row in rows
        ])
    values = [result["labels"], result["predictions"]]
    if require_gates:
        values.append(result["gates"])
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError(f"{path} contains non-finite values")
    return result


def metric(predictions, labels):
    coefficient, _ = spearmanr(predictions, labels)
    if not np.isfinite(coefficient):
        coefficient = 0.0
    return {
        "coef": float(100.0 * coefficient),
        "mse_raw": float(np.mean((predictions - labels) ** 2)),
    }


def compare(candidate, reference, labels):
    candidate_metric = metric(candidate, labels)
    reference_metric = metric(reference, labels)
    reference_mse = reference_metric["mse_raw"]
    return {
        "delta_coef": candidate_metric["coef"] - reference_metric["coef"],
        "relative_mse_gain": (
            (reference_mse - candidate_metric["mse_raw"]) / reference_mse
            if reference_mse > 0 else 0.0
        ),
    }


def fit_positive_affine(predictions, labels):
    x = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-12:
        raise ValueError("cannot fit an affine control to constant predictions")
    scale = float(np.dot(centered, y - y.mean()) / denominator)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("train affine control is not rank-preserving")
    shift = float(y.mean() - scale * x.mean())
    return scale, shift


def bootstrap(candidate, reference, labels, repeats, seed):
    rng = np.random.RandomState(seed)
    delta_coef = []
    relative_mse_gain = []
    for _ in range(repeats):
        indices = rng.randint(0, len(labels), len(labels))
        result = compare(
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


def gate_stats(gates):
    result = {
        name: {
            "mean": float(gates[:, index].mean()),
            "std": float(gates[:, index].std()),
            "min": float(gates[:, index].min()),
            "max": float(gates[:, index].max()),
        }
        for index, name in enumerate(GATE_NAMES)
    }
    result["max_abs_sum_minus_one"] = float(np.max(np.abs(
        gates.sum(axis=1) - 1.0
    )))
    return result


def endpoint_stats(tables):
    result = {}
    off = tables["off"]["predictions"]
    for name, bound in ENDPOINT_BOUNDS.items():
        correction = tables[name]["predictions"] - off
        result[name] = {
            "mean": float(correction.mean()),
            "std": float(correction.std()),
            "std_over_bound": float(correction.std() / bound),
            "min": float(correction.min()),
            "max": float(correction.max()),
            "saturation_fraction": float(np.mean(
                np.abs(correction) >= 0.95 * bound
            )),
        }
    return result


def aligned_tables(paths, expected_samples):
    tables = {
        name: read_table(
            path, expected_samples, require_gates=name == "auto"
        )
        for name, path in paths.items()
    }
    reference = tables["base"]
    for name, table in tables.items():
        if table["sample_ids"] != reference["sample_ids"] or not (
                np.array_equal(table["labels"], reference["labels"])):
            raise ValueError(f"{name} table is not aligned with Base")
    return tables


def analyze(train, calibration, bootstrap_repeats, seed):
    scale, shift = fit_positive_affine(
        train["base"]["predictions"], train["base"]["labels"]
    )
    affine_train = scale * train["base"]["predictions"] + shift
    affine_cal = scale * calibration["base"]["predictions"] + shift
    affine_anchor_error = {
        "train_max_abs_off_minus_affine": float(np.max(np.abs(
            train["off"]["predictions"] - affine_train
        ))),
        "calibration_max_abs_off_minus_affine": float(np.max(np.abs(
            calibration["off"]["predictions"] - affine_cal
        ))),
    }
    labels = calibration["base"]["labels"]
    auto = calibration["auto"]["predictions"]
    auto_comparison = compare(auto, affine_cal, labels)
    boot = bootstrap(
        auto, affine_cal, labels, bootstrap_repeats, seed
    )
    gates = gate_stats(calibration["auto"]["gates"])
    endpoints = {
        "train": endpoint_stats(train),
        "calibration": endpoint_stats(calibration),
    }
    endpoint_nonconstant = all(
        endpoints["train"][name]["std_over_bound"] >= 0.05
        for name in ENDPOINT_BOUNDS
    )
    endpoint_nonsaturated = all(
        endpoints["train"][name]["saturation_fraction"] < 0.90
        for name in ENDPOINT_BOUNDS
    )
    gate_dynamic = max(
        gates["motion"]["std"], gates["skeleton"]["std"]
    ) >= 0.02
    checks = {
        "off_matches_train_fitted_positive_affine": (
            max(affine_anchor_error.values()) <= 1e-6
        ),
        "competitive_gate_sums_to_one": (
            gates["max_abs_sum_minus_one"] <= 1e-6
        ),
        "both_train_endpoints_nonconstant": endpoint_nonconstant,
        "train_endpoints_not_saturated": endpoint_nonsaturated,
        "sample_gate_nonconstant": gate_dynamic,
        "calibration_delta_coef_at_least_0_5": (
            auto_comparison["delta_coef"] >= 0.5
        ),
        "calibration_mse_not_worse_by_over_2pct": (
            auto_comparison["relative_mse_gain"] >= -0.02
        ),
        "bootstrap_coef_positive_probability_at_least_90pct": (
            boot["delta_coef_positive_probability"] >= 0.90
        ),
        "bootstrap_mse_guardrail_probability_at_least_90pct": (
            boot["mse_guardrail_probability"] >= 0.90
        ),
    }
    endpoint_matrix = np.stack([
        affine_cal,
        calibration["motion"]["predictions"],
        calibration["skeleton"]["predictions"],
    ], axis=1)
    oracle = endpoint_matrix[
        np.arange(len(labels)),
        np.argmin((endpoint_matrix - labels[:, None]) ** 2, axis=1),
    ]
    if all(checks.values()):
        status = "GO_SEED_CONFIRMATION"
    elif not endpoint_nonconstant or not endpoint_nonsaturated:
        status = "STOP_AUTOGATE_CONSTANT_BIAS"
    else:
        status = "NO_GO"
    constant_shift = float(np.mean(
        train["base"]["labels"] - train["base"]["predictions"]
    ))
    return {
        "affine_anchor": {
            "scale": scale,
            "shift": shift,
            **affine_anchor_error,
        },
        "metrics": {
            split: {
                **{
                    name: metric(table["predictions"], table["labels"])
                    for name, table in tables.items()
                },
                "constant_control": metric(
                    tables["base"]["predictions"] + constant_shift,
                    tables["base"]["labels"],
                ),
                "affine_control": metric(
                    (affine_train if split == "train" else affine_cal),
                    tables["base"]["labels"],
                ),
            }
            for split, tables in (
                ("train", train), ("calibration", calibration)
            )
        },
        "auto_vs_affine": auto_comparison,
        "paired_bootstrap": boot,
        "endpoint_diagnostics": endpoints,
        "calibration_gate_stats": gates,
        "endpoint_oracle_over_auto": compare(oracle, auto, labels),
        "checks": checks,
        "status": status,
    }


def self_test():
    labels = np.linspace(0.1, 0.9, 20)
    base = labels.copy()
    base[[3, 4, 10, 11, 16, 17]] = base[[4, 3, 11, 10, 17, 16]]
    scale, shift = fit_positive_affine(base, labels)
    off = scale * base + shift
    tables = {}
    for name, prediction in (
            ("base", base), ("off", off),
            ("motion", off + 0.006 * np.sin(np.arange(20))),
            ("skeleton", off + 0.008 * np.cos(np.arange(20))),
            ("auto", labels)):
        table = {
            "sample_ids": [str(index) for index in range(20)],
            "labels": labels.copy(),
            "predictions": prediction.copy(),
        }
        if name == "auto":
            motion = 0.2 + 0.1 * labels
            skeleton = 0.3 - 0.1 * labels
            table["gates"] = np.stack(
                (1 - motion - skeleton, motion, skeleton), axis=1
            )
        tables[name] = table
    result = analyze(tables, tables, 100, 0)
    assert result["affine_anchor"][
        "train_max_abs_off_minus_affine"] < 1e-12
    assert result["auto_vs_affine"]["delta_coef"] > 0
    assert result["checks"]["competitive_gate_sums_to_one"]
    print("decoupled competitive AutoGate analyzer self-test passed")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    train = aligned_tables(
        {name: getattr(args, f"{name}_train") for name in VARIANTS},
        args.expected_train_samples,
    )
    calibration = aligned_tables(
        {name: getattr(args, f"{name}_cal") for name in VARIANTS},
        args.expected_cal_samples,
    )
    if set(train["base"]["sample_ids"]) & set(
            calibration["base"]["sample_ids"]):
        raise ValueError("train and calibration sample IDs overlap")
    result = analyze(train, calibration, args.bootstrap, args.seed)
    result.update({
        "schema_version": 2,
        "protocol": (
            "Frozen PAMFN; train-only positive affine Base anchor; "
            "zero-mean cue experts trained before a Base/Motion/Skeleton "
            "softmax gate; final epoch; no calibration checkpoint selection."
        ),
        "decision": (
            "STOP_AUTOGATE_CONSTANT_BIAS means stop this sample-level gate "
            "with the current cue features, not discard Motion/Skeleton."
        ),
        "warning": (
            "The historical 320/80 split is development-only and cannot "
            "serve as final independent evidence."
        ),
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"status={result['status']}")
    print(
        "auto_vs_affine_delta_coef="
        f"{result['auto_vs_affine']['delta_coef']:+.4f}"
    )
    print(
        "auto_vs_affine_relative_mse_gain="
        f"{result['auto_vs_affine']['relative_mse_gain']:+.4%}"
    )
    print(f"report={os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
