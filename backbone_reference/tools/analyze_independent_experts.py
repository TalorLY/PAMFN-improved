import argparse
import csv
import hashlib
import json
import os
import tempfile

import numpy as np
from scipy.stats import spearmanr


EXPERTS = ("base", "motion", "skeleton")
REQUIRED_COLUMNS = ("sample_id", "group_id", "label", "prediction")
MAX_MIXTURE_UNITS = 200


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure sample-level headroom among independently trained "
            "Base, Motion, and Skeleton experts"
        )
    )
    parser.add_argument("--dataset", default="TES")
    for split in ("train", "cal"):
        for expert in EXPERTS:
            parser.add_argument(f"--{expert}_{split}")
    parser.add_argument("--output_dir")
    parser.add_argument("--expected_train_samples", type=int, default=320)
    parser.add_argument(
        "--expected_calibration_samples", type=int, default=80
    )
    parser.add_argument("--mixture_step", type=float, default=0.01)
    parser.add_argument("--score_bins", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    required = [
        getattr(args, f"{expert}_{split}")
        for split in ("train", "cal")
        for expert in EXPERTS
    ] + [args.output_dir]
    if any(not value for value in required):
        parser.error(
            "six expert CSV paths and --output_dir are required"
        )
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    if args.score_bins < 2:
        parser.error("--score_bins must be at least 2")
    if args.expected_train_samples < 0:
        parser.error("--expected_train_samples must be non-negative")
    if args.expected_calibration_samples < 0:
        parser.error(
            "--expected_calibration_samples must be non-negative"
        )
    validate_mixture_step(args.mixture_step)
    if not args.dataset or any(
            value in args.dataset for value in ("/", "\\")):
        parser.error("--dataset must be a non-empty file-name token")
    return args


def validate_mixture_step(step):
    if not np.isfinite(step) or not 0 < step <= 1:
        raise ValueError("mixture_step must be finite and in (0, 1]")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0, atol=1e-10, rtol=0):
        raise ValueError("1 / mixture_step must be an integer")
    if units > MAX_MIXTURE_UNITS:
        raise ValueError(
            "mixture_step creates too many simplex weights; use a step "
            f"of at least {1.0 / MAX_MIXTURE_UNITS:g}"
        )
    return units


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_expert_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"{path} contains duplicate header names")
        missing = set(REQUIRED_COLUMNS) - set(fieldnames)
        if missing:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no samples")

    sample_ids = []
    group_ids = []
    labels = []
    predictions = []
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(
                row.get(column) is None for column in REQUIRED_COLUMNS):
            raise ValueError(
                f"{path}:{row_number} has a malformed CSV row"
            )
        sample_id = row["sample_id"].strip()
        group_id = row["group_id"].strip()
        if not sample_id or not group_id:
            raise ValueError(
                f"{path}:{row_number} has an empty sample/group ID"
            )
        try:
            label = float(row["label"])
            prediction = float(row["prediction"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path}:{row_number} has a non-numeric value"
            ) from error
        if not np.isfinite(label) or not np.isfinite(prediction):
            raise ValueError(
                f"{path}:{row_number} contains NaN or infinity"
            )
        sample_ids.append(sample_id)
        group_ids.append(group_id)
        labels.append(label)
        predictions.append(prediction)

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{path} contains duplicate sample IDs")
    if len(group_ids) != len(set(group_ids)):
        raise ValueError(f"{path} contains duplicate group IDs")
    return {
        "path": os.path.abspath(path),
        "sha256": file_sha256(path),
        "sample_ids": sample_ids,
        "group_ids": group_ids,
        "labels": np.asarray(labels, dtype=np.float64),
        "predictions": np.asarray(predictions, dtype=np.float64)
    }


def merge_expert_tables(paths):
    tables = {
        expert: read_expert_csv(paths[expert]) for expert in EXPERTS
    }
    reference = tables["base"]
    reference_ids = reference["sample_ids"]
    reference_set = set(reference_ids)
    aligned_predictions = []

    for expert in EXPERTS:
        table = tables[expert]
        if set(table["sample_ids"]) != reference_set:
            raise ValueError(
                f"{expert} CSV has a different sample ID set"
            )
        index_by_id = {
            sample_id: index
            for index, sample_id in enumerate(table["sample_ids"])
        }
        indices = [index_by_id[sample_id] for sample_id in reference_ids]
        groups = [table["group_ids"][index] for index in indices]
        labels = table["labels"][indices]
        if groups != reference["group_ids"]:
            raise ValueError(
                f"{expert} CSV has mismatched group IDs"
            )
        if not np.allclose(
                labels, reference["labels"], atol=1e-7, rtol=0):
            raise ValueError(f"{expert} CSV has mismatched labels")
        aligned_predictions.append(table["predictions"][indices])

    return {
        "sample_ids": list(reference_ids),
        "group_ids": list(reference["group_ids"]),
        "labels": reference["labels"].copy(),
        "predictions": np.column_stack(aligned_predictions),
        "inputs": {
            expert: {
                "path": tables[expert]["path"],
                "sha256": tables[expert]["sha256"]
            }
            for expert in EXPERTS
        }
    }


def validate_splits(
        train, calibration, expected_train=0,
        expected_calibration=0):
    if expected_train and len(train["sample_ids"]) != expected_train:
        raise ValueError(
            f"expected {expected_train} train samples, got "
            f"{len(train['sample_ids'])}"
        )
    if expected_calibration and (
            len(calibration["sample_ids"]) != expected_calibration):
        raise ValueError(
            f"expected {expected_calibration} calibration samples, got "
            f"{len(calibration['sample_ids'])}"
        )
    if set(train["sample_ids"]) & set(calibration["sample_ids"]):
        raise ValueError("train and calibration sample IDs overlap")
    if set(train["group_ids"]) & set(calibration["group_ids"]):
        raise ValueError("train and calibration group IDs overlap")
    for split_name, table in (
            ("train", train), ("calibration", calibration)):
        if np.std(table["labels"]) <= 1e-12:
            raise ValueError(f"{split_name} labels are constant")
        if table["predictions"].shape != (
                len(table["labels"]), len(EXPERTS)):
            raise ValueError(f"{split_name} prediction shape is invalid")


def safe_spearman(predictions, labels):
    coefficient, _ = spearmanr(predictions, labels)
    return float(coefficient) if np.isfinite(coefficient) else None


def prediction_metrics(predictions, labels):
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    rho = safe_spearman(predictions, labels)
    mse = float(np.mean((predictions - labels) ** 2))
    return {
        "rho_defined": rho is not None,
        "rho": rho,
        "coef": None if rho is None else float(rho * 100),
        "mse_raw": mse,
        "loss_x100": float(mse * 100),
        "prediction_mean": float(np.mean(predictions)),
        "prediction_std": float(np.std(predictions))
    }


def metrics_by_expert(predictions, labels):
    return {
        expert: prediction_metrics(predictions[:, index], labels)
        for index, expert in enumerate(EXPERTS)
    }


def fit_positive_affine(predictions, labels):
    parameters = []
    calibrated = np.empty_like(predictions, dtype=np.float64)
    label_mean = float(np.mean(labels))
    for index, expert in enumerate(EXPERTS):
        values = predictions[:, index]
        centered = values - np.mean(values)
        variance = float(np.mean(centered ** 2))
        if variance <= 1e-12:
            slope = 0.0
        else:
            slope = max(
                float(np.mean(centered * (labels - label_mean)))
                / variance,
                0.0
            )
        intercept = label_mean - slope * float(np.mean(values))
        calibrated[:, index] = slope * values + intercept
        parameters.append({
            "expert": expert,
            "slope": float(slope),
            "intercept": float(intercept)
        })
    return parameters, calibrated


def apply_affine(predictions, parameters):
    calibrated = np.empty_like(predictions, dtype=np.float64)
    for index, parameter in enumerate(parameters):
        calibrated[:, index] = (
            parameter["slope"] * predictions[:, index]
            + parameter["intercept"]
        )
    return calibrated


def simplex_weights(step):
    units = validate_mixture_step(step)
    return np.asarray([
        (base / units, motion / units, skeleton / units)
        for base in range(units + 1)
        for motion in range(units - base + 1)
        for skeleton in (units - base - motion,)
    ], dtype=np.float64)


def select_fixed_mixture(predictions, labels, step):
    weights = simplex_weights(step)
    mixtures = predictions @ weights.T
    mse = np.mean((mixtures - labels[:, None]) ** 2, axis=0)
    minimum = float(np.min(mse))
    candidates = np.flatnonzero(
        np.isclose(mse, minimum, atol=1e-12, rtol=0)
    )
    if len(candidates) == 1:
        selected = int(candidates[0])
    else:
        selected = min(
            candidates,
            key=lambda index: -(
                safe_spearman(mixtures[:, index], labels)
                if safe_spearman(mixtures[:, index], labels) is not None
                else float("-inf")
            )
        )
    prediction = mixtures[:, selected]
    return weights[selected], prediction, prediction_metrics(
        prediction, labels
    )


def apply_mixture(predictions, weights):
    return predictions @ np.asarray(weights, dtype=np.float64)


def fit_score_only_controller(
        base_scores, predictions, labels, mixture_step, score_bins):
    boundaries = np.unique(np.quantile(
        base_scores,
        np.linspace(0.0, 1.0, score_bins + 1)[1:-1]
    ))
    assignments = np.digitize(base_scores, boundaries)
    global_weights, _, _ = select_fixed_mixture(
        predictions, labels, mixture_step
    )
    weights_by_bin = []
    for bin_index in range(len(boundaries) + 1):
        rows = assignments == bin_index
        if np.sum(rows) < 2:
            weights = global_weights
        else:
            weights, _, _ = select_fixed_mixture(
                predictions[rows], labels[rows], mixture_step
            )
        weights_by_bin.append(weights)
    parameters = {
        "requested_bins": int(score_bins),
        "boundaries": [float(value) for value in boundaries],
        "weights_by_bin": [
            {
                expert: float(weights[index])
                for index, expert in enumerate(EXPERTS)
            }
            for weights in weights_by_bin
        ]
    }
    return parameters, apply_score_only_controller(
        base_scores, predictions, parameters
    )


def apply_score_only_controller(base_scores, predictions, parameters):
    assignments = np.digitize(
        base_scores, np.asarray(parameters["boundaries"])
    )
    output = np.empty(len(base_scores), dtype=np.float64)
    for bin_index, weight_dict in enumerate(
            parameters["weights_by_bin"]):
        rows = assignments == bin_index
        weights = [weight_dict[expert] for expert in EXPERTS]
        output[rows] = apply_mixture(predictions[rows], weights)
    return output


def score_determinism_control(
        train_base_scores, train_predictions,
        calibration_base_scores, calibration_predictions,
        calibration_labels):
    order = np.argsort(train_base_scores, kind="mergesort")
    sorted_scores = train_base_scores[order]
    unique_scores, inverse = np.unique(
        sorted_scores, return_inverse=True
    )
    support_min = float(unique_scores[0])
    support_max = float(unique_scores[-1])
    in_support = (
        (calibration_base_scores >= support_min)
        & (calibration_base_scores <= support_max)
    )
    in_support_count = int(np.sum(in_support))
    support_ratio = float(np.mean(in_support))
    report = {}
    label_std = float(np.std(calibration_labels))
    for index, expert in enumerate(EXPERTS[1:], start=1):
        sorted_targets = train_predictions[order, index]
        sums = np.bincount(inverse, weights=sorted_targets)
        counts = np.bincount(inverse)
        mean_targets = sums / counts
        predicted_from_base = np.interp(
            calibration_base_scores, unique_scores, mean_targets
        )
        unexplained_all = (
            calibration_predictions[:, index] - predicted_from_base
        )
        expert_difference_all = (
            calibration_predictions[:, index]
            - calibration_predictions[:, 0]
        )
        unexplained = unexplained_all[in_support]
        expert_difference = expert_difference_all[in_support]
        difference_std = (
            float(np.std(expert_difference))
            if in_support_count else 0.0
        )
        unexplained_std = (
            float(np.std(unexplained))
            if in_support_count else 0.0
        )
        unexplained_over_label = (
            unexplained_std / label_std if label_std > 0 else 0.0
        )
        unique_fraction = (
            unexplained_std / difference_std
            if difference_std > 1e-8 else 0.0
        )
        passes_relative = bool(unique_fraction >= 0.10)
        passes_absolute = bool(unexplained_over_label >= 0.01)
        report[expert] = {
            "method": "train-only piecewise-linear expert-from-base map",
            "train_base_support": [support_min, support_max],
            "calibration_in_support_count": in_support_count,
            "calibration_in_support_ratio": support_ratio,
            "expert_difference_std": difference_std,
            "unexplained_std": unexplained_std,
            "unexplained_std_over_label_std": unexplained_over_label,
            "unique_fraction_of_expert_difference": float(
                unique_fraction
            ),
            "passes_10pct_unique_fraction": passes_relative,
            "passes_1pct_label_scale": passes_absolute,
            "passes_non_score_component": bool(
                support_ratio >= 0.90
                and passes_relative
                and passes_absolute
            )
        }
    return report


def sample_oracle(predictions, labels, sample_ids):
    squared_error = (predictions - labels[:, None]) ** 2
    minimum = np.min(squared_error, axis=1, keepdims=True)
    ties = np.isclose(squared_error, minimum, atol=1e-12, rtol=0)
    selected = np.argmin(squared_error, axis=1)
    oracle_prediction = predictions[np.arange(len(labels)), selected]
    strict_counts = {expert: 0 for expert in EXPERTS}
    winning_ids = {expert: [] for expert in EXPERTS}
    for row, index in enumerate(selected):
        if int(np.sum(ties[row])) == 1:
            expert = EXPERTS[index]
            strict_counts[expert] += 1
            winning_ids[expert].append(sample_ids[row])
    return {
        "prediction": oracle_prediction,
        "metrics": prediction_metrics(oracle_prediction, labels),
        "selection_count": {
            expert: int(np.sum(selected == index))
            for index, expert in enumerate(EXPERTS)
        },
        "strict_selection_count": strict_counts,
        "strict_selection_ratio": {
            expert: float(strict_counts[expert] / len(labels))
            for expert in EXPERTS
        },
        "winning_sample_ids": winning_ids,
        "tie_ratio": float(np.mean(np.sum(ties, axis=1) > 1))
    }


def constant_offset_oracle(predictions, labels, sample_ids):
    base = predictions[:, 0]
    offsets = np.mean(predictions - base[:, None], axis=0)
    candidates = base[:, None] + offsets[None, :]
    result = sample_oracle(candidates, labels, sample_ids)
    result["mean_offsets_from_base"] = {
        expert: float(offsets[index])
        for index, expert in enumerate(EXPERTS)
    }
    return result


def relative_mse_gain(reference, candidate):
    return (
        float((reference - candidate) / reference)
        if reference > 0 else 0.0
    )


def coef_delta(candidate_metrics, reference_metrics):
    if (
            candidate_metrics["coef"] is None
            or reference_metrics["coef"] is None):
        return None
    return float(candidate_metrics["coef"] - reference_metrics["coef"])


def percentile_interval(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    lower, upper = np.percentile(values, [2.5, 97.5])
    return [float(lower), float(upper)]


def bootstrap_summary(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "95ci": percentile_interval(finite),
        "positive_probability": (
            float(np.mean(finite > 0)) if len(finite) else None
        ),
        "finite_replicates": int(len(finite))
    }


def group_bootstrap(
        labels, group_ids, fixed_prediction, score_prediction,
        oracle_prediction, expert_predictions,
        iterations, seed):
    groups = list(dict.fromkeys(group_ids))
    group_rows = {
        group: np.flatnonzero(np.asarray(group_ids) == group)
        for group in groups
    }
    rng = np.random.RandomState(seed)
    values = {
        "oracle_relative_mse_gain_over_fixed": [],
        "oracle_relative_mse_gain_over_score": [],
        "oracle_relative_mse_gain_over_constant": [],
        "oracle_delta_coef_over_fixed": [],
        "oracle_delta_coef_over_score": [],
        "oracle_delta_coef_over_constant": []
    }
    for _ in range(iterations):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([
            group_rows[group] for group in sampled_groups
        ])
        sampled_labels = labels[indices]
        fixed = fixed_prediction[indices]
        score = score_prediction[indices]
        oracle = oracle_prediction[indices]
        sampled_experts = expert_predictions[indices]
        constant = constant_offset_oracle(
            sampled_experts,
            sampled_labels,
            [str(index) for index in range(len(indices))]
        )["prediction"]
        fixed_metrics = prediction_metrics(fixed, sampled_labels)
        score_metrics = prediction_metrics(score, sampled_labels)
        oracle_metrics = prediction_metrics(oracle, sampled_labels)
        constant_metrics = prediction_metrics(constant, sampled_labels)
        values["oracle_relative_mse_gain_over_fixed"].append(
            relative_mse_gain(
                fixed_metrics["mse_raw"], oracle_metrics["mse_raw"]
            )
        )
        values["oracle_relative_mse_gain_over_score"].append(
            relative_mse_gain(
                score_metrics["mse_raw"], oracle_metrics["mse_raw"]
            )
        )
        values["oracle_relative_mse_gain_over_constant"].append(
            relative_mse_gain(
                constant_metrics["mse_raw"],
                oracle_metrics["mse_raw"]
            )
        )
        for name, reference in (
                ("fixed", fixed_metrics),
                ("score", score_metrics),
                ("constant", constant_metrics)):
            delta = coef_delta(oracle_metrics, reference)
            values[f"oracle_delta_coef_over_{name}"].append(
                np.nan if delta is None else delta
            )
    return {
        name: bootstrap_summary(samples)
        for name, samples in values.items()
    }


def pairwise_complementarity(predictions, labels):
    squared_error = (predictions - labels[:, None]) ** 2
    report = {}
    for left in range(len(EXPERTS)):
        for right in range(left + 1, len(EXPERTS)):
            key = f"{EXPERTS[left]}__{EXPERTS[right]}"
            error_rho = safe_spearman(
                squared_error[:, left], squared_error[:, right]
            )
            report[key] = {
                "prediction_rho": safe_spearman(
                    predictions[:, left], predictions[:, right]
                ),
                "squared_error_rho": error_rho,
                "left_better_ratio": float(np.mean(
                    squared_error[:, left] < squared_error[:, right]
                )),
                "right_better_ratio": float(np.mean(
                    squared_error[:, right] < squared_error[:, left]
                ))
            }
    return report


def analyze_experts(
        train, calibration, mixture_step,
        score_bins, bootstrap, seed):
    affine_parameters, train_calibrated = fit_positive_affine(
        train["predictions"], train["labels"]
    )
    calibration_calibrated = apply_affine(
        calibration["predictions"], affine_parameters
    )
    weights, train_fixed, train_fixed_metrics = select_fixed_mixture(
        train_calibrated, train["labels"], mixture_step
    )
    calibration_fixed = apply_mixture(
        calibration_calibrated, weights
    )
    calibration_fixed_metrics = prediction_metrics(
        calibration_fixed, calibration["labels"]
    )
    score_parameters, train_score_control = fit_score_only_controller(
        train["predictions"][:, 0], train_calibrated,
        train["labels"], mixture_step, score_bins
    )
    calibration_score_control = apply_score_only_controller(
        calibration["predictions"][:, 0],
        calibration_calibrated, score_parameters
    )
    calibration_score_metrics = prediction_metrics(
        calibration_score_control, calibration["labels"]
    )
    determinism = score_determinism_control(
        train["predictions"][:, 0], train_calibrated,
        calibration["predictions"][:, 0],
        calibration_calibrated, calibration["labels"]
    )
    oracle = sample_oracle(
        calibration_calibrated,
        calibration["labels"], calibration["sample_ids"]
    )
    constant = constant_offset_oracle(
        calibration_calibrated,
        calibration["labels"], calibration["sample_ids"]
    )

    oracle_gain_fixed = relative_mse_gain(
        calibration_fixed_metrics["mse_raw"],
        oracle["metrics"]["mse_raw"]
    )
    oracle_gain_constant = relative_mse_gain(
        constant["metrics"]["mse_raw"],
        oracle["metrics"]["mse_raw"]
    )
    oracle_gain_score = relative_mse_gain(
        calibration_score_metrics["mse_raw"],
        oracle["metrics"]["mse_raw"]
    )
    oracle_delta_fixed = coef_delta(
        oracle["metrics"], calibration_fixed_metrics
    )
    oracle_delta_constant = coef_delta(
        oracle["metrics"], constant["metrics"]
    )
    oracle_delta_score = coef_delta(
        oracle["metrics"], calibration_score_metrics
    )
    nonconstant = {
        expert: bool(np.std(calibration_calibrated[:, index]) > 1e-8)
        for index, expert in enumerate(EXPERTS)
    }
    checks = {
        "oracle_mse_gain_over_fixed_at_least_2pct": bool(
            oracle_gain_fixed >= 0.02
        ),
        "oracle_coef_above_fixed": bool(
            oracle_delta_fixed is not None and oracle_delta_fixed > 0
        ),
        "oracle_mse_gain_over_score_at_least_2pct": bool(
            oracle_gain_score >= 0.02
        ),
        "oracle_coef_above_score": bool(
            oracle_delta_score is not None and oracle_delta_score > 0
        ),
        "oracle_mse_gain_over_constant_at_least_2pct": bool(
            oracle_gain_constant >= 0.02
        ),
        "oracle_coef_above_constant": bool(
            oracle_delta_constant is not None
            and oracle_delta_constant > 0
        ),
        "motion_strict_win_ratio_at_least_10pct": bool(
            oracle["strict_selection_ratio"]["motion"] >= 0.10
        ),
        "skeleton_strict_win_ratio_at_least_10pct": bool(
            oracle["strict_selection_ratio"]["skeleton"] >= 0.10
        ),
        "motion_has_non_score_component": determinism["motion"][
            "passes_non_score_component"
        ],
        "skeleton_has_non_score_component": determinism["skeleton"][
            "passes_non_score_component"
        ],
        "all_calibration_experts_nonconstant": bool(
            all(nonconstant.values())
        )
    }
    candidate = bool(all(checks.values()))

    calibration_single_metrics = metrics_by_expert(
        calibration_calibrated, calibration["labels"]
    )
    best_single = min(
        EXPERTS,
        key=lambda expert: calibration_single_metrics[expert]["mse_raw"]
    )
    return {
        "schema_version": 1,
        "protocol": (
            "Affine calibration and convex fixed-mixture weights are fit "
            "on train only. Score-only routing and score-determinism "
            "controls test whether apparent complementarity is just a "
            "function of the Base score. Calibration is evaluated once. "
            "Sample and constant-offset oracles are label-selected "
            "diagnostics only."
        ),
        "warnings": [
            "Oracle outputs are optimistic upper bounds and must never be "
            "used as gate inputs or reported as deployable performance.",
            "A positive candidate authorizes a gate experiment; it is not "
            "paper evidence and still requires group OOF confirmation."
        ],
        "sample_counts": {
            "train": int(len(train["labels"])),
            "calibration": int(len(calibration["labels"]))
        },
        "expert_order": list(EXPERTS),
        "raw_metrics": {
            "train": metrics_by_expert(
                train["predictions"], train["labels"]
            ),
            "calibration": metrics_by_expert(
                calibration["predictions"], calibration["labels"]
            )
        },
        "train_only_affine": {
            "parameters": affine_parameters,
            "train_metrics": metrics_by_expert(
                train_calibrated, train["labels"]
            ),
            "calibration_metrics": calibration_single_metrics
        },
        "fixed_mixture": {
            "selection_split": "train",
            "selection_metric": "MSE; Spearman breaks exact MSE ties",
            "mixture_step": float(mixture_step),
            "weights": {
                expert: float(weights[index])
                for index, expert in enumerate(EXPERTS)
            },
            "train_metrics": train_fixed_metrics,
            "calibration_metrics": calibration_fixed_metrics
        },
        "score_only_control": {
            "selection_split": "train",
            "input": "raw Base prediction only",
            "controller": "quantile-bin convex expert mixture",
            "parameters": score_parameters,
            "train_metrics": prediction_metrics(
                train_score_control, train["labels"]
            ),
            "calibration_metrics": calibration_score_metrics
        },
        "score_determinism_control": determinism,
        "calibration_best_single": {
            "expert": best_single,
            "metrics": calibration_single_metrics[best_single],
            "warning": "Label-selected calibration diagnostic only."
        },
        "sample_oracle": {
            key: value for key, value in oracle.items()
            if key != "prediction"
        },
        "constant_offset_oracle": {
            key: value for key, value in constant.items()
            if key != "prediction"
        },
        "pairwise_complementarity": pairwise_complementarity(
            calibration_calibrated, calibration["labels"]
        ),
        "headroom": {
            "oracle_relative_mse_gain_over_fixed_mixture":
                float(oracle_gain_fixed),
            "oracle_delta_coef_over_fixed_mixture": oracle_delta_fixed,
            "oracle_relative_mse_gain_over_score_only_control":
                float(oracle_gain_score),
            "oracle_delta_coef_over_score_only_control":
                oracle_delta_score,
            "oracle_relative_mse_gain_over_constant_offset_oracle":
                float(oracle_gain_constant),
            "oracle_delta_coef_over_constant_offset_oracle":
                oracle_delta_constant
        },
        "bootstrap": group_bootstrap(
            calibration["labels"], calibration["group_ids"],
            calibration_fixed, calibration_score_control,
            oracle["prediction"],
            calibration_calibrated, bootstrap, seed
        ),
        "preliminary_decisions": {
            "checks": checks,
            "calibration_expert_nonconstant": nonconstant,
            "candidate_for_gate": candidate,
            "next_step": (
                "train cue/score/cue_score gate with group OOF scorers"
                if candidate else
                "redesign residual experts; do not run OOF, PCS, or test"
            )
        }
    }


def write_merged_csv(table, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "sample_id", "group_id", "label",
            "pred_base", "pred_motion", "pred_skeleton"
        ))
        writer.writeheader()
        for row, sample_id in enumerate(table["sample_ids"]):
            writer.writerow({
                "sample_id": sample_id,
                "group_id": table["group_ids"][row],
                "label": table["labels"][row],
                "pred_base": table["predictions"][row, 0],
                "pred_motion": table["predictions"][row, 1],
                "pred_skeleton": table["predictions"][row, 2]
            })


def expect_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _write_test_csv(path, ids, groups, labels, predictions):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for values in zip(ids, groups, labels, predictions):
            writer.writerow(dict(zip(REQUIRED_COLUMNS, values)))


def _synthetic_table(prefix, labels, predictions):
    count = len(labels)
    return {
        "sample_ids": [f"{prefix}_sample_{index}" for index in range(count)],
        "group_ids": [f"{prefix}_group_{index}" for index in range(count)],
        "labels": np.asarray(labels, dtype=np.float64),
        "predictions": np.asarray(predictions, dtype=np.float64),
        "inputs": {}
    }


def run_self_test():
    rng = np.random.RandomState(7)
    train_labels = rng.uniform(0.1, 0.9, 120)
    cal_labels = rng.uniform(0.1, 0.9, 60)

    def complementary(labels):
        count = len(labels)
        split = np.arange(count) % 2 == 0
        base = labels + rng.normal(0, 0.10, count)
        motion = labels + np.where(
            split, rng.normal(0, 0.005, count),
            rng.normal(0, 0.16, count)
        )
        skeleton = labels + np.where(
            split, rng.normal(0, 0.16, count),
            rng.normal(0, 0.005, count)
        )
        return np.column_stack((base, motion, skeleton))

    train = _synthetic_table(
        "train", train_labels, complementary(train_labels)
    )
    calibration = _synthetic_table(
        "cal", cal_labels, complementary(cal_labels)
    )
    report = analyze_experts(
        train, calibration, mixture_step=0.05,
        score_bins=5, bootstrap=100, seed=0
    )
    assert report["preliminary_decisions"]["candidate_for_gate"]
    weights = report["fixed_mixture"]["weights"]
    assert all(value >= 0 for value in weights.values())
    assert np.isclose(sum(weights.values()), 1.0)
    json.dumps(report, allow_nan=False)

    shared_noise_train = rng.normal(0, 0.04, len(train_labels))
    shared_noise_cal = rng.normal(0, 0.04, len(cal_labels))
    negative_train = np.column_stack([
        train_labels + shared_noise_train + offset
        for offset in (0.0, 0.02, -0.02)
    ])
    negative_cal = np.column_stack([
        cal_labels + shared_noise_cal + offset
        for offset in (0.0, 0.02, -0.02)
    ])
    negative_report = analyze_experts(
        _synthetic_table("negative_train", train_labels, negative_train),
        _synthetic_table("negative_cal", cal_labels, negative_cal),
        mixture_step=0.05, score_bins=5, bootstrap=50, seed=0
    )
    assert not negative_report[
        "preliminary_decisions"
    ]["candidate_for_gate"]
    assert not negative_report["preliminary_decisions"]["checks"][
        "oracle_mse_gain_over_constant_at_least_2pct"
    ]

    score_rng = np.random.RandomState(0)
    score_train = score_rng.uniform(0.05, 0.95, 320)
    score_cal = score_rng.uniform(0.05, 0.95, 80)
    score_train_labels = np.clip(
        0.05 + 0.9 * score_train
        + 0.16 * np.sin(2 * np.pi * score_train)
        + score_rng.normal(0, 0.05, len(score_train)),
        0, 1
    )
    score_cal_labels = np.clip(
        0.05 + 0.9 * score_cal
        + 0.16 * np.sin(2 * np.pi * score_cal)
        + score_rng.normal(0, 0.05, len(score_cal)),
        0, 1
    )
    score_only_report = analyze_experts(
        _synthetic_table(
            "score_train", score_train_labels,
            np.column_stack((
                score_train, score_train ** 2, np.sqrt(score_train)
            ))
        ),
        _synthetic_table(
            "score_cal", score_cal_labels,
            np.column_stack((
                score_cal, score_cal ** 2, np.sqrt(score_cal)
            ))
        ),
        mixture_step=0.05, score_bins=5, bootstrap=50, seed=0
    )
    score_checks = score_only_report["preliminary_decisions"]["checks"]
    assert not score_only_report[
        "preliminary_decisions"
    ]["candidate_for_gate"]
    assert not score_checks["motion_has_non_score_component"]
    assert not score_checks["skeleton_has_non_score_component"]
    expect_value_error(lambda: validate_mixture_step(0.001))

    outside_train_scores = np.linspace(0.0, 1.0, 20)
    outside_cal_scores = np.linspace(2.0, 3.0, 10)
    outside_report = analyze_experts(
        _synthetic_table(
            "outside_train", outside_train_scores,
            np.column_stack((
                outside_train_scores,
                outside_train_scores ** 2,
                np.sqrt(outside_train_scores)
            ))
        ),
        _synthetic_table(
            "outside_cal", outside_cal_scores / 3.0,
            np.column_stack((
                outside_cal_scores,
                outside_cal_scores ** 2,
                np.sqrt(outside_cal_scores)
            ))
        ),
        mixture_step=0.05, score_bins=5, bootstrap=10, seed=0
    )
    assert not outside_report[
        "preliminary_decisions"
    ]["candidate_for_gate"]
    json.dumps(outside_report, allow_nan=False)

    with tempfile.TemporaryDirectory() as directory:
        ids = ["a", "b", "c"]
        groups = ["ga", "gb", "gc"]
        labels = [0.1, 0.2, 0.3]
        paths = {}
        for index, expert in enumerate(EXPERTS):
            order = [2, 0, 1] if expert == "motion" else [0, 1, 2]
            path = os.path.join(directory, f"{expert}.csv")
            _write_test_csv(
                path,
                [ids[row] for row in order],
                [groups[row] for row in order],
                [labels[row] for row in order],
                [labels[row] + index * 0.01 for row in order]
            )
            paths[expert] = path
        merged = merge_expert_tables(paths)
        assert merged["sample_ids"] == ids
        assert np.allclose(merged["predictions"][:, 1], [0.11, 0.21, 0.31])

        duplicate = os.path.join(directory, "duplicate.csv")
        _write_test_csv(
            duplicate, ["a", "a"], ["ga", "gb"],
            [0.1, 0.2], [0.1, 0.2]
        )
        expect_value_error(lambda: read_expert_csv(duplicate))

        nan_path = os.path.join(directory, "nan.csv")
        _write_test_csv(
            nan_path, ["a"], ["ga"], [0.1], [float("nan")]
        )
        expect_value_error(lambda: read_expert_csv(nan_path))

        malformed = os.path.join(directory, "malformed.csv")
        with open(malformed, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(REQUIRED_COLUMNS)
            writer.writerow(("a", "ga", 0.1))
        expect_value_error(lambda: read_expert_csv(malformed))

        mismatch = os.path.join(directory, "mismatch.csv")
        _write_test_csv(
            mismatch, ids, groups, [0.1, 0.2, 0.4], labels
        )
        mismatched_paths = dict(paths)
        mismatched_paths["skeleton"] = mismatch
        expect_value_error(
            lambda: merge_expert_tables(mismatched_paths)
        )

        overlap = dict(calibration)
        overlap["sample_ids"] = list(train["sample_ids"][:60])
        overlap["group_ids"] = list(train["group_ids"][:60])
        expect_value_error(
            lambda: validate_splits(train, overlap)
        )
    print("strict CSV merge and validation: passed")
    print("train-only affine and convex mixture: passed")
    print("complementary expert positive control: passed")
    print("constant-offset negative control: passed")
    print("score-transform negative control: passed")
    print("out-of-support score guard: passed")
    print("mixture-grid and malformed-CSV guards: passed")
    print("independent expert analyzer self-test passed")


def format_signed(value):
    return "undefined" if value is None else f"{value:+.3f}"


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    train_paths = {
        expert: getattr(args, f"{expert}_train") for expert in EXPERTS
    }
    calibration_paths = {
        expert: getattr(args, f"{expert}_cal") for expert in EXPERTS
    }
    train = merge_expert_tables(train_paths)
    calibration = merge_expert_tables(calibration_paths)
    validate_splits(
        train, calibration,
        args.expected_train_samples,
        args.expected_calibration_samples
    )
    report = analyze_experts(
        train, calibration, args.mixture_step,
        args.score_bins, args.bootstrap, args.seed
    )
    report["dataset"] = args.dataset
    report["inputs"] = {
        "train": train["inputs"],
        "calibration": calibration["inputs"]
    }

    os.makedirs(args.output_dir, exist_ok=True)
    train_output = os.path.join(
        args.output_dir, f"{args.dataset}_independent_train.csv"
    )
    calibration_output = os.path.join(
        args.output_dir,
        f"{args.dataset}_independent_calibration.csv"
    )
    report_output = os.path.join(
        args.output_dir, f"{args.dataset}_independent_headroom.json"
    )
    write_merged_csv(train, train_output)
    write_merged_csv(calibration, calibration_output)
    with open(report_output, "w", encoding="utf-8") as handle:
        json.dump(
            report, handle, indent=2,
            ensure_ascii=False, allow_nan=False
        )

    headroom = report["headroom"]
    print(
        "oracle_vs_fixed: "
        f"mse_gain={headroom['oracle_relative_mse_gain_over_fixed_mixture']:.2%} "
        "delta_coef="
        f"{format_signed(headroom['oracle_delta_coef_over_fixed_mixture'])}"
    )
    print(
        "oracle_vs_score_only: "
        "mse_gain="
        f"{headroom['oracle_relative_mse_gain_over_score_only_control']:.2%} "
        "delta_coef="
        f"{format_signed(headroom['oracle_delta_coef_over_score_only_control'])}"
    )
    print(
        "oracle_vs_constant: "
        f"mse_gain={headroom['oracle_relative_mse_gain_over_constant_offset_oracle']:.2%} "
        "delta_coef="
        f"{format_signed(headroom['oracle_delta_coef_over_constant_offset_oracle'])}"
    )
    print(
        "candidate_for_gate="
        f"{report['preliminary_decisions']['candidate_for_gate']}"
    )
    print(f"report={report_output}")


if __name__ == "__main__":
    main()
