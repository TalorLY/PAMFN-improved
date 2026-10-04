import argparse
import csv
import hashlib
import json
import os
import tempfile
import warnings

import numpy as np
from scipy.stats import spearmanr


VARIANTS = (
    "base",
    "motion_on",
    "motion_off",
    "skeleton_on",
    "skeleton_off",
)
CUES = ("motion", "skeleton")
DEFAULT_CUE_BOUNDS = {"motion": 0.015, "skeleton": 0.02}
REQUIRED_COLUMNS = ("sample_id", "group_id", "label", "prediction")
EPSILON = 1e-12
MAX_GATE_UNITS = 1000


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Separate cue-on gains into direct cue effects and checkpoint "
            "or co-adaptation components on a held-out calibration split"
        )
    )
    parser.add_argument("--dataset", default="TES")
    for split in ("train", "cal"):
        for variant in VARIANTS:
            parser.add_argument(f"--{variant}_{split}")
    parser.add_argument("--output")
    parser.add_argument("--expected_train_samples", type=int, default=320)
    parser.add_argument(
        "--expected_calibration_samples", type=int, default=80
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--gate_step", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    anchor_group = parser.add_mutually_exclusive_group()
    anchor_group.add_argument("--require_anchored", action="store_true")
    anchor_group.add_argument(
        "--require_bias_separated", action="store_true",
        help=(
            "require cue-off = raw Base + one train-fitted scalar shared "
            "across samples, splits, and cue checkpoints"
        )
    )
    anchor_group.add_argument(
        "--paired_motion_exposure", action="store_true",
        help=(
            "interpret Base as the matched off/off control and Motion "
            "on/off as endpoints from one paired-trained checkpoint"
        )
    )
    parser.add_argument("--anchor_atol", type=float, default=1e-6)
    parser.add_argument("--motion_bound", type=float, default=0.015)
    parser.add_argument("--skeleton_bound", type=float, default=0.02)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args

    required = [
        getattr(args, f"{variant}_{split}")
        for split in ("train", "cal")
        for variant in VARIANTS
    ] + [args.output]
    if any(not value for value in required):
        parser.error("ten prediction CSV paths and --output are required")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    validate_gate_step(args.gate_step)
    if args.paired_motion_exposure and not np.isclose(
            args.gate_step, 1.0, atol=1e-12, rtol=0):
        parser.error(
            "--paired_motion_exposure requires --gate_step 1; only the "
            "pre-registered endpoints 0 and 1 are allowed"
        )
    if args.expected_train_samples < 0:
        parser.error("--expected_train_samples must be non-negative")
    if args.expected_calibration_samples < 0:
        parser.error(
            "--expected_calibration_samples must be non-negative"
        )
    if not np.isfinite(args.anchor_atol) or args.anchor_atol < 0:
        parser.error("--anchor_atol must be finite and non-negative")
    for name, value in (
            ("--motion_bound", args.motion_bound),
            ("--skeleton_bound", args.skeleton_bound)):
        if not np.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if not args.dataset or any(
            token in args.dataset for token in ("/", "\\")):
        parser.error("--dataset must be a non-empty file-name token")
    return args


def validate_gate_step(step):
    if not np.isfinite(step) or not 0 < step <= 1:
        raise ValueError("gate_step must be finite and in (0, 1]")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0, atol=1e-10, rtol=0):
        raise ValueError("1 / gate_step must be an integer")
    if units > MAX_GATE_UNITS:
        raise ValueError(
            "gate_step creates too many candidates; use a step of at "
            f"least {1.0 / MAX_GATE_UNITS:g}"
        )
    return units


def gate_grid(step):
    units = validate_gate_step(step)
    return np.linspace(0.0, 1.0, units + 1, dtype=np.float64)


def fit_best_gated_constant(off, labels, bound):
    """Fit c on train for the oracle min(e^2, (e-c)^2), g in {0,1}."""
    errors = np.asarray(labels, dtype=np.float64) - np.asarray(
        off, dtype=np.float64
    )
    if not np.isfinite(errors).all() or not np.isfinite(bound) or bound <= 0:
        raise ValueError("gated constant fit requires finite values and bound")
    candidates = {0.0, -float(bound), float(bound)}
    positive = np.sort(errors[errors > 0])[::-1]
    negative = np.sort(errors[errors < 0])
    for values in (positive, negative):
        if len(values):
            means = np.cumsum(values) / np.arange(1, len(values) + 1)
            candidates.update(
                float(np.clip(value, -bound, bound)) for value in means
            )
    candidates.update(
        float(np.clip(2.0 * value, -bound, bound)) for value in errors
    )

    def objective(value):
        return float(np.mean(np.minimum(
            errors ** 2,
            (errors - value) ** 2,
        )))

    return min(
        candidates,
        key=lambda value: (objective(value), abs(value), value),
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_prediction_csv(path):
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
            raise ValueError(f"{path}:{row_number} has a malformed row")
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
    return {
        "path": os.path.abspath(path),
        "sha256": file_sha256(path),
        "sample_ids": sample_ids,
        "group_ids": group_ids,
        "labels": np.asarray(labels, dtype=np.float64),
        "predictions": np.asarray(predictions, dtype=np.float64),
    }


def merge_prediction_tables(paths):
    tables = {
        variant: read_prediction_csv(paths[variant])
        for variant in VARIANTS
    }
    reference = tables["base"]
    reference_ids = reference["sample_ids"]
    reference_id_set = set(reference_ids)
    predictions = []

    for variant in VARIANTS:
        table = tables[variant]
        if set(table["sample_ids"]) != reference_id_set:
            raise ValueError(
                f"{variant} CSV has a different sample ID set"
            )
        index_by_id = {
            sample_id: index
            for index, sample_id in enumerate(table["sample_ids"])
        }
        indices = [index_by_id[sample_id] for sample_id in reference_ids]
        groups = [table["group_ids"][index] for index in indices]
        labels = table["labels"][indices]
        if groups != reference["group_ids"]:
            raise ValueError(f"{variant} CSV has mismatched group IDs")
        if not np.allclose(
                labels, reference["labels"], atol=1e-7, rtol=0):
            raise ValueError(f"{variant} CSV has mismatched labels")
        predictions.append(table["predictions"][indices])

    return {
        "sample_ids": list(reference_ids),
        "group_ids": list(reference["group_ids"]),
        "labels": reference["labels"].copy(),
        "predictions": np.column_stack(predictions),
        "inputs": {
            variant: {
                "path": tables[variant]["path"],
                "sha256": tables[variant]["sha256"],
            }
            for variant in VARIANTS
        },
    }


def validate_splits(
        train, calibration, expected_train=0,
        expected_calibration=0):
    if expected_train and len(train["labels"]) != expected_train:
        raise ValueError(
            f"expected {expected_train} train samples, got "
            f"{len(train['labels'])}"
        )
    if expected_calibration and (
            len(calibration["labels"]) != expected_calibration):
        raise ValueError(
            f"expected {expected_calibration} calibration samples, got "
            f"{len(calibration['labels'])}"
        )
    if set(train["sample_ids"]) & set(calibration["sample_ids"]):
        raise ValueError("train and calibration sample IDs overlap")
    if set(train["group_ids"]) & set(calibration["group_ids"]):
        raise ValueError("train and calibration group IDs overlap")
    for split_name, table in (
            ("train", train), ("calibration", calibration)):
        if np.std(table["labels"]) <= EPSILON:
            raise ValueError(f"{split_name} labels are constant")
        expected_shape = (len(table["labels"]), len(VARIANTS))
        if table["predictions"].shape != expected_shape:
            raise ValueError(
                f"{split_name} prediction shape is invalid"
            )


def anchor_parity(table, atol):
    base = table["predictions"][:, VARIANTS.index("base")]
    motion_off = table["predictions"][:, VARIANTS.index("motion_off")]
    skeleton_off = table[
        "predictions"
    ][:, VARIANTS.index("skeleton_off")]
    differences = {
        "motion_off_vs_base": float(np.max(np.abs(motion_off - base))),
        "skeleton_off_vs_base": float(
            np.max(np.abs(skeleton_off - base))
        ),
        "motion_off_vs_skeleton_off": float(
            np.max(np.abs(motion_off - skeleton_off))
        ),
    }
    return {
        "anchor_atol": float(atol),
        "max_abs_differences": differences,
        "passed": bool(all(value <= atol for value in differences.values())),
    }


def bias_separated_anchor_parity(train, calibration, atol):
    base_index = VARIANTS.index("base")
    off_indices = {
        cue: VARIANTS.index(f"{cue}_off") for cue in CUES
    }

    split_reports = {}
    split_biases = {}
    for split_name, table in (
            ("train", train), ("calibration", calibration)):
        base = table["predictions"][:, base_index]
        cue_biases = {
            cue: table["predictions"][:, index] - base
            for cue, index in off_indices.items()
        }
        means = {
            cue: float(np.mean(values))
            for cue, values in cue_biases.items()
        }
        deviations = {
            cue: float(np.max(np.abs(values - means[cue])))
            for cue, values in cue_biases.items()
        }
        cue_difference = float(np.max(np.abs(
            cue_biases["motion"] - cue_biases["skeleton"]
        )))
        split_reports[split_name] = {
            "cue_bias_means": means,
            "max_abs_deviation_from_split_mean": deviations,
            "max_abs_motion_vs_skeleton_off_difference": cue_difference,
            "passed": bool(
                all(value <= atol for value in deviations.values())
                and cue_difference <= atol
            ),
        }
        split_biases[split_name] = float(np.mean(list(means.values())))

    cross_split_difference = abs(
        split_biases["train"] - split_biases["calibration"]
    )
    train_base = train["predictions"][:, base_index]
    expected_train_fitted_bias = float(np.mean(
        train["labels"] - train_base
    ))
    train_fit_abs_error = abs(
        split_biases["train"] - expected_train_fitted_bias
    )
    return {
        "mode": "bias_separated_shared_scalar",
        "anchor_atol": float(atol),
        "train": split_reports["train"],
        "calibration": split_reports["calibration"],
        "shared_bias": split_biases,
        "expected_train_fitted_bias": expected_train_fitted_bias,
        "train_fitted_bias_abs_error": float(train_fit_abs_error),
        "cross_split_abs_difference": float(cross_split_difference),
        "passed": bool(
            split_reports["train"]["passed"]
            and split_reports["calibration"]["passed"]
            and cross_split_difference <= atol
            and train_fit_abs_error <= atol
        ),
    }
def safe_spearman(predictions, labels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
        "prediction_std": float(np.std(predictions)),
    }


def metrics_by_variant(predictions, labels):
    return {
        variant: prediction_metrics(predictions[:, index], labels)
        for index, variant in enumerate(VARIANTS)
    }


def fit_positive_affine(predictions, labels):
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    centered = predictions - np.mean(predictions)
    variance = float(np.mean(centered ** 2))
    if variance <= EPSILON:
        slope = 0.0
    else:
        slope = max(0.0, float(np.mean(
            centered * (labels - np.mean(labels))
        ) / variance))
    intercept = float(np.mean(labels) - slope * np.mean(predictions))
    return slope, intercept


def calibration_controls(train, calibration):
    base_index = VARIANTS.index("base")
    train_base = train["predictions"][:, base_index]
    cal_base = calibration["predictions"][:, base_index]
    train_labels = train["labels"]
    cal_labels = calibration["labels"]
    base_train_metrics = prediction_metrics(train_base, train_labels)
    base_cal_metrics = prediction_metrics(cal_base, cal_labels)

    scalar_bias = float(np.mean(train_labels - train_base))
    scalar_train = train_base + scalar_bias
    scalar_cal = cal_base + scalar_bias
    scalar_train_metrics = prediction_metrics(scalar_train, train_labels)
    scalar_cal_metrics = prediction_metrics(scalar_cal, cal_labels)

    slope, intercept = fit_positive_affine(train_base, train_labels)
    affine_train = slope * train_base + intercept
    affine_cal = slope * cal_base + intercept
    affine_train_metrics = prediction_metrics(affine_train, train_labels)
    affine_cal_metrics = prediction_metrics(affine_cal, cal_labels)
    return {
        "fit_split": "train",
        "raw_base": {
            "train_metrics": base_train_metrics,
            "calibration_metrics": base_cal_metrics,
        },
        "train_scalar_bias": {
            "bias": scalar_bias,
            "train_metrics": scalar_train_metrics,
            "calibration_metrics": scalar_cal_metrics,
            "calibration_comparison_to_raw_base": comparison(
                scalar_cal_metrics, base_cal_metrics
            ),
        },
        "train_positive_affine": {
            "slope": slope,
            "intercept": intercept,
            "train_metrics": affine_train_metrics,
            "calibration_metrics": affine_cal_metrics,
            "calibration_comparison_to_raw_base": comparison(
                affine_cal_metrics, base_cal_metrics
            ),
        },
    }


def coef_delta(candidate, reference):
    if candidate["coef"] is None or reference["coef"] is None:
        return None
    return float(candidate["coef"] - reference["coef"])


def relative_mse_gain(reference_mse, candidate_mse):
    denominator = max(float(reference_mse), EPSILON)
    return float((reference_mse - candidate_mse) / denominator)


def comparison(candidate, reference):
    return {
        "delta_coef": coef_delta(candidate, reference),
        "relative_mse_gain": relative_mse_gain(
            reference["mse_raw"], candidate["mse_raw"]
        ),
        "delta_mse_raw": float(
            candidate["mse_raw"] - reference["mse_raw"]
        ),
    }


def effect_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "mean_abs": float(np.mean(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values ** 2))),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def ratio_or_none(numerator, denominator):
    if denominator <= EPSILON:
        return None
    return float(numerator / denominator)


def gate_candidates(off, direct, grid):
    return off[:, None] + direct[:, None] * grid[None, :]


def select_fixed_gate(off, direct, labels, grid):
    candidates = gate_candidates(off, direct, grid)
    mse = np.mean((candidates - labels[:, None]) ** 2, axis=0)
    selected = int(np.argmin(mse))
    prediction = candidates[:, selected]
    return {
        "gate": float(grid[selected]),
        "grid_index": selected,
        "prediction": prediction,
        "metrics": prediction_metrics(prediction, labels),
    }


def sample_gate_oracle(
        off, direct, labels, sample_ids, grid, name="direct"):
    candidates = gate_candidates(off, direct, grid)
    squared_error = (candidates - labels[:, None]) ** 2
    minimum = np.min(squared_error, axis=1, keepdims=True)
    ties = np.isclose(squared_error, minimum, atol=EPSILON, rtol=0)
    selected = np.argmin(squared_error, axis=1)
    prediction = candidates[np.arange(len(labels)), selected]
    selected_gates = grid[selected]
    strict = np.sum(ties, axis=1) == 1
    unique_indices, counts = np.unique(selected, return_counts=True)
    return {
        "prediction": prediction,
        "metrics": prediction_metrics(prediction, labels),
        "candidate": name,
        "gate_histogram": {
            f"{grid[index]:.12g}": int(count)
            for index, count in zip(unique_indices, counts)
        },
        "gate_mean": float(np.mean(selected_gates)),
        "gate_std": float(np.std(selected_gates)),
        "off_endpoint_ratio": float(np.mean(selected == 0)),
        "on_endpoint_ratio": float(np.mean(selected == len(grid) - 1)),
        "interior_ratio": float(np.mean(
            (selected > 0) & (selected < len(grid) - 1)
        )),
        "strict_selection_ratio": float(np.mean(strict)),
        "strict_winning_sample_ids": [
            sample_ids[index] for index in np.flatnonzero(strict)
        ],
        "tie_ratio": float(np.mean(np.sum(ties, axis=1) > 1)),
    }


def percentile_interval(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    lower, upper = np.percentile(values, [2.5, 97.5])
    return [float(lower), float(upper)]


def bootstrap_statistic(values, threshold):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "95ci": percentile_interval(values),
        "positive_probability": (
            float(np.mean(values > 0)) if len(values) else None
        ),
        "criterion_threshold": float(threshold),
        "probability_meeting_criterion": (
            float(np.mean(values >= threshold)) if len(values) else None
        ),
        "finite_replicates": int(len(values)),
    }


def group_bootstrap(
        labels, group_ids, base, on, off, fixed_gate_prediction,
        train_mean_direct, best_train_gated_constant, grid, iterations, seed):
    groups = list(dict.fromkeys(group_ids))
    group_array = np.asarray(group_ids, dtype=object)
    group_rows = {
        group: np.flatnonzero(group_array == group) for group in groups
    }
    rng = np.random.RandomState(seed)
    values = {
        "cue_off_delta_coef_over_base": [],
        "cue_off_relative_mse_gain_over_base": [],
        "cue_on_delta_coef_over_off": [],
        "cue_on_relative_mse_gain_over_off": [],
        "fixed_gate_delta_coef_over_off": [],
        "fixed_gate_relative_mse_gain_over_off": [],
        "oracle_relative_mse_gain_over_fixed_gate": [],
        "oracle_delta_coef_over_fixed_gate": [],
        "oracle_relative_mse_gain_over_constant_offset_oracle": [],
        "oracle_delta_coef_over_constant_offset_oracle": [],
        "oracle_relative_mse_gain_over_best_train_gated_constant_oracle": [],
        "oracle_delta_coef_over_best_train_gated_constant_oracle": [],
        "direct_std_over_total_std": [],
    }
    for _ in range(iterations):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([
            group_rows[group] for group in sampled_groups
        ])
        sampled_labels = labels[indices]
        sampled_on = on[indices]
        sampled_off = off[indices]
        sampled_base = base[indices]
        sampled_fixed = fixed_gate_prediction[indices]
        on_metrics = prediction_metrics(sampled_on, sampled_labels)
        off_metrics = prediction_metrics(sampled_off, sampled_labels)
        base_metrics = prediction_metrics(sampled_base, sampled_labels)
        fixed_metrics = prediction_metrics(sampled_fixed, sampled_labels)
        direct = sampled_on - sampled_off
        oracle = sample_gate_oracle(
            sampled_off,
            direct,
            sampled_labels,
            [str(index) for index in range(len(indices))],
            grid,
        )
        constant_oracle = sample_gate_oracle(
            sampled_off,
            np.full(len(indices), train_mean_direct),
            sampled_labels,
            [str(index) for index in range(len(indices))],
            grid,
            name="train_mean_direct_constant",
        )
        best_constant_oracle = sample_gate_oracle(
            sampled_off,
            np.full(len(indices), best_train_gated_constant),
            sampled_labels,
            [str(index) for index in range(len(indices))],
            grid,
            name="best_train_gated_constant",
        )
        delta = coef_delta(on_metrics, off_metrics)
        off_delta = coef_delta(off_metrics, base_metrics)
        values["cue_off_delta_coef_over_base"].append(
            np.nan if off_delta is None else off_delta
        )
        values["cue_off_relative_mse_gain_over_base"].append(
            relative_mse_gain(
                base_metrics["mse_raw"], off_metrics["mse_raw"]
            )
        )
        values["cue_on_delta_coef_over_off"].append(
            np.nan if delta is None else delta
        )
        values["cue_on_relative_mse_gain_over_off"].append(
            relative_mse_gain(
                off_metrics["mse_raw"], on_metrics["mse_raw"]
            )
        )
        fixed_delta = coef_delta(fixed_metrics, off_metrics)
        values["fixed_gate_delta_coef_over_off"].append(
            np.nan if fixed_delta is None else fixed_delta
        )
        values["fixed_gate_relative_mse_gain_over_off"].append(
            relative_mse_gain(
                off_metrics["mse_raw"], fixed_metrics["mse_raw"]
            )
        )
        values["oracle_relative_mse_gain_over_fixed_gate"].append(
            relative_mse_gain(
                fixed_metrics["mse_raw"],
                oracle["metrics"]["mse_raw"],
            )
        )
        oracle_delta = coef_delta(oracle["metrics"], fixed_metrics)
        values["oracle_delta_coef_over_fixed_gate"].append(
            np.nan if oracle_delta is None else oracle_delta
        )
        values[
            "oracle_relative_mse_gain_over_constant_offset_oracle"
        ].append(relative_mse_gain(
            constant_oracle["metrics"]["mse_raw"],
            oracle["metrics"]["mse_raw"],
        ))
        constant_delta = coef_delta(
            oracle["metrics"], constant_oracle["metrics"]
        )
        values[
            "oracle_delta_coef_over_constant_offset_oracle"
        ].append(np.nan if constant_delta is None else constant_delta)
        values[
            "oracle_relative_mse_gain_over_best_train_gated_constant_oracle"
        ].append(relative_mse_gain(
            best_constant_oracle["metrics"]["mse_raw"],
            oracle["metrics"]["mse_raw"],
        ))
        best_constant_delta = coef_delta(
            oracle["metrics"], best_constant_oracle["metrics"]
        )
        values[
            "oracle_delta_coef_over_best_train_gated_constant_oracle"
        ].append(
            np.nan if best_constant_delta is None else best_constant_delta
        )
        direct_std = float(np.std(sampled_on - sampled_off))
        total_std = float(np.std(sampled_on - sampled_base))
        ratio = ratio_or_none(direct_std, total_std)
        values["direct_std_over_total_std"].append(
            np.nan if ratio is None else ratio
        )

    thresholds = {
        "cue_off_delta_coef_over_base": -0.5,
        "cue_off_relative_mse_gain_over_base": -0.02,
        "cue_on_delta_coef_over_off": 0.5,
        "cue_on_relative_mse_gain_over_off": 0.0,
        "fixed_gate_delta_coef_over_off": 0.5,
        "fixed_gate_relative_mse_gain_over_off": -0.02,
        "oracle_relative_mse_gain_over_fixed_gate": 0.02,
        "oracle_delta_coef_over_fixed_gate": 0.0,
        "oracle_relative_mse_gain_over_constant_offset_oracle": 0.02,
        "oracle_delta_coef_over_constant_offset_oracle": 0.0,
        "oracle_relative_mse_gain_over_best_train_gated_constant_oracle": 0.02,
        "oracle_delta_coef_over_best_train_gated_constant_oracle": 0.0,
        "direct_std_over_total_std": 0.10,
    }
    return {
        name: bootstrap_statistic(samples, thresholds[name])
        for name, samples in values.items()
    }


def analyze_cue(
        cue, train, calibration, gate_step, bootstrap, seed,
        residual_bound, require_bias_separated=False):
    base_index = VARIANTS.index("base")
    on_index = VARIANTS.index(f"{cue}_on")
    off_index = VARIANTS.index(f"{cue}_off")

    train_base = train["predictions"][:, base_index]
    train_on = train["predictions"][:, on_index]
    train_off = train["predictions"][:, off_index]
    cal_base = calibration["predictions"][:, base_index]
    cal_on = calibration["predictions"][:, on_index]
    cal_off = calibration["predictions"][:, off_index]

    train_metrics = {
        "base": prediction_metrics(train_base, train["labels"]),
        "on": prediction_metrics(train_on, train["labels"]),
        "off": prediction_metrics(train_off, train["labels"]),
    }
    cal_metrics = {
        "base": prediction_metrics(cal_base, calibration["labels"]),
        "on": prediction_metrics(cal_on, calibration["labels"]),
        "off": prediction_metrics(cal_off, calibration["labels"]),
    }

    grid = gate_grid(gate_step)
    train_direct = train_on - train_off
    fixed_gate = select_fixed_gate(
        train_off, train_direct, train["labels"], grid
    )
    cal_fixed = cal_off + fixed_gate["gate"] * (cal_on - cal_off)
    cal_fixed_metrics = prediction_metrics(cal_fixed, calibration["labels"])

    total = cal_on - cal_base
    direct = cal_on - cal_off
    retrain = cal_off - cal_base
    total_summary = effect_summary(total)
    direct_summary = effect_summary(direct)
    retrain_summary = effect_summary(retrain)
    direct_std_ratio = ratio_or_none(
        direct_summary["std"], total_summary["std"]
    )
    direct_retrain_rho = safe_spearman(direct, retrain)
    direct_retrain_covariance = float(np.mean(
        (direct - np.mean(direct)) * (retrain - np.mean(retrain))
    ))

    identity_error = float(np.max(np.abs(total - direct - retrain)))
    if identity_error > 1e-10:
        raise AssertionError(
            "direct/checkpoint decomposition identity failed"
        )

    oracle = sample_gate_oracle(
        cal_off,
        direct,
        calibration["labels"],
        calibration["sample_ids"],
        grid,
    )
    train_mean_direct = float(np.mean(train_direct))
    constant_fixed_prediction = (
        cal_off + fixed_gate["gate"] * train_mean_direct
    )
    constant_fixed_metrics = prediction_metrics(
        constant_fixed_prediction, calibration["labels"]
    )
    best_train_always_on_constant = float(np.clip(
        np.mean(train["labels"] - train_off),
        -residual_bound,
        residual_bound,
    ))
    best_always_on_prediction = cal_off + best_train_always_on_constant
    best_always_on_metrics = prediction_metrics(
        best_always_on_prediction, calibration["labels"]
    )
    best_train_gated_constant = fit_best_gated_constant(
        train_off, train["labels"], residual_bound
    )
    constant_oracle = sample_gate_oracle(
        cal_off,
        np.full(len(cal_off), train_mean_direct),
        calibration["labels"],
        calibration["sample_ids"],
        grid,
        name="train_mean_direct_constant",
    )
    best_constant_oracle = sample_gate_oracle(
        cal_off,
        np.full(len(cal_off), best_train_gated_constant),
        calibration["labels"],
        calibration["sample_ids"],
        grid,
        name="best_train_gated_constant",
    )
    on_vs_off = comparison(cal_metrics["on"], cal_metrics["off"])
    fixed_vs_off = comparison(cal_fixed_metrics, cal_metrics["off"])
    oracle_vs_fixed = comparison(
        oracle["metrics"], cal_fixed_metrics
    )
    oracle_vs_constant = comparison(
        oracle["metrics"], constant_oracle["metrics"]
    )
    oracle_vs_best_constant = comparison(
        oracle["metrics"], best_constant_oracle["metrics"]
    )
    oracle_vs_off = comparison(
        oracle["metrics"], cal_metrics["off"]
    )
    fixed_vs_train_mean_constant = comparison(
        cal_fixed_metrics, constant_fixed_metrics
    )
    fixed_vs_best_always_on = comparison(
        cal_fixed_metrics, best_always_on_metrics
    )
    bootstrap_report = group_bootstrap(
        calibration["labels"], calibration["group_ids"],
        cal_base, cal_on, cal_off, cal_fixed,
        train_mean_direct, best_train_gated_constant, grid, bootstrap, seed
    )
    average_checks = {
        "train_locked_fixed_gate_delta_coef_over_off_at_least_0_5": bool(
            fixed_vs_off["delta_coef"] is not None
            and fixed_vs_off["delta_coef"] >= 0.5
        ),
        "train_locked_fixed_gate_nonzero": bool(
            fixed_gate["gate"] > 0
        ),
        "train_locked_fixed_gate_mse_not_worse_than_off_by_over_2pct": bool(
            fixed_vs_off["relative_mse_gain"] >= -0.02
        ),
        "bootstrap_fixed_delta_coef_at_least_0_5_probability_at_least_90pct":
            bool(bootstrap_report[
                "fixed_gate_delta_coef_over_off"
            ]["probability_meeting_criterion"] is not None
            and bootstrap_report[
                "fixed_gate_delta_coef_over_off"
            ]["probability_meeting_criterion"] >= 0.90),
        "bootstrap_fixed_mse_not_worse_than_minus_2pct_probability_at_least_90pct":
            bool(bootstrap_report[
                "fixed_gate_relative_mse_gain_over_off"
            ]["probability_meeting_criterion"] is not None
            and bootstrap_report[
                "fixed_gate_relative_mse_gain_over_off"
            ]["probability_meeting_criterion"] >= 0.90),
    }
    residual_diagnostics = {
        "bound": float(residual_bound),
        "train": effect_summary(train_direct),
        "calibration": direct_summary,
        "train_mean_over_bound": float(
            train_mean_direct / residual_bound
        ),
        "calibration_mean_over_bound": float(
            direct_summary["mean"] / residual_bound
        ),
        "calibration_std_over_bound": float(
            direct_summary["std"] / residual_bound
        ),
        "calibration_abs_at_least_95pct_bound_ratio": float(np.mean(
            np.abs(direct) >= 0.95 * residual_bound
        )),
        "train_residual_vs_required_correction_spearman": safe_spearman(
            train_direct, train["labels"] - train_off
        ),
        "calibration_residual_vs_required_correction_spearman": (
            safe_spearman(
                direct, calibration["labels"] - cal_off
            )
        ),
    }
    if require_bias_separated:
        average_checks.update({
            "train_residual_mean_within_10pct_bound": bool(
                abs(train_mean_direct) <= 0.10 * residual_bound
            ),
            "calibration_residual_std_at_least_10pct_bound": bool(
                direct_summary["std"] >= 0.10 * residual_bound
            ),
            "calibration_bound_saturation_below_80pct": bool(
                residual_diagnostics[
                    "calibration_abs_at_least_95pct_bound_ratio"
                ] <= 0.80
            ),
        })
    dynamic_checks = {
        "sample_oracle_mse_gain_over_fixed_gate_at_least_2pct":
            bool(oracle_vs_fixed["relative_mse_gain"] >= 0.02),
        "sample_oracle_mse_gain_over_constant_offset_at_least_2pct":
            bool(oracle_vs_constant["relative_mse_gain"] >= 0.02),
        "sample_oracle_delta_coef_over_fixed_gate_positive": bool(
            oracle_vs_fixed["delta_coef"] is not None
            and oracle_vs_fixed["delta_coef"] > 0
        ),
        "sample_oracle_delta_coef_over_constant_offset_positive": bool(
            oracle_vs_constant["delta_coef"] is not None
            and oracle_vs_constant["delta_coef"] > 0
        ),
        "sample_oracle_mse_gain_over_best_train_gated_constant_oracle_at_least_2pct":
            bool(oracle_vs_best_constant["relative_mse_gain"] >= 0.02),
        "sample_oracle_delta_coef_over_best_train_gated_constant_oracle_positive":
            bool(
                oracle_vs_best_constant["delta_coef"] is not None
                and oracle_vs_best_constant["delta_coef"] > 0
            ),
        "direct_effect_nonconstant": bool(
            direct_summary["std"] > 1e-8
        ),
        "sample_oracle_gate_nonconstant": bool(
            oracle["gate_std"] > 1e-8
        ),
        "bootstrap_oracle_mse_gain_over_fixed_2pct_probability_at_least_90pct":
            bool(bootstrap_report[
                "oracle_relative_mse_gain_over_fixed_gate"
            ]["probability_meeting_criterion"] is not None
            and bootstrap_report[
                "oracle_relative_mse_gain_over_fixed_gate"
            ]["probability_meeting_criterion"] >= 0.90),
        "bootstrap_oracle_mse_gain_over_constant_2pct_probability_at_least_90pct":
            bool(bootstrap_report[
                "oracle_relative_mse_gain_over_constant_offset_oracle"
            ]["probability_meeting_criterion"] is not None
            and bootstrap_report[
                "oracle_relative_mse_gain_over_constant_offset_oracle"
            ]["probability_meeting_criterion"] >= 0.90),
        "bootstrap_oracle_delta_coef_over_fixed_positive_probability_at_least_90pct":
            bool(bootstrap_report[
                "oracle_delta_coef_over_fixed_gate"
            ]["positive_probability"] is not None and bootstrap_report[
                "oracle_delta_coef_over_fixed_gate"
            ]["positive_probability"] >= 0.90),
        "bootstrap_oracle_delta_coef_over_constant_positive_probability_at_least_90pct":
            bool(bootstrap_report[
                "oracle_delta_coef_over_constant_offset_oracle"
            ]["positive_probability"] is not None and bootstrap_report[
                "oracle_delta_coef_over_constant_offset_oracle"
            ]["positive_probability"] >= 0.90),
        "bootstrap_oracle_mse_gain_over_best_train_gated_constant_2pct_probability_at_least_90pct":
            bool(bootstrap_report[
                "oracle_relative_mse_gain_over_best_train_gated_constant_oracle"
            ]["probability_meeting_criterion"] is not None
            and bootstrap_report[
                "oracle_relative_mse_gain_over_best_train_gated_constant_oracle"
            ]["probability_meeting_criterion"] >= 0.90),
        "bootstrap_oracle_delta_coef_over_best_train_gated_constant_positive_probability_at_least_90pct":
            bool(bootstrap_report[
                "oracle_delta_coef_over_best_train_gated_constant_oracle"
            ]["positive_probability"] is not None and bootstrap_report[
                "oracle_delta_coef_over_best_train_gated_constant_oracle"
            ]["positive_probability"] >= 0.90),
    }
    average_supported = bool(all(average_checks.values()))
    dynamic_supported = bool(all(dynamic_checks.values()))
    supported = bool(average_supported and dynamic_supported)

    return {
        "variant_metrics": {
            "train": train_metrics,
            "calibration": cal_metrics,
        },
        "cue_on_vs_off": {
            "train": comparison(
                train_metrics["on"], train_metrics["off"]
            ),
            "calibration": on_vs_off,
        },
        "decomposition": {
            "identity": "total = direct + checkpoint_component",
            "definitions": {
                "total": "cue_on - reference_base",
                "direct": "cue_on - same_checkpoint_cue_off",
                "retrain": (
                    "same_checkpoint_cue_off - "
                    "reference_base; legacy field name for the "
                    "checkpoint/co-adaptation component"
                ),
            },
            "calibration_effects": {
                "total": total_summary,
                "direct": direct_summary,
                "retrain": retrain_summary,
            },
            "max_abs_identity_error": identity_error,
            "direct_std_over_total_std": direct_std_ratio,
            "direct_retrain_spearman": direct_retrain_rho,
            "direct_retrain_covariance": direct_retrain_covariance,
        },
        "fixed_gate": {
            "selection_split": "train",
            "selection_metric": "MSE; exact ties prefer smaller gate",
            "grid_step": float(gate_step),
            "gate": fixed_gate["gate"],
            "train_metrics": fixed_gate["metrics"],
            "calibration_metrics": cal_fixed_metrics,
            "calibration_comparison_to_off": fixed_vs_off,
        },
        "constant_controls": {
            "train_mean_direct_constant_at_fixed_gate": {
                "constant": train_mean_direct,
                "calibration_metrics": constant_fixed_metrics,
                "cue_fixed_comparison_to_constant": (
                    fixed_vs_train_mean_constant
                ),
            },
            "best_train_always_on_constant_within_cue_bound": {
                "constant": best_train_always_on_constant,
                "calibration_metrics": best_always_on_metrics,
                "cue_fixed_comparison_to_constant": (
                    fixed_vs_best_always_on
                ),
            },
        },
        "residual_diagnostics": residual_diagnostics,
        "sample_gate_oracle": {
            "oracle_upper_bound_not_deployable": True,
            **{
                key: value for key, value in oracle.items()
                if key != "prediction"
            },
        },
        "constant_offset_oracle": {
            "oracle_upper_bound_not_deployable": True,
            "train_mean_direct": train_mean_direct,
            **{
                key: value for key, value in constant_oracle.items()
                if key != "prediction"
            },
        },
        "best_train_fitted_gated_constant_oracle": {
            "oracle_upper_bound_not_deployable": True,
            "fit_split": "train",
            "fit_objective": "mean min(e^2, (e-c)^2), gate in {0,1}",
            "best_train_gated_constant": best_train_gated_constant,
            **{
                key: value for key, value in best_constant_oracle.items()
                if key != "prediction"
            },
        },
        "headroom": {
            "oracle_over_fixed_gate": oracle_vs_fixed,
            "oracle_over_constant_offset_oracle": oracle_vs_constant,
            "oracle_over_best_train_fitted_gated_constant_oracle": (
                oracle_vs_best_constant
            ),
            "oracle_over_cue_off": oracle_vs_off,
        },
        "bootstrap": bootstrap_report,
        "preliminary_decision": {
            "average_effect_checks": average_checks,
            "dynamic_gate_checks": dynamic_checks,
            "average_effect_supported": average_supported,
            "fixed_cue_supported": average_supported,
            "oracle_headroom_supported": dynamic_supported,
            "dynamic_gate_supported": dynamic_supported,
            "direct_effect_supported": supported,
            "interpretation": (
                "The inference-time intervention on this fixed cue-trained "
                "checkpoint is large and heterogeneous enough to justify "
                "an OOF gate experiment."
                if supported else
                (
                    "The cue has a stable average effect but no evidence of "
                    "sample-level routing headroom; keep it fixed, not gated."
                    if average_supported else
                    "The cue lacks a stable average direct benefit; redesign "
                    "the residual path before expensive retraining."
                )
            ),
        },
    }


def analyze_direct_effects(
        train, calibration, gate_step, bootstrap, seed,
        require_anchored=False, anchor_atol=1e-6,
        require_bias_separated=False, cue_bounds=None):
    validate_gate_step(gate_step)
    if require_anchored and require_bias_separated:
        raise ValueError(
            "exact and bias-separated anchor requirements are exclusive"
        )
    cue_bounds = dict(DEFAULT_CUE_BOUNDS if cue_bounds is None else cue_bounds)
    if set(cue_bounds) != set(CUES) or any(
            not np.isfinite(value) or value <= 0
            for value in cue_bounds.values()):
        raise ValueError("cue_bounds must contain positive Motion/Skeleton bounds")

    if require_bias_separated:
        anchor_report = bias_separated_anchor_parity(
            train, calibration, anchor_atol
        )
        anchor_report["required"] = True
    else:
        anchor_report = {
            "mode": "exact_cue_off_equals_raw_base",
            "required": bool(require_anchored),
            "train": anchor_parity(train, anchor_atol),
            "calibration": anchor_parity(calibration, anchor_atol),
        }
        anchor_report["passed"] = bool(
            anchor_report["train"]["passed"]
            and anchor_report["calibration"]["passed"]
        )
    if (
            (require_anchored or require_bias_separated)
            and not anchor_report["passed"]):
        raise ValueError(
            "required anchor identity failed; inspect cue-off and Base exports"
        )
    sample_level_grouping = all(
        sample_id == group_id
        for table in (train, calibration)
        for sample_id, group_id in zip(
            table["sample_ids"], table["group_ids"]
        )
    )
    cue_effects = {
        cue: analyze_cue(
            cue, train, calibration, gate_step, bootstrap, seed,
            cue_bounds[cue],
            require_bias_separated=require_bias_separated,
        )
        for cue in CUES
    }
    average_supported = {
        cue: cue_effects[cue]["preliminary_decision"][
            "average_effect_supported"
        ]
        for cue in CUES
    }
    dynamic_supported = {
        cue: cue_effects[cue]["preliminary_decision"][
            "dynamic_gate_supported"
        ]
        for cue in CUES
    }
    supported = {
        cue: cue_effects[cue]["preliminary_decision"][
            "direct_effect_supported"
        ]
        for cue in CUES
    }
    decision_valid = bool(anchor_report["passed"])
    for cue in CUES:
        cue_decision = cue_effects[cue]["preliminary_decision"]
        cue_decision["anchored_decision_valid"] = decision_valid
        cue_decision["authorize_oof_gate"] = bool(
            decision_valid and supported[cue]
        )
        if not decision_valid:
            cue_decision["interpretation"] = (
                "The direct-effect evidence is not actionable because "
                "cue-off does not satisfy the required shared anchor. Rerun "
                "the anchored export before authorizing any OOF gate."
            )
    if not decision_valid:
        next_step = (
            "The cue-off checkpoints do not share the required Base anchor; "
            "do not authorize OOF from this report. Rerun the frozen-Base "
            "adapter experiment with --require_anchored."
        )
    elif all(supported.values()):
        next_step = (
            "Proceed to a group-OOF three-way Base/Motion/Skeleton gate; "
            "keep PCS and test untouched."
        )
    elif any(supported.values()):
        passing = [cue for cue in CUES if supported[cue]]
        failing = [cue for cue in CUES if not supported[cue]]
        next_step = (
            f"Gate only {', '.join(passing)} for now; redesign the "
            f"{', '.join(failing)} residual path before combining cues."
        )
    elif any(average_supported.values()):
        passing = [cue for cue in CUES if average_supported[cue]]
        next_step = (
            f"Keep {', '.join(passing)} as fixed residual adapters; "
            "sample-level AutoGate is not supported by this probe."
        )
    else:
        next_step = (
            "Do not run the expensive OOF gate yet; redesign bounded "
            "residual adapters anchored to the same frozen Base scorer."
        )

    report_warnings = [
        "The on/off sample oracle uses calibration labels and is an "
        "optimistic diagnostic, never a deployable gate result.",
        "A positive decision only authorizes a group-OOF gate "
        "experiment; it is not final paper or test evidence.",
        "Cue-off must be exported from the same cue checkpoint as "
        "cue-on. Otherwise the direct/checkpoint decomposition is invalid.",
        "Paired group bootstrap quantifies sampling stability but does "
        "not remove the optimistic bias of a label-selected oracle.",
        "OOF gate authorization is conservative: a cue must pass both "
        "the train-locked fixed-effect checks and the optimistic oracle "
        "headroom checks.",
        "direct_std_over_total_std is an algebraic identity under an exact "
        "shared anchor; use residual std/bound and saturation diagnostics "
        "as evidence of sample variation instead.",
    ]
    if sample_level_grouping:
        report_warnings.append(
            "Every group_id equals sample_id, so group bootstrap is "
            "sample-level unless higher-level subject/source groups are "
            "provided."
        )

    return {
        "schema_version": 3,
        "oracle_upper_bound_not_deployable": True,
        "protocol": (
            "All variants remain on the exported raw regression-score "
            "scale; no per-variant affine is allowed. For each cue, one "
            "fixed gate g in [0,1] is selected by train MSE and locked for "
            "calibration. Sample and constant-offset oracles use the same "
            "gate grid. Bootstrap resamples calibration groups jointly "
            "while keeping every train-fitted parameter locked."
        ),
        "prediction_semantics": (
            {
                "base": "raw frozen cue-free Base",
                "cue_off": "raw Base + one train-fitted shared scalar bias",
                "cue_on": "cue_off + bounded sample-conditioned residual",
            }
            if require_bias_separated else
            {
                "base": "raw frozen cue-free Base",
                "cue_off": "same frozen Base with cue gate forced to zero",
                "cue_on": "cue_off + bounded cue intervention",
            }
        ),
        "warnings": report_warnings,
        "sample_counts": {
            "train": int(len(train["labels"])),
            "calibration": int(len(calibration["labels"])),
            "train_unique_groups": int(len(set(train["group_ids"]))),
            "calibration_unique_groups": int(
                len(set(calibration["group_ids"]))
            ),
        },
        "variant_order": list(VARIANTS),
        "grouping": {
            "all_group_ids_equal_sample_ids": sample_level_grouping,
        },
        "anchor_parity": anchor_report,
        "calibration_controls": calibration_controls(train, calibration),
        "shared_raw_scale_metrics": {
            "train": metrics_by_variant(
                train["predictions"], train["labels"]
            ),
            "calibration": metrics_by_variant(
                calibration["predictions"], calibration["labels"]
            ),
        },
        "scale_control": {
            "method": "raw shared regression-score scale",
            "per_variant_affine_forbidden": True,
            "reason": (
                "Independent transforms would change cue_on - cue_off "
                "and invalidate the direct/retrain identity."
            ),
        },
        "cue_effects": cue_effects,
        "preliminary_decisions": {
            "average_effect_supported": average_supported,
            "fixed_cue_supported": average_supported,
            "oracle_headroom_supported": dynamic_supported,
            "dynamic_gate_supported": dynamic_supported,
            "direct_effect_supported": supported,
            "gate_requires_average_effect": True,
            "anchored_decision_valid": decision_valid,
            "both_cues_supported": bool(
                decision_valid and all(supported.values())
            ),
            "any_cue_supported": bool(
                decision_valid and any(supported.values())
            ),
            "next_step": next_step,
        },
    }


def apply_paired_motion_decision(report):
    motion = report["cue_effects"]["motion"]
    skeleton = report["cue_effects"]["skeleton"]
    skeleton_effects = skeleton["decomposition"]["calibration_effects"]
    if any(
            skeleton_effects[name]["rms"] > 1e-12
            for name in ("total", "direct", "retrain")):
        raise ValueError(
            "paired Motion mode requires Skeleton inputs to be an exact "
            "matched-control null"
        )
    if not np.isclose(
            motion["fixed_gate"]["grid_step"], 1.0,
            atol=1e-12, rtol=0):
        raise ValueError(
            "paired Motion decision requires the binary endpoint grid {0,1}"
        )

    calibration_metrics = motion["variant_metrics"]["calibration"]
    off_vs_control = comparison(
        calibration_metrics["off"], calibration_metrics["base"]
    )
    bootstrap = motion["bootstrap"]
    guardrail_checks = {
        "paired_g0_delta_coef_vs_control_at_least_minus_0_5": bool(
            off_vs_control["delta_coef"] is not None
            and off_vs_control["delta_coef"] >= -0.5
        ),
        "paired_g0_mse_not_worse_than_control_by_over_2pct": bool(
            off_vs_control["relative_mse_gain"] >= -0.02
        ),
        "bootstrap_g0_delta_coef_guardrail_probability_at_least_90pct":
            bool(bootstrap[
                "cue_off_delta_coef_over_base"
            ]["probability_meeting_criterion"] is not None and bootstrap[
                "cue_off_delta_coef_over_base"
            ]["probability_meeting_criterion"] >= 0.90),
        "bootstrap_g0_mse_guardrail_probability_at_least_90pct": bool(
            bootstrap[
                "cue_off_relative_mse_gain_over_base"
            ]["probability_meeting_criterion"] is not None and bootstrap[
                "cue_off_relative_mse_gain_over_base"
            ]["probability_meeting_criterion"] >= 0.90),
    }
    guardrail_supported = bool(all(guardrail_checks.values()))
    average_supported = bool(
        motion["preliminary_decision"]["average_effect_supported"]
    )
    dynamic_supported = bool(
        motion["preliminary_decision"]["dynamic_gate_supported"]
    )
    qualified = bool(
        guardrail_supported and average_supported and dynamic_supported
    )
    if qualified:
        status = "GO_AUTOGATE"
        next_step = (
            "Freeze this endpoint mechanism and implement the preregistered "
            "label-free group-OOF AutoGate without tuning on this lockbox."
        )
    elif guardrail_supported and average_supported:
        status = "FIXED_MOTION_ONLY"
        next_step = (
            "The paired scorer supports fixed Motion but not sample routing; "
            "do not build AutoGate from this mechanism."
        )
    else:
        status = "NO_GO"
        next_step = (
            "Stop the stage-Motion AutoGate route; do not sweep alpha, "
            "epochs, batch size, or endpoint weights on this lockbox."
        )

    report["generic_anchor_diagnostic"] = report["anchor_parity"]
    report["anchor_parity"] = {
        "mode": "paired_control_guardrail",
        "required": False,
        "not_applicable": True,
        "passed": None,
        "reason": (
            "The paired g0 checkpoint may co-adapt away from the separately "
            "trained off/off control; stability is decided by the explicit "
            "guardrail, not exact anchor identity."
        ),
    }
    for cue in CUES:
        generic = report["cue_effects"][cue]["preliminary_decision"]
        generic["authorize_oof_gate"] = False
        generic["generic_decision_superseded"] = True
        generic["interpretation"] = (
            "Superseded in paired mode. Read paired_motion_decision only."
        )
    report["decision_authority"] = "paired_motion_decision_only"
    report["prediction_semantics"] = {
        "base": (
            "matched cue-free off/off model trained from scratch with the "
            "same duplicated-batch protocol"
        ),
        "motion_off": (
            "paired off/on checkpoint evaluated with Motion gate forced to 0"
        ),
        "motion_on": (
            "the same paired off/on checkpoint evaluated with Motion gate "
            "forced to 1"
        ),
        "skeleton": "exact matched-control null used only by the shared analyzer",
    }
    report["paired_motion_decision"] = {
        "status": status,
        "guardrail": {
            "calibration_comparison_paired_g0_vs_control": off_vs_control,
            "bootstrap": {
                "delta_coef": bootstrap[
                    "cue_off_delta_coef_over_base"
                ],
                "relative_mse_gain": bootstrap[
                    "cue_off_relative_mse_gain_over_base"
                ],
            },
            "checks": guardrail_checks,
            "supported": guardrail_supported,
        },
        "fixed_endpoint_supported": average_supported,
        "sample_routing_headroom_supported": dynamic_supported,
        "mechanism_qualified": qualified,
        "authorize_next_stage_oof_autogate_experiment": qualified,
        "autogate_result_claimed": False,
        "authorize_official_test": False,
        "authorize_deployment": False,
        "authorization_note": (
            "A positive result authorizes only the next label-free, group-"
            "OOF AutoGate experiment. It is not an AutoGate result, a test "
            "result, or evidence of training-seed stability."
        ),
        "next_step": next_step,
    }
    report["preliminary_decisions"]["oof_authorized"] = False
    report["preliminary_decisions"][
        "generic_decision_superseded"
    ] = True
    report["preliminary_decisions"][
        "anchored_decision_valid"
    ] = None
    report["preliminary_decisions"]["next_step"] = next_step
    report["warnings"].insert(
        0,
        "In paired mode, Base means the matched off/off training control; "
        "it is not a reused or conventionally trained PAMFN checkpoint.",
    )
    report["warnings"].insert(
        1,
        "This is one training seed. Sample bootstrap does not establish "
        "stability across retraining seeds.",
    )
    return report


def _write_test_csv(path, ids, groups, labels, predictions):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for values in zip(ids, groups, labels, predictions):
            writer.writerow(dict(zip(REQUIRED_COLUMNS, values)))


def _synthetic_table(prefix, labels, predictions, group_size=1):
    count = len(labels)
    return {
        "sample_ids": [f"{prefix}_sample_{i}" for i in range(count)],
        "group_ids": [
            f"{prefix}_group_{i // group_size}" for i in range(count)
        ],
        "labels": np.asarray(labels, dtype=np.float64),
        "predictions": np.asarray(predictions, dtype=np.float64),
        "inputs": {},
    }


def expect_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def run_self_test():
    rng = np.random.RandomState(31)
    train_labels = rng.uniform(0.1, 0.9, 240)
    cal_labels = rng.uniform(0.1, 0.9, 120)

    def positive_predictions(labels):
        count = len(labels)
        base_noise = rng.normal(0, 0.13, count)
        base = labels + base_noise

        motion_off = base.copy()
        motion_good = np.arange(count) % 4 != 0
        motion_noise = np.where(
            motion_good,
            rng.normal(0, 0.025, count),
            rng.normal(0, 0.23, count),
        )
        motion_on = labels + motion_noise + 0.01

        skeleton_off = base.copy()
        skeleton_good = np.arange(count) % 5 != 0
        skeleton_noise = np.where(
            skeleton_good,
            rng.normal(0, 0.01, count),
            rng.normal(0, 0.20, count),
        )
        skeleton_on = labels + skeleton_noise - 0.01
        return np.column_stack((
            base, motion_on, motion_off, skeleton_on, skeleton_off
        ))

    train = _synthetic_table(
        "train", train_labels, positive_predictions(train_labels),
        group_size=2,
    )
    calibration = _synthetic_table(
        "cal", cal_labels, positive_predictions(cal_labels),
        group_size=2,
    )
    validate_splits(train, calibration, 240, 120)
    report = analyze_direct_effects(
        train, calibration, gate_step=0.05, bootstrap=100, seed=0
    )
    assert report["preliminary_decisions"]["both_cues_supported"], {
        cue: report["cue_effects"][cue]["preliminary_decision"]
        for cue in CUES
    }
    for cue in CUES:
        decomposition = report["cue_effects"][cue]["decomposition"]
        assert decomposition["max_abs_identity_error"] < 1e-12
        assert decomposition["direct_std_over_total_std"] >= 0.10
    json.dumps(report, allow_nan=False)

    def retrain_only_predictions(labels):
        count = len(labels)
        base = labels + rng.normal(0, 0.12, count)
        motion = labels + rng.normal(0, 0.07, count) + 0.02
        skeleton = labels + rng.normal(0, 0.06, count) - 0.02
        return np.column_stack((
            base, motion, motion, skeleton, skeleton
        ))

    negative_report = analyze_direct_effects(
        _synthetic_table(
            "negative_train", train_labels,
            retrain_only_predictions(train_labels),
        ),
        _synthetic_table(
            "negative_cal", cal_labels,
            retrain_only_predictions(cal_labels),
        ),
        gate_step=0.05,
        bootstrap=50,
        seed=0,
    )
    assert not negative_report[
        "preliminary_decisions"
    ]["any_cue_supported"]
    for cue in CUES:
        effect = negative_report["cue_effects"][cue]
        assert effect["decomposition"][
            "calibration_effects"
        ]["direct"]["std"] == 0.0
        assert effect["sample_gate_oracle"]["tie_ratio"] == 1.0
    json.dumps(negative_report, allow_nan=False)

    def anchored_null_predictions(labels):
        base = labels + rng.normal(0, 0.12, len(labels))
        return np.column_stack((base, base, base, base, base))

    anchored_null_report = analyze_direct_effects(
        _synthetic_table(
            "anchored_null_train", train_labels,
            anchored_null_predictions(train_labels),
        ),
        _synthetic_table(
            "anchored_null_cal", cal_labels,
            anchored_null_predictions(cal_labels),
        ),
        gate_step=0.05,
        bootstrap=20,
        seed=0,
        require_anchored=True,
    )
    assert anchored_null_report["anchor_parity"]["passed"]
    assert not anchored_null_report[
        "preliminary_decisions"
    ]["any_cue_supported"]
    for cue in CUES:
        decision = anchored_null_report[
            "cue_effects"
        ][cue]["preliminary_decision"]
        assert decision["anchored_decision_valid"]
        assert not decision["authorize_oof_gate"]
    json.dumps(anchored_null_report, allow_nan=False)

    anchored = dict(train)
    anchored["predictions"] = train["predictions"].copy()
    anchored["predictions"][:, VARIANTS.index("motion_off")] = (
        anchored["predictions"][:, VARIANTS.index("base")]
    )
    anchored["predictions"][:, VARIANTS.index("skeleton_off")] = (
        anchored["predictions"][:, VARIANTS.index("base")]
    )
    assert anchor_parity(anchored, 1e-12)["passed"]
    anchored_calibration = dict(calibration)
    anchored_calibration["predictions"] = (
        calibration["predictions"].copy()
    )
    for variant in ("motion_off", "skeleton_off"):
        anchored_calibration["predictions"][:, VARIANTS.index(variant)] = (
            anchored_calibration[
                "predictions"
            ][:, VARIANTS.index("base")]
        )
    anchored_report = analyze_direct_effects(
        anchored,
        anchored_calibration,
        gate_step=0.05,
        bootstrap=10,
        seed=0,
        require_anchored=True,
    )
    assert anchored_report["anchor_parity"]["passed"]
    unanchored_train = dict(train)
    unanchored_train["predictions"] = train["predictions"].copy()
    unanchored_train[
        "predictions"
    ][:, VARIANTS.index("motion_off")] += 0.01
    unanchored_report = analyze_direct_effects(
        unanchored_train,
        calibration,
        gate_step=0.05,
        bootstrap=10,
        seed=0,
    )
    assert not unanchored_report[
        "preliminary_decisions"
    ]["anchored_decision_valid"]
    for cue in CUES:
        decision = unanchored_report[
            "cue_effects"
        ][cue]["preliminary_decision"]
        assert not decision["anchored_decision_valid"]
        assert not decision["authorize_oof_gate"]
    expect_value_error(lambda: analyze_direct_effects(
        unanchored_train,
        calibration,
        gate_step=0.05,
        bootstrap=10,
        seed=0,
        require_anchored=True,
    ))

    bias_train_base = train_labels + rng.normal(0.04, 0.08, len(train_labels))
    fitted_bias = float(np.mean(train_labels - bias_train_base))
    bias_cal_base = cal_labels + rng.normal(0.04, 0.08, len(cal_labels))

    def bias_predictions(base):
        shared = base + fitted_bias
        return np.column_stack((
            base,
            shared - DEFAULT_CUE_BOUNDS["motion"],
            shared,
            shared - DEFAULT_CUE_BOUNDS["skeleton"],
            shared,
        ))

    bias_train = _synthetic_table(
        "bias_train", train_labels, bias_predictions(bias_train_base)
    )
    bias_calibration = _synthetic_table(
        "bias_cal", cal_labels, bias_predictions(bias_cal_base)
    )
    bias_report = analyze_direct_effects(
        bias_train,
        bias_calibration,
        gate_step=0.05,
        bootstrap=10,
        seed=0,
        require_bias_separated=True,
    )
    assert bias_report["anchor_parity"]["passed"]
    assert abs(
        bias_report["anchor_parity"]["shared_bias"]["train"]
        - fitted_bias
    ) <= 1e-12
    for cue in CUES:
        decision = bias_report["cue_effects"][cue][
            "preliminary_decision"
        ]
        assert not decision["fixed_cue_supported"]
        assert not decision["authorize_oof_gate"]

    wrong_fit_train = dict(bias_train)
    wrong_fit_train["predictions"] = bias_train["predictions"].copy()
    for variant in ("motion_off", "skeleton_off"):
        wrong_fit_train["predictions"][:, VARIANTS.index(variant)] += 0.001
    assert not bias_separated_anchor_parity(
        wrong_fit_train, bias_calibration, 1e-6
    )["passed"]

    drifted_calibration = dict(bias_calibration)
    drifted_calibration["predictions"] = (
        bias_calibration["predictions"].copy()
    )
    for variant in ("motion_off", "skeleton_off"):
        drifted_calibration[
            "predictions"
        ][:, VARIANTS.index(variant)] += 0.001
    assert not bias_separated_anchor_parity(
        bias_train, drifted_calibration, 1e-6
    )["passed"]

    mismatched_cue_train = dict(bias_train)
    mismatched_cue_train["predictions"] = bias_train["predictions"].copy()
    mismatched_cue_train[
        "predictions"
    ][:, VARIANTS.index("motion_off")] += 0.001
    assert not bias_separated_anchor_parity(
        mismatched_cue_train, bias_calibration, 1e-6
    )["passed"]

    with tempfile.TemporaryDirectory() as directory:
        ids = ["a", "b", "c", "d"]
        groups = ["g1", "g1", "g2", "g2"]
        labels = [0.1, 0.2, 0.3, 0.4]
        paths = {}
        for index, variant in enumerate(VARIANTS):
            order = [2, 0, 3, 1] if variant == "motion_on" else range(4)
            order = list(order)
            path = os.path.join(directory, f"{variant}.csv")
            _write_test_csv(
                path,
                [ids[row] for row in order],
                [groups[row] for row in order],
                [labels[row] for row in order],
                [labels[row] + index * 0.01 for row in order],
            )
            paths[variant] = path
        merged = merge_prediction_tables(paths)
        assert merged["sample_ids"] == ids
        assert np.allclose(
            merged["predictions"][:, VARIANTS.index("motion_on")],
            np.asarray(labels) + 0.01,
        )

        duplicate = os.path.join(directory, "duplicate.csv")
        _write_test_csv(
            duplicate, ["a", "a"], ["g1", "g2"],
            [0.1, 0.2], [0.1, 0.2],
        )
        expect_value_error(lambda: read_prediction_csv(duplicate))

        nan_path = os.path.join(directory, "nan.csv")
        _write_test_csv(
            nan_path, ["a"], ["g1"], [0.1], [float("nan")]
        )
        expect_value_error(lambda: read_prediction_csv(nan_path))

        mismatched = os.path.join(directory, "mismatched.csv")
        _write_test_csv(
            mismatched, ids, groups, [0.1, 0.2, 0.3, 0.5], labels
        )
        mismatch_paths = dict(paths)
        mismatch_paths["skeleton_off"] = mismatched
        expect_value_error(
            lambda: merge_prediction_tables(mismatch_paths)
        )

        group_mismatch = os.path.join(directory, "group_mismatch.csv")
        _write_test_csv(
            group_mismatch, ids, ["g1", "g1", "g2", "wrong"],
            labels, labels
        )
        group_mismatch_paths = dict(paths)
        group_mismatch_paths["motion_off"] = group_mismatch
        expect_value_error(
            lambda: merge_prediction_tables(group_mismatch_paths)
        )

        malformed = os.path.join(directory, "malformed.csv")
        with open(malformed, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(REQUIRED_COLUMNS)
            writer.writerow(("a", "g1", 0.1))
        expect_value_error(lambda: read_prediction_csv(malformed))

        overlap = dict(calibration)
        overlap["sample_ids"] = list(train["sample_ids"][:120])
        expect_value_error(lambda: validate_splits(train, overlap))

        group_overlap = dict(calibration)
        group_overlap["group_ids"] = list(train["group_ids"][:120])
        expect_value_error(lambda: validate_splits(train, group_overlap))

    fitted_constant = fit_best_gated_constant(
        np.zeros(4), np.asarray([0.0, 0.0, 0.4, 0.4]), 1.0
    )
    if not np.isclose(fitted_constant, 0.4, atol=1e-12, rtol=0):
        raise AssertionError(
            "best train-fitted gated constant used the always-on objective"
        )

    print("strict ten-CSV alignment and split validation: passed")
    print("raw shared-scale protocol: passed")
    print("direct/checkpoint additive decomposition: passed")
    print("heterogeneous direct-effect positive control: passed")
    print("retraining-only negative control: passed")
    print("anchored zero-effect negative control: passed")
    print("group bootstrap and JSON finiteness: passed")
    print("best train-fitted gated constant control: passed")
    print("direct cue effect analyzer self-test passed")


def format_signed(value):
    return "undefined" if value is None else f"{value:+.3f}"


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    train_paths = {
        variant: getattr(args, f"{variant}_train")
        for variant in VARIANTS
    }
    calibration_paths = {
        variant: getattr(args, f"{variant}_cal")
        for variant in VARIANTS
    }
    train = merge_prediction_tables(train_paths)
    calibration = merge_prediction_tables(calibration_paths)
    validate_splits(
        train,
        calibration,
        args.expected_train_samples,
        args.expected_calibration_samples,
    )
    if args.paired_motion_exposure:
        base_index = VARIANTS.index("base")
        skeleton_indices = (
            VARIANTS.index("skeleton_on"),
            VARIANTS.index("skeleton_off"),
        )
        for split_name, table in (
                ("train", train), ("calibration", calibration)):
            base = table["predictions"][:, base_index]
            if any(
                    not np.array_equal(
                        table["predictions"][:, index], base
                    ) for index in skeleton_indices):
                raise ValueError(
                    f"paired Motion {split_name} Skeleton null is not "
                    "exactly equal to the matched control"
                )
    report = analyze_direct_effects(
        train,
        calibration,
        args.gate_step,
        args.bootstrap,
        args.seed,
        require_anchored=args.require_anchored,
        anchor_atol=args.anchor_atol,
        require_bias_separated=args.require_bias_separated,
        cue_bounds={
            "motion": args.motion_bound,
            "skeleton": args.skeleton_bound,
        },
    )
    report["dataset"] = args.dataset
    report["inputs"] = {
        "train": train["inputs"],
        "calibration": calibration["inputs"],
    }
    if args.paired_motion_exposure:
        report = apply_paired_motion_decision(report)

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    output_cues = ("motion",) if args.paired_motion_exposure else CUES
    for cue in output_cues:
        effect = report["cue_effects"][cue]
        direct = effect["cue_on_vs_off"]["calibration"]
        fixed = effect["fixed_gate"]
        fixed_effect = fixed["calibration_comparison_to_off"]
        headroom = effect["headroom"][
            "oracle_over_fixed_gate"
        ]
        ratio = effect["decomposition"]["direct_std_over_total_std"]
        ratio_text = "undefined" if ratio is None else f"{ratio:.2%}"
        supported = effect["preliminary_decision"][
            "direct_effect_supported"
        ]
        average = effect["preliminary_decision"][
            "average_effect_supported"
        ]
        dynamic = effect["preliminary_decision"][
            "dynamic_gate_supported"
        ]
        print(
            f"{cue}: on_vs_off_delta_coef="
            f"{format_signed(direct['delta_coef'])} "
            f"fixed_gate={fixed['gate']:.2f} "
            f"fixed_delta_coef={format_signed(fixed_effect['delta_coef'])} "
            f"oracle_mse_gain={headroom['relative_mse_gain']:.2%} "
            f"direct_std_ratio={ratio_text} average={average} "
            f"dynamic={dynamic} supported={supported}"
        )
    if args.paired_motion_exposure:
        decision = report["paired_motion_decision"]
        guardrail = decision["guardrail"][
            "calibration_comparison_paired_g0_vs_control"
        ]
        print(
            "paired_motion_status="
            f"{decision['status']} "
            "g0_vs_control_delta_coef="
            f"{format_signed(guardrail['delta_coef'])} "
            "g0_vs_control_mse_gain="
            f"{guardrail['relative_mse_gain']:.2%} "
            "fixed_endpoint_supported="
            f"{decision['fixed_endpoint_supported']} "
            "routing_headroom_supported="
            f"{decision['sample_routing_headroom_supported']}"
        )
    if args.paired_motion_exposure:
        print("decision_authority=paired_motion_decision_only")
    else:
        print(
            "anchor_parity_passed="
            f"{report['anchor_parity']['passed']}"
        )
        print(
            "both_cues_supported="
            f"{report['preliminary_decisions']['both_cues_supported']}"
        )
    print(f"report={output}")


if __name__ == "__main__":
    main()
