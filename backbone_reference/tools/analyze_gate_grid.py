import argparse
import csv
import json
import os
from collections import Counter

import numpy as np
from scipy.stats import spearmanr


GATE_COLUMNS = (
    "pred_m0_s0",
    "pred_m0_s05",
    "pred_m0_s1",
    "pred_m05_s0",
    "pred_m05_s05",
    "pred_m05_s1",
    "pred_m1_s0",
    "pred_m1_s05",
    "pred_m1_s1"
)
DEFAULT_COLUMN = "pred_m1_s05"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze calibration gate-grid predictions"
    )
    parser.add_argument("--after", required=True)
    parser.add_argument("--before", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    return args


def safe_spearman(predictions, labels):
    coefficient, _ = spearmanr(predictions, labels)
    return float(coefficient)


def read_grid(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no samples")
    required = {"sample_id", "group_id", "label", *GATE_COLUMNS}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{path} contains duplicate sample IDs")
    labels = np.asarray(
        [float(row["label"]) for row in rows], dtype=np.float64
    )
    predictions = np.asarray([
        [float(row[column]) for column in GATE_COLUMNS]
        for row in rows
    ], dtype=np.float64)
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError(f"{path} contains NaN or infinite values")
    return sample_ids, labels, predictions


def gate_metrics(predictions, labels):
    mse = np.mean((predictions - labels[:, None]) ** 2, axis=0)
    rho = np.asarray([
        safe_spearman(predictions[:, index], labels)
        for index in range(predictions.shape[1])
    ])
    return rho, mse


def choose_best_gate(rho, mse):
    valid_rho = np.where(np.isfinite(rho), rho, -np.inf)
    return min(
        range(len(GATE_COLUMNS)),
        key=lambda index: (-valid_rho[index], mse[index])
    )


def percentile_interval(values):
    lower, upper = np.percentile(values, [2.5, 97.5])
    return [float(lower), float(upper)]


def analyze_grid(labels, predictions, bootstrap, seed):
    rho, mse = gate_metrics(predictions, labels)
    best_index = choose_best_gate(rho, mse)
    default_index = GATE_COLUMNS.index(DEFAULT_COLUMN)
    base_index = GATE_COLUMNS.index("pred_m0_s0")
    oracle_index = np.argmin(
        (predictions - labels[:, None]) ** 2, axis=1
    )
    oracle_predictions = predictions[
        np.arange(len(labels)), oracle_index
    ]
    oracle_mse = float(
        np.mean((oracle_predictions - labels) ** 2)
    )
    oracle_rho = safe_spearman(oracle_predictions, labels)

    base_predictions = predictions[:, base_index]
    mean_offsets = np.mean(
        predictions - base_predictions[:, None], axis=0
    )
    constant_offset_predictions = (
        base_predictions[:, None] + mean_offsets[None, :]
    )
    constant_oracle_index = np.argmin(
        (constant_offset_predictions - labels[:, None]) ** 2,
        axis=1
    )
    constant_oracle_predictions = constant_offset_predictions[
        np.arange(len(labels)), constant_oracle_index
    ]
    constant_oracle_mse = float(
        np.mean((constant_oracle_predictions - labels) ** 2)
    )
    constant_oracle_rho = safe_spearman(
        constant_oracle_predictions, labels
    )
    actual_gain_over_constant = (
        (constant_oracle_mse - oracle_mse) / constant_oracle_mse
        if constant_oracle_mse > 0 else 0.0
    )
    actual_delta_rho_over_constant = (
        oracle_rho - constant_oracle_rho
    )
    oracle_selection_agreement = float(
        np.mean(oracle_index == constant_oracle_index)
    )

    rng = np.random.RandomState(seed)
    selection_counts = Counter()
    fixed_delta_rho = []
    fixed_delta_mse = []
    oracle_delta_mse = []
    for _ in range(bootstrap):
        indices = rng.randint(0, len(labels), len(labels))
        sampled_labels = labels[indices]
        sampled_predictions = predictions[indices]
        sampled_rho, sampled_mse = gate_metrics(
            sampled_predictions, sampled_labels
        )
        selected = choose_best_gate(sampled_rho, sampled_mse)
        selection_counts[GATE_COLUMNS[selected]] += 1
        fixed_delta_rho.append(
            safe_spearman(
                sampled_predictions[:, best_index], sampled_labels
            )
            - safe_spearman(
                sampled_predictions[:, default_index], sampled_labels
            )
        )
        fixed_delta_mse.append(
            np.mean(
                (
                    sampled_predictions[:, best_index]
                    - sampled_labels
                ) ** 2
            )
            - np.mean(
                (
                    sampled_predictions[:, default_index]
                    - sampled_labels
                ) ** 2
            )
        )
        sampled_oracle_index = np.argmin(
            (
                sampled_predictions
                - sampled_labels[:, None]
            ) ** 2,
            axis=1
        )
        sampled_oracle = sampled_predictions[
            np.arange(len(indices)), sampled_oracle_index
        ]
        oracle_delta_mse.append(
            np.mean((sampled_oracle - sampled_labels) ** 2)
            - np.mean(
                (
                    sampled_predictions[:, best_index]
                    - sampled_labels
                ) ** 2
            )
        )

    per_gate = {}
    for index, column in enumerate(GATE_COLUMNS):
        per_gate[column] = {
            "rho": float(rho[index]),
            "coef": float(rho[index] * 100),
            "mse_raw": float(mse[index]),
            "loss_x100": float(mse[index] * 100)
        }

    default_mse = float(mse[default_index])
    best_mse = float(mse[best_index])
    sample_oracle_mse_gain = (
        (best_mse - oracle_mse) / best_mse
        if best_mse > 0 else 0.0
    )
    best_frequency = (
        selection_counts[GATE_COLUMNS[best_index]] / bootstrap
    )
    best_delta_coef = float(
        (rho[best_index] - rho[default_index]) * 100
    )
    best_relative_mse = (
        (best_mse - default_mse) / default_mse
        if default_mse > 0 else 0.0
    )
    raw_sample_headroom = bool(
        sample_oracle_mse_gain >= 0.05
        and oracle_rho > rho[best_index]
    )
    constant_offset_control_pass = bool(
        actual_gain_over_constant >= 0.02
        and actual_delta_rho_over_constant > 0.0
    )
    return {
        "samples": int(len(labels)),
        "primary_metric": "Spearman",
        "mse_role": "guardrail",
        "per_gate": per_gate,
        "default_gate": DEFAULT_COLUMN,
        "task_best_gate": GATE_COLUMNS[best_index],
        "task_best_delta_coef": best_delta_coef,
        "task_best_relative_mse_change": float(best_relative_mse),
        "task_best_bootstrap_frequency": float(best_frequency),
        "sample_oracle": {
            "rho": float(oracle_rho),
            "coef": float(oracle_rho * 100),
            "mse_raw": oracle_mse,
            "relative_mse_gain_over_task_best":
                float(sample_oracle_mse_gain),
            "relative_mse_gain_over_constant_offset_oracle":
                float(actual_gain_over_constant),
            "delta_rho_over_constant_offset_oracle":
                float(actual_delta_rho_over_constant),
            "selection_agreement_with_constant_offset_oracle":
                oracle_selection_agreement,
            "different_from_task_best_ratio": float(
                np.mean(oracle_index != best_index)
            ),
            "warning": (
                "Label-selected optimistic upper bound; "
                "not a deployable selector."
            )
        },
        "constant_offset_oracle": {
            "base_gate": GATE_COLUMNS[base_index],
            "rho": float(constant_oracle_rho),
            "coef": float(constant_oracle_rho * 100),
            "mse_raw": constant_oracle_mse,
            "selection_agreement_with_sample_oracle":
                oracle_selection_agreement,
            "actual_oracle_relative_mse_gain":
                float(actual_gain_over_constant),
            "actual_oracle_delta_rho":
                float(actual_delta_rho_over_constant),
            "actual_oracle_delta_coef":
                float(actual_delta_rho_over_constant * 100),
            "warning": (
                "Label-selected control using the base prediction plus "
                "each gate's dataset-wide mean offset."
            )
        },
        "bootstrap_95ci": {
            "task_best_minus_default_rho":
                percentile_interval(fixed_delta_rho),
            "task_best_minus_default_mse":
                percentile_interval(fixed_delta_mse),
            "sample_oracle_minus_task_best_mse":
                percentile_interval(oracle_delta_mse)
        },
        "bootstrap_selection_frequency": {
            column: selection_counts[column] / bootstrap
            for column in GATE_COLUMNS
        },
        "preliminary_decisions": {
            "stable_task_region": bool(best_frequency >= 0.70),
            "task_headroom": bool(
                best_delta_coef >= 0.5
                and best_relative_mse <= 0.02
                and best_frequency >= 0.70
            ),
            "sample_headroom_without_constant_offset_control":
                raw_sample_headroom,
            "constant_offset_control_pass":
                constant_offset_control_pass,
            "sample_headroom": bool(
                raw_sample_headroom
                and constant_offset_control_pass
            )
        }
    }


def compare_before_after(
        before_ids, before_labels, before_predictions,
        after_ids, after_labels, after_predictions):
    if before_ids != after_ids:
        raise ValueError(
            "before and after CSV files have different sample order"
        )
    if not np.allclose(before_labels, after_labels):
        raise ValueError(
            "before and after CSV files have different labels"
        )
    default_index = GATE_COLUMNS.index(DEFAULT_COLUMN)
    before_rho, before_mse = gate_metrics(
        before_predictions, before_labels
    )
    after_rho, after_mse = gate_metrics(
        after_predictions, after_labels
    )
    relative_mse = (
        (
            after_mse[default_index]
            - before_mse[default_index]
        ) / before_mse[default_index]
        if before_mse[default_index] > 0 else 0.0
    )
    delta_coef = (
        after_rho[default_index] - before_rho[default_index]
    ) * 100
    return {
        "default_gate_delta_coef": float(delta_coef),
        "default_gate_relative_mse_change": float(relative_mse),
        "exposure_guardrail_pass": bool(
            delta_coef >= -0.3 and relative_mse <= 0.02
        )
    }


def main():
    args = parse_args()
    after_ids, after_labels, after_predictions = read_grid(args.after)
    report = {
        "after": analyze_grid(
            after_labels, after_predictions,
            args.bootstrap, args.seed
        )
    }
    if args.before:
        before_ids, before_labels, before_predictions = read_grid(
            args.before
        )
        report["before"] = analyze_grid(
            before_labels, before_predictions,
            args.bootstrap, args.seed
        )
        report["exposure_comparison"] = compare_before_after(
            before_ids, before_labels, before_predictions,
            after_ids, after_labels, after_predictions
        )

    parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    after = report["after"]
    print(
        f"task_best={after['task_best_gate']} "
        f"delta_coef={after['task_best_delta_coef']:+.2f} "
        f"bootstrap_frequency="
        f"{after['task_best_bootstrap_frequency']:.1%}"
    )
    print(
        "sample_oracle_mse_gain="
        f"{after['sample_oracle']['relative_mse_gain_over_task_best']:.1%}"
    )
    print(
        "constant_offset_control_pass="
        f"{after['preliminary_decisions']['constant_offset_control_pass']} "
        "actual_mse_gain_over_constant="
        f"{after['constant_offset_oracle']['actual_oracle_relative_mse_gain']:.1%} "
        "sample_headroom="
        f"{after['preliminary_decisions']['sample_headroom']}"
    )
    if "exposure_comparison" in report:
        comparison = report["exposure_comparison"]
        print(
            "exposure_guardrail_pass="
            f"{comparison['exposure_guardrail_pass']} "
            f"default_delta_coef="
            f"{comparison['default_gate_delta_coef']:+.2f}"
        )
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
