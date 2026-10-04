"""Analyze the TES/PCS/Ball criterion-conditioned screening experiment."""

import argparse
import csv
import json
import os
import tempfile

import numpy as np
from scipy.stats import spearmanr


ACTIONS = ("TES", "PCS", "Ball")
BRANCHES = ("single", "shared", "conditioned")


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no rows")
    indexed = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in indexed:
            raise ValueError(f"{path} contains duplicate sample IDs")
        converted = {"sample_id": sample_id}
        for key, value in row.items():
            if key not in ("sample_id", "group_id"):
                converted[key] = float(value)
        indexed[sample_id] = converted
    return indexed


def aligned(left, right, description):
    if set(left) != set(right):
        raise ValueError(f"{description} sample IDs do not align")
    sample_ids = list(left)
    labels = np.array([left[key]["label"] for key in sample_ids])
    left_predictions = np.array([
        left[key]["prediction"] for key in sample_ids
    ])
    right_predictions = np.array([
        right[key]["prediction"] for key in sample_ids
    ])
    right_labels = np.array([right[key]["label"] for key in sample_ids])
    if not np.array_equal(labels, right_labels):
        raise ValueError(f"{description} labels do not align")
    return sample_ids, labels, left_predictions, right_predictions


def coefficient(predictions, labels):
    value = spearmanr(predictions, labels).statistic
    if not np.isfinite(value):
        raise ValueError("Spearman coefficient is undefined")
    return float(value * 100)


def affine_fit(predictions, labels):
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    centered = predictions - predictions.mean()
    denominator = float(np.dot(centered, centered))
    slope = (
        0.0 if denominator <= 1e-12 else
        max(0.0, float(np.dot(centered, labels - labels.mean())
                       / denominator))
    )
    shift = float(labels.mean() - slope * predictions.mean())
    return slope, shift


def affine_mse(train_prediction, train_label, cal_prediction, cal_label):
    slope, shift = affine_fit(train_prediction, train_label)
    calibrated = slope * cal_prediction + shift
    return float(np.mean((calibrated - cal_label) ** 2)), slope, shift


def bootstrap(labels, baseline, candidate, iterations, seed):
    rng = np.random.default_rng(seed)
    deltas = []
    sample_count = len(labels)
    for _ in range(iterations):
        indices = rng.integers(0, sample_count, sample_count)
        try:
            delta = (
                coefficient(candidate[indices], labels[indices])
                - coefficient(baseline[indices], labels[indices])
            )
        except ValueError:
            continue
        deltas.append(delta)
    if not deltas:
        raise ValueError("all bootstrap samples had undefined coefficient")
    values = np.asarray(deltas)
    return {
        "iterations_requested": iterations,
        "iterations_valid": len(values),
        "p_delta_gt_0": float(np.mean(values > 0)),
        "ci95": np.percentile(values, [2.5, 97.5]).tolist(),
    }


def criterion_names(row):
    names = []
    for key in row:
        if key.endswith("_label") and key != "label":
            names.append(key[:-6])
    expected = (
        ["tes", "pcs"]
        if "tes" in names else
        ["difficulty", "execution", "penalty", "total"]
    )
    if set(names) != set(expected):
        raise ValueError(
            f"criterion columns are incomplete: {sorted(names)}"
        )
    return expected


def criterion_report(rows):
    first = next(iter(rows.values()))
    names = criterion_names(first)
    report = {}
    for name in names:
        labels = np.array([
            row[f"{name}_label"] for row in rows.values()
        ])
        predictions = np.array([
            row[f"{name}_prediction"] for row in rows.values()
        ])
        report[name] = {
            "coef": coefficient(predictions, labels),
            "mse": float(np.mean((predictions - labels) ** 2)),
        }
    return report


def attention_report(rows, args):
    first = next(iter(rows.values()))
    names = criterion_names(first)
    centers = np.array([
        [row[f"{name}_attention_center"] for name in names]
        for row in rows.values()
    ])
    entropies = np.array([
        [row[f"{name}_attention_entropy"] for name in names]
        for row in rows.values()
    ])
    divergences = np.array([
        row["criterion_attention_divergence"]
        for row in rows.values()
    ])
    mean_entropy = float(entropies.mean())
    mean_divergence = float(divergences.mean())
    passed = bool(
        args.min_attention_entropy <= mean_entropy
        <= args.max_attention_entropy
        and mean_divergence >= args.min_attention_js
    )
    return {
        "criterion_names": names,
        "mean_centers": centers.mean(axis=0).tolist(),
        "center_std": centers.std(axis=0).tolist(),
        "mean_entropies": entropies.mean(axis=0).tolist(),
        "mean_entropy": mean_entropy,
        "mean_pairwise_js": mean_divergence,
        "passed": passed,
    }


def consistency_report(rows, max_gap):
    first = next(iter(rows.values()))
    required = {
        "difficulty_label", "execution_label", "penalty_label",
        "total_label", "composed_total_prediction",
        "direct_total_prediction", "total_consistency_gap",
    }
    if not required <= set(first):
        raise ValueError("Ball export is missing score consistency columns")
    label_identity_error = np.array([
        abs(
            row["difficulty_label"] + row["execution_label"]
            - row["penalty_label"] - row["total_label"]
        )
        for row in rows.values()
    ])
    prediction_gap = np.array([
        abs(row["total_consistency_gap"]) for row in rows.values()
    ])
    return {
        "maximum_label_identity_error": float(
            label_identity_error.max()
        ),
        "mean_absolute_direct_composed_gap": float(
            prediction_gap.mean()
        ),
        "maximum_absolute_direct_composed_gap": float(
            prediction_gap.max()
        ),
        "passed": bool(
            label_identity_error.max() <= 0.0005
            and prediction_gap.mean() <= max_gap
        ),
    }


def analyze_action(args, action, action_index):
    directory = os.path.join(args.root, action)
    rows = {}
    for branch in BRANCHES:
        for split in ("train", "cal"):
            rows[(branch, split)] = read_csv(os.path.join(
                directory, f"{action}_{branch}_{split}.csv"
            ))
    rows[("uniform", "cal")] = read_csv(os.path.join(
        directory, f"{action}_conditioned_uniform_cal.csv"
    ))

    baseline_train = rows[("single", "train")]
    baseline_cal = rows[("single", "cal")]
    reports = {}
    for branch_index, branch in enumerate(BRANCHES):
        _, train_labels, baseline_train_prediction, train_prediction = (
            aligned(
                baseline_train, rows[(branch, "train")],
                f"{action} {branch} train"
            )
        )
        _, cal_labels, baseline_cal_prediction, cal_prediction = aligned(
            baseline_cal, rows[(branch, "cal")],
            f"{action} {branch} calibration"
        )
        baseline_mse, baseline_slope, baseline_shift = affine_mse(
            baseline_train_prediction, train_labels,
            baseline_cal_prediction, cal_labels
        )
        candidate_mse, candidate_slope, candidate_shift = affine_mse(
            train_prediction, train_labels, cal_prediction, cal_labels
        )
        baseline_coef = coefficient(
            baseline_cal_prediction, cal_labels
        )
        candidate_coef = coefficient(cal_prediction, cal_labels)
        reports[branch] = {
            "base_coef": baseline_coef,
            "candidate_coef": candidate_coef,
            "delta_coef": candidate_coef - baseline_coef,
            "affine_mse": {
                "base": baseline_mse,
                "candidate": candidate_mse,
                "relative_gain_pct": (
                    (baseline_mse - candidate_mse)
                    / baseline_mse * 100
                ),
                "base_scale_shift": [
                    baseline_slope, baseline_shift
                ],
                "candidate_scale_shift": [
                    candidate_slope, candidate_shift
                ],
            },
            "bootstrap": bootstrap(
                cal_labels, baseline_cal_prediction, cal_prediction,
                args.bootstrap,
                args.seed + action_index * 101 + branch_index * 17,
            ),
        }
        if branch != "single":
            reports[branch]["criteria"] = criterion_report(
                rows[(branch, "cal")]
            )
        if branch == "conditioned":
            reports[branch]["attention"] = attention_report(
                rows[(branch, "cal")], args
            )
            if action == "Ball":
                reports[branch]["consistency"] = consistency_report(
                    rows[(branch, "cal")],
                    args.max_consistency_gap
                )

    _, labels, _, uniform_prediction = aligned(
        baseline_cal, rows[("uniform", "cal")],
        f"{action} conditioned uniform calibration"
    )
    uniform_coef = coefficient(uniform_prediction, labels)
    reports["conditioned_uniform"] = {
        "candidate_coef": uniform_coef,
        "drop_from_conditioned": (
            reports["conditioned"]["candidate_coef"] - uniform_coef
        ),
    }
    return reports


def analyze(args):
    actions = {
        action: analyze_action(args, action, index)
        for index, action in enumerate(ACTIONS)
    }
    conditioned_delta = np.array([
        actions[action]["conditioned"]["delta_coef"]
        for action in ACTIONS
    ])
    shared_delta = np.array([
        actions[action]["shared"]["delta_coef"]
        for action in ACTIONS
    ])
    uniform_drop = np.array([
        actions[action]["conditioned_uniform"][
            "drop_from_conditioned"
        ]
        for action in ACTIONS
    ])
    qualified = {
        action for action in ACTIONS
        if actions[action]["conditioned"]["delta_coef"]
        >= args.min_action_delta
    }
    checks = {
        "macro_delta": bool(
            conditioned_delta.mean() >= args.min_macro_delta
        ),
        "two_of_three_actions": bool(len(qualified) >= 2),
        "fisv_and_rg_coverage": bool(
            "Ball" in qualified
            and bool({"TES", "PCS"} & qualified)
        ),
        "beats_shared": bool(
            (conditioned_delta - shared_delta).mean()
            >= args.min_vs_shared
        ),
        "uniform_attention_control": bool(
            uniform_drop.mean() >= args.min_uniform_drop
        ),
        "affine_mse_guardrail": bool(all(
            actions[action]["conditioned"]["affine_mse"][
                "relative_gain_pct"
            ] >= -args.max_affine_mse_drop_pct
            for action in ACTIONS
        )),
        "criterion_attention_not_collapsed": bool(all(
            actions[action]["conditioned"]["attention"]["passed"]
            for action in ACTIONS
        )),
        "rg_score_consistency": bool(
            actions["Ball"]["conditioned"][
                "consistency"]["passed"]
        ),
    }
    metrics = {
        "conditioned_macro_delta_coef": float(
            conditioned_delta.mean()
        ),
        "shared_macro_delta_coef": float(shared_delta.mean()),
        "conditioned_minus_shared_macro": float(
            (conditioned_delta - shared_delta).mean()
        ),
        "uniform_attention_drop_macro": float(
            uniform_drop.mean()
        ),
        "qualified_actions": sorted(qualified),
    }
    return {
        "schema_version": 1,
        "actions": actions,
        "decision": {
            "status": (
                "GO_TO_THREE_SEED_SIX_ACTION_CRITERION_MODEL"
                if all(checks.values()) else
                "STOP_CRITERION_CONSISTENCY_DIRECTION"
            ),
            "checks": checks,
            "metrics": metrics,
            "rule": (
                "All pre-registered checks must pass before the "
                "six-action, three-seed experiment."
            ),
        },
    }


def write_synthetic_csv(path, action, branch, split, count):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    names = (
        ("tes", "pcs") if action in ("TES", "PCS") else
        ("difficulty", "execution", "penalty", "total")
    )
    fields = ["sample_id", "group_id", "label", "prediction"]
    if branch != "single":
        for name in names:
            fields.extend((
                f"{name}_label", f"{name}_prediction",
                f"{name}_attention_center",
                f"{name}_attention_entropy",
            ))
        fields.append("criterion_attention_divergence")
        if action == "Ball":
            fields.extend((
                "composed_total_prediction",
                "direct_total_prediction",
                "total_consistency_gap",
            ))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(count):
            base = (index + 1) / (count + 2)
            if action == "Ball":
                labels = {
                    "difficulty": 0.25 + 0.35 * base,
                    "execution": 0.15 + 0.20 * base,
                    "penalty": 0.01 * (index % 3),
                }
                labels["total"] = (
                    labels["difficulty"] + labels["execution"]
                    - labels["penalty"]
                )
                primary = labels["total"]
            else:
                labels = {"tes": base, "pcs": 0.2 + 0.6 * base}
                primary = labels[action.lower()]
            noise = ((index * 7) % 5 - 2) * 0.004
            scale = {
                "single": 1.0, "shared": 0.5,
                "conditioned": 0.1, "uniform": 0.8,
            }[branch]
            row = {
                "sample_id": str(index),
                "group_id": str(index),
                "label": primary,
                "prediction": primary + scale * noise,
            }
            if branch != "single":
                for criterion_index, name in enumerate(names):
                    row.update({
                        f"{name}_label": labels[name],
                        f"{name}_prediction": (
                            labels[name] + scale * noise
                        ),
                        f"{name}_attention_center": (
                            0.3 + 0.15 * criterion_index
                        ),
                        f"{name}_attention_entropy": 0.7,
                    })
                row["criterion_attention_divergence"] = 0.02
                if action == "Ball":
                    composed = (
                        row["difficulty_prediction"]
                        + row["execution_prediction"]
                        - row["penalty_prediction"]
                    )
                    direct = composed + 0.001
                    row.update({
                        "prediction": composed,
                        "composed_total_prediction": composed,
                        "direct_total_prediction": direct,
                        "total_consistency_gap": direct - composed,
                    })
            writer.writerow(row)


def self_test():
    with tempfile.TemporaryDirectory() as directory:
        for action in ACTIONS:
            for branch in BRANCHES:
                for split, count in (("train", 30), ("cal", 20)):
                    write_synthetic_csv(
                        os.path.join(
                            directory, action,
                            f"{action}_{branch}_{split}.csv"
                        ),
                        action, branch, split, count
                    )
            write_synthetic_csv(
                os.path.join(
                    directory, action,
                    f"{action}_conditioned_uniform_cal.csv"
                ),
                action, "uniform", "cal", 20
            )
        args = argparse.Namespace(
            root=directory,
            bootstrap=50,
            seed=0,
            min_macro_delta=0.5,
            min_action_delta=0.5,
            min_vs_shared=0.3,
            min_uniform_drop=0.2,
            max_affine_mse_drop_pct=2.0,
            min_attention_js=0.005,
            min_attention_entropy=0.1,
            max_attention_entropy=0.98,
            max_consistency_gap=0.02,
        )
        report = analyze(args)
        assert set(report["actions"]) == set(ACTIONS)
        assert "status" in report["decision"]
    print("criterion consistency analyzer self-test passed")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--output")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_macro_delta", type=float, default=0.5)
    parser.add_argument("--min_action_delta", type=float, default=0.5)
    parser.add_argument("--min_vs_shared", type=float, default=0.3)
    parser.add_argument("--min_uniform_drop", type=float, default=0.2)
    parser.add_argument(
        "--max_affine_mse_drop_pct", type=float, default=2.0
    )
    parser.add_argument(
        "--min_attention_js", type=float, default=0.005
    )
    parser.add_argument(
        "--min_attention_entropy", type=float, default=0.1
    )
    parser.add_argument(
        "--max_attention_entropy", type=float, default=0.98
    )
    parser.add_argument(
        "--max_consistency_gap", type=float, default=0.02
    )
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    if not args.root or not args.output:
        parser.error("--root and --output are required")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    return args


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    report = analyze(args)
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output)), exist_ok=True
    )
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(
            report, handle, indent=2, ensure_ascii=False,
            allow_nan=False
        )
        handle.write("\n")
    print(f"decision: {report['decision']['status']}")
    for name, passed in report["decision"]["checks"].items():
        print(f"{name}: {passed}")
    print(f"report={os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
