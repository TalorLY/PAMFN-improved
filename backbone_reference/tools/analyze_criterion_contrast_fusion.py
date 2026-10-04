"""Analyze the TES/PCS/Ball criterion-contrast fusion screen."""

import argparse
import csv
import json
import os
import tempfile

import numpy as np
from scipy.stats import rankdata, spearmanr


ACTIONS = ("TES", "PCS", "Ball")
BRANCHES = ("single", "rank_shared", "contrast_fusion")
MODALITIES = ("rgb", "flow", "audio")


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
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if predictions.size < 2 or np.ptp(predictions) <= 1e-12:
        return None
    value = spearmanr(predictions, labels).statistic
    return None if not np.isfinite(value) else float(value * 100)


def prediction_health(predictions):
    predictions = np.asarray(predictions, dtype=np.float64)
    unique = int(len(np.unique(predictions)))
    std = float(predictions.std())
    return {
        "count": int(len(predictions)),
        "unique": unique,
        "std": std,
        "minimum": float(predictions.min()),
        "maximum": float(predictions.max()),
        "status": "ok" if unique > 1 and std > 1e-12 else "collapsed",
    }


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
    if coefficient(candidate, labels) is None:
        return {
            "status": "collapsed",
            "iterations_requested": iterations,
            "iterations_valid": 0,
            "p_delta_gt_0": None,
            "ci95": None,
        }
    rng = np.random.default_rng(seed)
    deltas = []
    sample_count = len(labels)
    for _ in range(iterations):
        indices = rng.integers(0, sample_count, sample_count)
        base_coef = coefficient(baseline[indices], labels[indices])
        candidate_coef = coefficient(candidate[indices], labels[indices])
        if base_coef is not None and candidate_coef is not None:
            deltas.append(candidate_coef - base_coef)
    if not deltas:
        return {
            "status": "undefined",
            "iterations_requested": iterations,
            "iterations_valid": 0,
            "p_delta_gt_0": None,
            "ci95": None,
        }
    values = np.asarray(deltas)
    return {
        "status": "ok",
        "iterations_requested": iterations,
        "iterations_valid": len(values),
        "p_delta_gt_0": float(np.mean(values > 0)),
        "ci95": np.percentile(values, [2.5, 97.5]).tolist(),
    }


def pair_accuracy(labels, predictions, low_margin):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    target_difference = labels[:, None] - labels[None, :]
    prediction_difference = (
        predictions[:, None] - predictions[None, :]
    )
    upper = np.triu(np.ones_like(target_difference, dtype=bool), 1)
    valid = upper & (np.abs(target_difference) > 1e-12)
    if low_margin is not None:
        valid &= np.abs(target_difference) <= low_margin
    count = int(valid.sum())
    if count == 0:
        return {"count": 0, "accuracy": None}
    ordered = prediction_difference[valid] * np.sign(
        target_difference[valid]
    )
    return {
        "count": count,
        "accuracy": float(np.mean(ordered > 0)),
    }


def criterion_names(row):
    names = [
        key[:-6] for key in row
        if key.endswith("_label") and key != "label"
    ]
    expected = (
        ["tes", "pcs"] if "tes" in names else
        ["difficulty", "execution", "penalty", "total"]
    )
    if set(names) != set(expected):
        raise ValueError(
            f"criterion columns are incomplete: {sorted(names)}"
        )
    return expected


def criterion_report(rows):
    names = criterion_names(next(iter(rows.values())))
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
            "prediction_health": prediction_health(predictions),
        }
    return report


def js_divergence(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mixture = 0.5 * (left + right)
    left_term = np.where(
        left > 0, left * np.log(left / np.maximum(mixture, 1e-12)), 0
    )
    right_term = np.where(
        right > 0, right * np.log(right / np.maximum(mixture, 1e-12)), 0
    )
    return float(0.5 * (left_term.sum() + right_term.sum()))


def fusion_report(rows):
    first = next(iter(rows.values()))
    names = criterion_names(first)
    weights = np.array([
        [
            [row[f"{name}_modality_weight_{modality}"]
             for modality in MODALITIES]
            for name in names
        ]
        for row in rows.values()
    ])
    if weights.std(axis=0).max() > 1e-6:
        raise ValueError("global criterion modality weights vary by sample")
    mean_weights = weights.mean(axis=0)
    pairwise_js = [
        js_divergence(mean_weights[left], mean_weights[right])
        for left in range(len(names))
        for right in range(left + 1, len(names))
    ]
    private_norm = np.array([
        [row[f"{name}_private_norm"] for name in names]
        for row in rows.values()
    ])
    private_cosine = np.array([
        [row[f"{name}_private_cosine"] for name in names]
        for row in rows.values()
    ])
    pre_std = np.array([
        [row[f"{name}_pre_context_std"] for name in names]
        for row in rows.values()
    ])
    post_std = np.array([
        [row[f"{name}_post_context_std"] for name in names]
        for row in rows.values()
    ])
    return {
        "criterion_names": names,
        "modality_names": list(MODALITIES),
        "mean_modality_weights": mean_weights.tolist(),
        "mean_pairwise_weight_js": float(np.mean(pairwise_js)),
        "mean_private_norm": private_norm.mean(axis=0).tolist(),
        "mean_absolute_private_cosine": float(
            np.abs(private_cosine).mean()
        ),
        "mean_pre_context_std": float(pre_std.mean()),
        "mean_post_context_std": float(post_std.mean()),
    }


def fisv_contrast_report(rows):
    labels = np.array([
        0.5 * (row["tes_label"] - row["pcs_label"])
        for row in rows.values()
    ])
    predictions = np.array([
        row["criterion_contrast_prediction"] for row in rows.values()
    ])
    return {
        "coef": coefficient(predictions, labels),
        "mse": float(np.mean((predictions - labels) ** 2)),
        "prediction_health": prediction_health(predictions),
    }


def binary_auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(scores, method="average")
    return float(
        (
            ranks[labels].sum()
            - positives * (positives + 1) / 2
        ) / (positives * negatives)
    )


def rg_report(rows, max_gap):
    labels = np.array([
        row["penalty_label"] > 1e-8 for row in rows.values()
    ])
    probabilities = np.array([
        row["penalty_event_probability"] for row in rows.values()
    ])
    identity_error = np.array([
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
        "penalty_positive_count": int(labels.sum()),
        "penalty_event_auc": binary_auc(labels, probabilities),
        "maximum_label_identity_error": float(identity_error.max()),
        "mean_absolute_direct_composed_gap": float(
            prediction_gap.mean()
        ),
        "maximum_absolute_direct_composed_gap": float(
            prediction_gap.max()
        ),
        "score_consistency_passed": bool(
            identity_error.max() <= 0.0005
            and prediction_gap.mean() <= max_gap
        ),
    }


def branch_report(
        args, action, branch, action_index, branch_index,
        baseline_train, baseline_cal, candidate_train, candidate_cal):
    _, train_labels, baseline_train_prediction, train_prediction = aligned(
        baseline_train, candidate_train, f"{action} {branch} train"
    )
    _, cal_labels, baseline_cal_prediction, cal_prediction = aligned(
        baseline_cal, candidate_cal, f"{action} {branch} calibration"
    )
    baseline_mse, baseline_slope, baseline_shift = affine_mse(
        baseline_train_prediction, train_labels,
        baseline_cal_prediction, cal_labels,
    )
    candidate_mse, candidate_slope, candidate_shift = affine_mse(
        train_prediction, train_labels, cal_prediction, cal_labels,
    )
    baseline_coef = coefficient(baseline_cal_prediction, cal_labels)
    candidate_coef = coefficient(cal_prediction, cal_labels)
    base_pair = pair_accuracy(
        cal_labels, baseline_cal_prediction, args.low_margin
    )
    candidate_pair = pair_accuracy(
        cal_labels, cal_prediction, args.low_margin
    )
    pair_gain = (
        None
        if base_pair["accuracy"] is None
        or candidate_pair["accuracy"] is None
        else candidate_pair["accuracy"] - base_pair["accuracy"]
    )
    report = {
        "base_coef": baseline_coef,
        "candidate_coef": candidate_coef,
        "delta_coef": (
            None if candidate_coef is None or baseline_coef is None
            else candidate_coef - baseline_coef
        ),
        "prediction_health": prediction_health(cal_prediction),
        "affine_mse": {
            "base": baseline_mse,
            "candidate": candidate_mse,
            "relative_gain_pct": (
                (baseline_mse - candidate_mse) / baseline_mse * 100
            ),
            "base_scale_shift": [baseline_slope, baseline_shift],
            "candidate_scale_shift": [
                candidate_slope, candidate_shift
            ],
        },
        "low_margin_pair_accuracy": {
            "base": base_pair,
            "candidate": candidate_pair,
            "gain": pair_gain,
        },
        "bootstrap": bootstrap(
            cal_labels, baseline_cal_prediction, cal_prediction,
            args.bootstrap,
            args.seed + action_index * 101 + branch_index * 17,
        ),
    }
    if branch != "single":
        report["criteria"] = criterion_report(candidate_cal)
        report["fusion"] = fusion_report(candidate_cal)
    if branch == "contrast_fusion":
        if action in ("TES", "PCS"):
            report["criterion_contrast"] = fisv_contrast_report(
                candidate_cal
            )
        else:
            report["rg_structure"] = rg_report(
                candidate_cal, args.max_consistency_gap
            )
    return report


def analyze_action(args, action, action_index):
    directory = os.path.join(args.root, action)
    rows = {}
    for branch in BRANCHES:
        for split in ("train", "cal"):
            rows[(branch, split)] = read_csv(os.path.join(
                directory, f"{action}_{branch}_{split}.csv"
            ))
    uniform = read_csv(os.path.join(
        directory, f"{action}_contrast_fusion_uniform_cal.csv"
    ))
    baseline_train = rows[("single", "train")]
    baseline_cal = rows[("single", "cal")]
    reports = {
        branch: branch_report(
            args, action, branch, action_index, branch_index,
            baseline_train, baseline_cal,
            rows[(branch, "train")], rows[(branch, "cal")],
        )
        for branch_index, branch in enumerate(BRANCHES)
    }
    _, labels, _, uniform_prediction = aligned(
        baseline_cal, uniform, f"{action} uniform fusion control"
    )
    uniform_coef = coefficient(uniform_prediction, labels)
    full_coef = reports["contrast_fusion"]["candidate_coef"]
    reports["contrast_fusion_uniform"] = {
        "candidate_coef": uniform_coef,
        "prediction_health": prediction_health(uniform_prediction),
        "drop_from_contrast_fusion": (
            None if full_coef is None or uniform_coef is None
            else full_coef - uniform_coef
        ),
    }
    return reports


def mean_if_complete(values):
    return (
        None if any(value is None for value in values)
        else float(np.mean(values))
    )


def analyze(args):
    actions = {
        action: analyze_action(args, action, index)
        for index, action in enumerate(ACTIONS)
    }
    full_delta = [
        actions[action]["contrast_fusion"]["delta_coef"]
        for action in ACTIONS
    ]
    shared_delta = [
        actions[action]["rank_shared"]["delta_coef"]
        for action in ACTIONS
    ]
    uniform_drop = [
        actions[action]["contrast_fusion_uniform"][
            "drop_from_contrast_fusion"
        ]
        for action in ACTIONS
    ]
    low_margin_gain = [
        actions[action]["contrast_fusion"][
            "low_margin_pair_accuracy"]["gain"]
        for action in ACTIONS
    ]
    macro_delta = mean_if_complete(full_delta)
    shared_macro = mean_if_complete(shared_delta)
    contrast_minus_shared = mean_if_complete([
        None if full is None or shared is None else full - shared
        for full, shared in zip(full_delta, shared_delta)
    ])
    uniform_drop_macro = mean_if_complete(uniform_drop)
    low_margin_macro = mean_if_complete(low_margin_gain)
    qualified = {
        action for action, delta in zip(ACTIONS, full_delta)
        if delta is not None and delta >= args.min_action_delta
    }
    contrast_coefs = [
        actions[action]["contrast_fusion"][
            "criterion_contrast"]["coef"]
        for action in ("TES", "PCS")
    ]
    rg_auc = actions["Ball"]["contrast_fusion"][
        "rg_structure"]["penalty_event_auc"
    ]
    checks = {
        "all_predictions_valid": bool(all(
            actions[action]["contrast_fusion"][
                "prediction_health"]["status"] == "ok"
            for action in ACTIONS
        )),
        "macro_delta": bool(
            macro_delta is not None
            and macro_delta >= args.min_macro_delta
        ),
        "two_of_three_actions": bool(len(qualified) >= 2),
        "fisv_and_rg_coverage": bool(
            "Ball" in qualified and bool({"TES", "PCS"} & qualified)
        ),
        "beats_rank_shared": bool(
            contrast_minus_shared is not None
            and contrast_minus_shared >= args.min_vs_shared
        ),
        "uniform_fusion_control": bool(
            uniform_drop_macro is not None
            and uniform_drop_macro >= args.min_uniform_drop
        ),
        "low_margin_pair_accuracy": bool(
            low_margin_macro is not None
            and low_margin_macro > args.min_low_margin_gain
        ),
        "affine_mse_guardrail": bool(all(
            actions[action]["contrast_fusion"]["affine_mse"][
                "relative_gain_pct"
            ] >= -args.max_affine_mse_drop_pct
            for action in ACTIONS
        )),
        "criterion_weights_not_uniform": bool(all(
            actions[action]["contrast_fusion"]["fusion"][
                "mean_pairwise_weight_js"
            ] >= args.min_modality_weight_js
            for action in ACTIONS
        )),
        "fisv_contrast_signal": bool(
            all(value is not None for value in contrast_coefs)
            and np.mean(contrast_coefs) >= args.min_contrast_coef
        ),
        "rg_penalty_signal": bool(
            rg_auc is not None and rg_auc >= args.min_penalty_auc
        ),
        "rg_score_consistency": bool(
            actions["Ball"]["contrast_fusion"][
                "rg_structure"]["score_consistency_passed"]
        ),
    }
    metrics = {
        "contrast_fusion_macro_delta_coef": macro_delta,
        "rank_shared_macro_delta_coef": shared_macro,
        "contrast_minus_rank_shared_macro": contrast_minus_shared,
        "uniform_fusion_drop_macro": uniform_drop_macro,
        "low_margin_pair_accuracy_gain_macro": low_margin_macro,
        "fisv_contrast_coef_macro": mean_if_complete(contrast_coefs),
        "ball_penalty_event_auc": rg_auc,
        "qualified_actions": sorted(qualified),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "actions": actions,
        "decision": {
            "status": (
                "GO_TO_THREE_SEED_SIX_ACTION_CRITERION_CONTRAST"
                if passed else
                "STOP_OR_REDESIGN_CRITERION_CONTRAST"
            ),
            "checks": checks,
            "metrics": metrics,
            "interpretation": (
                "The matched criterion-contrast fusion passed every "
                "pre-registered ranking, calibration, and mechanism check."
                if passed else
                "Do not run the six-action experiment. Inspect failed "
                "checks; collapsed predictions are recorded instead of "
                "crashing the analyzer."
            ),
        },
    }


def synthetic_row(action, branch, index, count):
    base = (index + 1) / (count + 2)
    if action == "Ball":
        labels = {
            "difficulty": 0.25 + 0.25 * base,
            "execution": 0.15 + 0.20 * base,
            "penalty": 0.01 if index % 5 == 0 else 0.0,
        }
        labels["total"] = (
            labels["difficulty"] + labels["execution"]
            - labels["penalty"]
        )
        primary = labels["total"]
    else:
        labels = {
            "tes": 0.25 + 0.65 * base,
            "pcs": 0.20 + 0.55 * base + 0.02 * ((index % 3) - 1),
        }
        primary = labels[action.lower()]
    noise = ((index * 7) % 5 - 2) * 0.004
    scale = {
        "single": 1.0,
        "rank_shared": 0.6,
        "contrast_fusion": 0.1,
        "uniform": 0.8,
    }[branch]
    row = {
        "sample_id": str(index),
        "group_id": str(index),
        "label": primary,
        "prediction": primary + scale * noise,
    }
    if branch == "single":
        return row
    names = list(labels)
    diverse_weights = [
        [0.70, 0.20, 0.10],
        [0.10, 0.70, 0.20],
        [0.20, 0.10, 0.70],
        [0.45, 0.45, 0.10],
    ]
    for criterion_index, name in enumerate(names):
        row[f"{name}_label"] = labels[name]
        row[f"{name}_prediction"] = labels[name] + scale * noise
        weights = (
            [1 / 3] * 3
            if branch in ("rank_shared", "uniform")
            else diverse_weights[criterion_index]
        )
        for modality, weight in zip(MODALITIES, weights):
            row[f"{name}_modality_weight_{modality}"] = weight
        row[f"{name}_private_norm"] = (
            0.0 if branch in ("rank_shared", "uniform") else 0.2
        )
        row[f"{name}_private_cosine"] = 0.05
        row[f"{name}_pre_context_std"] = 0.7
        row[f"{name}_post_context_std"] = 1.0
    row["criterion_shared_norm"] = 2.0
    if action in ("TES", "PCS"):
        row["common_quality_prediction"] = 0.5 * (
            labels["tes"] + labels["pcs"]
        )
        row["criterion_contrast_prediction"] = 0.5 * (
            labels["tes"] - labels["pcs"]
        ) + scale * noise
    else:
        composed = (
            row["difficulty_prediction"]
            + row["execution_prediction"]
            - row["penalty_prediction"]
        )
        direct = composed + 0.001
        row.update({
            "prediction": composed,
            "penalty_event_probability": (
                0.9 if labels["penalty"] > 0 else 0.1
            ),
            "penalty_amount_prediction": labels["penalty"],
            "composed_total_prediction": composed,
            "direct_total_prediction": direct,
            "total_consistency_gap": direct - composed,
        })
    return row


def write_synthetic_csv(path, action, branch, count):
    rows = [
        synthetic_row(action, branch, index, count)
        for index in range(count)
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    with tempfile.TemporaryDirectory() as directory:
        for action in ACTIONS:
            for branch in BRANCHES:
                for split, count in (("train", 30), ("cal", 20)):
                    write_synthetic_csv(
                        os.path.join(
                            directory, action,
                            f"{action}_{branch}_{split}.csv",
                        ),
                        action, branch, count,
                    )
            write_synthetic_csv(
                os.path.join(
                    directory, action,
                    f"{action}_contrast_fusion_uniform_cal.csv",
                ),
                action, "uniform", 20,
            )
        args = argparse.Namespace(
            root=directory,
            bootstrap=50,
            seed=0,
            low_margin=0.1,
            min_macro_delta=0.5,
            min_action_delta=0.5,
            min_vs_shared=0.3,
            min_uniform_drop=0.2,
            min_low_margin_gain=0.0,
            max_affine_mse_drop_pct=2.0,
            min_modality_weight_js=0.005,
            min_contrast_coef=10.0,
            min_penalty_auc=0.55,
            max_consistency_gap=0.02,
        )
        report = analyze(args)
        assert set(report["actions"]) == set(ACTIONS)
        assert "status" in report["decision"]
        assert report["actions"]["Ball"]["contrast_fusion"][
            "rg_structure"]["penalty_event_auc"
        ] > 0.9
    print("criterion contrast fusion analyzer self-test passed")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--output")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--low_margin", type=float, default=0.05)
    parser.add_argument("--min_macro_delta", type=float, default=0.5)
    parser.add_argument("--min_action_delta", type=float, default=0.5)
    parser.add_argument("--min_vs_shared", type=float, default=0.3)
    parser.add_argument("--min_uniform_drop", type=float, default=0.2)
    parser.add_argument("--min_low_margin_gain", type=float, default=0.0)
    parser.add_argument(
        "--max_affine_mse_drop_pct", type=float, default=2.0
    )
    parser.add_argument(
        "--min_modality_weight_js", type=float, default=0.005
    )
    parser.add_argument("--min_contrast_coef", type=float, default=10.0)
    parser.add_argument("--min_penalty_auc", type=float, default=0.55)
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
    if args.low_margin <= 0:
        parser.error("--low_margin must be positive")
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
            allow_nan=False,
        )
        handle.write("\n")
    print(f"decision: {report['decision']['status']}")
    for name, passed in report["decision"]["checks"].items():
        print(f"{name}: {passed}")
    print(f"report={os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
