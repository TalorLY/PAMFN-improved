"""Analyze the three-branch C-OPRR screen without using official test data."""

import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import spearmanr


BRANCHES = ("global", "fixed", "ordered")
CONTROLS = ("ordered_reverse", "ordered_shuffle")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--output")
    parser.add_argument("--actions", nargs="+", default=("TES", "PCS", "Ball"))
    parser.add_argument("--pair_audit_dir")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_macro_delta", type=float, default=0.5)
    parser.add_argument("--min_action_delta", type=float, default=0.5)
    parser.add_argument("--min_vs_control", type=float, default=0.3)
    parser.add_argument("--max_affine_mse_drop_pct", type=float, default=2.0)
    parser.add_argument("--min_phase_mass", type=float, default=0.1)
    parser.add_argument("--min_center_gap", type=float, default=0.1)
    parser.add_argument("--min_order_control_drop", type=float, default=0.1)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    for name in ("root", "output", "pair_audit_dir"):
        if not getattr(args, name):
            parser.error(f"--{name} is required")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    return args


def read_predictions(path):
    rows = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = row["sample_id"]
            if sample_id in rows:
                raise ValueError(f"{path} contains duplicate sample IDs")
            converted = dict(row)
            for key, value in row.items():
                if key in ("sample_id", "group_id") or value == "":
                    continue
                try:
                    converted[key] = float(value)
                except ValueError:
                    pass
            rows[sample_id] = converted
    if not rows:
        raise ValueError(f"{path} contains no predictions")
    return rows


def aligned(reference, candidate, description):
    if set(reference) != set(candidate):
        raise ValueError(f"{description} sample IDs do not align")
    sample_ids = sorted(reference)
    labels = np.array(
        [reference[sample_id]["label"] for sample_id in sample_ids],
        dtype=np.float64,
    )
    candidate_labels = np.array(
        [candidate[sample_id]["label"] for sample_id in sample_ids],
        dtype=np.float64,
    )
    if not np.allclose(labels, candidate_labels, atol=1e-8):
        raise ValueError(f"{description} labels do not align")
    return (
        sample_ids,
        labels,
        np.array(
            [reference[sample_id]["prediction"] for sample_id in sample_ids],
            dtype=np.float64,
        ),
        np.array(
            [candidate[sample_id]["prediction"] for sample_id in sample_ids],
            dtype=np.float64,
        ),
    )


def coefficient(prediction, label):
    value = spearmanr(prediction, label).statistic
    return float(value * 100)


def fit_affine(prediction, label):
    centered = prediction - prediction.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-15:
        slope = 0.0
    else:
        slope = max(
            0.0,
            float(np.dot(centered, label - label.mean()) / denominator),
        )
    shift = float(label.mean() - slope * prediction.mean())
    return slope, shift


def affine_mse(train_prediction, train_label, prediction, label):
    slope, shift = fit_affine(train_prediction, train_label)
    calibrated = slope * prediction + shift
    return float(np.mean((calibrated - label) ** 2)), slope, shift


def bootstrap_probability(
        labels, base_prediction, candidate_prediction, iterations, seed):
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(iterations):
        indices = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        base = coefficient(base_prediction[indices], sampled_labels)
        candidate = coefficient(candidate_prediction[indices], sampled_labels)
        if np.isfinite(base) and np.isfinite(candidate):
            deltas.append(candidate - base)
    if not deltas:
        return None
    values = np.asarray(deltas)
    return {
        "iterations": int(len(values)),
        "p_delta_gt_0": float(np.mean(values > 0)),
        "ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def pair_audit(action, variant, sample_ids, labels, base, candidate):
    records = []
    base_margins = []
    comparable = []
    for left in range(len(sample_ids)):
        for right in range(left + 1, len(sample_ids)):
            label_margin = labels[left] - labels[right]
            if abs(label_margin) <= 1e-8:
                continue
            base_margin = base[left] - base[right]
            candidate_margin = candidate[left] - candidate[right]
            comparable.append((
                left, right, label_margin, base_margin, candidate_margin
            ))
            base_margins.append(abs(base_margin))
    if not comparable:
        raise ValueError(f"{action} has no comparable calibration pairs")
    low_threshold = float(np.quantile(base_margins, 0.25))
    high_threshold = float(np.quantile(base_margins, 0.75))
    for left, right, label_margin, base_margin, candidate_margin in comparable:
        base_correct = base_margin * label_margin > 0
        candidate_correct = candidate_margin * label_margin > 0
        low_margin = abs(base_margin) <= low_threshold
        high_confidence = (
            base_correct and abs(base_margin) >= high_threshold
        )
        records.append({
            "action": action,
            "variant": variant,
            "pair_id": f"{sample_ids[left]}__{sample_ids[right]}",
            "left_id": sample_ids[left],
            "right_id": sample_ids[right],
            "label_margin": float(label_margin),
            "base_margin": float(base_margin),
            "candidate_margin": float(candidate_margin),
            "base_correct": bool(base_correct),
            "candidate_correct": bool(candidate_correct),
            "low_margin": bool(low_margin),
            "high_confidence": bool(high_confidence),
            "high_confidence_inversion": bool(
                high_confidence and not candidate_correct
            ),
        })
    low = [record for record in records if record["low_margin"]]
    protected = [
        record for record in records if record["high_confidence"]
    ]
    return records, {
        "comparable_pairs": len(records),
        "low_margin_threshold": low_threshold,
        "high_confidence_threshold": high_threshold,
        "base_pair_accuracy": float(np.mean([
            record["base_correct"] for record in records
        ])),
        "candidate_pair_accuracy": float(np.mean([
            record["candidate_correct"] for record in records
        ])),
        "low_margin_pairs": len(low),
        "low_margin_delta_pair_accuracy": float(
            np.mean([record["candidate_correct"] for record in low])
            - np.mean([record["base_correct"] for record in low])
        ),
        "protected_pairs": len(protected),
        "high_confidence_inversion_rate": (
            float(np.mean([
                record["high_confidence_inversion"]
                for record in protected
            ]))
            if protected else 0.0
        ),
    }


def phase_diagnostics(rows, min_mass, min_gap):
    first = next(iter(rows.values()))
    mass_columns = sorted(
        (key for key in first if key.startswith("phase_mass_")),
        key=lambda key: int(key.rsplit("_", 1)[1]),
    )
    center_columns = sorted(
        (key for key in first if key.startswith("phase_center_")),
        key=lambda key: int(key.rsplit("_", 1)[1]),
    )
    if not mass_columns or len(mass_columns) != len(center_columns):
        raise ValueError("ordered export is missing phase diagnostics")
    masses = np.array([
        [row[key] for key in mass_columns] for row in rows.values()
    ])
    centers = np.array([
        [row[key] for key in center_columns] for row in rows.values()
    ])
    residuals = np.array([
        row["phase_residual"] for row in rows.values()
    ])
    entropies = np.array([
        row["phase_entropy"] for row in rows.values()
    ])
    mean_mass = masses.mean(axis=0)
    mean_center = centers.mean(axis=0)
    center_gaps = np.diff(mean_center)
    ordered_fraction = float(np.mean(np.all(
        np.diff(centers, axis=1) > 0, axis=1
    )))
    passed = bool(
        mean_mass.min() >= min_mass
        and center_gaps.min() >= min_gap
        and ordered_fraction == 1.0
    )
    return {
        "phase_count": len(mass_columns),
        "mean_mass": mean_mass.tolist(),
        "mean_center": mean_center.tolist(),
        "minimum_mean_mass": float(mean_mass.min()),
        "minimum_center_gap": float(center_gaps.min()),
        "strictly_ordered_sample_fraction": ordered_fraction,
        "mean_entropy": float(entropies.mean()),
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "passed": passed,
    }


def write_pair_audit(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "action", "variant", "pair_id", "left_id", "right_id",
        "label_margin", "base_margin", "candidate_margin",
        "base_correct", "candidate_correct", "low_margin",
        "high_confidence", "high_confidence_inversion",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def analyze_action(args, action, action_index):
    directory = os.path.join(args.root, action)
    base_train_rows = read_predictions(
        os.path.join(directory, f"{action}_base_train.csv")
    )
    base_cal_rows = read_predictions(
        os.path.join(directory, f"{action}_base_cal.csv")
    )
    reports = {}
    all_pair_records = []
    for branch_index, branch in enumerate(BRANCHES):
        train_rows = read_predictions(
            os.path.join(directory, f"{action}_{branch}_train.csv")
        )
        cal_rows = read_predictions(
            os.path.join(directory, f"{action}_{branch}_cal.csv")
        )
        _, train_labels, base_train, candidate_train = aligned(
            base_train_rows, train_rows, f"{action} {branch} train"
        )
        sample_ids, cal_labels, base_cal, candidate_cal = aligned(
            base_cal_rows, cal_rows, f"{action} {branch} calibration"
        )
        base_coef = coefficient(base_cal, cal_labels)
        candidate_coef = coefficient(candidate_cal, cal_labels)
        base_mse, base_slope, base_shift = affine_mse(
            base_train, train_labels, base_cal, cal_labels
        )
        candidate_mse, candidate_slope, candidate_shift = affine_mse(
            candidate_train, train_labels, candidate_cal, cal_labels
        )
        records, pair_report = pair_audit(
            action, branch, sample_ids, cal_labels, base_cal, candidate_cal
        )
        all_pair_records.extend(records)
        reports[branch] = {
            "base_coef": base_coef,
            "candidate_coef": candidate_coef,
            "delta_coef": candidate_coef - base_coef,
            "affine_mse": {
                "base": base_mse,
                "candidate": candidate_mse,
                "relative_gain_pct": (
                    (base_mse - candidate_mse) / base_mse * 100
                ),
                "base_scale_shift": [base_slope, base_shift],
                "candidate_scale_shift": [
                    candidate_slope, candidate_shift
                ],
            },
            "pairwise": pair_report,
            "bootstrap": bootstrap_probability(
                cal_labels, base_cal, candidate_cal,
                args.bootstrap,
                args.seed + action_index * 101 + branch_index * 13,
            ),
        }
        if branch == "ordered":
            reports[branch]["phase"] = phase_diagnostics(
                cal_rows, args.min_phase_mass, args.min_center_gap
            )

    for control_index, control in enumerate(CONTROLS):
        control_rows = read_predictions(
            os.path.join(directory, f"{action}_{control}_cal.csv")
        )
        sample_ids, labels, base, candidate = aligned(
            base_cal_rows, control_rows, f"{action} {control}"
        )
        records, pair_report = pair_audit(
            action, control, sample_ids, labels, base, candidate
        )
        all_pair_records.extend(records)
        reports[control] = {
            "base_coef": coefficient(base, labels),
            "candidate_coef": coefficient(candidate, labels),
            "delta_coef": (
                coefficient(candidate, labels) - coefficient(base, labels)
            ),
            "pairwise": pair_report,
        }
    reports["ordered"]["order_control"] = {
        "reverse_drop": (
            reports["ordered"]["candidate_coef"]
            - reports["ordered_reverse"]["candidate_coef"]
        ),
        "shuffle_drop": (
            reports["ordered"]["candidate_coef"]
            - reports["ordered_shuffle"]["candidate_coef"]
        ),
    }
    write_pair_audit(
        os.path.join(args.pair_audit_dir, f"{action}_pair_audit.csv"),
        all_pair_records,
    )
    return reports


def analyze(args):
    actions = {
        action: analyze_action(args, action, index)
        for index, action in enumerate(args.actions)
    }
    ordered_deltas = np.array([
        actions[action]["ordered"]["delta_coef"] for action in args.actions
    ])
    global_deltas = np.array([
        actions[action]["global"]["delta_coef"] for action in args.actions
    ])
    fixed_deltas = np.array([
        actions[action]["fixed"]["delta_coef"] for action in args.actions
    ])
    low_pair_deltas = np.array([
        actions[action]["ordered"]["pairwise"][
            "low_margin_delta_pair_accuracy"
        ]
        for action in args.actions
    ])
    reverse_drops = np.array([
        actions[action]["ordered"]["order_control"]["reverse_drop"]
        for action in args.actions
    ])
    shuffle_drops = np.array([
        actions[action]["ordered"]["order_control"]["shuffle_drop"]
        for action in args.actions
    ])
    qualified = {
        action for action in args.actions
        if actions[action]["ordered"]["delta_coef"] >= args.min_action_delta
    }
    family_coverage = (
        "Ball" in qualified
        and bool({"TES", "PCS"} & qualified)
    )
    checks = {
        "macro_delta": bool(
            ordered_deltas.mean() >= args.min_macro_delta
        ),
        "two_of_three_actions": bool(len(qualified) >= 2),
        "fisv_and_rg_coverage": bool(family_coverage),
        "beats_global": bool(
            (ordered_deltas - global_deltas).mean()
            >= args.min_vs_control
        ),
        "beats_fixed": bool(
            (ordered_deltas - fixed_deltas).mean()
            >= args.min_vs_control
        ),
        "low_margin_pair_accuracy": bool(low_pair_deltas.mean() > 0),
        "affine_mse_guardrail": bool(all(
            actions[action]["ordered"]["affine_mse"][
                "relative_gain_pct"
            ] >= -args.max_affine_mse_drop_pct
            for action in args.actions
        )),
        "phase_not_collapsed": bool(all(
            actions[action]["ordered"]["phase"]["passed"]
            for action in args.actions
        )),
        "reverse_control": bool(
            reverse_drops.mean() >= args.min_order_control_drop
        ),
        "shuffle_control": bool(
            shuffle_drops.mean() >= args.min_order_control_drop
        ),
    }
    metrics = {
        "ordered_macro_delta_coef": float(ordered_deltas.mean()),
        "global_macro_delta_coef": float(global_deltas.mean()),
        "fixed_macro_delta_coef": float(fixed_deltas.mean()),
        "ordered_minus_global_macro": float(
            (ordered_deltas - global_deltas).mean()
        ),
        "ordered_minus_fixed_macro": float(
            (ordered_deltas - fixed_deltas).mean()
        ),
        "ordered_low_margin_pair_delta_macro": float(
            low_pair_deltas.mean()
        ),
        "reverse_drop_macro": float(reverse_drops.mean()),
        "shuffle_drop_macro": float(shuffle_drops.mean()),
        "qualified_actions": sorted(qualified),
    }
    return {
        "schema_version": 1,
        "actions": actions,
        "decision": {
            "status": (
                "GO_TO_THREE_SEED_SIX_ACTION"
                if all(checks.values())
                else "STOP_PAMFN_ADAPTER_GATE_FAMILY"
            ),
            "checks": checks,
            "metrics": metrics,
            "rule": (
                "All pre-registered checks must pass. A failure stops the "
                "PAMFN adapter/gate family rather than starting another gate."
            ),
        },
    }


def self_test():
    labels = np.array([0.1, 0.2, 0.4, 0.8])
    base = np.array([0.1, 0.3, 0.2, 0.7])
    improved = np.array([0.1, 0.2, 0.4, 0.8])
    ids = ["a", "b", "c", "d"]
    records, report = pair_audit(
        "TES", "ordered", ids, labels, base, improved
    )
    assert records
    assert report["candidate_pair_accuracy"] > report["base_pair_accuracy"]
    slope, _ = fit_affine(base, labels)
    assert slope >= 0
    boot = bootstrap_probability(labels, base, improved, 100, 0)
    assert 0 <= boot["p_delta_gt_0"] <= 1
    phase_rows = {
        sample_id: {
            "phase_mass_0": 0.3,
            "phase_mass_1": 0.4,
            "phase_mass_2": 0.3,
            "phase_center_0": 0.16,
            "phase_center_1": 0.50,
            "phase_center_2": 0.84,
            "phase_residual": index * 0.01,
            "phase_entropy": 0.2,
        }
        for index, sample_id in enumerate(ids)
    }
    assert phase_diagnostics(phase_rows, 0.1, 0.1)["passed"]
    print("phase rank analyzer self-test passed")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    report = analyze(args)
    parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print("decision:", report["decision"]["status"])
    for key, value in report["decision"]["checks"].items():
        print(f"{key}: {value}")
    print(f"report={os.path.abspath(args.output)}")
    print(f"pair_audits={os.path.abspath(args.pair_audit_dir)}")


if __name__ == "__main__":
    main()
