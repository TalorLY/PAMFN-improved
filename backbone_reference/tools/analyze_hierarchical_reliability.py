"""Audit the TES task-prior x temporal-reliability development probe."""

import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import spearmanr


VARIANTS = ("off", "motion", "skeleton", "universal", "adaptive")
SEVERITIES = (25, 50, 75, 100)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory")
    parser.add_argument("--output")
    parser.add_argument("--expected_train_samples", type=int, default=320)
    parser.add_argument("--expected_cal_samples", type=int, default=80)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and (not args.directory or not args.output):
        parser.error("--directory and --output are required")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    return args


def read_table(path, expected):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "sample_id", "label", "prediction", "motion_gate",
        "skeleton_gate", "motion_reliability", "skeleton_reliability",
        "motion_prior", "skeleton_prior",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} is missing hierarchical diagnostics")
    if len(rows) != expected:
        raise ValueError(
            f"{path} contains {len(rows)} samples; expected {expected}"
        )
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{path} contains duplicate sample IDs")
    result = {"sample_ids": sample_ids}
    for key in required - {"sample_id"}:
        result[key] = np.asarray([float(row[key]) for row in rows])
        if not np.isfinite(result[key]).all():
            raise ValueError(f"{path} contains non-finite {key}")
    for cue in ("motion", "skeleton"):
        prior = result[f"{cue}_prior"]
        if np.max(prior) - np.min(prior) > 2e-6:
            raise ValueError(
                f"{path} has sample-varying {cue} task priors"
            )
        error = np.max(np.abs(
            result[f"{cue}_gate"]
            - prior * result[f"{cue}_reliability"]
        ))
        if error > 2e-6:
            raise ValueError(
                f"{path} violates gate=prior*reliability for {cue}: {error}"
            )
    return result


def assert_aligned(reference, candidate, name):
    if candidate["sample_ids"] != reference["sample_ids"] or not (
            np.array_equal(candidate["label"], reference["label"])):
        raise ValueError(f"{name} is not aligned with the clean export")


def metric(prediction, label):
    if np.all(prediction == prediction[0]) or np.all(label == label[0]):
        coefficient = 0.0
    else:
        coefficient, _ = spearmanr(prediction, label)
        if not np.isfinite(coefficient):
            coefficient = 0.0
    return {
        "coef": float(100 * coefficient),
        "mse_raw": float(np.mean((prediction - label) ** 2)),
    }


def bootstrap(candidate, reference, labels, repeats, seed):
    rng = np.random.RandomState(seed)
    deltas = []
    mse_gains = []
    for _ in range(repeats):
        indices = rng.randint(0, len(labels), len(labels))
        candidate_metric = metric(candidate[indices], labels[indices])
        reference_metric = metric(reference[indices], labels[indices])
        deltas.append(
            candidate_metric["coef"] - reference_metric["coef"]
        )
        denominator = reference_metric["mse_raw"]
        mse_gains.append(
            (denominator - candidate_metric["mse_raw"]) / denominator
            if denominator > 0 else 0.0
        )
    return {
        "delta_coef_95ci": np.percentile(deltas, (2.5, 97.5)).tolist(),
        "relative_mse_gain_95ci": np.percentile(
            mse_gains, (2.5, 97.5)
        ).tolist(),
        "delta_coef_positive_probability": float(np.mean(
            np.asarray(deltas) > 0
        )),
    }


def monotonic_audit(clean, corrupted, cue):
    reliability_key = f"{cue}_reliability"
    gate_key = f"{cue}_gate"
    reliability = [float(clean[reliability_key].mean())]
    gates = [float(clean[gate_key].mean())]
    for severity in SEVERITIES:
        reliability.append(float(
            corrupted[(cue, severity)][reliability_key].mean()
        ))
        gates.append(float(corrupted[(cue, severity)][gate_key].mean()))
    reliability_monotonic = all(
        left + 2e-6 >= right
        for left, right in zip(reliability, reliability[1:])
    )
    gate_monotonic = all(
        left + 2e-6 >= right for left, right in zip(gates, gates[1:])
    )
    full_zero = (
        np.max(np.abs(
            corrupted[(cue, 100)][reliability_key]
        )) <= 2e-6
        and np.max(np.abs(corrupted[(cue, 100)][gate_key])) <= 2e-6
    )
    return {
        "severities": [0, *SEVERITIES],
        "mean_reliability": reliability,
        "mean_gate": gates,
        "reliability_monotonic": reliability_monotonic,
        "gate_monotonic": gate_monotonic,
        "full_corruption_zero": bool(full_zero),
    }


def analyze(args):
    tables = {}
    for split, expected in (
            ("train", args.expected_train_samples),
            ("cal", args.expected_cal_samples)):
        for variant in VARIANTS:
            path = os.path.join(
                args.directory, f"TES_hierarchical_{variant}_{split}.csv"
            )
            tables[(split, variant)] = read_table(path, expected)
        reference = tables[(split, "off")]
        for variant in VARIANTS[1:]:
            assert_aligned(
                reference, tables[(split, variant)],
                f"{split}/{variant}",
            )

    clean = tables[("cal", "adaptive")]
    corrupted = {}
    for cue in ("motion", "skeleton"):
        for severity in SEVERITIES:
            path = os.path.join(
                args.directory,
                f"TES_hierarchical_adaptive_cal_{cue}{severity}.csv",
            )
            table = read_table(path, args.expected_cal_samples)
            assert_aligned(clean, table, f"{cue}/{severity}")
            corrupted[(cue, severity)] = table

    metrics = {
        split: {
            variant: metric(
                tables[(split, variant)]["prediction"],
                tables[(split, variant)]["label"],
            )
            for variant in VARIANTS
        }
        for split in ("train", "cal")
    }
    fixed = ("off", "motion", "skeleton", "universal")
    best_mse = min(metrics["train"][name]["mse_raw"] for name in fixed)
    admissible = [
        name for name in fixed
        if metrics["train"][name]["mse_raw"] <= best_mse * 1.02 + 1e-15
    ]
    reference_name = max(
        admissible,
        key=lambda name: (
            metrics["train"][name]["coef"],
            -metrics["train"][name]["mse_raw"],
        ),
    )
    adaptive = tables[("cal", "adaptive")]["prediction"]
    reference = tables[("cal", reference_name)]["prediction"]
    labels = tables[("cal", "adaptive")]["label"]
    delta_coef = (
        metrics["cal"]["adaptive"]["coef"]
        - metrics["cal"][reference_name]["coef"]
    )
    reference_mse = metrics["cal"][reference_name]["mse_raw"]
    relative_mse_gain = (
        (reference_mse - metrics["cal"]["adaptive"]["mse_raw"])
        / reference_mse if reference_mse > 0 else 0.0
    )
    audits = {
        cue: monotonic_audit(clean, corrupted, cue)
        for cue in ("motion", "skeleton")
    }
    structural_pass = all(
        audit[check]
        for audit in audits.values()
        for check in (
            "reliability_monotonic", "gate_monotonic",
            "full_corruption_zero",
        )
    )
    clean_no_regret = delta_coef >= -0.10 and relative_mse_gain >= -0.02
    report = {
        "schema_version": 1,
        "protocol": "train-fitted TES development probe; not final OOF",
        "metrics": metrics,
        "train_selected_fixed_reference": reference_name,
        "adaptive_vs_reference_cal": {
            "delta_coef": float(delta_coef),
            "relative_mse_gain": float(relative_mse_gain),
            "bootstrap": bootstrap(
                adaptive, reference, labels, args.bootstrap, args.seed
            ),
        },
        "task_priors": {
            "motion": float(clean["motion_prior"].mean()),
            "skeleton": float(clean["skeleton_prior"].mean()),
            "motion_range": [
                float(clean["motion_prior"].min()),
                float(clean["motion_prior"].max()),
            ],
            "skeleton_range": [
                float(clean["skeleton_prior"].min()),
                float(clean["skeleton_prior"].max()),
            ],
        },
        "corruption_audit": audits,
        "decision": {
            "structural_pass": bool(structural_pass),
            "clean_no_regret": bool(clean_no_regret),
            "status": (
                "GO_NEXT_TASKS" if structural_pass and clean_no_regret
                else "STOP_OR_REVISE"
            ),
            "note": (
                "GO authorizes PCS/Ball development checks, not final claims."
            ),
        },
    }
    return report


def self_test():
    clean = {
        "motion_reliability": np.asarray([0.8, 1.0]),
        "motion_gate": np.asarray([0.4, 0.5]),
    }
    corrupted = {}
    for severity, value in zip(SEVERITIES, (0.7, 0.5, 0.2, 0.0)):
        corrupted[("motion", severity)] = {
            "motion_reliability": np.asarray([value, value]),
            "motion_gate": np.asarray([value / 2, value / 2]),
        }
    audit = monotonic_audit(clean, corrupted, "motion")
    assert audit["reliability_monotonic"]
    assert audit["gate_monotonic"]
    assert audit["full_corruption_zero"]
    print("hierarchical reliability analyzer self-test: PASS")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    report = analyze(args)
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
