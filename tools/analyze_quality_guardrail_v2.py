"""Audit the TES task-prior x quality-guardrail v2 development probe."""

import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import spearmanr


VARIANTS = (
    "off", "motion", "skeleton", "universal", "prior_only", "guardrail"
)
SEVERITIES = (25, 50, 75, 100)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action", default="TES",
        choices=("TES", "PCS", "Ball", "Clubs", "Hoop", "Ribbon"),
    )
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


def read_table(path, expected, reliability_disabled=False):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "sample_id", "label", "prediction", "motion_gate",
        "skeleton_gate", "motion_reliability", "skeleton_reliability",
        "motion_prior", "skeleton_prior", "motion_quality_raw",
        "skeleton_quality_raw",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} is missing guardrail-v2 diagnostics")
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
        if np.ptp(prior) > 2e-6:
            raise ValueError(f"{path} has sample-varying {cue} task priors")
        expected_gate = (
            prior if reliability_disabled
            else prior * result[f"{cue}_reliability"]
        )
        error = np.max(np.abs(result[f"{cue}_gate"] - expected_gate))
        if error > 2e-6:
            policy = "prior" if reliability_disabled else "prior*quality"
            raise ValueError(
                f"{path} violates gate={policy} for {cue}: {error}"
            )
    return result


def assert_aligned(reference, candidate, name):
    if candidate["sample_ids"] != reference["sample_ids"] or not (
            np.array_equal(candidate["label"], reference["label"])):
        raise ValueError(f"{name} is not sample/label aligned")


def metric(prediction, label):
    coefficient, _ = spearmanr(prediction, label)
    if not np.isfinite(coefficient):
        coefficient = 0.0
    return {
        "coef": float(100 * coefficient),
        "mse_raw": float(np.mean((prediction - label) ** 2)),
    }


def bootstrap(candidate, reference, labels, repeats, seed):
    rng = np.random.RandomState(seed)
    delta_coef = []
    relative_mse_gain = []
    for _ in range(repeats):
        indices = rng.randint(0, len(labels), len(labels))
        candidate_metric = metric(candidate[indices], labels[indices])
        reference_metric = metric(reference[indices], labels[indices])
        delta_coef.append(
            candidate_metric["coef"] - reference_metric["coef"]
        )
        denominator = reference_metric["mse_raw"]
        relative_mse_gain.append(
            (denominator - candidate_metric["mse_raw"]) / denominator
            if denominator > 0 else 0.0
        )
    return {
        "delta_coef_95ci": np.percentile(
            delta_coef, (2.5, 97.5)
        ).tolist(),
        "relative_mse_gain_95ci": np.percentile(
            relative_mse_gain, (2.5, 97.5)
        ).tolist(),
        "delta_coef_positive_probability": float(
            np.mean(np.asarray(delta_coef) > 0)
        ),
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
    full = corrupted[(cue, 100)]
    return {
        "severities": [0, *SEVERITIES],
        "mean_reliability": reliability,
        "mean_gate": gates,
        "reliability_monotonic": all(
            left + 2e-6 >= right
            for left, right in zip(reliability, reliability[1:])
        ),
        "gate_monotonic": all(
            left + 2e-6 >= right
            for left, right in zip(gates, gates[1:])
        ),
        "full_corruption_zero": bool(
            np.max(np.abs(full[reliability_key])) <= 2e-6
            and np.max(np.abs(full[gate_key])) <= 2e-6
        ),
    }


def analyze(args):
    tables = {}
    disabled_variants = set(VARIANTS) - {"guardrail"}
    for split, expected in (
            ("train", args.expected_train_samples),
            ("cal", args.expected_cal_samples)):
        for variant in VARIANTS:
            path = os.path.join(
                args.directory,
                f"{args.action}_guardrail_v2_{variant}_{split}.csv",
            )
            tables[(split, variant)] = read_table(
                path, expected, variant in disabled_variants
            )
        reference = tables[(split, "off")]
        for variant in VARIANTS[1:]:
            assert_aligned(
                reference, tables[(split, variant)],
                f"{split}/{variant}",
            )

    clean = tables[("cal", "guardrail")]
    corrupted = {"guardrail": {}, "prior_only": {}}
    for policy in corrupted:
        for cue in ("motion", "skeleton"):
            for severity in SEVERITIES:
                path = os.path.join(
                    args.directory,
                    f"{args.action}_guardrail_v2_{policy}_cal_"
                    f"{cue}{severity}.csv",
                )
                table = read_table(
                    path, args.expected_cal_samples,
                    reliability_disabled=(policy == "prior_only"),
                )
                assert_aligned(clean, table, f"{policy}/{cue}/{severity}")
                corrupted[policy][(cue, severity)] = table

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
    guard_metric = metrics["cal"]["guardrail"]
    prior_metric = metrics["cal"]["prior_only"]
    delta_coef = guard_metric["coef"] - prior_metric["coef"]
    relative_mse_gain = (
        (prior_metric["mse_raw"] - guard_metric["mse_raw"])
        / prior_metric["mse_raw"] if prior_metric["mse_raw"] > 0 else 0.0
    )
    audits = {
        cue: monotonic_audit(clean, corrupted["guardrail"], cue)
        for cue in ("motion", "skeleton")
    }
    structural_pass = all(
        audit[key]
        for audit in audits.values()
        for key in (
            "reliability_monotonic", "gate_monotonic",
            "full_corruption_zero",
        )
    )
    priors = {
        cue: float(clean[f"{cue}_prior"].mean())
        for cue in ("motion", "skeleton")
    }
    full_corruption = {}
    active_robustness_pass = True
    labels = clean["label"]
    for cue in ("motion", "skeleton"):
        guarded = metric(
            corrupted["guardrail"][(cue, 100)]["prediction"], labels
        )
        unguarded = metric(
            corrupted["prior_only"][(cue, 100)]["prediction"], labels
        )
        active = priors[cue] > 1e-6
        passed = (not active) or (
            guarded["mse_raw"] <= unguarded["mse_raw"] + 1e-12
        )
        active_robustness_pass = active_robustness_pass and passed
        full_corruption[cue] = {
            "active_prior": active,
            "guardrail": guarded,
            "prior_only": unguarded,
            "mse_guardrail_not_worse": bool(passed),
        }
    clean_no_regret = delta_coef >= -0.10 and relative_mse_gain >= -0.02
    status = (
        "GO_NEXT_TASKS"
        if structural_pass and clean_no_regret and active_robustness_pass
        else "STOP_OR_REVISE"
    )
    return {
        "schema_version": 2,
        "action": args.action,
        "protocol": (
            f"train-fitted {args.action} development probe; not final OOF"
        ),
        "metrics": metrics,
        "task_priors": priors,
        "guardrail_vs_prior_only_cal": {
            "delta_coef": float(delta_coef),
            "relative_mse_gain": float(relative_mse_gain),
            "bootstrap": bootstrap(
                clean["prediction"],
                tables[("cal", "prior_only")]["prediction"],
                labels, args.bootstrap, args.seed,
            ),
        },
        "corruption_audit": audits,
        "full_corruption_robustness": full_corruption,
        "decision": {
            "structural_pass": bool(structural_pass),
            "clean_no_regret": bool(clean_no_regret),
            "active_cue_robustness_pass": bool(active_robustness_pass),
            "status": status,
            "note": (
                "GO permits PCS/Ball seed-0 development checks, not final "
                "multi-seed claims."
            ),
        },
    }


def self_test():
    clean = {
        "motion_reliability": np.asarray([1.0, 1.0]),
        "motion_gate": np.asarray([0.5, 0.5]),
    }
    corrupted = {}
    for severity, value in zip(SEVERITIES, (0.75, 0.5, 0.25, 0.0)):
        corrupted[("motion", severity)] = {
            "motion_reliability": np.asarray([value, value]),
            "motion_gate": np.asarray([value / 2, value / 2]),
        }
    audit = monotonic_audit(clean, corrupted, "motion")
    assert audit["reliability_monotonic"]
    assert audit["gate_monotonic"]
    assert audit["full_corruption_zero"]
    print("quality guardrail v2 analyzer self-test: PASS")


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
