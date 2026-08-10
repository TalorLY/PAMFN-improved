#!/usr/bin/env python3
"""Cross-action decision for the minimal feature-level temporal screen."""

import argparse
import json
import os

import numpy as np

import analyze_direct_cue_effect as direct


ACTIONS = ("TES", "PCS", "Ball")
CUES = ("motion", "skeleton")
EXPECTED = {"TES": (320, 80), "PCS": (320, 80), "Ball": (160, 40)}
EPSILON = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze fixed endpoint feature-level temporal experts"
    )
    parser.add_argument("--root")
    parser.add_argument("--output")
    parser.add_argument("--actions", nargs="+", default=list(ACTIONS))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anchor_atol", type=float, default=1e-6)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    if not args.root or not args.output:
        parser.error("--root and --output are required")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    if any(action not in ACTIONS for action in args.actions):
        parser.error("actions must be TES, PCS, or Ball")
    if not np.isfinite(args.anchor_atol) or args.anchor_atol < 0:
        parser.error("--anchor_atol must be finite and non-negative")
    return args


def ensure_same_rows(reference, candidate, description):
    if reference["sample_ids"] != candidate["sample_ids"]:
        raise ValueError(description + " has mismatched sample IDs/order")
    if reference["group_ids"] != candidate["group_ids"]:
        raise ValueError(description + " has mismatched group IDs")
    if not np.array_equal(reference["labels"], candidate["labels"]):
        raise ValueError(description + " has mismatched labels")


def pairwise_order_diagnostics(labels, candidate, control):
    labels = np.asarray(labels, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    left, right = np.triu_indices(labels.size, k=1)
    target_delta = labels[left] - labels[right]
    usable = np.abs(target_delta) > EPSILON
    if not np.any(usable):
        return {"comparable_pairs": 0}
    target_sign = np.sign(target_delta[usable])
    control_delta = (control[left] - control[right])[usable]
    candidate_delta = (candidate[left] - candidate[right])[usable]
    control_sign = np.sign(control_delta)
    candidate_sign = np.sign(candidate_delta)
    control_correct = control_sign == target_sign
    candidate_correct = candidate_sign == target_sign
    control_tie = control_sign == 0
    candidate_tie = candidate_sign == 0
    margins = np.abs(control_delta)
    low_margin = margins <= np.percentile(margins, 25)

    def summarize(mask):
        count = int(np.sum(mask))
        if not count:
            return {"pairs": 0, "delta_pair_accuracy": None}
        old = (control_correct[mask] + 0.5 * control_tie[mask]).astype(float)
        new = (candidate_correct[mask] + 0.5 * candidate_tie[mask]).astype(float)
        return {
            "pairs": count,
            "control_pair_accuracy": float(np.mean(old)),
            "candidate_pair_accuracy": float(np.mean(new)),
            "delta_pair_accuracy": float(np.mean(new) - np.mean(old)),
            "wrong_to_correct_pairs": int(np.sum(
                (~control_correct[mask]) & candidate_correct[mask]
            )),
            "correct_to_wrong_pairs": int(np.sum(
                control_correct[mask] & (~candidate_correct[mask])
            )),
        }

    return {
        "comparable_pairs": int(np.sum(usable)),
        "all_pairs": summarize(np.ones(np.sum(usable), dtype=bool)),
        "lowest_control_margin_quartile": summarize(low_margin),
    }


def group_bootstrap(table, on, off, iterations, seed):
    group_rows = {}
    for index, group_id in enumerate(table["group_ids"]):
        group_rows.setdefault(group_id, []).append(index)
    groups = sorted(group_rows)
    if not groups:
        raise ValueError("calibration contains no groups")
    rng = np.random.RandomState(seed)
    coef_gains = []
    mse_gains = []
    for _ in range(iterations):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.asarray([
            index for group in sampled for index in group_rows[group]
        ], dtype=np.int64)
        on_metrics = direct.prediction_metrics(
            on[indices], table["labels"][indices]
        )
        off_metrics = direct.prediction_metrics(
            off[indices], table["labels"][indices]
        )
        gain = direct.comparison(on_metrics, off_metrics)
        if gain["delta_coef"] is not None:
            coef_gains.append(gain["delta_coef"])
        mse_gains.append(gain["relative_mse_gain"])
    coef_array = np.asarray(coef_gains, dtype=np.float64)
    mse_array = np.asarray(mse_gains, dtype=np.float64)

    def interval(values):
        if not len(values):
            return [None, None]
        return [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]

    return {
        "iterations": int(iterations),
        "resampling_unit": "group_id",
        "delta_coef": {
            "ci95": interval(coef_array),
            "probability_at_least_0p5": (
                None if not len(coef_array) else
                float(np.mean(coef_array >= 0.5))
            ),
        },
        "relative_mse_gain": {
            "ci95": interval(mse_array),
            "probability_not_worse_than_minus_2pct": float(
                np.mean(mse_array >= -0.02)
            ),
        },
    }


def read_split(root, action, cue, split):
    action_dir = os.path.join(root, action)
    paths = {
        "base": os.path.join(action_dir, f"{action}_base_{split}.csv"),
        "off": os.path.join(
            action_dir, f"{action}_{cue}_off_{split}.csv"
        ),
        "on": os.path.join(
            action_dir, f"{action}_{cue}_on_{split}.csv"
        ),
    }
    tables = {name: direct.read_prediction_csv(path)
              for name, path in paths.items()}
    ensure_same_rows(tables["base"], tables["off"], f"{action}/{cue}/{split}/off")
    ensure_same_rows(tables["base"], tables["on"], f"{action}/{cue}/{split}/on")
    return tables


def verify_log(root, action, cue):
    path = os.path.join(root, action, f"{action}_{cue}_seed0.log")
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    required = (
        "Temporal feature screen:",
        "Frozen Base invariant: passed",
        "Blind training complete;",
        "Initialized from ",
        f"{action}_inner_base_seed0.pth",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise ValueError(f"{path} is missing protocol evidence: {missing}")
    return os.path.abspath(path)


def analyze_action_cue(root, action, cue, bootstrap, seed, atol):
    train = read_split(root, action, cue, "train")
    cal = read_split(root, action, cue, "cal")
    expected_train, expected_cal = EXPECTED[action]
    if len(train["base"]["labels"]) != expected_train:
        raise ValueError(f"{action} expected {expected_train} train samples")
    if len(cal["base"]["labels"]) != expected_cal:
        raise ValueError(f"{action} expected {expected_cal} calibration samples")
    if set(train["base"]["sample_ids"]) & set(cal["base"]["sample_ids"]):
        raise ValueError(f"{action} train/calibration sample IDs overlap")
    if set(train["base"]["group_ids"]) & set(cal["base"]["group_ids"]):
        raise ValueError(f"{action} train/calibration group IDs overlap")

    anchor_errors = {
        split: float(np.max(np.abs(
            tables["off"]["predictions"] - tables["base"]["predictions"]
        )))
        for split, tables in (("train", train), ("calibration", cal))
    }
    anchor_passed = all(value <= atol for value in anchor_errors.values())
    if not anchor_passed:
        raise ValueError(f"{action}/{cue} gate-off Base parity failed: {anchor_errors}")

    split_reports = {}
    for split, tables in (("train", train), ("calibration", cal)):
        on_metrics = direct.prediction_metrics(
            tables["on"]["predictions"], tables["on"]["labels"]
        )
        off_metrics = direct.prediction_metrics(
            tables["off"]["predictions"], tables["off"]["labels"]
        )
        split_reports[split] = {
            "on_metrics": on_metrics,
            "off_metrics": off_metrics,
            "on_vs_off": direct.comparison(on_metrics, off_metrics),
            "pairwise_order": pairwise_order_diagnostics(
                tables["on"]["labels"],
                tables["on"]["predictions"],
                tables["off"]["predictions"],
            ),
        }
    boot = group_bootstrap(
        cal["on"], cal["on"]["predictions"], cal["off"]["predictions"],
        bootstrap, seed
    )
    train_gain = split_reports["train"]["on_vs_off"]
    cal_gain = split_reports["calibration"]["on_vs_off"]
    cal_pair_gain = split_reports["calibration"]["pairwise_order"][
        "all_pairs"
    ]["delta_pair_accuracy"]
    checks = {
        "gate_off_matches_base": anchor_passed,
        "train_delta_coef_at_least_0p5": bool(
            train_gain["delta_coef"] is not None
            and train_gain["delta_coef"] >= 0.5
        ),
        "train_mse_not_worse_than_minus_2pct": bool(
            train_gain["relative_mse_gain"] >= -0.02
        ),
        "calibration_delta_coef_positive": bool(
            cal_gain["delta_coef"] is not None and cal_gain["delta_coef"] > 0
        ),
        "calibration_mse_not_worse_than_minus_2pct": bool(
            cal_gain["relative_mse_gain"] >= -0.02
        ),
        "calibration_pair_accuracy_improves": bool(
            cal_pair_gain is not None and cal_pair_gain > 0
        ),
        "bootstrap_p_delta_coef_ge_0p5_at_least_0p90": bool(
            boot["delta_coef"]["probability_at_least_0p5"] is not None
            and boot["delta_coef"]["probability_at_least_0p5"] >= 0.90
        ),
        "bootstrap_p_mse_guardrail_at_least_0p90": bool(
            boot["relative_mse_gain"][
                "probability_not_worse_than_minus_2pct"
            ] >= 0.90
        ),
    }
    return {
        "training_log": verify_log(root, action, cue),
        "anchor_max_abs_error": anchor_errors,
        "splits": split_reports,
        "calibration_group_bootstrap": boot,
        "support_checks": checks,
        "feature_expert_supported": bool(all(checks.values())),
    }


def analyze(root, actions, bootstrap, seed, atol):
    root = os.path.abspath(root)
    reports = {
        action: {
            cue: analyze_action_cue(
                root, action, cue,
                bootstrap, seed + index * 17 + cue_index, atol
            )
            for cue_index, cue in enumerate(CUES)
        }
        for index, action in enumerate(actions)
    }
    support_counts = {
        cue: sum(reports[action][cue]["feature_expert_supported"]
                 for action in actions)
        for cue in CUES
    }
    transferable = {cue: count >= 2 for cue, count in support_counts.items()}
    go = any(transferable.values())
    return {
        "schema_version": 1,
        "screen": "minimal_feature_level_temporal_expert",
        "protocol": {
            "injection": "stage1_pre_pool",
            "expert_gate": "fixed endpoints 0/1; no gate fitting",
            "selection": "none; one pre-registered configuration",
            "base": "clean inner-train Base frozen during expert training",
            "required_supported_actions_per_cue": 2,
        },
        "actions": reports,
        "support_counts": support_counts,
        "transferable_cues": transferable,
        "decision": {
            "status": (
                "GO_TEMPORAL_RELIABILITY_GATE" if go
                else "STOP_MOTION_SKELETON_AUTOGATE"
            ),
            "full_experiment_authorized": False,
            "next_step": (
                "Build one small task-prior x temporal-reliability gate "
                "using only the transferable cue(s)."
                if go else
                "Stop Motion/Skeleton AutoGate and choose a different "
                "innovation source."
            ),
        },
    }


def self_test():
    labels = np.asarray([0.1, 0.2, 0.3, 0.4])
    control = np.asarray([0.4, 0.3, 0.2, 0.1])
    candidate = labels.copy()
    diagnostics = pairwise_order_diagnostics(labels, candidate, control)
    assert diagnostics["all_pairs"]["delta_pair_accuracy"] > 0
    table = {"labels": labels, "group_ids": ["a", "b", "c", "d"]}
    boot = group_bootstrap(table, candidate, control, 50, 0)
    assert boot["delta_coef"]["probability_at_least_0p5"] > 0.9
    print("temporal feature expert analyzer self-test passed")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    report = analyze(
        args.root, args.actions, args.bootstrap, args.seed, args.anchor_atol
    )
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    for action in args.actions:
        for cue in CUES:
            item = report["actions"][action][cue]
            gain = item["splits"]["calibration"]["on_vs_off"]
            print(
                f"{action:4s} {cue:8s} cal_dcoef={gain['delta_coef']!s:>9s} "
                f"cal_mse_gain={gain['relative_mse_gain']:+.4f} "
                f"supported={item['feature_expert_supported']}"
            )
    print("decision:", report["decision"]["status"])
    print("report:", output)


if __name__ == "__main__":
    main()
