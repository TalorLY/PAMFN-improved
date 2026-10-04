#!/usr/bin/env python3
"""Analyze whether pairwise ranking supervision rescues cue experts.

The rank weight and fixed cue gate are selected without calibration labels:
the gate is selected inside each direct-effect report on the train split, and
the non-zero rank weight is selected against the rank=0 control on train,
subject to a train MSE guardrail.  Calibration is then used once for the
rescue decision.
"""

import argparse
import json
import os
import tempfile

import numpy as np

import analyze_direct_cue_effect as direct


ACTIONS = ("TES", "PCS", "Ball")
CUES = ("motion", "skeleton")
EPSILON = 1e-12


def weight_tag(weight):
    return ("%g" % float(weight)).replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-action ranking-aware independent-expert screen"
    )
    parser.add_argument("--root")
    parser.add_argument("--output")
    parser.add_argument(
        "--actions", nargs="+", default=list(ACTIONS), choices=ACTIONS
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, default=[0.0, 0.01, 0.05]
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--require_clean_inner_base", action="store_true",
        help=(
            "require blind inner-train logs for all three modality branches "
            "and the multimodal Base"
        ),
    )
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    if not args.root or not args.output:
        parser.error("--root and --output are required")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    if len(args.weights) < 2:
        parser.error("include rank=0 and at least one positive rank weight")
    if not np.isclose(args.weights[0], 0.0, atol=1e-12, rtol=0):
        parser.error("the first weight must be the MSE-only rank=0 control")
    if any(not np.isfinite(value) or value < 0 for value in args.weights):
        parser.error("rank weights must be finite and non-negative")
    if len(set(args.weights)) != len(args.weights):
        parser.error("rank weights must be unique")
    return args


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_same_rows(reference, candidate, description):
    if reference["sample_ids"] != candidate["sample_ids"]:
        raise ValueError("%s has mismatched sample IDs or order" % description)
    if reference["group_ids"] != candidate["group_ids"]:
        raise ValueError("%s has mismatched group IDs" % description)
    if not np.array_equal(reference["labels"], candidate["labels"]):
        raise ValueError("%s has mismatched labels" % description)


def verify_clean_inner_base(root, action):
    action_dir = os.path.join(root, action)
    branch_dir = os.path.join(action_dir, "branches")
    artifacts = [
        (
            os.path.join(action_dir, "%s_inner_rgb_seed0.log" % action),
            os.path.join(branch_dir, "%s_rgb_UMT.pth" % action),
            "branch",
        ),
        (
            os.path.join(action_dir, "%s_inner_flow_seed0.log" % action),
            os.path.join(branch_dir, "%s_flow_I3D.pth" % action),
            "branch",
        ),
        (
            os.path.join(action_dir, "%s_inner_audio_seed0.log" % action),
            os.path.join(branch_dir, "%s_audio_MAST.pth" % action),
            "branch",
        ),
        (
            os.path.join(action_dir, "%s_inner_base_seed0.log" % action),
            os.path.join(action_dir, "%s_inner_base_seed0.pth" % action),
            "base",
        ),
    ]
    verified = []
    for log_path, checkpoint_path, kind in artifacts:
        if not os.path.isfile(log_path):
            raise ValueError("missing clean inner-training log: %s" % log_path)
        if not os.path.isfile(checkpoint_path):
            raise ValueError("missing clean inner checkpoint: %s" % checkpoint_path)
        with open(log_path, "r", encoding="utf-8") as handle:
            text = handle.read()
        if "Blind training complete;" not in text:
            raise ValueError("held-out loader was not proven blind: %s" % log_path)
        if "Initialized from " in text:
            raise ValueError("inner model used a forbidden init checkpoint: %s" % log_path)
        if kind == "branch" and (
                "Saved the final blind-training branch model;" not in text):
            raise ValueError("branch was not saved by blind final-epoch path: %s" % log_path)
        if kind == "base" and (
                "Saved the final blind-training checkpoint;" not in text):
            raise ValueError("Base was not saved by blind final-epoch path: %s" % log_path)
        if kind == "base" and (
                "Modality branches: %s" % branch_dir not in text):
            raise ValueError(
                "Base did not declare the clean inner branch directory: %s"
                % log_path
            )
        verified.append({
            "kind": kind,
            "log": os.path.abspath(log_path),
            "checkpoint": os.path.abspath(checkpoint_path),
        })
    return {
        "verified": True,
        "calibration_loader_iterated_during_training": False,
        "forbidden_multimodal_init_used": False,
        "artifacts": verified,
    }


def candidate_paths(root, action, weight, cue, split):
    tag = weight_tag(weight)
    prefix = os.path.join(root, action, "%s_rank%s" % (action, tag))
    return (
        "%s_%s_on_%s.csv" % (prefix, cue, split),
        "%s_%s_off_%s.csv" % (prefix, cue, split),
    )


def load_candidate(root, action, weight, cue):
    tag = weight_tag(weight)
    report_path = os.path.join(
        root, action, "%s_rank%s_direct_effect.json" % (action, tag)
    )
    report = load_json(report_path)
    if report.get("schema_version") != 3:
        raise ValueError("unexpected direct-effect schema: %s" % report_path)
    if not report.get("anchor_parity", {}).get("passed", False):
        raise ValueError("anchor parity failed: %s" % report_path)
    effect = report["cue_effects"][cue]
    gate = float(effect["fixed_gate"]["gate"])
    tables = {}
    fixed_predictions = {}
    for split in ("train", "cal"):
        on_path, off_path = candidate_paths(
            root, action, weight, cue, split
        )
        on = direct.read_prediction_csv(on_path)
        off = direct.read_prediction_csv(off_path)
        ensure_same_rows(off, on, "%s/%s/rank=%g/%s" % (
            action, cue, weight, split
        ))
        tables[split] = {"on": on, "off": off}
        fixed_predictions[split] = (
            off["predictions"]
            + gate * (on["predictions"] - off["predictions"])
        )

    train_fixed = direct.prediction_metrics(
        fixed_predictions["train"], tables["train"]["off"]["labels"]
    )
    train_off = direct.prediction_metrics(
        tables["train"]["off"]["predictions"],
        tables["train"]["off"]["labels"],
    )
    calibration_fixed = direct.prediction_metrics(
        fixed_predictions["cal"], tables["cal"]["off"]["labels"]
    )
    calibration_off = direct.prediction_metrics(
        tables["cal"]["off"]["predictions"],
        tables["cal"]["off"]["labels"],
    )
    recorded = effect["fixed_gate"]
    for name, computed, expected in (
        ("train", train_fixed, recorded["train_metrics"]),
        ("calibration", calibration_fixed, recorded["calibration_metrics"]),
    ):
        if not np.isclose(
            computed["mse_raw"], expected["mse_raw"], atol=1e-10, rtol=0
        ):
            raise ValueError("%s fixed prediction disagrees with %s" % (
                report_path, name
            ))

    train_gain = direct.comparison(train_fixed, train_off)
    eligible = bool(
        gate > 0
        and train_gain["delta_coef"] is not None
        and train_gain["delta_coef"] > 0
        and train_gain["relative_mse_gain"] >= -0.02
    )
    return {
        "weight": float(weight),
        "tag": tag,
        "report_path": os.path.abspath(report_path),
        "report": report,
        "effect": effect,
        "gate": gate,
        "tables": tables,
        "fixed_predictions": fixed_predictions,
        "train_fixed_metrics": train_fixed,
        "train_off_metrics": train_off,
        "train_gain_over_off": train_gain,
        "calibration_fixed_metrics": calibration_fixed,
        "calibration_off_metrics": calibration_off,
        "calibration_gain_over_off": direct.comparison(
            calibration_fixed, calibration_off
        ),
        "train_selection_eligible": eligible,
    }


def select_rank_candidate(candidates, control):
    positive = [item for item in candidates if item["weight"] > 0]
    for item in positive:
        ensure_same_rows(
            control["tables"]["train"]["off"],
            item["tables"]["train"]["off"],
            "rank candidate versus rank=0 train",
        )
        comparison = direct.comparison(
            item["train_fixed_metrics"], control["train_fixed_metrics"]
        )
        item["train_gain_over_rank0"] = comparison
        item["strict_train_selection_eligible"] = bool(
            item["train_selection_eligible"]
            and comparison["delta_coef"] is not None
            and comparison["delta_coef"] > 0
            and comparison["relative_mse_gain"] >= -0.02
        )
    eligible = [
        item for item in positive
        if item["strict_train_selection_eligible"]
    ]
    pool = eligible if eligible else positive
    if not pool:
        raise ValueError("at least one positive rank-weight candidate is required")

    def key(item):
        gain = item["train_gain_over_rank0"]
        delta_coef = gain["delta_coef"]
        return (
            -float("inf") if delta_coef is None else delta_coef,
            gain["relative_mse_gain"],
            -item["weight"],
        )

    return max(pool, key=key), bool(eligible)


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
    threshold = float(np.percentile(margins, 25))
    low_margin = margins <= threshold

    def summarize(mask):
        count = int(np.sum(mask))
        if count == 0:
            return {"pairs": 0}
        control_score = (
            control_correct[mask].astype(np.float64)
            + 0.5 * control_tie[mask].astype(np.float64)
        )
        candidate_score = (
            candidate_correct[mask].astype(np.float64)
            + 0.5 * candidate_tie[mask].astype(np.float64)
        )
        return {
            "pairs": count,
            "control_pair_accuracy": float(np.mean(control_score)),
            "candidate_pair_accuracy": float(np.mean(candidate_score)),
            "delta_pair_accuracy": float(
                np.mean(candidate_score) - np.mean(control_score)
            ),
            "wrong_to_correct_pairs": int(np.sum(
                (~control_correct[mask]) & candidate_correct[mask]
            )),
            "correct_to_wrong_pairs": int(np.sum(
                control_correct[mask] & (~candidate_correct[mask])
            )),
        }

    return {
        "comparable_pairs": int(np.sum(usable)),
        "all_pairs": summarize(np.ones(target_sign.size, dtype=bool)),
        "lowest_control_margin_quartile": {
            "control_margin_threshold": threshold,
            **summarize(low_margin),
        },
    }


def percentile_summary(values):
    if not values:
        return {"median": None, "ci95": [None, None]}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "ci95": [
            float(np.percentile(array, 2.5)),
            float(np.percentile(array, 97.5)),
        ],
    }


def paired_bootstrap(labels, group_ids, selected, control, iterations, seed):
    group_to_indices = {}
    for index, group_id in enumerate(group_ids):
        group_to_indices.setdefault(group_id, []).append(index)
    groups = sorted(group_to_indices)
    if not groups:
        raise ValueError("calibration has no groups")
    rng = np.random.RandomState(seed)
    delta_coefs = []
    relative_mse_gains = []
    for _ in range(iterations):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = np.asarray([
            index
            for group_id in sampled_groups
            for index in group_to_indices[group_id]
        ], dtype=np.int64)
        selected_metrics = direct.prediction_metrics(
            selected[indices], labels[indices]
        )
        control_metrics = direct.prediction_metrics(
            control[indices], labels[indices]
        )
        comparison = direct.comparison(selected_metrics, control_metrics)
        if comparison["delta_coef"] is not None:
            delta_coefs.append(comparison["delta_coef"])
        relative_mse_gains.append(comparison["relative_mse_gain"])
    return {
        "iterations": int(iterations),
        "resampling_unit": "group_id",
        "delta_coef": {
            **percentile_summary(delta_coefs),
            "positive_probability": (
                None if not delta_coefs else
                float(np.mean(np.asarray(delta_coefs) > 0))
            ),
        },
        "relative_mse_gain": {
            **percentile_summary(relative_mse_gains),
            "not_worse_than_minus_2pct_probability": float(np.mean(
                np.asarray(relative_mse_gains) >= -0.02
            )),
        },
    }


def public_candidate(candidate):
    result = {
        "rank_weight": candidate["weight"],
        "train_selected_fixed_gate": candidate["gate"],
        "train_selection_eligible": candidate["train_selection_eligible"],
        "train_fixed_metrics": candidate["train_fixed_metrics"],
        "train_gain_over_cue_off": candidate["train_gain_over_off"],
        "calibration_fixed_metrics": candidate["calibration_fixed_metrics"],
        "calibration_gain_over_cue_off": candidate[
            "calibration_gain_over_off"
        ],
        "direct_effect_average_supported": candidate["effect"][
            "preliminary_decision"
        ]["average_effect_supported"],
        "direct_effect_report": candidate["report_path"],
    }
    if "train_gain_over_rank0" in candidate:
        result["train_gain_over_rank0"] = candidate["train_gain_over_rank0"]
        result["strict_train_selection_eligible"] = candidate[
            "strict_train_selection_eligible"
        ]
    return result


def analyze_action_cue(root, action, weights, cue, bootstrap, seed):
    candidates = [
        load_candidate(root, action, weight, cue) for weight in weights
    ]
    control = candidates[0]
    selected, had_eligible = select_rank_candidate(candidates, control)
    control_cal = control["tables"]["cal"]
    selected_cal = selected["tables"]["cal"]
    ensure_same_rows(
        control_cal["off"], selected_cal["off"],
        "%s/%s selected-vs-control calibration" % (action, cue),
    )
    selected_vs_control = direct.comparison(
        selected["calibration_fixed_metrics"],
        control["calibration_fixed_metrics"],
    )
    bootstrap_report = paired_bootstrap(
        selected_cal["off"]["labels"],
        selected_cal["off"]["group_ids"],
        selected["fixed_predictions"]["cal"],
        control["fixed_predictions"]["cal"],
        bootstrap,
        seed,
    )
    pair_diagnostics = {
        split: pairwise_order_diagnostics(
            selected["tables"][split]["off"]["labels"],
            selected["fixed_predictions"][split],
            control["fixed_predictions"][split],
        )
        for split in ("train", "cal")
    }
    checks = {
        "positive_rank_candidate_passed_train_selection": had_eligible,
        "selected_rank_direct_average_effect_supported": bool(
            selected["effect"]["preliminary_decision"][
                "average_effect_supported"
            ]
        ),
        "selected_rank_improves_calibration_coef_over_rank0": bool(
            selected_vs_control["delta_coef"] is not None
            and selected_vs_control["delta_coef"] > 0
        ),
        "selected_rank_calibration_mse_not_worse_than_rank0_by_over_2pct": bool(
            selected_vs_control["relative_mse_gain"] >= -0.02
        ),
        "bootstrap_coef_improvement_probability_at_least_80pct": bool(
            bootstrap_report["delta_coef"]["positive_probability"] is not None
            and bootstrap_report["delta_coef"]["positive_probability"] >= 0.80
        ),
        "bootstrap_mse_guardrail_probability_at_least_80pct": bool(
            bootstrap_report["relative_mse_gain"][
                "not_worse_than_minus_2pct_probability"
            ] >= 0.80
        ),
    }
    supported = bool(all(checks.values()))
    return {
        "selection_protocol": (
            "A positive rank weight must first beat both its own cue-off "
            "prediction and the rank=0 fixed-gate control on train Spearman, "
            "with non-zero gate and <=2% train-MSE regret in both comparisons. "
            "Among eligible weights, maximize train Spearman gain over rank=0; "
            "ties prefer train MSE gain then smaller weight. Calibration is "
            "not used for rank-weight or fixed-gate selection."
        ),
        "candidates": [public_candidate(item) for item in candidates],
        "rank0_control": public_candidate(control),
        "selected_rank_candidate": public_candidate(selected),
        "calibration_selected_vs_rank0": selected_vs_control,
        "paired_group_bootstrap_selected_vs_rank0": bootstrap_report,
        "pairwise_order_diagnostics_selected_vs_rank0": pair_diagnostics,
        "rescue_checks": checks,
        "rank_rescue_supported": supported,
    }


def analyze(
        root, actions, weights, bootstrap, seed,
        require_clean_inner_base=False):
    clean_base = (
        {
            action: verify_clean_inner_base(root, action)
            for action in actions
        }
        if require_clean_inner_base else None
    )
    action_reports = {}
    for action_index, action in enumerate(actions):
        action_reports[action] = {
            cue: analyze_action_cue(
                root, action, weights, cue, bootstrap,
                seed + 1000 * action_index + 100 * cue_index,
            )
            for cue_index, cue in enumerate(CUES)
        }
    support_counts = {
        cue: int(sum(
            action_reports[action][cue]["rank_rescue_supported"]
            for action in actions
        ))
        for cue in CUES
    }
    transferable = {
        cue: support_counts[cue] >= 2 for cue in CUES
    }
    go_gate_screen = bool(any(transferable.values()))
    return {
        "schema_version": 2,
        "screen": "clean-inner-Base ranking-aware independent expert rescue",
        "test_set_used": False,
        "clean_inner_base_required": bool(require_clean_inner_base),
        "clean_inner_base_audit": clean_base,
        "full_experiment_authorized": False,
        "actions": list(actions),
        "rank_weights": [float(value) for value in weights],
        "action_reports": action_reports,
        "cross_action": {
            "required_supported_actions_per_cue": 2,
            "support_counts": support_counts,
            "transferable_cues": transferable,
        },
        "decision": {
            "status": (
                "GO_GATE_SCREEN" if go_gate_screen
                else "PIVOT_FEATURE_LEVEL_TEMPORAL_ADAPTERS"
            ),
            "reason": (
                "At least one ranking-aware cue expert transfers to two or "
                "more representative actions; restore only that cue in a "
                "small action-prior x reliability Gate screen."
                if go_gate_screen else
                "No ranking-aware scalar-score residual expert transfers to "
                "two actions. Stop tuning scalar gates/rank weights and move "
                "the cue adapter before temporal pooling."
            ),
            "next_step": (
                "Run a small TES/PCS/Ball Gate screen using only transferable "
                "cue experts; do not start the six-action multi-seed study yet."
                if go_gate_screen else
                "Implement feature-level temporal Motion/Skeleton adapters, "
                "then repeat the frozen-Base expert screen before any Gate."
            ),
        },
    }


def run_self_test():
    control = {
        "weight": 0.0,
        "train_fixed_metrics": {"coef": 50.0, "mse_raw": 1.0},
        "tables": {"train": {"off": {
            "sample_ids": ["a", "b"],
            "group_ids": ["g1", "g2"],
            "labels": np.asarray([0.0, 1.0]),
        }}},
    }
    candidates = [
        {
            "weight": 0.01,
            "train_selection_eligible": True,
            "train_fixed_metrics": {"coef": 51.0, "mse_raw": 0.99},
            "tables": control["tables"],
            "train_gain_over_off": {
                "delta_coef": 1.0, "relative_mse_gain": 0.01
            },
        },
        {
            "weight": 0.05,
            "train_selection_eligible": True,
            "train_fixed_metrics": {"coef": 52.0, "mse_raw": 1.01},
            "tables": control["tables"],
            "train_gain_over_off": {
                "delta_coef": 2.0, "relative_mse_gain": -0.01
            },
        },
    ]
    selected, eligible = select_rank_candidate(candidates, control)
    assert eligible and np.isclose(selected["weight"], 0.05)
    losing = {
        "weight": 0.1,
        "train_selection_eligible": True,
        "train_fixed_metrics": {"coef": 49.0, "mse_raw": 0.95},
        "tables": control["tables"],
        "train_gain_over_off": {
            "delta_coef": 3.0, "relative_mse_gain": 0.03
        },
    }
    _, eligible = select_rank_candidate([losing], control)
    assert not eligible
    labels = np.arange(12, dtype=np.float64)
    control = labels[::-1]
    rescued = labels.copy()
    boot = paired_bootstrap(
        labels, ["g%d" % value for value in range(12)],
        rescued, control, 100, 0,
    )
    assert boot["delta_coef"]["positive_probability"] == 1.0
    pair_report = pairwise_order_diagnostics(labels, rescued, control)
    assert pair_report["all_pairs"]["delta_pair_accuracy"] > 0
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "self_test.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(boot, handle)
        assert load_json(path)["iterations"] == 100
        action_dir = os.path.join(directory, "TES")
        branch_dir = os.path.join(action_dir, "branches")
        os.makedirs(branch_dir)
        blind_text = (
            "Blind training complete; the held-out calibration loader was "
            "never iterated\n"
        )
        for descriptor, checkpoint in (
                ("rgb", "TES_rgb_UMT.pth"),
                ("flow", "TES_flow_I3D.pth"),
                ("audio", "TES_audio_MAST.pth")):
            log_path = os.path.join(
                action_dir, "TES_inner_%s_seed0.log" % descriptor
            )
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write(
                    blind_text
                    + "Saved the final blind-training branch model;\n"
                )
            open(os.path.join(branch_dir, checkpoint), "wb").close()
        with open(
                os.path.join(action_dir, "TES_inner_base_seed0.log"),
                "w", encoding="utf-8") as handle:
            handle.write(
                blind_text
                + "Modality branches: %s\n" % branch_dir
                + "Saved the final blind-training checkpoint;\n"
            )
        open(
            os.path.join(action_dir, "TES_inner_base_seed0.pth"), "wb"
        ).close()
        assert verify_clean_inner_base(directory, "TES")["verified"]
    print("ranking expert rescue analyzer self-test: passed")


def print_summary(report):
    for action in report["actions"]:
        for cue in CUES:
            item = report["action_reports"][action][cue]
            selected = item["selected_rank_candidate"]
            comparison = item["calibration_selected_vs_rank0"]
            print(
                "%s %-8s rank=%g gate=%.2f cal_delta_coef=%s "
                "cal_mse_gain=%+.4f supported=%s" % (
                    action,
                    cue,
                    selected["rank_weight"],
                    selected["train_selected_fixed_gate"],
                    ("undefined" if comparison["delta_coef"] is None else
                     "%+.3f" % comparison["delta_coef"]),
                    comparison["relative_mse_gain"],
                    item["rank_rescue_supported"],
                )
            )
    print("decision.status = %s" % report["decision"]["status"])
    print(report["decision"]["next_step"])


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    report = analyze(
        args.root, args.actions, args.weights, args.bootstrap, args.seed,
        require_clean_inner_base=args.require_clean_inner_base,
    )
    output_parent = os.path.dirname(os.path.abspath(args.output))
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print_summary(report)


if __name__ == "__main__":
    main()
