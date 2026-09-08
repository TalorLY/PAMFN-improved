#!/usr/bin/env python
"""Summarize deterministic/MC re-evaluation of Stage-A checkpoints."""

import argparse
import csv
import json
import math
import os
import statistics


ACTIONS = ("TES", "Clubs", "Ribbon")
SEEDS = (0, 1, 2)
METHODS = (
    "base",
    "ordered_discrete",
    "ordered_continuous",
    "coqm_no_coord",
    "coqm",
)
POLICIES = ("legacy_gumbel", "argmax", "mc_gumbel")
STABLE_POLICIES = ("argmax", "mc_gumbel")


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def finite(value, field):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"non-finite {field}: {value}")
    return float(value)


def mean(values):
    return statistics.fmean(float(value) for value in values)


def _new_result_path(root, action, seed, method, policy):
    return os.path.join(
        root, action, f"seed{seed}", method, policy,
        "test_result.json",
    )


def _legacy_result_path(root, action, seed, method):
    return os.path.join(
        root, action, f"seed{seed}", method, "test_result.json"
    )


def load_runs(checkpoint_root, root, eval_seed=20260908,
              mc_samples=20):
    rows = []
    for action in ACTIONS:
        for seed in SEEDS:
            for method in METHODS:
                for policy in POLICIES:
                    path = (
                        _legacy_result_path(
                            checkpoint_root, action, seed, method
                        )
                        if policy == "legacy_gumbel"
                        else _new_result_path(
                            root, action, seed, method, policy
                        )
                    )
                    if not os.path.isfile(path):
                        raise ValueError(
                            "missing result for "
                            f"{action}/seed{seed}/{method}/{policy}: {path}"
                        )
                    result = read_json(path)
                    metadata = result.get("run", {})
                    if (
                            metadata.get("tag") != method
                            or metadata.get("action") != action
                            or metadata.get("seed") != seed):
                        raise ValueError(
                            "metadata mismatch for "
                            f"{action}/seed{seed}/{method}/{policy}"
                        )
                    if policy != "legacy_gumbel":
                        evaluation = result.get("evaluation", {})
                        expected_passes = (
                            mc_samples if policy == "mc_gumbel" else 1
                        )
                        if (
                                metadata.get("eval_policy") != policy
                                or metadata.get("eval_seed") != eval_seed
                                or evaluation.get("policy") != policy
                                or evaluation.get("passes") != expected_passes
                                or evaluation.get("seed") != eval_seed
                                or not evaluation.get("rng_isolated")):
                            raise ValueError(
                                "evaluation protocol mismatch for "
                                f"{action}/seed{seed}/{method}/{policy}"
                            )
                    rows.append({
                        "action": action,
                        "seed": seed,
                        "method": method,
                        "policy": policy,
                        "test_srcc": finite(
                            result.get("coef"), "coef"
                        ) / 100,
                        "test_mse": finite(
                            result.get("loss_x100"), "loss_x100"
                        ) / 100,
                        "prediction_std_ratio": finite(
                            result.get("prediction_std_ratio"),
                            "prediction_std_ratio",
                        ),
                        "prediction_std": finite(
                            result.get("prediction_std"),
                            "prediction_std",
                        ),
                        "target_std": finite(
                            result.get("target_std"), "target_std"
                        ),
                        "mae": finite(result.get("mae"), "mae"),
                        "bias": finite(result.get("bias"), "bias"),
                    })
    return rows


def paired(rows, action, seed, method, policy):
    return next(
        row for row in rows
        if row["action"] == action
        and row["seed"] == seed
        and row["method"] == method
        and row["policy"] == policy
    )


def summarize_policy(rows, policy):
    srcc = {
        method: mean(
            row["test_srcc"] for row in rows
            if row["method"] == method and row["policy"] == policy
        )
        for method in METHODS
    }
    mse = {
        method: mean(
            row["test_mse"] for row in rows
            if row["method"] == method and row["policy"] == policy
        )
        for method in METHODS
    }
    action_deltas = {
        action: mean(
            paired(
                rows, action, seed, "coqm_no_coord", policy
            )["test_srcc"]
            - paired(rows, action, seed, "base", policy)["test_srcc"]
            for seed in SEEDS
        )
        for action in ACTIONS
    }
    seed_deltas = {
        str(seed): mean(
            paired(
                rows, action, seed, "coqm_no_coord", policy
            )["test_srcc"]
            - paired(rows, action, seed, "base", policy)["test_srcc"]
            for action in ACTIONS
        )
        for seed in SEEDS
    }
    spread_pairs = []
    for action in ACTIONS:
        for seed in SEEDS:
            base_ratio = paired(
                rows, action, seed, "base", policy
            )["prediction_std_ratio"]
            candidate_ratio = paired(
                rows, action, seed, "coqm_no_coord", policy
            )["prediction_std_ratio"]
            spread_pairs.append({
                "action": action,
                "seed": seed,
                "base": base_ratio,
                "coqm_no_coord": candidate_ratio,
                "safe": candidate_ratio >= max(0.1, 0.5 * base_ratio),
            })
    base_delta = srcc["coqm_no_coord"] - srcc["base"]
    ordered_delta = (
        srcc["coqm_no_coord"] - srcc["ordered_continuous"]
    )
    checks = {
        "macro_delta_at_least_0p4pp": base_delta * 100 >= 0.4,
        "beats_ordered_continuous": ordered_delta > 0,
        "at_least_two_positive_actions": sum(
            value > 0 for value in action_deltas.values()
        ) >= 2,
        "prediction_spread_not_collapsed": all(
            item["safe"] for item in spread_pairs
        ),
    }
    return {
        "signal_pass": all(checks.values()),
        "checks": checks,
        "mean_test_srcc": srcc,
        "mean_test_mse": mse,
        "coqm_no_coord_minus_base_pp": base_delta * 100,
        "coqm_no_coord_minus_ordered_continuous_pp": (
            ordered_delta * 100
        ),
        "coqm_no_coord_mse_relative_to_base": (
            (mse["coqm_no_coord"] - mse["base"]) / mse["base"]
        ),
        "action_delta_pp": {
            key: value * 100 for key, value in action_deltas.items()
        },
        "seed_delta_pp": {
            key: value * 100 for key, value in seed_deltas.items()
        },
        "prediction_spread_pairs": spread_pairs,
    }


def aggregate(rows):
    policies = {
        policy: summarize_policy(rows, policy) for policy in POLICIES
    }
    stable_passes = [
        policies[policy]["signal_pass"] for policy in STABLE_POLICIES
    ]
    if all(stable_passes):
        decision = "COQM_EVAL_B0_SIGNAL_PERSISTS"
        next_step = (
            "Proceed to fresh deterministic-evaluation Stage A2 training; "
            "do not use this reused-checkpoint result as a paper result."
        )
    elif any(stable_passes):
        decision = "COQM_EVAL_B0_SIGNAL_AMBIGUOUS"
        next_step = (
            "Do not launch Stage A2 yet; inspect argmax/MC disagreement "
            "and routing sensitivity."
        )
    else:
        decision = "COQM_EVAL_B0_SIGNAL_NOT_CONFIRMED"
        next_step = (
            "Stop COQM-Geom promotion and return to mechanism diagnosis."
        )
    return {
        "decision": decision,
        "diagnostic_only": True,
        "reason": (
            "These checkpoints were selected under legacy stochastic "
            "evaluation, so re-evaluation cannot be a final paper table."
        ),
        "next_step": next_step,
        "policies": policies,
    }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, gate):
    lines = [
        "# COQM B0 评测稳定化诊断",
        "",
        f"结论：`{gate['decision']}`",
        "",
        "> 本表仅用于决定下一步。旧 checkpoint 是在随机 Gumbel 评测下选择的，",
        "> 因此不能直接作为论文最终结果。",
        "",
        "## 三种评测策略的 Macro 结果",
        "",
        "| 策略 | 方法 | SRCC | MSE |",
        "|---|---|---:|---:|",
    ]
    for policy in POLICIES:
        summary = gate["policies"][policy]
        for method in METHODS:
            lines.append(
                f"| {policy} | {method} | "
                f"{summary['mean_test_srcc'][method]:.4f} | "
                f"{summary['mean_test_mse'][method]:.6f} |"
            )
    lines.extend(["", "## COQM-NoCoord 稳定信号判据", ""])
    for policy in STABLE_POLICIES:
        summary = gate["policies"][policy]
        lines.extend([
            f"### {policy}",
            "",
            "- NoCoord − Base："
            f"{summary['coqm_no_coord_minus_base_pp']:+.3f} pp。",
            "- NoCoord − Ordered-Continuous："
            f"{summary['coqm_no_coord_minus_ordered_continuous_pp']:+.3f} pp。",
            "- NoCoord 相对 Base MSE："
            f"{summary['coqm_no_coord_mse_relative_to_base'] * 100:+.2f}%。",
        ])
        for name, passed in summary["checks"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
        lines.append("")
    lines.extend(["## 下一步", "", gate["next_step"]])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--eval_seed", type=int, default=20260908)
    parser.add_argument("--mc_samples", type=int, default=20)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()
    rows = load_runs(
        args.checkpoint_root,
        args.root,
        eval_seed=args.eval_seed,
        mc_samples=args.mc_samples,
    )
    gate = aggregate(rows)
    write_csv(args.output_csv, rows)
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True
    )
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(gate, handle, indent=2, sort_keys=True)
    write_markdown(args.output_md, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
