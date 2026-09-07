#!/usr/bin/env python
"""Summarize the pre-registered five-method COQM Stage-A experiment."""

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


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def finite(value, field):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"non-finite {field}: {value}")
    return float(value)


def mean(values):
    return statistics.fmean(float(value) for value in values)


def load_runs(root):
    rows = []
    for action in ACTIONS:
        for seed in SEEDS:
            for method in METHODS:
                run_root = os.path.join(
                    root, action, f"seed{seed}", method
                )
                train_path = os.path.join(run_root, "train_result.json")
                test_path = os.path.join(run_root, "test_result.json")
                if not os.path.isfile(train_path) or not os.path.isfile(test_path):
                    raise ValueError(
                        f"missing result for {action}/seed{seed}/{method}"
                    )
                train = read_json(train_path)
                test = read_json(test_path)
                metadata = test.get("run", {})
                if (
                        metadata.get("tag") != method
                        or metadata.get("action") != action
                        or metadata.get("seed") != seed):
                    raise ValueError(
                        f"metadata mismatch for {action}/seed{seed}/{method}"
                    )
                if method != "base" and not metadata.get(
                        "quality_aux_rng_isolated"):
                    raise ValueError(
                        f"RNG isolation missing for {action}/seed{seed}/{method}"
                    )
                expected_target = (
                    "continuous" if method == "ordered_continuous"
                    else "discrete"
                )
                if metadata.get(
                        "ordinal_prototype_target_mode") != expected_target:
                    raise ValueError(
                        f"target mode mismatch for {action}/seed{seed}/{method}"
                    )
                expected_coord = 0.0 if method == "coqm_no_coord" else 1.0
                if finite(
                        metadata.get("coqm_coord_scale"),
                        "coqm_coord_scale") != expected_coord:
                    raise ValueError(
                        f"coordinate scale mismatch for {action}/seed{seed}/{method}"
                    )
                row = {
                    "action": action,
                    "seed": seed,
                    "method": method,
                    "best_srcc": finite(train.get("best_srcc"), "best_srcc"),
                    "best_avg_srcc": finite(
                        train.get("best_avg_srcc"), "best_avg_srcc"
                    ),
                    "saved_model_srcc": finite(
                        train.get("saved_model_srcc"), "saved_model_srcc"
                    ),
                    "test_srcc": finite(test.get("coef"), "coef") / 100,
                    "test_mse": finite(
                        test.get("loss_x100"), "loss_x100"
                    ) / 100,
                    "prediction_std_ratio": finite(
                        test.get("prediction_std_ratio"),
                        "prediction_std_ratio",
                    ),
                    "prediction_std": finite(
                        test.get("prediction_std"), "prediction_std"
                    ),
                    "target_std": finite(
                        test.get("target_std"), "target_std"
                    ),
                    "mae": finite(test.get("mae"), "mae"),
                    "bias": finite(test.get("bias"), "bias"),
                    "rng_isolated": bool(
                        metadata.get("quality_aux_rng_isolated")
                    ),
                    "ordinal_prototype_weight": finite(
                        metadata.get("ordinal_prototype_weight"),
                        "ordinal_prototype_weight",
                    ),
                    "coqm_weight": finite(
                        metadata.get("coqm_weight"), "coqm_weight"
                    ),
                    "coqm_coord_scale": finite(
                        metadata.get("coqm_coord_scale"),
                        "coqm_coord_scale",
                    ),
                }
                diagnostic = test.get("coqm")
                if method.startswith("coqm"):
                    if not isinstance(diagnostic, dict):
                        raise ValueError(
                            f"missing COQM diagnostics for {action}/{seed}/{method}"
                        )
                    for field in (
                            "curve_cosine", "coordinate_mae",
                            "quality_std_ratio", "orthogonal_error",
                            "score_q_spearman", "score_q_pearson"):
                        row[field] = finite(diagnostic.get(field), field)
                    for field in (
                            "soft_occupancy", "hard_occupancy",
                            "adjacent_cosine_distance"):
                        values = diagnostic.get(field, [])
                        expected = 3 if field == "adjacent_cosine_distance" else 4
                        if len(values) != expected:
                            raise ValueError(f"invalid {field}")
                        values = [finite(value, field) for value in values]
                        row[field] = json.dumps(values)
                rows.append(row)
    return rows


def paired(rows, action, seed, method):
    return next(
        row for row in rows
        if row["action"] == action
        and row["seed"] == seed
        and row["method"] == method
    )


def aggregate(rows):
    means = {
        method: mean(
            row["test_srcc"] for row in rows if row["method"] == method
        )
        for method in METHODS
    }
    mse_means = {
        method: mean(
            row["test_mse"] for row in rows if row["method"] == method
        )
        for method in METHODS
    }
    seed_deltas = {
        str(seed): mean(
            paired(rows, action, seed, "coqm")["test_srcc"]
            - paired(rows, action, seed, "base")["test_srcc"]
            for action in ACTIONS
        )
        for seed in SEEDS
    }
    action_deltas = {
        action: mean(
            paired(rows, action, seed, "coqm")["test_srcc"]
            - paired(rows, action, seed, "base")["test_srcc"]
            for seed in SEEDS
        )
        for action in ACTIONS
    }
    coqm_delta = means["coqm"] - means["base"]
    manifold_delta = means["coqm"] - means["ordered_continuous"]
    mse_relative = (
        (mse_means["coqm"] - mse_means["base"])
        / mse_means["base"]
    )
    spread_pairs = []
    for action in ACTIONS:
        for seed in SEEDS:
            base_ratio = paired(
                rows, action, seed, "base"
            )["prediction_std_ratio"]
            coqm_ratio = paired(
                rows, action, seed, "coqm"
            )["prediction_std_ratio"]
            spread_pairs.append({
                "action": action,
                "seed": seed,
                "base": base_ratio,
                "coqm": coqm_ratio,
                "safe": coqm_ratio >= max(0.1, 0.5 * base_ratio),
            })
    coqm_rows = [row for row in rows if row["method"] == "coqm"]
    geometry_valid = all(
        row["orthogonal_error"] <= 1e-4
        and abs(sum(json.loads(row["soft_occupancy"])) - 1.0) <= 1e-5
        and abs(sum(json.loads(row["hard_occupancy"])) - 1.0) <= 1e-5
        and min(json.loads(row["adjacent_cosine_distance"])) > 0
        for row in coqm_rows
    )
    checks = {
        "macro_delta_at_least_0p4pp": coqm_delta * 100 >= 0.4,
        "at_least_two_positive_seeds": sum(
            value > 0 for value in seed_deltas.values()
        ) >= 2,
        "at_least_two_positive_actions": sum(
            value > 0 for value in action_deltas.values()
        ) >= 2,
        "full_beats_ordered_continuous": manifold_delta > 0,
        "mse_deterioration_at_most_5pct": mse_relative <= 0.05,
        "prediction_spread_not_collapsed": all(
            item["safe"] for item in spread_pairs
        ),
        "all_auxiliary_runs_rng_isolated": all(
            row["rng_isolated"]
            for row in rows if row["method"] != "base"
        ),
        "coqm_geometry_valid": geometry_valid,
    }
    return {
        "decision": "COQM_STAGE_A_PASS" if all(checks.values())
                    else "COQM_STAGE_A_FAIL",
        "checks": checks,
        "mean_test_srcc": means,
        "mean_test_mse": mse_means,
        "coqm_minus_base_pp": coqm_delta * 100,
        "coqm_minus_ordered_continuous_pp": manifold_delta * 100,
        "coqm_mse_relative_change": mse_relative,
        "seed_delta_pp": {
            seed: value * 100 for seed, value in seed_deltas.items()
        },
        "action_delta_pp": {
            action: value * 100 for action, value in action_deltas.items()
        },
        "prediction_spread_pairs": spread_pairs,
        "coqm_diagnostics_mean": {
            field: mean(row[field] for row in coqm_rows)
            for field in (
                "curve_cosine", "coordinate_mae", "quality_std_ratio",
                "orthogonal_error", "score_q_spearman", "score_q_pearson",
            )
        },
    }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, gate):
    lines = [
        "# COQM Stage A 预注册裁决",
        "",
        f"结论：`{gate['decision']}`",
        "",
        "## 同 checkpoint 核心结果",
        "",
        "| 方法 | Macro SRCC | Macro MSE |",
        "|---|---:|---:|",
    ]
    for method in METHODS:
        lines.append(
            f"| {method} | {gate['mean_test_srcc'][method]:.4f} | "
            f"{gate['mean_test_mse'][method]:.6f} |"
        )
    lines.extend([
        "",
        f"- Full COQM − Base：{gate['coqm_minus_base_pp']:+.3f} pp。",
        "- Full COQM − Ordered-Continuous："
        f"{gate['coqm_minus_ordered_continuous_pp']:+.3f} pp。",
        "- Full COQM 相对 MSE 变化："
        f"{gate['coqm_mse_relative_change'] * 100:+.2f}%。",
        "",
        "## 门槛",
        "",
    ])
    for name, passed in gate["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend([
        "",
        "## 配对差值（百分点）",
        "",
        "- Seed：" + ", ".join(
            f"{key}={value:+.3f}"
            for key, value in gate["seed_delta_pp"].items()
        ),
        "- Task：" + ", ".join(
            f"{key}={value:+.3f}"
            for key, value in gate["action_delta_pp"].items()
        ),
    ])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()
    rows = load_runs(args.root)
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
