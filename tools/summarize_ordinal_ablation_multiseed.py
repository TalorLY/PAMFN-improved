#!/usr/bin/env python
"""Summarize four ordinal variants across actions and random seeds."""

import argparse
import csv
import math
import os
import statistics


RUN_NAME = "ablation_runs.csv"
FEATURE_NAME = "feature_evidence_summary.csv"
MODES = ("base", "grade", "unordered", "prototype")
FEATURE_MODES = ("grade", "unordered", "prototype")
METRICS = (
    "best_coef",
    "best_avg_coef",
    "min_loss",
    "min_avg_loss",
)


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    os.makedirs(
        os.path.dirname(os.path.abspath(path)),
        exist_ok=True,
    )
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_seed_results(values):
    results = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"seed result must use SEED=ROOT format: {value}"
            )
        seed_text, root = value.split("=", 1)
        seed = int(seed_text)
        if seed in results:
            raise ValueError(f"duplicate seed result: {seed}")
        results[seed] = root
    return dict(sorted(results.items()))


def mean_std(values):
    values = [float(value) for value in values]
    return (
        statistics.fmean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def optional_mean_std(rows, field):
    values = [
        float(row[field])
        for row in rows
        if row.get(field, "") not in ("", None)
    ]
    if not values:
        return "", ""
    return mean_std(values)


def load_results(seed_roots, actions):
    results = {}
    features = {}
    run_rows = []
    feature_rows = []
    expected_runs = {
        (action, mode) for action in actions for mode in MODES
    }
    expected_features = {
        (action, mode)
        for action in actions
        for mode in FEATURE_MODES
    }

    for seed, root in seed_roots.items():
        rows = read_csv(os.path.join(root, RUN_NAME))
        observed = set()
        for row in rows:
            key = (row["action"], row["mode"])
            if key in observed:
                raise ValueError(
                    f"duplicate run row for seed {seed}: {key}"
                )
            observed.add(key)
            result_key = (seed,) + key
            metrics = {
                metric: float(row[metric]) for metric in METRICS
            }
            if not all(math.isfinite(value) for value in metrics.values()):
                raise ValueError(
                    f"non-finite official metric for seed {seed}: {key}"
                )
            results[result_key] = metrics
            run_rows.append({
                "seed": seed,
                "bundle_root": root,
                **row,
            })
        if observed != expected_runs:
            missing = sorted(expected_runs - observed)
            extra = sorted(observed - expected_runs)
            raise ValueError(
                f"incomplete runs for seed {seed}; "
                f"missing={missing}, extra={extra}"
            )

        rows = read_csv(os.path.join(root, FEATURE_NAME))
        observed = set()
        for row in rows:
            key = (row["action"], row["mode"])
            if key in observed:
                raise ValueError(
                    f"duplicate feature row for seed {seed}: {key}"
                )
            observed.add(key)
            features[(seed,) + key] = row
            feature_rows.append({
                "seed": seed,
                "bundle_root": root,
                **row,
            })
        if observed != expected_features:
            missing = sorted(expected_features - observed)
            extra = sorted(observed - expected_features)
            raise ValueError(
                f"incomplete features for seed {seed}; "
                f"missing={missing}, extra={extra}"
            )

    return results, features, run_rows, feature_rows


def _seed_scope_values(
        results, seeds, actions, mode, metric):
    return [
        statistics.fmean(
            results[(seed, action, mode)][metric]
            for action in actions
        )
        for seed in seeds
    ]


def aggregate_scope(label, actions, seeds, results):
    row = {
        "scope": label,
        "action_count": len(actions),
        "seed_count": len(seeds),
        "paired_runs": len(actions) * len(seeds),
    }
    for metric in METRICS:
        mode_values = {}
        for mode in MODES:
            values = _seed_scope_values(
                results, seeds, actions, mode, metric
            )
            mode_values[mode] = values
            mean, std = mean_std(values)
            row[f"{mode}_{metric}_mean"] = mean
            row[f"{mode}_{metric}_std"] = std

        base_values = mode_values["base"]
        for mode in MODES[1:]:
            deltas = [
                value - base
                for value, base in zip(
                    mode_values[mode], base_values
                )
            ]
            mean, std = mean_std(deltas)
            row[f"{mode}_delta_{metric}_mean"] = mean
            row[f"{mode}_delta_{metric}_std"] = std

        for control in ("grade", "unordered"):
            deltas = [
                ordered - baseline
                for ordered, baseline in zip(
                    mode_values["prototype"],
                    mode_values[control],
                )
            ]
            mean, std = mean_std(deltas)
            prefix = f"prototype_minus_{control}_{metric}"
            row[f"{prefix}_mean"] = mean
            row[f"{prefix}_std"] = std

    for mode in MODES[1:]:
        row[f"{mode}_best_wins_vs_base"] = sum(
            results[(seed, action, mode)]["best_coef"]
            > results[(seed, action, "base")]["best_coef"]
            for seed in seeds
            for action in actions
        )
        row[f"{mode}_avg_wins_vs_base"] = sum(
            results[(seed, action, mode)]["best_avg_coef"]
            > results[(seed, action, "base")]["best_avg_coef"]
            for seed in seeds
            for action in actions
        )
    for control in ("grade", "unordered"):
        row[f"prototype_best_wins_vs_{control}"] = sum(
            results[(seed, action, "prototype")]["best_coef"]
            > results[(seed, action, control)]["best_coef"]
            for seed in seeds
            for action in actions
        )
        row[f"prototype_avg_wins_vs_{control}"] = sum(
            results[
                (seed, action, "prototype")
            ]["best_avg_coef"]
            > results[
                (seed, action, control)
            ]["best_avg_coef"]
            for seed in seeds
            for action in actions
        )
    return row


def aggregate_features(actions, seeds, features):
    rows = []
    scopes = [(action, [action]) for action in actions]
    scopes.append(("MACRO", actions))
    for label, scope_actions in scopes:
        for mode in FEATURE_MODES:
            selected = [
                features[(seed, action, mode)]
                for seed in seeds
                for action in scope_actions
            ]
            row = {
                "scope": label,
                "mode": mode,
                "run_count": len(selected),
            }
            for field in (
                "expected_grade_srcc",
                "grade_exact_accuracy",
                "grade_adjacent_accuracy",
                "test_grade_soft_ce",
                "prototype_order_loss",
                "min_prototype_gap",
                "max_prototype_cosine",
                "axis_label_srcc",
            ):
                mean, std = optional_mean_std(selected, field)
                row[f"{field}_mean"] = mean
                row[f"{field}_std"] = std

            active_counts = []
            for item in selected:
                active_counts.append(sum(
                    float(item[f"predicted_count_bin{index}"]) > 0
                    for index in range(4)
                ))
            row["active_grade_count_mean"] = (
                statistics.fmean(active_counts)
            )
            row["hard_grade_collapses"] = sum(
                count <= 1 for count in active_counts
            )
            row["monotonic_runs"] = sum(
                item["monotonic_bin_means"].lower() == "true"
                for item in selected
            )
            rows.append(row)
    return rows


def _fmt(row, prefix):
    return (
        f'{float(row[prefix + "_mean"]):.2f}'
        f'±{float(row[prefix + "_std"]):.2f}'
    )


def _delta(row, prefix):
    return (
        f'{float(row[prefix + "_mean"]):+.2f}'
        f'±{float(row[prefix + "_std"]):.2f}'
    )


def _markdown_metric_table(rows, metric):
    lines = [
        "| 范围 | Base | Grade | Unordered | Ordered | "
        "Ordered−Base | Ordered−Unordered |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f'| {row["scope"]} | '
            f'{_fmt(row, "base_" + metric)} | '
            f'{_fmt(row, "grade_" + metric)} | '
            f'{_fmt(row, "unordered_" + metric)} | '
            f'{_fmt(row, "prototype_" + metric)} | '
            f'{_delta(row, "prototype_delta_" + metric)} | '
            f'{_delta(row, "prototype_minus_unordered_" + metric)} |'
        )
    return lines


def write_report(path, display_rows, source_rows, feature_rows):
    macro = next(
        row for row in source_rows if row["scope"] == "MACRO"
    )
    prototype_features = [
        row for row in feature_rows
        if row["mode"] == "prototype"
    ]
    report = [
        "# 有序质量原型：三随机种子关键消融汇总",
        "",
        "所有 Coef 数值均按原论文格式乘以 100；表中为均值±标准差。",
        "",
        "## Best Coef",
        "",
        *_markdown_metric_table(display_rows, "best_coef"),
        "",
        "## Best Avg Coef",
        "",
        *_markdown_metric_table(
            display_rows, "best_avg_coef"
        ),
        "",
        "## 宏平均配对结论",
        "",
        (
            "- Ordered−Base：Best "
            f'{_delta(macro, "prototype_delta_best_coef")}，'
            "Best Avg "
            f'{_delta(macro, "prototype_delta_best_avg_coef")}。'
        ),
        (
            "- Ordered−Unordered：Best "
            f'{_delta(macro, "prototype_minus_unordered_best_coef")}，'
            "Best Avg "
            f'{_delta(macro, "prototype_minus_unordered_best_avg_coef")}。'
        ),
        (
            "- Ordered−Grade：Best "
            f'{_delta(macro, "prototype_minus_grade_best_coef")}，'
            "Best Avg "
            f'{_delta(macro, "prototype_minus_grade_best_avg_coef")}。'
        ),
        (
            "- 18 个动作×种子配对中，Ordered 胜过 Base："
            f'{macro["prototype_best_wins_vs_base"]}/18 Best，'
            f'{macro["prototype_avg_wins_vs_base"]}/18 Best Avg；'
            "胜过 Unordered："
            f'{macro["prototype_best_wins_vs_unordered"]}/18 Best，'
            f'{macro["prototype_avg_wins_vs_unordered"]}/18 Best Avg。'
        ),
        "",
        "## Ordered 特征证据",
        "",
        "| 范围 | Expected-grade SRCC | 相邻等级准确率 | "
        "质量轴 SRCC | 最小原型间隔 | 单等级硬塌缩 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in prototype_features:
        axis = row["axis_label_srcc_mean"]
        gap = row["min_prototype_gap_mean"]
        report.append(
            f'| {row["scope"]} | '
            f'{float(row["expected_grade_srcc_mean"]):.3f} | '
            f'{float(row["grade_adjacent_accuracy_mean"]):.3f} | '
            f'{float(axis):.3f} | '
            f'{float(gap):.3f} | '
            f'{row["hard_grade_collapses"]}/{row["run_count"]} |'
        )
    report.append("")
    os.makedirs(
        os.path.dirname(os.path.abspath(path)),
        exist_ok=True,
    )
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(report))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-results", nargs="+", required=True,
        help="entries in SEED=ROOT format",
    )
    parser.add_argument("--actions", nargs="+", required=True)
    parser.add_argument("--output-runs", required=True)
    parser.add_argument("--output-action-table", required=True)
    parser.add_argument("--output-source-table", required=True)
    parser.add_argument("--output-feature-runs", required=True)
    parser.add_argument("--output-feature-table", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()

    seed_roots = parse_seed_results(args.seed_results)
    if len(seed_roots) < 2:
        parser.error("at least two seed results are required")
    seeds = list(seed_roots)
    results, features, run_rows, feature_run_rows = load_results(
        seed_roots, args.actions
    )

    action_rows = [
        aggregate_scope(
            action, [action], seeds, results
        )
        for action in args.actions
    ]
    action_rows.append(
        aggregate_scope("MACRO", args.actions, seeds, results)
    )
    source_rows = [
        aggregate_scope(
            "FIS-V",
            [action for action in args.actions
             if action in ("TES", "PCS")],
            seeds,
            results,
        ),
        aggregate_scope(
            "RG",
            [action for action in args.actions
             if action not in ("TES", "PCS")],
            seeds,
            results,
        ),
        aggregate_scope("MACRO", args.actions, seeds, results),
    ]
    feature_rows = aggregate_features(
        args.actions, seeds, features
    )
    display_rows = action_rows[:-1] + source_rows

    write_csv(args.output_runs, run_rows)
    write_csv(args.output_action_table, action_rows)
    write_csv(args.output_source_table, source_rows)
    write_csv(args.output_feature_runs, feature_run_rows)
    write_csv(args.output_feature_table, feature_rows)
    write_report(
        args.output_report,
        display_rows,
        source_rows,
        feature_rows,
    )
    print(
        f"combined {len(seeds)} seeds, "
        f"{len(args.actions)} actions, and {len(MODES)} modes"
    )
    print(f"runs: {args.output_runs}")
    print(f"action table: {args.output_action_table}")
    print(f"source table: {args.output_source_table}")
    print(f"feature table: {args.output_feature_table}")
    print(f"mentor report: {args.output_report}")


if __name__ == "__main__":
    main()
