#!/usr/bin/env python
"""Combine paired Base/Prototype results across random seeds."""

import argparse
import csv
import os
import statistics


SUMMARY_NAME = "original_protocol_summary.csv"
DIAGNOSTIC_NAME = "ordinal_prototype_diagnostics.csv"
METRICS = (
    "best_coef",
    "best_avg_coef",
    "min_loss",
    "min_avg_loss",
)


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    os.makedirs(
        os.path.dirname(os.path.abspath(path)),
        exist_ok=True,
    )
    if fieldnames is None:
        fieldnames = list(rows[0])
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def aggregate_row(action, base_rows, candidate_rows, diagnostics):
    row = {
        "action": action,
        "seed_count": len(base_rows),
        "comparisons": len(candidate_rows),
    }
    for metric in METRICS:
        base_values = [item[metric] for item in base_rows]
        candidate_values = [
            item[metric] for item in candidate_rows
        ]
        delta_values = [
            float(candidate) - float(base)
            for base, candidate in zip(
                base_values, candidate_values
            )
        ]
        base_mean, base_std = mean_std(base_values)
        candidate_mean, candidate_std = mean_std(
            candidate_values
        )
        delta_mean, delta_std = mean_std(delta_values)
        row[f"base_{metric}_mean"] = base_mean
        row[f"base_{metric}_std"] = base_std
        row[f"prototype_{metric}_mean"] = candidate_mean
        row[f"prototype_{metric}_std"] = candidate_std
        row[f"delta_{metric}_mean"] = delta_mean
        row[f"delta_{metric}_std"] = delta_std

    row["best_wins"] = sum(
        float(candidate["best_coef"]) > float(base["best_coef"])
        for base, candidate in zip(base_rows, candidate_rows)
    )
    row["avg_wins"] = sum(
        float(candidate["best_avg_coef"])
        > float(base["best_avg_coef"])
        for base, candidate in zip(base_rows, candidate_rows)
    )
    row["prototype_collapses"] = sum(
        item["collapsed"].lower() == "true"
        for item in diagnostics
    )
    row["order_violations"] = sum(
        int(item["order_violations"]) for item in diagnostics
    )
    row["margin_violations"] = sum(
        int(item["margin_violations"]) for item in diagnostics
    )
    row["min_prototype_gap"] = min(
        float(item[name])
        for item in diagnostics
        for name in ("gap_0_1", "gap_1_2", "gap_2_3")
    )
    row["max_prototype_cosine"] = max(
        float(item["max_prototype_cosine"])
        for item in diagnostics
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-results",
        nargs="+",
        required=True,
        help="entries in SEED=ROOT format",
    )
    parser.add_argument("--actions", nargs="+", required=True)
    parser.add_argument("--output-runs", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-diagnostics", required=True)
    args = parser.parse_args()

    seed_roots = parse_seed_results(args.seed_results)
    if len(seed_roots) < 2:
        parser.error("at least two seed results are required")

    summaries = {}
    diagnostics = {}
    run_rows = []
    diagnostic_rows = []
    summary_fieldnames = None
    diagnostic_fieldnames = None

    for seed, root in seed_roots.items():
        summary_path = os.path.join(root, SUMMARY_NAME)
        diagnostic_path = os.path.join(root, DIAGNOSTIC_NAME)
        seed_summary = read_csv(summary_path)
        seed_diagnostics = read_csv(diagnostic_path)
        if summary_fieldnames is None:
            summary_fieldnames = list(seed_summary[0])
        if diagnostic_fieldnames is None:
            diagnostic_fieldnames = list(seed_diagnostics[0])

        for row in seed_summary:
            key = (seed, row["action"], row["mode"])
            if key in summaries:
                raise ValueError(f"duplicate summary row: {key}")
            summaries[key] = row
            run_rows.append({"seed": seed, **row})
        for row in seed_diagnostics:
            key = (seed, row["action"])
            if key in diagnostics:
                raise ValueError(f"duplicate diagnostic row: {key}")
            diagnostics[key] = row
            diagnostic_rows.append({"seed": seed, **row})

    seeds = list(seed_roots)
    aggregate_rows = []
    for action in args.actions:
        base_rows = [
            summaries[(seed, action, "base")] for seed in seeds
        ]
        candidate_rows = [
            summaries[(seed, action, "prototype")]
            for seed in seeds
        ]
        action_diagnostics = [
            diagnostics[(seed, action)] for seed in seeds
        ]
        aggregate_rows.append(
            aggregate_row(
                action,
                base_rows,
                candidate_rows,
                action_diagnostics,
            )
        )

    macro_base_rows = []
    macro_candidate_rows = []
    all_diagnostics = []
    for seed in seeds:
        base_row = {}
        candidate_row = {}
        for metric in METRICS:
            base_row[metric] = statistics.fmean(
                float(summaries[(seed, action, "base")][metric])
                for action in args.actions
            )
            candidate_row[metric] = statistics.fmean(
                float(
                    summaries[
                        (seed, action, "prototype")
                    ][metric]
                )
                for action in args.actions
            )
        macro_base_rows.append(base_row)
        macro_candidate_rows.append(candidate_row)
        all_diagnostics.extend(
            diagnostics[(seed, action)]
            for action in args.actions
        )
    macro_row = aggregate_row(
        "MACRO",
        macro_base_rows,
        macro_candidate_rows,
        all_diagnostics,
    )
    macro_row["comparisons"] = len(seeds) * len(args.actions)
    macro_row["best_wins"] = sum(
        float(
            summaries[(seed, action, "prototype")]["best_coef"]
        )
        > float(summaries[(seed, action, "base")]["best_coef"])
        for seed in seeds
        for action in args.actions
    )
    macro_row["avg_wins"] = sum(
        float(
            summaries[
                (seed, action, "prototype")
            ]["best_avg_coef"]
        )
        > float(
            summaries[(seed, action, "base")]["best_avg_coef"]
        )
        for seed in seeds
        for action in args.actions
    )
    aggregate_rows.append(macro_row)

    write_csv(
        args.output_runs,
        run_rows,
        fieldnames=["seed"] + summary_fieldnames,
    )
    write_csv(args.output_summary, aggregate_rows)
    write_csv(
        args.output_diagnostics,
        diagnostic_rows,
        fieldnames=["seed"] + diagnostic_fieldnames,
    )
    print(
        f"combined {len(seeds)} seeds and "
        f"{len(args.actions)} actions"
    )
    print(f"runs: {args.output_runs}")
    print(f"summary: {args.output_summary}")
    print(f"diagnostics: {args.output_diagnostics}")


if __name__ == "__main__":
    main()
