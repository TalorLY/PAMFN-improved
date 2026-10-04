#!/usr/bin/env python
"""Combine Base, grade-head, unordered, and ordered prototype results."""

import argparse
import csv
import os
import statistics

from summarize_original_protocol import read_summary


MODES = ("base", "grade", "unordered", "prototype")
METRICS = (
    "best_coef",
    "best_avg_coef",
    "min_loss",
    "min_avg_loss",
)


def _read_csv(path):
    with open(
            path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    os.makedirs(
        os.path.dirname(os.path.abspath(path)),
        exist_ok=True,
    )
    with open(
            path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def _candidate_summary(root, action, mode, seed):
    path = os.path.join(
        root, action, f"{action}_{mode}_seed{seed}.log"
    )
    return read_summary(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-summary", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--actions", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-runs", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-advantages", required=True)
    args = parser.parse_args()

    reference = {
        (row["action"], row["mode"]): row
        for row in _read_csv(args.reference_summary)
    }
    results = {}
    for action in args.actions:
        for mode in ("base", "prototype"):
            row = reference.get((action, mode))
            if row is None:
                raise ValueError(
                    f"reference summary missing {action}/{mode}"
                )
            results[(action, mode)] = {
                metric: float(row[metric])
                for metric in METRICS
            }
        for mode in ("grade", "unordered"):
            results[(action, mode)] = _candidate_summary(
                args.candidate_root,
                action,
                mode,
                args.seed,
            )

    run_rows = []
    for action in args.actions:
        base = results[(action, "base")]
        for mode in MODES:
            result = results[(action, mode)]
            run_rows.append({
                "action": action,
                "mode": mode,
                **result,
                "delta_best_coef": (
                    result["best_coef"] - base["best_coef"]
                ),
                "delta_best_avg_coef": (
                    result["best_avg_coef"]
                    - base["best_avg_coef"]
                ),
                "delta_min_loss": (
                    result["min_loss"] - base["min_loss"]
                ),
                "delta_min_avg_loss": (
                    result["min_avg_loss"]
                    - base["min_avg_loss"]
                ),
            })

    summary_rows = []
    for mode in MODES:
        mode_results = [
            results[(action, mode)] for action in args.actions
        ]
        base_results = [
            results[(action, "base")] for action in args.actions
        ]
        row = {
            "mode": mode,
            "action_count": len(args.actions),
        }
        for metric in METRICS:
            values = [result[metric] for result in mode_results]
            base_values = [
                result[metric] for result in base_results
            ]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"delta_{metric}_mean"] = statistics.fmean(
                value - base
                for value, base in zip(values, base_values)
            )
        row["best_wins_vs_base"] = sum(
            result["best_coef"] > base["best_coef"]
            for result, base in zip(mode_results, base_results)
        )
        row["avg_wins_vs_base"] = sum(
            result["best_avg_coef"] > base["best_avg_coef"]
            for result, base in zip(mode_results, base_results)
        )
        summary_rows.append(row)

    advantage_rows = []
    for action in args.actions:
        ordered = results[(action, "prototype")]
        unordered = results[(action, "unordered")]
        grade = results[(action, "grade")]
        advantage_rows.append({
            "action": action,
            "ordered_minus_unordered_best": (
                ordered["best_coef"] - unordered["best_coef"]
            ),
            "ordered_minus_unordered_avg": (
                ordered["best_avg_coef"]
                - unordered["best_avg_coef"]
            ),
            "ordered_minus_grade_best": (
                ordered["best_coef"] - grade["best_coef"]
            ),
            "ordered_minus_grade_avg": (
                ordered["best_avg_coef"]
                - grade["best_avg_coef"]
            ),
        })
    advantage_rows.append({
        "action": "MACRO",
        "ordered_minus_unordered_best": statistics.fmean(
            row["ordered_minus_unordered_best"]
            for row in advantage_rows
        ),
        "ordered_minus_unordered_avg": statistics.fmean(
            row["ordered_minus_unordered_avg"]
            for row in advantage_rows
        ),
        "ordered_minus_grade_best": statistics.fmean(
            row["ordered_minus_grade_best"]
            for row in advantage_rows
        ),
        "ordered_minus_grade_avg": statistics.fmean(
            row["ordered_minus_grade_avg"]
            for row in advantage_rows
        ),
    })

    _write_csv(args.output_runs, run_rows)
    _write_csv(args.output_summary, summary_rows)
    _write_csv(args.output_advantages, advantage_rows)
    print(f"runs: {args.output_runs}")
    print(f"summary: {args.output_summary}")
    print(f"advantages: {args.output_advantages}")


if __name__ == "__main__":
    main()
