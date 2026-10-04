#!/usr/bin/env python
"""Collect original-protocol PAMFN metrics from Base/candidate logs."""

import argparse
import csv
import os
import re


SUMMARY_PATTERN = re.compile(
    r"best avg coef:\s*(?P<best_avg_coef>[-+.\deE]+)\s+"
    r"best coef:\s*(?P<best_coef>[-+.\deE]+)\s+"
    r"min avg loss:\s*(?P<min_avg_loss>[-+.\deE]+)\s+"
    r"min loss:\s*(?P<min_loss>[-+.\deE]+)"
)


def read_summary(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    matches = list(SUMMARY_PATTERN.finditer(text))
    if not matches:
        raise ValueError(f"original-protocol summary not found: {path}")
    summary = {
        name: float(value)
        for name, value in matches[-1].groupdict().items()
    }
    # The final logger prints correlations in [0, 1], while the paper table
    # and per-epoch BestCoef fields use percentage points.
    summary["best_coef"] *= 100.0
    summary["best_avg_coef"] *= 100.0
    return summary


def find_log(roots, action, mode, seed):
    name = f"{action}_{mode}_seed{seed}.log"
    matches = [
        os.path.join(root, action, name)
        for root in roots
        if os.path.isfile(os.path.join(root, action, name))
    ]
    if not matches:
        raise FileNotFoundError(
            f"{name} not found under: {', '.join(roots)}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"{name} is ambiguous; found: {', '.join(matches)}"
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        help="root containing both Base and Pairwise logs",
    )
    parser.add_argument(
        "--base-roots",
        nargs="+",
        help="one or more roots containing reusable Base logs",
    )
    parser.add_argument(
        "--pairwise-root",
        help="legacy alias for --candidate-root with pairwise mode",
    )
    parser.add_argument(
        "--candidate-root",
        help="root containing candidate-mode logs",
    )
    parser.add_argument(
        "--candidate-mode",
        default="pairwise",
        help="candidate filename mode, for example pairwise or listwise",
    )
    parser.add_argument("--actions", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.pairwise_root and args.candidate_root:
        parser.error(
            "--pairwise-root and --candidate-root are mutually exclusive"
        )
    candidate_root = args.candidate_root or args.pairwise_root
    if args.pairwise_root and args.candidate_mode != "pairwise":
        parser.error(
            "--pairwise-root requires --candidate-mode pairwise"
        )

    if args.root:
        if args.base_roots or candidate_root:
            parser.error(
                "--root cannot be combined with --base-roots or "
                "a candidate root"
            )
        base_roots = [args.root]
        candidate_roots = [args.root]
    else:
        if not args.base_roots or not candidate_root:
            parser.error(
                "provide --root, or both --base-roots and a "
                "candidate root"
            )
        base_roots = args.base_roots
        candidate_roots = [candidate_root]

    rows = []
    for action in args.actions:
        summaries = {
            "base": read_summary(
                find_log(base_roots, action, "base", args.seed)
            ),
            args.candidate_mode: read_summary(
                find_log(
                    candidate_roots,
                    action,
                    args.candidate_mode,
                    args.seed,
                )
            ),
        }
        for mode in ("base", args.candidate_mode):
            summary = summaries[mode]
            rows.append({
                "action": action,
                "mode": mode,
                **summary,
                "delta_best_coef": (
                    summary["best_coef"]
                    - summaries["base"]["best_coef"]
                ),
                "delta_best_avg_coef": (
                    summary["best_avg_coef"]
                    - summaries["base"]["best_avg_coef"]
                ),
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fieldnames = [
        "action",
        "mode",
        "best_coef",
        "best_avg_coef",
        "min_loss",
        "min_avg_loss",
        "delta_best_coef",
        "delta_best_avg_coef",
    ]
    with open(args.output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
