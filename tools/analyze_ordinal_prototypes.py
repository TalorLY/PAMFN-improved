#!/usr/bin/env python
"""Export ordered-prototype occupancy and collapse diagnostics."""

import argparse
import csv
import os

import torch
import torch.nn.functional as F

from datasets import read_labels


def load_state_dict(path):
    try:
        value = torch.load(
            path, map_location="cpu", weights_only=False
        )
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict) and "state_dict" in value:
        return value["state_dict"]
    return value


def analyze_checkpoint(
        path, action, require_order=True, order_margin=0.1):
    state = load_state_dict(path)
    required = (
        "quality_prototypes",
        "quality_axis",
        "ordinal_thresholds",
    )
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(
            f"{path} is missing prototype tensors: {missing}"
        )

    prototypes = F.normalize(
        state["quality_prototypes"].float(), dim=1
    )
    num_grades = prototypes.shape[0]
    axis = F.normalize(state["quality_axis"].float(), dim=0)
    thresholds = state["ordinal_thresholds"].float()
    if thresholds.numel() != num_grades - 1:
        raise ValueError(
            f"{path} has {num_grades} prototypes but "
            f"{thresholds.numel()} thresholds"
        )
    positions = prototypes @ axis
    gaps = positions[1:] - positions[:-1]

    similarity = prototypes @ prototypes.transpose(0, 1)
    off_diagonal = similarity[
        ~torch.eye(num_grades, dtype=torch.bool)
    ]
    max_cosine = float(off_diagonal.max())
    labels = torch.tensor(
        [
            label
            for _, label in read_labels(action, is_train=True)
        ],
        dtype=torch.float32,
    )
    bins = torch.bucketize(labels, thresholds)
    counts = torch.bincount(bins, minlength=num_grades)
    order_violations = int((gaps <= 0).sum())
    margin_violations = int((gaps < order_margin).sum())
    collapsed = (
        max_cosine >= 0.98
        or (require_order and order_violations > 0)
    )

    row = {
        "action": action,
        "checkpoint": path,
        "num_grades": num_grades,
        "max_prototype_cosine": max_cosine,
        "order_violations": order_violations,
        "margin_violations": margin_violations,
        "collapsed": collapsed,
    }
    for index, value in enumerate(thresholds, start=1):
        row[f"q{index}"] = float(value)
    for index, value in enumerate(counts):
        row[f"count_grade_{index}"] = int(value)
    if num_grades == 4:
        row.update({
            "count_poor": int(counts[0]),
            "count_fair": int(counts[1]),
            "count_good": int(counts[2]),
            "count_excellent": int(counts[3]),
        })
    for index, value in enumerate(positions):
        row[f"position_{index}"] = float(value)
    for index, value in enumerate(gaps):
        row[f"gap_{index}_{index + 1}"] = float(value)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--actions", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="prototype")
    parser.add_argument("--allow-unordered", action="store_true")
    parser.add_argument(
        "--order-margin", type=float, default=0.1
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for action in args.actions:
        checkpoint = os.path.join(
            args.root,
            action,
            f"{action}_{args.mode}_seed{args.seed}.pth",
        )
        rows.append(analyze_checkpoint(
            checkpoint,
            action,
            require_order=not args.allow_unordered,
            order_margin=args.order_margin,
        ))

    os.makedirs(
        os.path.dirname(os.path.abspath(args.output)),
        exist_ok=True,
    )
    fieldnames = list(rows[0])
    with open(
            args.output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
