#!/usr/bin/env python
"""Summarize the six-action COQM anchor screen or formal comparison."""

import argparse
import csv
import json
import math
import os
import statistics


ACTIONS = ("TES", "PCS", "Ball", "Clubs", "Hoop", "Ribbon")
FORMAL_SEEDS = (0, 1, 2, 42, 3407)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _finite(value, field):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"non-finite {field}: {value}")
    return float(value)


def _mean(values):
    return statistics.fmean(float(value) for value in values)


def experiment_grid(stage):
    if stage == "screen":
        return (0,), ("coqm_uniform", "coqm_train_center")
    if stage == "formal":
        return FORMAL_SEEDS, ("base", "coqm_geom")
    raise ValueError(f"unsupported stage: {stage}")


def load_runs(root, stage, candidate=""):
    seeds, methods = experiment_grid(stage)
    rows = []
    for action in ACTIONS:
        for seed in seeds:
            for method in methods:
                run_root = os.path.join(
                    root, action, f"seed{seed}", method
                )
                train_path = os.path.join(run_root, "train_result.json")
                test_path = os.path.join(run_root, "test_result.json")
                if not os.path.isfile(train_path) or not os.path.isfile(
                        test_path):
                    raise ValueError(
                        f"missing result for {action}/seed{seed}/{method}"
                    )
                train = _read_json(train_path)
                test = _read_json(test_path)
                metadata = test.get("run", {})
                if (
                        metadata.get("tag") != method
                        or metadata.get("action") != action
                        or metadata.get("seed") != seed):
                    raise ValueError(
                        f"metadata mismatch for {action}/seed{seed}/{method}"
                    )
                if metadata.get("eval_policy") != "argmax":
                    raise ValueError(
                        f"non-argmax evaluation for {action}/{seed}/{method}"
                    )
                if method != "base":
                    expected_mode = (
                        "uniform" if method == "coqm_uniform"
                        else "train_centers" if method == "coqm_train_center"
                        else candidate
                    )
                    if metadata.get("coqm_anchor_mode") != expected_mode:
                        raise ValueError(
                            f"anchor mode mismatch for {action}/{seed}/{method}"
                        )
                    if not metadata.get("quality_aux_rng_isolated"):
                        raise ValueError(
                            f"RNG isolation missing for {action}/{seed}/{method}"
                        )
                    if _finite(
                            metadata.get("coqm_coord_scale"),
                            "coqm_coord_scale") != 0.0:
                        raise ValueError(
                            f"coordinate loss enabled for {action}/{seed}/{method}"
                        )
                    anchor_scores = metadata.get("coqm_anchor_scores", [])
                    if (
                            len(anchor_scores) != 4
                            or any(not math.isfinite(float(value))
                                   for value in anchor_scores)
                            or any(float(right) <= float(left)
                                   for left, right in zip(
                                       anchor_scores, anchor_scores[1:]))):
                        raise ValueError(
                            f"invalid anchors for {action}/{seed}/{method}"
                        )
                row = {
                    "action": action,
                    "seed": seed,
                    "method": method,
                    "best_srcc": _finite(
                        train.get("best_srcc"), "best_srcc"
                    ),
                    "saved_model_srcc": _finite(
                        train.get("saved_model_srcc"),
                        "saved_model_srcc",
                    ),
                    "test_srcc": _finite(test.get("coef"), "coef") / 100,
                    "test_mse": _finite(
                        test.get("loss_x100"), "loss_x100"
                    ) / 100,
                    "prediction_std_ratio": _finite(
                        test.get("prediction_std_ratio"),
                        "prediction_std_ratio",
                    ),
                    "mae": _finite(test.get("mae"), "mae"),
                    "bias": _finite(test.get("bias"), "bias"),
                    "anchor_mode": metadata.get("coqm_anchor_mode", ""),
                    "anchor_scores": json.dumps(
                        metadata.get("coqm_anchor_scores", [])
                    ),
                }
                rows.append(row)
    return rows


def _paired(rows, action, seed, method):
    return next(
        row for row in rows
        if row["action"] == action
        and row["seed"] == seed
        and row["method"] == method
    )


def summarize_screen(rows):
    uniform = _mean(
        row["test_srcc"] for row in rows
        if row["method"] == "coqm_uniform"
    )
    train_center = _mean(
        row["test_srcc"] for row in rows
        if row["method"] == "coqm_train_center"
    )
    selected = (
        "train_centers" if train_center > uniform else "uniform"
    )
    action_deltas = {
        action: (
            _paired(rows, action, 0, "coqm_train_center")["test_srcc"]
            - _paired(rows, action, 0, "coqm_uniform")["test_srcc"]
        ) * 100
        for action in ACTIONS
    }
    return {
        "decision": (
            "COQM_TRAIN_CENTER_SELECTED"
            if selected == "train_centers"
            else "COQM_UNIFORM_RETAINED"
        ),
        "selected_anchor_mode": selected,
        "mean_test_srcc": {
            "coqm_uniform": uniform,
            "coqm_train_center": train_center,
        },
        "train_center_minus_uniform_pp": (
            train_center - uniform
        ) * 100,
        "train_center_action_delta_pp": action_deltas,
        "train_center_action_wins": sum(
            value > 0 for value in action_deltas.values()
        ),
    }


def summarize_formal(rows, candidate):
    means = {
        method: _mean(
            row["test_srcc"] for row in rows
            if row["method"] == method
        )
        for method in ("base", "coqm_geom")
    }
    action_deltas = {
        action: _mean(
            _paired(rows, action, seed, "coqm_geom")["test_srcc"]
            - _paired(rows, action, seed, "base")["test_srcc"]
            for seed in FORMAL_SEEDS
        ) * 100
        for action in ACTIONS
    }
    seed_deltas = {
        str(seed): _mean(
            _paired(rows, action, seed, "coqm_geom")["test_srcc"]
            - _paired(rows, action, seed, "base")["test_srcc"]
            for action in ACTIONS
        ) * 100
        for seed in FORMAL_SEEDS
    }
    paired_deltas = [
        _paired(rows, action, seed, "coqm_geom")["test_srcc"]
        - _paired(rows, action, seed, "base")["test_srcc"]
        for action in ACTIONS
        for seed in FORMAL_SEEDS
    ]
    return {
        "decision": (
            "COQM_ALL_ACTION_FORMAL_POSITIVE"
            if means["coqm_geom"] > means["base"]
            else "COQM_ALL_ACTION_FORMAL_NOT_POSITIVE"
        ),
        "selected_anchor_mode": candidate,
        "mean_test_srcc": means,
        "coqm_minus_base_pp": (
            means["coqm_geom"] - means["base"]
        ) * 100,
        "action_delta_pp": action_deltas,
        "seed_delta_pp": seed_deltas,
        "paired_wins": sum(value > 0 for value in paired_deltas),
        "paired_total": len(paired_deltas),
    }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, stage, report):
    if stage == "screen":
        means = report["mean_test_srcc"]
        lines = [
            "# COQM 六动作锚点筛选",
            "",
            f"结论：`{report['decision']}`",
            "",
            "| 方法 | 六动作 Macro SRCC |",
            "|---|---:|",
            f"| Uniform | {means['coqm_uniform']:.4f} |",
            f"| TrainCenter | {means['coqm_train_center']:.4f} |",
            "",
            "TrainCenter − Uniform："
            f"{report['train_center_minus_uniform_pp']:+.3f} pp；"
            f"动作胜场 {report['train_center_action_wins']}/6。",
            "",
            "动作差值（pp）：" + ", ".join(
                f"{key}={value:+.3f}" for key, value in
                report["train_center_action_delta_pp"].items()
            ),
        ]
    else:
        means = report["mean_test_srcc"]
        lines = [
            "# COQM 六动作五种子正式结果",
            "",
            f"结论：`{report['decision']}`",
            "",
            f"锚点：`{report['selected_anchor_mode']}`",
            "",
            "| 方法 | 六动作五种子 Macro SRCC |",
            "|---|---:|",
            f"| Base | {means['base']:.4f} |",
            f"| COQM-Geom | {means['coqm_geom']:.4f} |",
            "",
            f"COQM-Geom − Base：{report['coqm_minus_base_pp']:+.3f} pp；"
            f"配对胜场 {report['paired_wins']}/{report['paired_total']}。",
        ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--stage", choices=("screen", "formal"), required=True
    )
    parser.add_argument(
        "--candidate", choices=("uniform", "train_centers"), default=""
    )
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()
    if args.stage == "formal" and not args.candidate:
        parser.error("--candidate is required for the formal stage")
    rows = load_runs(args.root, args.stage, args.candidate)
    report = (
        summarize_screen(rows) if args.stage == "screen"
        else summarize_formal(rows, args.candidate)
    )
    write_csv(args.output_csv, rows)
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_json)),
        exist_ok=True,
    )
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    write_markdown(args.output_md, args.stage, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
