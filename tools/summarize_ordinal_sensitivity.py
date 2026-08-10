#!/usr/bin/env python
"""Summarize one-factor ordinal-prototype sensitivity experiments."""

import argparse
import csv
import math
import os


SETTING_META = {
    "k3": ("num_grades", 3.0, "K=3"),
    "k4": ("num_grades", 4.0, "K=4"),
    "k5": ("num_grades", 5.0, "K=5"),
    "w5e5": ("weight", 0.00005, "w=5e-5"),
    "w1e4": ("weight", 0.0001, "w=1e-4"),
    "w2e4": ("weight", 0.0002, "w=2e-4"),
}
ANCHORS = {"num_grades": "k4", "weight": "w1e4"}


def _read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _settings(entries):
    result = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"setting must use NAME=ROOT syntax: {entry}"
            )
        name, root = entry.split("=", 1)
        if name not in SETTING_META:
            raise ValueError(f"unsupported setting name: {name}")
        if name in result:
            raise ValueError(f"duplicate setting name: {name}")
        result[name] = root
    missing = set(SETTING_META) - set(result)
    if missing:
        raise ValueError(
            "missing settings: " + ", ".join(sorted(missing))
        )
    return result


def _float(row, key):
    return float(row[key])


def _mean(values):
    return sum(values) / len(values)


def _scope_actions(actions):
    fisv = [action for action in actions if action in ("TES", "PCS")]
    rg = [
        action
        for action in actions
        if action in ("Ball", "Clubs", "Hoop", "Ribbon")
    ]
    if len(fisv) != 2 or len(rg) != 4:
        raise ValueError(
            "actions must contain TES, PCS, Ball, Clubs, Hoop, Ribbon"
        )
    scopes = [(action, [action]) for action in actions]
    scopes.extend([
        ("FIS-V", fisv),
        ("RG", rg),
        ("Macro", list(actions)),
    ])
    return scopes


def _load_metrics(root, actions):
    path = os.path.join(root, "original_protocol_summary.csv")
    rows = _read_csv(path)
    indexed = {
        (row["action"], row["mode"]): row
        for row in rows
    }
    required = {
        (action, mode)
        for action in actions
        for mode in ("base", "prototype")
    }
    missing = required - set(indexed)
    if missing:
        raise ValueError(
            f"{path} is missing rows: {sorted(missing)}"
        )
    return indexed


def _load_feature_rows(root, actions):
    path = os.path.join(root, "feature_evidence_summary.csv")
    rows = [
        row for row in _read_csv(path)
        if row["mode"] == "prototype"
    ]
    indexed = {row["action"]: row for row in rows}
    missing = set(actions) - set(indexed)
    if missing:
        raise ValueError(
            f"{path} is missing actions: {sorted(missing)}"
        )
    return indexed


def _metric_means(indexed, selected):
    base = [indexed[(action, "base")] for action in selected]
    candidate = [
        indexed[(action, "prototype")] for action in selected
    ]
    best = _mean([_float(row, "best_coef") for row in candidate])
    best_avg = _mean([
        _float(row, "best_avg_coef") for row in candidate
    ])
    base_best = _mean([_float(row, "best_coef") for row in base])
    base_best_avg = _mean([
        _float(row, "best_avg_coef") for row in base
    ])
    return {
        "best_coef": best,
        "base_best_coef": base_best,
        "delta_best_coef": best - base_best,
        "best_avg_coef": best_avg,
        "base_best_avg_coef": base_best_avg,
        "delta_best_avg_coef": best_avg - base_best_avg,
        "min_loss": _mean([
            _float(row, "min_loss") for row in candidate
        ]),
        "min_avg_loss": _mean([
            _float(row, "min_avg_loss") for row in candidate
        ]),
        "wins_best": sum(
            _float(row, "best_coef") > _float(base_row, "best_coef")
            for row, base_row in zip(candidate, base)
        ),
        "wins_best_avg": sum(
            _float(row, "best_avg_coef")
            > _float(base_row, "best_avg_coef")
            for row, base_row in zip(candidate, base)
        ),
    }


def _feature_means(indexed, selected):
    rows = [indexed[action] for action in selected]
    metrics = {}
    for key in (
            "expected_grade_srcc",
            "grade_adjacent_accuracy",
            "grade_mae",
            "axis_label_srcc",
            "min_prototype_gap",
            "max_prototype_cosine",
            "weighted_auxiliary_test"):
        values = [
            _float(row, key)
            for row in rows
            if row.get(key, "") not in ("", None)
            and math.isfinite(_float(row, key))
        ]
        metrics[key] = _mean(values) if values else ""
    metrics["monotonic_actions"] = sum(
        row["monotonic_bin_means"].lower() == "true"
        for row in rows
    )
    collapsed = 0
    for row in rows:
        count_keys = [
            key for key in row
            if key.startswith("predicted_count_bin")
        ]
        if any(int(float(row[key])) == 0 for key in count_keys):
            collapsed += 1
    metrics["collapsed_actions"] = collapsed
    return metrics


def _fmt(value, digits=2):
    if value == "":
        return ""
    return f"{float(value):.{digits}f}"


def _markdown_table(lines, headers, rows):
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines.append("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting-roots", nargs="+", required=True,
        help="six NAME=ROOT entries: k3 k4 k5 w5e5 w1e4 w2e4",
    )
    parser.add_argument("--actions", nargs="+", required=True)
    parser.add_argument("--output-runs", required=True)
    parser.add_argument("--output-k", required=True)
    parser.add_argument("--output-weight", required=True)
    parser.add_argument("--output-features", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()

    roots = _settings(args.setting_roots)
    scopes = _scope_actions(args.actions)
    metrics = {
        name: _load_metrics(root, args.actions)
        for name, root in roots.items()
    }
    features = {
        name: _load_feature_rows(root, args.actions)
        for name, root in roots.items()
    }

    run_rows = []
    factor_rows = {"num_grades": [], "weight": []}
    feature_rows = []
    for name, root in roots.items():
        factor, value, label = SETTING_META[name]
        anchor = ANCHORS[factor]
        for scope, selected in scopes:
            current = _metric_means(metrics[name], selected)
            reference = _metric_means(metrics[anchor], selected)
            row = {
                "factor": factor,
                "setting": name,
                "setting_label": label,
                "parameter_value": value,
                "scope": scope,
                "action_count": len(selected),
                **current,
                "delta_vs_anchor_best_coef": (
                    current["best_coef"] - reference["best_coef"]
                ),
                "delta_vs_anchor_best_avg_coef": (
                    current["best_avg_coef"]
                    - reference["best_avg_coef"]
                ),
            }
            run_rows.append(row)
            factor_rows[factor].append(row)

            feature_rows.append({
                "factor": factor,
                "setting": name,
                "setting_label": label,
                "parameter_value": value,
                "scope": scope,
                "action_count": len(selected),
                **_feature_means(features[name], selected),
            })

    _write_csv(args.output_runs, run_rows)
    _write_csv(args.output_k, factor_rows["num_grades"])
    _write_csv(args.output_weight, factor_rows["weight"])
    _write_csv(args.output_features, feature_rows)

    lines = [
        "# Ordinal Prototype Sensitivity (Seed 0)",
        "",
        "本报告采用单因素控制：K 实验固定 w=1e-4；权重实验固定 "
        "K=4。K=4,w=1e-4 是已完成的参考配置，没有重复训练。",
        "",
        "## 等级数 K 敏感性",
        "",
    ]
    report_scopes = {"FIS-V", "RG", "Macro"}
    for factor, title in (
            ("num_grades", "K"),
            ("weight", "辅助损失权重")):
        if factor == "weight":
            lines.extend(["## 权重鲁棒性", ""])
        rows = []
        for row in factor_rows[factor]:
            if row["scope"] not in report_scopes:
                continue
            rows.append([
                row["setting_label"],
                row["scope"],
                _fmt(row["best_coef"]),
                _fmt(row["delta_best_coef"]),
                _fmt(row["delta_vs_anchor_best_coef"]),
                _fmt(row["best_avg_coef"]),
                _fmt(row["delta_best_avg_coef"]),
                _fmt(row["delta_vs_anchor_best_avg_coef"]),
                f"{row['wins_best']}/{row['action_count']}",
            ])
        _markdown_table(
            lines,
            [
                title, "范围", "Best", "ΔBase", "Δ参考",
                "BestAvg", "ΔBase", "Δ参考", "胜Base",
            ],
            rows,
        )

    lines.extend(["## 特征与原型证据（六动作宏平均）", ""])
    feature_report = []
    for row in feature_rows:
        if row["scope"] != "Macro":
            continue
        feature_report.append([
            row["setting_label"],
            _fmt(row["expected_grade_srcc"], 3),
            _fmt(row["grade_adjacent_accuracy"], 3),
            _fmt(row["axis_label_srcc"], 3),
            _fmt(row["min_prototype_gap"], 3),
            _fmt(row["max_prototype_cosine"], 3),
            f"{row['monotonic_actions']}/6",
            str(row["collapsed_actions"]),
        ])
    _markdown_table(
        lines,
        [
            "配置", "等级SRCC", "邻级准确率", "轴SRCC",
            "最小间隔", "最大余弦", "单调动作", "塌缩动作",
        ],
        feature_report,
    )
    lines.extend([
        "## 解释原则",
        "",
        "- 不用 seed 0 敏感性结果重新挑选主配置；它只检验结论是否依赖单一 K 或单一权重。",
        "- 首先看 K=3/4/5 与三档权重下，宏平均和两个数据域是否保持同方向；再结合等级单调性、原型间隔与塌缩动作数判断结构是否健康。",
        "- 逐动作明细与损失值保存在 CSV 中，汇报时应同时说明负向动作，不能只展示提升动作。",
        "",
    ])
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_report)),
        exist_ok=True,
    )
    with open(args.output_report, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print(f"runs: {args.output_runs}")
    print(f"K sensitivity: {args.output_k}")
    print(f"weight sensitivity: {args.output_weight}")
    print(f"feature evidence: {args.output_features}")
    print(f"mentor report: {args.output_report}")


if __name__ == "__main__":
    main()
