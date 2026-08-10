#!/usr/bin/env python3
"""Diagnose clean calibration headroom in a frozen two-cue gate grid."""

import argparse
import csv
import json
import math
import os
import re
import tempfile
from collections import Counter

import numpy as np
from scipy.stats import spearmanr


PREDICTION_COLUMN = re.compile(r"^pred_m(.+)_s(.+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure label-selected clean gate headroom. This is an "
            "optimistic diagnostic and is not a deployable result."
        )
    )
    parser.add_argument("--action")
    parser.add_argument("--guardrail_grid")
    parser.add_argument("--prior_only_grid")
    parser.add_argument("--combine_reports", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--expected_samples", type=int)
    parser.add_argument(
        "--expected_grid_intervals", type=int,
        help=(
            "require the exact equally spaced gate grid from 0 to 1; "
            "10 requires the locked 11x11 oracle grid"
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--signal_threshold", type=float, default=0.5)
    parser.add_argument("--mse_guardrail", type=float, default=0.02)
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def decode_gate_token(token):
    legacy = {"0": 0.0, "05": 0.5, "1": 1.0}
    if token in legacy:
        return legacy[token]
    try:
        value = float(token.replace("p", "."))
    except ValueError as error:
        raise ValueError(f"invalid gate token: {token}") from error
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"gate token is outside [0, 1]: {token}")
    return value


def parse_prediction_column(column):
    match = PREDICTION_COLUMN.match(column)
    if match is None:
        return None
    return decode_gate_token(match.group(1)), decode_gate_token(match.group(2))


def read_grid(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"grid has no header: {path}")
        gate_columns = {}
        for column in reader.fieldnames:
            gates = parse_prediction_column(column)
            if gates is not None:
                if gates in gate_columns:
                    raise ValueError(f"duplicate gate pair in {path}: {gates}")
                gate_columns[gates] = column
        if not gate_columns:
            raise ValueError(f"grid has no prediction columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"grid has no samples: {path}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"grid contains duplicate sample IDs: {path}")
    labels = np.asarray([float(row["label"]) for row in rows])
    predictions = {
        gates: np.asarray([float(row[column]) for row in rows])
        for gates, column in gate_columns.items()
    }
    if not np.isfinite(labels).all() or any(
            not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError(f"grid contains non-finite values: {path}")
    motion_values = sorted({gates[0] for gates in predictions})
    skeleton_values = sorted({gates[1] for gates in predictions})
    expected_pairs = {
        (motion, skeleton)
        for motion in motion_values for skeleton in skeleton_values
    }
    if set(predictions) != expected_pairs:
        raise ValueError(f"gate columns do not form a complete grid: {path}")
    if (0.0, 0.0) not in predictions:
        raise ValueError(f"grid is missing the Base gate (0, 0): {path}")
    return {
        "sample_ids": sample_ids,
        "labels": labels,
        "predictions": predictions,
        "motion_values": motion_values,
        "skeleton_values": skeleton_values,
    }


def validate_grid_protocol(grid, path, expected_intervals):
    if expected_intervals is None:
        return
    if expected_intervals < 1:
        raise ValueError("expected grid intervals must be positive")
    expected_values = np.linspace(
        0.0, 1.0, expected_intervals + 1
    ).tolist()
    for cue in ("motion_values", "skeleton_values"):
        actual = grid[cue]
        if len(actual) != len(expected_values) or not np.allclose(
                actual, expected_values, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{path} does not match the locked "
                f"{expected_intervals + 1}x{expected_intervals + 1} "
                f"gate grid: {cue}={actual}"
            )
    expected_pairs = (expected_intervals + 1) ** 2
    if len(grid["predictions"]) != expected_pairs:
        raise ValueError(
            f"{path} has {len(grid['predictions'])} gate pairs; "
            f"expected {expected_pairs}"
        )


def safe_coef(prediction, labels):
    coefficient = spearmanr(prediction, labels).statistic
    return float(coefficient * 100) if np.isfinite(coefficient) else None


def metrics(prediction, labels):
    return {
        "coef": safe_coef(prediction, labels),
        "mse_raw": float(np.mean((prediction - labels) ** 2)),
    }


def candidate_metrics(predictions, labels):
    values = []
    for gates, prediction in predictions.items():
        result = metrics(prediction, labels)
        if result["coef"] is None:
            continue
        values.append({
            "motion_gate": float(gates[0]),
            "skeleton_gate": float(gates[1]),
            **result,
        })
    if not values:
        raise ValueError("all gate candidates have undefined Spearman")
    return values


def candidate_key(value):
    return (
        value["coef"], -value["mse_raw"],
        -(value["motion_gate"] + value["skeleton_gate"]),
        -value["motion_gate"], -value["skeleton_gate"],
    )


def select_candidates(predictions, labels, mse_guardrail):
    values = candidate_metrics(predictions, labels)
    pure = max(values, key=candidate_key)
    best_mse = min(value["mse_raw"] for value in values)
    mse_limit = best_mse * (1.0 + mse_guardrail)
    admissible = [
        value for value in values
        if value["mse_raw"] <= mse_limit + 1e-15
    ]
    return {
        "pure_coef_oracle": pure,
        "mse_admissible_oracle": max(admissible, key=candidate_key),
        "best_grid_mse": float(best_mse),
        "mse_limit": float(mse_limit),
        "admissible_candidates": len(admissible),
        "total_candidates": len(values),
    }


def percentile_interval(values):
    if not values:
        return [None, None]
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def paired_bootstrap_delta(
        selected_prediction, base_prediction, labels, iterations, seed):
    rng = np.random.default_rng(seed)
    deltas = []
    sample_count = len(labels)
    for _ in range(iterations):
        indices = rng.integers(0, sample_count, sample_count)
        selected_coef = safe_coef(
            selected_prediction[indices], labels[indices]
        )
        base_coef = safe_coef(base_prediction[indices], labels[indices])
        if selected_coef is not None and base_coef is not None:
            deltas.append(selected_coef - base_coef)
    if len(deltas) < max(10, iterations // 2):
        raise ValueError("too few valid bootstrap Spearman comparisons")
    return {
        "iterations_requested": int(iterations),
        "iterations_valid": len(deltas),
        "delta_coef_95ci": percentile_interval(deltas),
        "delta_coef_positive_probability": float(
            np.mean(np.asarray(deltas) > 0)
        ),
    }


def gate_tuple(value):
    return value["motion_gate"], value["skeleton_gate"]


def jackknife_selection(predictions, labels, mse_guardrail, selected_gate):
    selections = []
    sample_count = len(labels)
    for omitted in range(sample_count):
        keep = np.arange(sample_count) != omitted
        subset = {
            gates: prediction[keep]
            for gates, prediction in predictions.items()
        }
        chosen = select_candidates(
            subset, labels[keep], mse_guardrail
        )["mse_admissible_oracle"]
        selections.append(gate_tuple(chosen))
    counts = Counter(selections)
    mode_gate, mode_count = counts.most_common(1)[0]
    return {
        "leave_one_out_runs": sample_count,
        "distinct_selected_gates": len(counts),
        "full_sample_gate_selection_rate": float(
            counts[selected_gate] / sample_count
        ),
        "modal_gate": {
            "motion_gate": float(mode_gate[0]),
            "skeleton_gate": float(mode_gate[1]),
            "selection_rate": float(mode_count / sample_count),
        },
    }


def analyze_policy(
        grid, mse_guardrail, signal_threshold, bootstrap, seed):
    labels = grid["labels"]
    predictions = grid["predictions"]
    base_prediction = predictions[(0.0, 0.0)]
    base = metrics(base_prediction, labels)
    selection = select_candidates(predictions, labels, mse_guardrail)
    for name in ("pure_coef_oracle", "mse_admissible_oracle"):
        selected = selection[name]
        selected["delta_coef_vs_base"] = float(
            selected["coef"] - base["coef"]
        )
        selected["relative_mse_gain_vs_base"] = float(
            (base["mse_raw"] - selected["mse_raw"]) / base["mse_raw"]
        )
    chosen = selection["mse_admissible_oracle"]
    chosen_gate = gate_tuple(chosen)
    stability = paired_bootstrap_delta(
        predictions[chosen_gate], base_prediction, labels,
        bootstrap, seed,
    )
    jackknife = jackknife_selection(
        predictions, labels, mse_guardrail, chosen_gate
    )
    point_signal = chosen["delta_coef_vs_base"] >= signal_threshold
    direction_stable = (
        stability["delta_coef_positive_probability"] >= 0.90
    )
    if point_signal and direction_stable:
        status = "HEADROOM_SUPPORTED"
    elif point_signal:
        status = "HEADROOM_UNSTABLE"
    else:
        status = "NO_HEADROOM"
    return {
        "base": base,
        "selection": selection,
        "selected_gate_bootstrap": stability,
        "mse_admissible_selection_jackknife": jackknife,
        "decision": {
            "signal_threshold_delta_coef": float(signal_threshold),
            "point_signal_pass": bool(point_signal),
            "bootstrap_positive_probability_at_least_0_90": bool(
                direction_stable
            ),
            "status": status,
        },
    }


def root_cause(prior_result, guardrail_result, signal_threshold):
    prior_status = prior_result["decision"]["status"]
    guardrail_status = guardrail_result["decision"]["status"]
    if guardrail_status == "HEADROOM_SUPPORTED":
        return "ACTION_PRIOR_ESTIMATION_BOTTLENECK"
    if prior_status == "HEADROOM_SUPPORTED":
        return "CLEAN_RELIABILITY_ATTENUATION_BOTTLENECK"
    if "HEADROOM_UNSTABLE" in (prior_status, guardrail_status):
        return "INCONCLUSIVE_UNSTABLE_HEADROOM"
    if any(
            result["selection"]["pure_coef_oracle"]
            ["delta_coef_vs_base"] >= signal_threshold
            for result in (prior_result, guardrail_result)):
        return "COEF_MSE_GUARDRAIL_CONFLICT"
    return "CURRENT_RESIDUALS_HAVE_NO_CLEAN_COEF_HEADROOM"


def analyze_action(args):
    guardrail = read_grid(args.guardrail_grid)
    prior = read_grid(args.prior_only_grid)
    validate_grid_protocol(
        guardrail, args.guardrail_grid, args.expected_grid_intervals
    )
    validate_grid_protocol(
        prior, args.prior_only_grid, args.expected_grid_intervals
    )
    if guardrail["sample_ids"] != prior["sample_ids"] or not np.array_equal(
            guardrail["labels"], prior["labels"]):
        raise ValueError("guardrail and prior-only grids are not aligned")
    if set(guardrail["predictions"]) != set(prior["predictions"]):
        raise ValueError("guardrail and prior-only gate grids differ")
    if args.expected_samples is not None and len(
            guardrail["sample_ids"]) != args.expected_samples:
        raise ValueError(
            f"expected {args.expected_samples} samples, found "
            f"{len(guardrail['sample_ids'])}"
        )
    base_difference = float(np.max(np.abs(
        guardrail["predictions"][(0.0, 0.0)]
        - prior["predictions"][(0.0, 0.0)]
    )))
    if base_difference > 1e-7:
        raise ValueError(
            "guardrail and prior-only Base predictions are inconsistent"
        )
    prior_result = analyze_policy(
        prior, args.mse_guardrail, args.signal_threshold,
        args.bootstrap, args.seed,
    )
    guardrail_result = analyze_policy(
        guardrail, args.mse_guardrail, args.signal_threshold,
        args.bootstrap, args.seed + 1,
    )
    cause = root_cause(
        prior_result, guardrail_result, args.signal_threshold,
    )
    return {
        "schema_version": 1,
        "action": args.action,
        "protocol": (
            "label-selected clean calibration oracle; diagnostic only; "
            "not deployable and not a final result"
        ),
        "samples": len(guardrail["sample_ids"]),
        "grid": {
            "motion_values": guardrail["motion_values"],
            "skeleton_values": guardrail["skeleton_values"],
            "pairs": len(guardrail["predictions"]),
            "intervals": len(guardrail["motion_values"]) - 1,
        },
        "base_max_absolute_difference_between_policies": base_difference,
        "prior_only": prior_result,
        "guardrail": guardrail_result,
        "decision": {
            "root_cause": cause,
            "status": (
                "STOP_CURRENT_AUTOGATE_ACCURACY_ROUTE"
                if cause == "CURRENT_RESIDUALS_HAVE_NO_CLEAN_COEF_HEADROOM"
                else "REVISE_BEFORE_NEW_SCREEN"
            ),
            "note": (
                "Opening the calibration oracle consumes it for diagnosis; "
                "do not report the oracle as model performance."
            ),
        },
    }


def combine_reports(paths, expected_grid_intervals=None):
    reports = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("schema_version") != 1 or "decision" not in report:
            raise ValueError(f"invalid oracle report: {path}")
        if expected_grid_intervals is not None:
            grid = report.get("grid", {})
            expected_values = np.linspace(
                0.0, 1.0, expected_grid_intervals + 1
            ).tolist()
            expected_pairs = (expected_grid_intervals + 1) ** 2
            for cue in ("motion_values", "skeleton_values"):
                actual = grid.get(cue, [])
                if len(actual) != len(expected_values) or not np.allclose(
                        actual, expected_values, rtol=0.0, atol=1e-12):
                    raise ValueError(
                        f"oracle report has the wrong {cue}: {path}"
                    )
            if grid.get("pairs") != expected_pairs:
                raise ValueError(
                    f"oracle report has {grid.get('pairs')} gate pairs; "
                    f"expected {expected_pairs}: {path}"
                )
        reports.append(report)
    actions = {report["action"]: report for report in reports}
    if set(actions) != {"PCS", "Ball"}:
        raise ValueError("combined oracle report requires PCS and Ball")
    guardrail_supported = any(
        report["guardrail"]["decision"]["status"]
        == "HEADROOM_SUPPORTED" for report in reports
    )
    prior_supported = any(
        report["prior_only"]["decision"]["status"]
        == "HEADROOM_SUPPORTED" for report in reports
    )
    unstable = any(
        "HEADROOM_UNSTABLE" in (
            report["prior_only"]["decision"]["status"],
            report["guardrail"]["decision"]["status"],
        ) for report in reports
    )
    if guardrail_supported:
        status = "HEADROOM_FOUND_REVISE_ACTION_PRIOR_ESTIMATION"
    elif prior_supported:
        status = "HEADROOM_FOUND_REVISE_CLEAN_RELIABILITY"
    elif unstable:
        status = "INCONCLUSIVE_UNSTABLE_HEADROOM"
    elif any(
            report["decision"]["root_cause"]
            == "COEF_MSE_GUARDRAIL_CONFLICT" for report in reports):
        status = "HEADROOM_FOUND_REVISE_SELECTION_OBJECTIVE"
    else:
        status = "STOP_CURRENT_AUTOGATE_ACCURACY_ROUTE"
    return {
        "schema_version": 1,
        "protocol": "PCS/Ball final clean oracle headroom diagnosis",
        "actions": {
            action: {
                "root_cause": report["decision"]["root_cause"],
                "prior_only_delta_coef": report["prior_only"]["selection"]
                    ["mse_admissible_oracle"]["delta_coef_vs_base"],
                "prior_only_status": report["prior_only"]["decision"]
                    ["status"],
                "guardrail_delta_coef": report["guardrail"]["selection"]
                    ["mse_admissible_oracle"]["delta_coef_vs_base"],
                "guardrail_status": report["guardrail"]["decision"]
                    ["status"],
            }
            for action, report in sorted(actions.items())
        },
        "decision": {
            "status": status,
            "full_experiment_authorized": False,
            "note": (
                "An oracle can locate a bottleneck but cannot authorize a "
                "formal experiment. Any revision requires a fresh screen."
            ),
        },
    }


def write_report(report, output):
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if output:
        parent = os.path.dirname(os.path.abspath(output))
        os.makedirs(parent, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


def write_fixture(path, labels, base, improved):
    gates = (0.0, 0.5, 1.0)
    columns = [
        (motion, skeleton)
        for motion in gates for skeleton in gates
    ]
    fieldnames = ["sample_id", "group_id", "label"] + [
        "pred_m" + ("0" if m == 0 else "05" if m == 0.5 else "1")
        + "_s" + ("0" if s == 0 else "05" if s == 0.5 else "1")
        for m, s in columns
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, label in enumerate(labels):
            row = {
                "sample_id": str(index), "group_id": str(index),
                "label": float(label),
            }
            for motion, skeleton in columns:
                prediction = base + motion * (improved - base)
                mtoken = "0" if motion == 0 else "05" if motion == 0.5 else "1"
                stoken = "0" if skeleton == 0 else "05" if skeleton == 0.5 else "1"
                row[f"pred_m{mtoken}_s{stoken}"] = float(prediction[index])
            writer.writerow(row)


def self_test():
    assert parse_prediction_column("pred_m0p1_s0p9") == (0.1, 0.9)
    labels = np.arange(40, dtype=np.float64)
    base = labels + 8.0 * np.sin(np.arange(40))
    with tempfile.TemporaryDirectory() as directory:
        grid = os.path.join(directory, "grid.csv")
        write_fixture(grid, labels, base, labels)
        args = argparse.Namespace(
            action="PCS", guardrail_grid=grid, prior_only_grid=grid,
            expected_samples=40, bootstrap=200, seed=0,
            signal_threshold=0.5, mse_guardrail=0.02,
            expected_grid_intervals=2,
        )
        report = analyze_action(args)
        assert report["grid"]["pairs"] == 9
        assert report["guardrail"]["decision"]["status"] == (
            "HEADROOM_SUPPORTED"
        )
        assert report["decision"]["root_cause"] == (
            "ACTION_PRIOR_ESTIMATION_BOTTLENECK"
        )
        report_paths = []
        for action in ("PCS", "Ball"):
            action_report = dict(report)
            action_report["action"] = action
            path = os.path.join(directory, f"{action}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(action_report, handle)
            report_paths.append(path)
        summary = combine_reports(report_paths, expected_grid_intervals=2)
        assert summary["decision"]["status"] == (
            "HEADROOM_FOUND_REVISE_ACTION_PRIOR_ESTIMATION"
        )
        assert summary["decision"]["full_experiment_authorized"] is False
        flat_grid = os.path.join(directory, "flat_grid.csv")
        write_fixture(flat_grid, labels, base, base)
        args.guardrail_grid = flat_grid
        args.prior_only_grid = flat_grid
        flat_report = analyze_action(args)
        assert flat_report["decision"]["status"] == (
            "STOP_CURRENT_AUTOGATE_ACCURACY_ROUTE"
        )
        args.expected_grid_intervals = 10
        try:
            analyze_action(args)
        except ValueError as error:
            assert "does not match the locked 11x11" in str(error)
        else:
            raise AssertionError("coarse grid passed the locked protocol")
    print("clean gate oracle analyzer self-test: PASS")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.output:
        raise ValueError("--output is required")
    if (args.expected_grid_intervals is not None
            and args.expected_grid_intervals < 1):
        raise ValueError("--expected_grid_intervals must be positive")
    if args.combine_reports:
        if any((args.action, args.guardrail_grid, args.prior_only_grid)):
            raise ValueError("combine mode cannot include action grid inputs")
        report = combine_reports(
            args.combine_reports, args.expected_grid_intervals
        )
    else:
        if not all((args.action, args.guardrail_grid, args.prior_only_grid)):
            raise ValueError(
                "action mode requires --action, --guardrail_grid, and "
                "--prior_only_grid"
            )
        if args.bootstrap < 100:
            raise ValueError("--bootstrap must be at least 100")
        if not 0 <= args.mse_guardrail <= 1:
            raise ValueError("--mse_guardrail must be in [0, 1]")
        if args.signal_threshold < 0:
            raise ValueError("--signal_threshold must be non-negative")
        report = analyze_action(args)
    write_report(report, args.output)


if __name__ == "__main__":
    main()
