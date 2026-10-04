import argparse
import csv
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline import (  # noqa: E402
    CUE_FEATURE_COLUMNS,
    GATE_GRID,
    gate_column_name,
)


VARIANTS = ("cue", "score", "cue_score")
GRID_COLUMNS = tuple(
    gate_column_name(motion, skeleton)
    for motion in GATE_GRID
    for skeleton in GATE_GRID
)
DEFAULT_COLUMN = gate_column_name(1.0, 0.5)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny gate from frozen gate-grid predictions"
        )
    )
    parser.add_argument("--train_csv")
    parser.add_argument("--calibration_csv")
    parser.add_argument("--output_dir")
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS,
        default=list(VARIANTS)
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2]
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--expected_train_samples", type=int, default=320)
    parser.add_argument(
        "--expected_calibration_samples", type=int, default=80
    )
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return args
    for name in ("train_csv", "calibration_csv", "output_dir"):
        if not getattr(args, name):
            parser.error(f"--{name} is required")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch_size must be positive")
    if args.hidden_dim < 1 or args.lr <= 0:
        parser.error("hidden_dim and lr must be positive")
    if args.weight_decay < 0 or args.bootstrap < 1:
        parser.error("weight_decay must be non-negative and bootstrap positive")
    if (
            args.expected_train_samples < 0
            or args.expected_calibration_samples < 0
    ):
        parser.error("expected sample counts must be non-negative")
    return args


def read_probe_table(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no samples")

    required = {
        "sample_id", "label",
        *CUE_FEATURE_COLUMNS,
        *GRID_COLUMNS
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{path} contains duplicate sample IDs")

    labels = np.asarray(
        [float(row["label"]) for row in rows], dtype=np.float32
    )
    cue_features = np.asarray([
        [float(row[column]) for column in CUE_FEATURE_COLUMNS]
        for row in rows
    ], dtype=np.float32)
    grid_predictions = np.asarray([
        [float(row[column]) for column in GRID_COLUMNS]
        for row in rows
    ], dtype=np.float32)
    values = np.concatenate((
        labels[:, None], cue_features, grid_predictions
    ), axis=1)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains NaN or infinite values")
    return {
        "sample_ids": sample_ids,
        "labels": labels,
        "cue_features": cue_features,
        "grid_predictions": grid_predictions
    }


def variant_features(table, variant):
    cue = table["cue_features"]
    base = table["grid_predictions"][
        :, GRID_COLUMNS.index(gate_column_name(0.0, 0.0))
    ][:, None]
    if variant == "cue":
        return cue
    if variant == "score":
        return base
    if variant == "cue_score":
        return np.concatenate((cue, base), axis=1)
    raise ValueError(f"unsupported gate variant: {variant}")


def variant_feature_columns(variant):
    if variant == "cue":
        return tuple(CUE_FEATURE_COLUMNS)
    if variant == "score":
        return (gate_column_name(0.0, 0.0),)
    if variant == "cue_score":
        return (*CUE_FEATURE_COLUMNS, gate_column_name(0.0, 0.0))
    raise ValueError(f"unsupported gate variant: {variant}")


class TinyGate(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
        output = self.layers[-1]
        nn.init.zeros_(output.weight)
        output.bias.data.copy_(torch.tensor([
            math.log(0.99 / 0.01), 0.0
        ]))

    def forward(self, features):
        return torch.sigmoid(self.layers(features))


def linear_gate_weights(gate):
    gate = gate.clamp(0.0, 1.0)
    position = 2.0 * gate
    lower = torch.floor(position).long().clamp(0, 1)
    upper = lower + 1
    fraction = position - lower.to(position.dtype)
    lower_weight = nn.functional.one_hot(
        lower, num_classes=3
    ).to(position.dtype)
    upper_weight = nn.functional.one_hot(
        upper, num_classes=3
    ).to(position.dtype)
    return (
        lower_weight * (1.0 - fraction[:, None])
        + upper_weight * fraction[:, None]
    )


def interpolate_grid(grid_predictions, gates):
    grid = grid_predictions.reshape(-1, 3, 3)
    motion_weights = linear_gate_weights(gates[:, 0])
    skeleton_weights = linear_gate_weights(gates[:, 1])
    return torch.einsum(
        "bi,bij,bj->b", motion_weights, grid, skeleton_weights
    )


def safe_spearman(predictions, labels):
    coefficient, _ = spearmanr(predictions, labels)
    return float(coefficient) if np.isfinite(coefficient) else 0.0


def prediction_metrics(predictions, labels):
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    mse = float(np.mean((predictions - labels) ** 2))
    rho = safe_spearman(predictions, labels)
    return {
        "rho": rho,
        "coef": rho * 100,
        "mse_raw": mse,
        "loss_x100": mse * 100
    }


def paired_bootstrap(
        learned, default, labels, bootstrap, seed):
    rng = np.random.RandomState(seed)
    delta_coef = []
    delta_mse = []
    for _ in range(bootstrap):
        indices = rng.randint(0, len(labels), len(labels))
        sampled_labels = labels[indices]
        learned_values = learned[indices]
        default_values = default[indices]
        delta_coef.append(100 * (
            safe_spearman(learned_values, sampled_labels)
            - safe_spearman(default_values, sampled_labels)
        ))
        delta_mse.append(
            np.mean((learned_values - sampled_labels) ** 2)
            - np.mean((default_values - sampled_labels) ** 2)
        )
    return {
        "delta_coef_95ci": [
            float(value) for value in np.percentile(
                delta_coef, [2.5, 97.5]
            )
        ],
        "delta_mse_raw_95ci": [
            float(value) for value in np.percentile(
                delta_mse, [2.5, 97.5]
            )
        ],
        "coef_improvement_probability": float(
            np.mean(np.asarray(delta_coef) > 0)
        )
    }


def standardize(train_features, calibration_features):
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        (train_features - mean) / std,
        (calibration_features - mean) / std,
        mean.reshape(-1),
        std.reshape(-1)
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_gate(
        train_table, calibration_table, variant, seed, args):
    set_seed(seed)
    train_features = variant_features(train_table, variant)
    calibration_features = variant_features(
        calibration_table, variant
    )
    (
        train_features, calibration_features,
        feature_mean, feature_std
    ) = standardize(train_features, calibration_features)

    train_x = torch.from_numpy(train_features).float()
    train_grid = torch.from_numpy(
        train_table["grid_predictions"]
    ).float()
    train_labels = torch.from_numpy(train_table["labels"]).float()
    calibration_x = torch.from_numpy(
        calibration_features
    ).float()
    calibration_grid = torch.from_numpy(
        calibration_table["grid_predictions"]
    ).float()

    gate = TinyGate(train_x.shape[1], args.hidden_dim)
    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=args.lr,
        weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    for _ in range(args.epochs):
        permutation = torch.randperm(
            len(train_x), generator=generator
        )
        gate.train()
        for start in range(0, len(train_x), args.batch_size):
            indices = permutation[start:start + args.batch_size]
            gates = gate(train_x[indices])
            predictions = interpolate_grid(
                train_grid[indices], gates
            )
            loss = nn.functional.mse_loss(
                predictions, train_labels[indices]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    gate.eval()
    with torch.no_grad():
        train_gates = gate(train_x)
        train_predictions = interpolate_grid(
            train_grid, train_gates
        )
        calibration_gates = gate(calibration_x)
        calibration_predictions = interpolate_grid(
            calibration_grid, calibration_gates
        )
    return {
        "model": gate,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "train_predictions": train_predictions.numpy(),
        "train_gates": train_gates.numpy(),
        "calibration_predictions": calibration_predictions.numpy(),
        "calibration_gates": calibration_gates.numpy()
    }


def gate_statistics(gates):
    names = ("motion", "skeleton")
    result = {}
    for index, name in enumerate(names):
        values = gates[:, index]
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "near_zero_ratio": float(np.mean(values < 0.05)),
            "near_one_ratio": float(np.mean(values > 0.95))
        }
    return result


def write_prediction_csv(
        path, table, default, learned, gates):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "sample_id", "label", "fixed_prediction",
            "gate_prediction", "g_motion", "g_skeleton"
        ))
        writer.writeheader()
        for index, sample_id in enumerate(table["sample_ids"]):
            writer.writerow({
                "sample_id": sample_id,
                "label": float(table["labels"][index]),
                "fixed_prediction": float(default[index]),
                "gate_prediction": float(learned[index]),
                "g_motion": float(gates[index, 0]),
                "g_skeleton": float(gates[index, 1])
            })


def summarize_variant(runs):
    delta_coef = np.asarray([
        run["calibration"]["delta_coef"] for run in runs
    ])
    relative_mse = np.asarray([
        run["calibration"]["relative_mse_change"] for run in runs
    ])
    noncollapsed = [
        max(
            run["calibration"]["gate_stats"]["motion"]["std"],
            run["calibration"]["gate_stats"]["skeleton"]["std"]
        ) >= 0.02
        for run in runs
    ]
    return {
        "mean_delta_coef": float(delta_coef.mean()),
        "median_delta_coef": float(np.median(delta_coef)),
        "min_delta_coef": float(delta_coef.min()),
        "mean_relative_mse_change": float(relative_mse.mean()),
        "max_relative_mse_change": float(relative_mse.max()),
        "all_seeds_nonnegative": bool(np.all(delta_coef >= 0)),
        "all_seeds_noncollapsed": bool(all(noncollapsed)),
        "preliminary_pass": bool(
            np.all(delta_coef >= 0)
            and delta_coef.mean() >= 0.5
            and relative_mse.max() <= 0.02
            and all(noncollapsed)
        )
    }


def run_self_test():
    grid = torch.arange(18, dtype=torch.float32).reshape(2, 9)
    gates = torch.tensor([
        [0.0, 0.0], [1.0, 1.0]
    ], requires_grad=True)
    predictions = interpolate_grid(grid, gates)
    assert torch.allclose(predictions, torch.tensor([0.0, 17.0]))
    predictions.sum().backward()
    assert gates.grad is not None
    assert torch.isfinite(gates.grad).all()
    assert torch.all(gates.grad.abs().sum(dim=1) > 0)

    midpoint = torch.tensor(
        [[1.0, 0.5]], requires_grad=True
    )
    midpoint_prediction = interpolate_grid(grid[:1], midpoint)
    assert torch.allclose(midpoint_prediction, grid[:1, 7])
    midpoint_prediction.backward()
    assert midpoint.grad is not None
    assert torch.all(midpoint.grad.abs() > 0)

    model = TinyGate(6, 8)
    output = model(torch.zeros(4, 6))
    assert output.shape == (4, 2)
    assert torch.all((output >= 0) & (output <= 1))

    train_values = np.arange(32, dtype=np.float32) / 31.0
    calibration_values = (
        np.arange(12, dtype=np.float32) + 0.5
    ) / 12.0

    def synthetic_table(values, prefix):
        cue = np.stack((
            values,
            values ** 2,
            np.sin(values),
            1.0 - values,
            np.cos(values),
            np.full_like(values, 0.25)
        ), axis=1).astype(np.float32)
        labels = (0.3 + 0.4 * values).astype(np.float32)
        base = labels + 0.05 * (values - 0.5)
        grid_values = []
        for motion in GATE_GRID:
            for skeleton in GATE_GRID:
                grid_values.append(
                    base
                    + 0.02 * motion * (0.5 - values)
                    + 0.015 * skeleton * (values - 0.5)
                )
        return {
            "sample_ids": [
                f"{prefix}_{index}" for index in range(len(values))
            ],
            "labels": labels,
            "cue_features": cue,
            "grid_predictions": np.stack(
                grid_values, axis=1
            ).astype(np.float32)
        }

    fitted = fit_gate(
        synthetic_table(train_values, "train"),
        synthetic_table(calibration_values, "cal"),
        "cue", 0,
        argparse.Namespace(
            hidden_dim=8, lr=1e-3, weight_decay=1e-4,
            epochs=2, batch_size=8
        )
    )
    assert fitted["calibration_predictions"].shape == (12,)
    assert fitted["calibration_gates"].shape == (12, 2)
    assert np.isfinite(fitted["calibration_predictions"]).all()
    assert np.isfinite(fitted["calibration_gates"]).all()
    print("frozen gate interpolation: passed")
    print("tiny gate gradients: passed")
    print("midpoint interpolation gradient: passed")
    print("offline gate training loop: passed")
    print("frozen gate probe self-test passed")


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    train_table = read_probe_table(args.train_csv)
    calibration_table = read_probe_table(args.calibration_csv)
    overlap = set(train_table["sample_ids"]) & set(
        calibration_table["sample_ids"]
    )
    if overlap:
        raise ValueError(
            "train and calibration probe tables overlap"
        )
    expected_counts = (
        ("train", len(train_table["labels"]), args.expected_train_samples),
        (
            "calibration", len(calibration_table["labels"]),
            args.expected_calibration_samples
        )
    )
    for split_name, observed, expected in expected_counts:
        if expected and observed != expected:
            raise ValueError(
                f"expected {expected} {split_name} samples, got {observed}"
            )
    cue_standard_deviation = train_table["cue_features"].std(axis=0)
    constant_features = [
        CUE_FEATURE_COLUMNS[index]
        for index, value in enumerate(cue_standard_deviation)
        if value < 1e-8
    ]
    if constant_features:
        raise ValueError(
            "training cue features are constant: "
            f"{constant_features}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    report = {
        "protocol": (
            "Frozen scorer 3x3 grid with piecewise bilinear "
            "interpolation; only the tiny gate is trained. "
            "Calibration is evaluated once after the final epoch."
        ),
        "warning": (
            "This is a feasibility probe using in-sample scorer "
            "predictions on the 320 training groups. A positive result "
            "must be confirmed with group OOF scorers."
        ),
        "train_samples": len(train_table["labels"]),
        "calibration_samples": len(calibration_table["labels"]),
        "grid_columns": list(GRID_COLUMNS),
        "cue_feature_columns": list(CUE_FEATURE_COLUMNS),
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seeds": args.seeds,
            "checkpoint_rule": "final epoch; no calibration selection"
        },
        "train_cue_feature_stats": {
            column: {
                "mean": float(train_table["cue_features"][:, index].mean()),
                "std": float(train_table["cue_features"][:, index].std()),
                "min": float(train_table["cue_features"][:, index].min()),
                "max": float(train_table["cue_features"][:, index].max())
            }
            for index, column in enumerate(CUE_FEATURE_COLUMNS)
        },
        "runs": {},
        "summary": {}
    }
    default_index = GRID_COLUMNS.index(DEFAULT_COLUMN)
    train_default = train_table["grid_predictions"][:, default_index]
    calibration_default = calibration_table[
        "grid_predictions"
    ][:, default_index]

    for variant in args.variants:
        variant_runs = []
        for seed in args.seeds:
            fitted = fit_gate(
                train_table, calibration_table,
                variant, seed, args
            )
            learned = fitted["calibration_predictions"]
            learned_metrics = prediction_metrics(
                learned, calibration_table["labels"]
            )
            default_metrics = prediction_metrics(
                calibration_default,
                calibration_table["labels"]
            )
            relative_mse = (
                (learned_metrics["mse_raw"] - default_metrics["mse_raw"])
                / default_metrics["mse_raw"]
                if default_metrics["mse_raw"] > 0 else 0.0
            )
            run = {
                "seed": seed,
                "parameters": sum(
                    parameter.numel()
                    for parameter in fitted["model"].parameters()
                ),
                "feature_columns": list(
                    variant_feature_columns(variant)
                ),
                "train": {
                    "default": prediction_metrics(
                        train_default, train_table["labels"]
                    ),
                    "learned": prediction_metrics(
                        fitted["train_predictions"],
                        train_table["labels"]
                    ),
                    "gate_stats": gate_statistics(
                        fitted["train_gates"]
                    )
                },
                "calibration": {
                    "default": default_metrics,
                    "learned": learned_metrics,
                    "delta_coef": (
                        learned_metrics["coef"]
                        - default_metrics["coef"]
                    ),
                    "relative_mse_change": float(relative_mse),
                    "gate_stats": gate_statistics(
                        fitted["calibration_gates"]
                    ),
                    "paired_bootstrap": paired_bootstrap(
                        learned, calibration_default,
                        calibration_table["labels"],
                        args.bootstrap, seed + 1000
                    )
                }
            }
            variant_runs.append(run)

            prefix = f"{variant}_seed{seed}"
            write_prediction_csv(
                os.path.join(args.output_dir, f"{prefix}.csv"),
                calibration_table, calibration_default,
                learned, fitted["calibration_gates"]
            )
            torch.save({
                "state_dict": fitted["model"].state_dict(),
                "variant": variant,
                "seed": seed,
                "feature_columns": variant_feature_columns(variant),
                "feature_mean": fitted["feature_mean"],
                "feature_std": fitted["feature_std"],
                "hidden_dim": args.hidden_dim,
                "grid_columns": GRID_COLUMNS
            }, os.path.join(args.output_dir, f"{prefix}.pth"))
        report["runs"][variant] = variant_runs
        report["summary"][variant] = summarize_variant(
            variant_runs
        )

    if "cue" in report["summary"] and "score" in report["summary"]:
        cue_delta = report["summary"]["cue"]["mean_delta_coef"]
        score_delta = report["summary"]["score"]["mean_delta_coef"]
        cue_score_summary = report["summary"].get("cue_score")
        cue_score_delta = (
            cue_score_summary["mean_delta_coef"]
            if cue_score_summary is not None else None
        )
        cue_score_advantage = (
            cue_score_delta - score_delta
            if cue_score_delta is not None else None
        )
        cue_only_signal = bool(
            report["summary"]["cue"]["preliminary_pass"]
            and cue_delta > score_delta
        )
        cue_adds_to_score = bool(
            cue_score_summary is not None
            and cue_score_summary["preliminary_pass"]
            and cue_score_advantage >= 0.5
        )
        report["cue_vs_score_control"] = {
            "mean_delta_coef_difference": cue_delta - score_delta,
            "cue_exceeds_score": bool(cue_delta > score_delta),
            "cue_score_minus_score_mean_delta_coef": cue_score_advantage,
            "cue_only_reliability_signal": cue_only_signal,
            "cue_adds_to_score": cue_adds_to_score,
            "candidate_for_oof": bool(
                cue_only_signal or cue_adds_to_score
            )
        }

    output_path = os.path.join(
        args.output_dir, "frozen_gate_report.json"
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    for variant, summary in report["summary"].items():
        print(
            f"{variant}: mean_delta_coef="
            f"{summary['mean_delta_coef']:+.3f} "
            f"max_relative_mse="
            f"{summary['max_relative_mse_change']:+.2%} "
            f"pass={summary['preliminary_pass']}"
        )
    if "cue_vs_score_control" in report:
        print(
            "cue_exceeds_score="
            f"{report['cue_vs_score_control']['cue_exceeds_score']}"
        )
    print(f"report={output_path}")


if __name__ == "__main__":
    main()
