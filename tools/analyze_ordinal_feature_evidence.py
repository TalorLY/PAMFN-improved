#!/usr/bin/env python
"""Evaluate whether held-out samples use the learned quality ordering."""

import argparse
import csv
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets import AQADataset
from main import build_configs, load_config
from pipeline import build_model, setup_seed


MODES = ("grade", "unordered", "prototype")


def _load_state_dict(path):
    try:
        value = torch.load(
            path, map_location="cpu", weights_only=False
        )
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict) and "state_dict" in value:
        return value["state_dict"]
    return value


def _safe_spearman(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if (
            left.size < 2
            or np.std(left) == 0
            or np.std(right) == 0):
        return float("nan")
    return float(spearmanr(left, right)[0])


def _mode_roots(values):
    roots = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"mode root must use MODE=ROOT format: {value}"
            )
        mode, root = value.split("=", 1)
        if mode not in MODES:
            raise ValueError(f"unsupported mode: {mode}")
        if mode in roots:
            raise ValueError(f"duplicate mode: {mode}")
        roots[mode] = root
    return roots


def _experiment_args(
        action, feat_dir, mode, num_grades=4,
        prototype_weight=0.0001, order_scale=0.1):
    return SimpleNamespace(
        multi_modality=True,
        action=action,
        epochs=None,
        feat_dir=feat_dir,
        shared_rank_weight=0.0,
        shared_rank_margin=0.02,
        shared_rank_temperature=0.05,
        shared_list_weight=0.0,
        shared_list_margin=0.02,
        shared_list_temperature=0.05,
        ordinal_grade_weight=0.0001 if mode == "grade" else 0.0,
        ordinal_prototype_weight=(
            prototype_weight
            if mode in ("unordered", "prototype") else 0.0
        ),
        ordinal_num_grades=num_grades,
        ordinal_prototype_dim=128,
        ordinal_prototype_temperature=0.1,
        ordinal_prototype_order_margin=0.1,
        ordinal_prototype_order_scale=(
            order_scale if mode == "prototype" else 0.0
        ),
    )


def _build_test_loader(dataset_config):
    dataset = AQADataset(
        dataset_name=dataset_config["dataset_name"],
        feat_dir=dataset_config["feat_dir"],
        rgb_feat=dataset_config["rgb_feat"],
        flow_feat=dataset_config["flow_feat"],
        audio_feat=dataset_config["audio_feat"],
        squeeze_rgb_feat="mean",
        is_train=False,
        segments=-1,
        sequence_sampling="legacy",
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=dataset_config["num_workers"],
    )
    return dataset, loader


def _model_input(batch, device):
    video, audio, flow, label, _ = batch
    return {
        "V": video.to(device),
        "F": flow.to(device),
        "A": audio.to(device),
    }, label.float().to(device)


def _mean_or_nan(values):
    return (
        float(np.mean(values))
        if len(values) > 0 else float("nan")
    )


def analyze_mode(
        action, mode, root, seed, feat_dir, device,
        num_grades=4, prototype_weight=0.0001,
        order_scale=0.1):
    args = _experiment_args(
        action,
        feat_dir,
        mode,
        num_grades=num_grades,
        prototype_weight=prototype_weight,
        order_scale=order_scale,
    )
    config = load_config(action, 2)
    (
        _,
        _,
        dataset_config,
        model_config,
        _,
        _,
    ) = build_configs(args, config)
    checkpoint = os.path.join(
        root, action, f"{action}_{mode}_seed{seed}.pth"
    )
    model = build_model(**model_config)
    model.load_state_dict(_load_state_dict(checkpoint))
    model.to(device)
    model.eval()
    dataset, loader = _build_test_loader(dataset_config)

    labels = []
    predictions = []
    expected_grades = []
    predicted_grades = []
    true_grades = []
    confidences = []
    logit_margins = []
    soft_ce_losses = []
    axis_values = []
    sample_rows = []

    thresholds = model.ordinal_thresholds.detach().cpu().numpy()
    num_grades = thresholds.size + 1
    prototypes = None
    axis = None
    if mode in ("unordered", "prototype"):
        prototypes = F.normalize(
            model.quality_prototypes, dim=1
        ).detach()
        axis = F.normalize(
            model.quality_axis, dim=0
        ).detach()

    with torch.no_grad():
        for index, batch in enumerate(loader):
            model_input, target = _model_input(batch, device)
            prediction, info = model(model_input)
            embedding = info["quality_embedding"]
            if mode == "grade":
                logits = info["quality_logits"]
            else:
                logits = (
                    embedding @ prototypes.transpose(0, 1)
                    / model.ordinal_prototype_temperature
                )
            probabilities = F.softmax(logits, dim=1)
            class_ids = torch.arange(
                num_grades,
                device=device,
                dtype=probabilities.dtype,
            )
            expected = (probabilities * class_ids).sum(dim=1)
            predicted_grade = probabilities.argmax(dim=1)
            true_grade = torch.bucketize(
                target.reshape(-1), model.ordinal_thresholds
            )
            class_ids_int = torch.arange(
                num_grades, device=device
            ).unsqueeze(0)
            distance = (
                class_ids_int - true_grade[:, None]
            ).abs()
            soft_target = torch.where(
                distance == 0,
                torch.ones_like(
                    distance, dtype=logits.dtype
                ),
                torch.where(
                    distance == 1,
                    torch.full_like(
                        distance, 0.1, dtype=logits.dtype
                    ),
                    torch.zeros_like(
                        distance, dtype=logits.dtype
                    ),
                ),
            )
            soft_target = soft_target / soft_target.sum(
                dim=1, keepdim=True
            )
            soft_ce = -(
                soft_target * F.log_softmax(logits, dim=1)
            ).sum(dim=1)
            true_logit = logits.gather(
                1, true_grade[:, None]
            ).squeeze(1)
            masked = logits.clone()
            masked.scatter_(1, true_grade[:, None], float("-inf"))
            margin = true_logit - masked.max(dim=1).values
            axis_value = (
                embedding @ axis
                if mode == "prototype" else None
            )

            label_value = float(target.reshape(-1)[0])
            prediction_value = float(prediction.reshape(-1)[0])
            expected_value = float(expected[0])
            predicted_value = int(predicted_grade[0])
            true_value = int(true_grade[0])
            confidence = float(probabilities.max(dim=1).values[0])
            margin_value = float(margin[0])
            axis_value_float = (
                float(axis_value[0])
                if axis_value is not None else float("nan")
            )
            sample_id = dataset.labels[index][0]

            labels.append(label_value)
            predictions.append(prediction_value)
            expected_grades.append(expected_value)
            predicted_grades.append(predicted_value)
            true_grades.append(true_value)
            confidences.append(confidence)
            logit_margins.append(margin_value)
            soft_ce_losses.append(float(soft_ce[0]))
            if axis_value is not None:
                axis_values.append(axis_value_float)
            sample_rows.append({
                "action": action,
                "mode": mode,
                "seed": seed,
                "sample_id": sample_id,
                "label": label_value,
                "prediction": prediction_value,
                "true_grade": true_value,
                "predicted_grade": predicted_value,
                "expected_grade": expected_value,
                "confidence": confidence,
                "true_logit_margin": margin_value,
                "axis_projection": (
                    axis_value_float
                    if axis_value is not None else ""
                ),
            })

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    expected_grades = np.asarray(expected_grades)
    predicted_grades = np.asarray(predicted_grades)
    true_grades = np.asarray(true_grades)
    correct_distance = np.abs(predicted_grades - true_grades)
    bin_means = [
        _mean_or_nan(expected_grades[true_grades == grade])
        for grade in range(num_grades)
    ]
    finite_bin_means = [
        value for value in bin_means if np.isfinite(value)
    ]
    monotonic = (
        len(finite_bin_means) == num_grades
        and all(
            left < right
            for left, right in zip(
                finite_bin_means[:-1],
                finite_bin_means[1:],
            )
        )
    )
    order_loss = ""
    min_prototype_gap = ""
    max_prototype_cosine = ""
    if mode in ("unordered", "prototype"):
        positions = prototypes @ axis
        gaps = positions[1:] - positions[:-1]
        order_loss = float(F.relu(0.1 - gaps).mean())
        min_prototype_gap = float(gaps.min())
        similarity = prototypes @ prototypes.transpose(0, 1)
        max_prototype_cosine = float(similarity[
            ~torch.eye(
                num_grades, dtype=torch.bool, device=device
            )
        ].max())
    mean_soft_ce = float(np.mean(soft_ce_losses))
    auxiliary_test = mean_soft_ce
    if mode == "prototype":
        auxiliary_test += 0.1 * order_loss
    summary = {
        "action": action,
        "mode": mode,
        "seed": seed,
        "checkpoint": checkpoint,
        "sample_count": len(labels),
        "score_srcc": _safe_spearman(predictions, labels),
        "score_mse": float(np.mean((predictions - labels) ** 2)),
        "prediction_std": float(np.std(predictions)),
        "expected_grade_srcc": _safe_spearman(
            expected_grades, labels
        ),
        "grade_exact_accuracy": float(
            np.mean(correct_distance == 0)
        ),
        "grade_adjacent_accuracy": float(
            np.mean(correct_distance <= 1)
        ),
        "grade_mae": float(np.mean(correct_distance)),
        "mean_confidence": float(np.mean(confidences)),
        "mean_true_logit_margin": float(np.mean(logit_margins)),
        "test_grade_soft_ce": mean_soft_ce,
        "prototype_order_loss": order_loss,
        "weighted_auxiliary_test":
            prototype_weight * auxiliary_test,
        "min_prototype_gap": min_prototype_gap,
        "max_prototype_cosine": max_prototype_cosine,
        "axis_label_srcc": (
            _safe_spearman(axis_values, labels)
            if axis_values else ""
        ),
        "monotonic_bin_means": monotonic,
    }
    for grade, value in enumerate(bin_means):
        summary[f"expected_grade_mean_bin{grade}"] = value
        in_grade = true_grades == grade
        summary[f"score_mse_bin{grade}"] = (
            float(np.mean(
                (predictions[in_grade] - labels[in_grade]) ** 2
            ))
            if in_grade.any() else float("nan")
        )
        summary[f"predicted_count_bin{grade}"] = int(
            np.sum(predicted_grades == grade)
        )
        summary[f"true_count_bin{grade}"] = int(
            np.sum(true_grades == grade)
        )
    return summary, sample_rows


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode-roots", nargs="+", required=True,
        help="entries in MODE=ROOT format",
    )
    parser.add_argument("--actions", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feat-dir", default="./data/features")
    parser.add_argument("--gpu", default="5")
    parser.add_argument(
        "--ordinal-num-grades", type=int, default=4
    )
    parser.add_argument(
        "--ordinal-prototype-weight", type=float, default=0.0001
    )
    parser.add_argument(
        "--ordinal-prototype-order-scale", type=float, default=0.1
    )
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-samples", required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)
    roots = _mode_roots(args.mode_roots)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    summaries = []
    samples = []
    for mode, root in roots.items():
        for action in args.actions:
            summary, rows = analyze_mode(
                action,
                mode,
                root,
                args.seed,
                args.feat_dir,
                device,
                num_grades=args.ordinal_num_grades,
                prototype_weight=args.ordinal_prototype_weight,
                order_scale=args.ordinal_prototype_order_scale,
            )
            summaries.append(summary)
            samples.extend(rows)
            print(
                f"evidence action={action} mode={mode} "
                f"grade_srcc={summary['expected_grade_srcc']:.4f}"
            )
    _write_csv(args.output_summary, summaries)
    _write_csv(args.output_samples, samples)
    print(f"summary: {args.output_summary}")
    print(f"samples: {args.output_samples}")


if __name__ == "__main__":
    main()
