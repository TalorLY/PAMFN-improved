#!/usr/bin/env python
"""Fast structural checks for configurable ordinal grade counts."""

import inspect
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets import get_train_quality_thresholds, read_labels
from models.pamfn import PAMFN
from analyze_ordinal_prototypes import analyze_checkpoint
from summarize_ordinal_sensitivity import (
    _feature_means,
    _metric_means,
    _scope_actions,
)


def _stub(num_grades):
    dim = max(5, num_grades)
    positions = torch.linspace(-1.0, 1.0, num_grades)
    prototypes = torch.zeros(num_grades, dim)
    prototypes[:, 0] = positions
    prototypes[:, 1] = torch.sqrt(
        torch.clamp(1.0 - positions.square(), min=0.0)
    )
    model = SimpleNamespace(
        ordinal_thresholds=torch.linspace(
            0.2, 0.8, num_grades - 1
        ),
        quality_prototypes=torch.nn.Parameter(prototypes),
        quality_axis=torch.nn.Parameter(
            torch.tensor([1.0] + [0.0] * (dim - 1))
        ),
        ordinal_prototype_temperature=0.1,
        ordinal_prototype_order_margin=0.1,
        ordinal_prototype_order_scale=0.1,
    )
    model._ordinal_soft_targets = (
        PAMFN._ordinal_soft_targets.__get__(model)
    )
    model._ordinal_prototype_loss_components = (
        PAMFN._ordinal_prototype_loss_components.__get__(model)
    )
    return model, dim


def _verify_num_grades(num_grades):
    model, dim = _stub(num_grades)
    embedding = torch.randn(7, dim, requires_grad=True)
    target = torch.linspace(0.0, 1.0, 7)
    soft_targets = model._ordinal_soft_targets(
        target, embedding.dtype
    )
    assert soft_targets.shape == (7, num_grades)
    assert torch.allclose(
        soft_targets.sum(dim=1), torch.ones(7)
    )
    ce_loss, order_loss = (
        model._ordinal_prototype_loss_components(embedding, target)
    )
    loss = ce_loss + 0.1 * order_loss
    assert torch.isfinite(loss)
    loss.backward()
    assert embedding.grad is not None
    assert model.quality_prototypes.grad is not None

    thresholds = np.asarray(
        get_train_quality_thresholds("TES", num_grades)
    )
    assert thresholds.shape == (num_grades - 1,)
    labels = np.asarray([
        label for _, label in read_labels("TES", is_train=True)
    ])
    counts = np.bincount(
        np.searchsorted(thresholds, labels, side="left"),
        minlength=num_grades,
    )
    assert counts.shape == (num_grades,)
    assert counts.min() > 0

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = os.path.join(
            directory, "TES_prototype_seed0.pth"
        )
        torch.save(
            {
                "quality_prototypes":
                    model.quality_prototypes.detach(),
                "quality_axis": model.quality_axis.detach(),
                "ordinal_thresholds":
                    torch.tensor(thresholds),
            },
            checkpoint,
        )
        report = analyze_checkpoint(checkpoint, "TES")
        assert report["num_grades"] == num_grades
        assert sum(
            report[f"count_grade_{grade}"]
            for grade in range(num_grades)
        ) == labels.size


def main():
    for num_grades in (3, 4, 5):
        _verify_num_grades(num_grades)

    signature = inspect.signature(get_train_quality_thresholds)
    assert signature.parameters["num_grades"].default == 4
    actions = ("TES", "PCS", "Ball", "Clubs", "Hoop", "Ribbon")
    assert _scope_actions(actions)[-1] == ("Macro", list(actions))
    metric_rows = {}
    for index, action in enumerate(actions):
        metric_rows[(action, "base")] = {
            "best_coef": "80",
            "best_avg_coef": "79",
            "min_loss": "0.02",
            "min_avg_loss": "0.03",
        }
        metric_rows[(action, "prototype")] = {
            "best_coef": str(81 + index),
            "best_avg_coef": str(80 + index),
            "min_loss": "0.019",
            "min_avg_loss": "0.029",
        }
    macro = _metric_means(metric_rows, actions)
    assert macro["delta_best_coef"] == 3.5
    assert macro["wins_best"] == 6
    feature = _feature_means(
        {
            action: {
                "expected_grade_srcc": "0.8",
                "grade_adjacent_accuracy": "0.9",
                "grade_mae": "0.1",
                "axis_label_srcc": "0.7",
                "min_prototype_gap": "0.2",
                "max_prototype_cosine": "0.5",
                "weighted_auxiliary_test": "0.0001",
                "monotonic_bin_means": "True",
                "predicted_count_bin0": "1",
                "predicted_count_bin1": "1",
                "predicted_count_bin2": "1",
            }
            for action in actions
        },
        actions,
    )
    assert feature["monotonic_actions"] == 6
    assert feature["collapsed_actions"] == 0

    runner = os.path.join(
        ROOT, "tools", "run_ordinal_prototype_sensitivity.sh"
    )
    with open(runner, "r", encoding="utf-8") as handle:
        source = handle.read()
    for token in (
            "k3 3 0.0001",
            "k5 5 0.0001",
            "w5e5 4 0.00005",
            "w2e4 4 0.0002",
            "--ordinal_num_grades",
            "k4_reference",
            'BASE_ROOT_PRIMARY="${BASE_ROOT_PRIMARY:-results/shared_rank_official_protocol_v2}"',
            '--base-roots "${BASE_ROOTS[@]}"',
            "summarize_ordinal_sensitivity.py"):
        assert token in source, token
    assert "w1e4 4 0.0001" not in source
    print("ordinal sensitivity verification passed")


if __name__ == "__main__":
    main()
