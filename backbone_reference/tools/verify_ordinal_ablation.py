#!/usr/bin/env python
"""Fast checks for the decisive ordinal-prototype ablations."""

import inspect
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.pamfn import PAMFN
from pipeline import _format_loss_components


def _stub(grade_weight=0.0, prototype_weight=0.0, order_scale=0.1):
    stub = SimpleNamespace(
        shared_rank_weight=0.0,
        shared_list_weight=0.0,
        ordinal_grade_weight=grade_weight,
        ordinal_prototype_weight=prototype_weight,
        ordinal_prototype_temperature=0.1,
        ordinal_prototype_order_margin=0.1,
        ordinal_prototype_order_scale=order_scale,
        ordinal_thresholds=torch.tensor([0.25, 0.50, 0.75]),
        quality_prototypes=nn.Parameter(torch.eye(4)),
        quality_axis=nn.Parameter(
            torch.tensor([-1.5, -0.5, 0.5, 1.5])
        ),
        mse=nn.MSELoss(),
    )
    stub._ordinal_soft_targets = (
        lambda target, dtype:
        PAMFN._ordinal_soft_targets(stub, target, dtype)
    )
    stub._ordinal_grade_loss = (
        lambda logits, target:
        PAMFN._ordinal_grade_loss(stub, logits, target)
    )
    stub._ordinal_prototype_loss_components = (
        lambda embedding, target:
        PAMFN._ordinal_prototype_loss_components(
            stub, embedding, target
        )
    )
    return stub


def main():
    prediction = torch.tensor([0.20, 0.40, 0.60, 0.80])
    target = torch.tensor([0.10, 0.40, 0.60, 0.90])
    embedding = torch.eye(4, requires_grad=True)
    logits = torch.eye(4, requires_grad=True)

    grade_stub = _stub(grade_weight=0.0001)
    grade_components = PAMFN.call_training_loss_components(
        grade_stub,
        prediction,
        target,
        quality_embedding=embedding,
        quality_logits=logits,
    )
    assert set(grade_components) == {
        "score_mse",
        "grade_classification",
        "weighted_auxiliary",
        "objective",
    }
    assert grade_components["objective"] > grade_components["score_mse"]
    assert "GradeCE" in _format_loss_components(grade_components)
    grade_components["objective"].backward()
    assert logits.grad is not None

    ordered_stub = _stub(
        prototype_weight=0.0001, order_scale=0.1
    )
    ordered_components = PAMFN.call_training_loss_components(
        ordered_stub,
        prediction,
        target,
        quality_embedding=embedding,
    )
    assert set(ordered_components) == {
        "score_mse",
        "prototype_classification",
        "prototype_order",
        "weighted_auxiliary",
        "objective",
    }
    assert "ProtoCE" in _format_loss_components(ordered_components)
    assert "Order" in _format_loss_components(ordered_components)

    unordered_stub = _stub(
        prototype_weight=0.0001, order_scale=0.0
    )
    unordered_components = PAMFN.call_training_loss_components(
        unordered_stub,
        prediction,
        target,
        quality_embedding=embedding,
    )
    expected = (
        unordered_components["prototype_classification"] * 0.0001
    )
    assert torch.allclose(
        unordered_components["weighted_auxiliary"], expected
    )

    model_source = inspect.getsource(PAMFN)
    assert "ordinal_grade_weight" in model_source
    assert "_ordinal_prototype_loss_components" in model_source
    assert "quality_classifier" in model_source

    runner_path = os.path.join(
        ROOT, "tools", "run_ordinal_ablation_evidence.sh"
    )
    with open(runner_path, "r", encoding="utf-8") as handle:
        runner = handle.read()
    assert 'ACTIONS=(TES PCS Ball Clubs Hoop Ribbon)' in runner
    assert 'AUX_WEIGHT="0.0001"' in runner
    assert 'run_mode "$action" grade' in runner
    assert 'run_mode "$action" unordered' in runner
    assert "--ordinal_prototype_order_scale 0" in runner

    print("grade-head loss and gradients: passed")
    print("separate prototype/order losses: passed")
    print("unordered prototype ablation: passed")
    print("six-action ablation runner: passed")


if __name__ == "__main__":
    main()
