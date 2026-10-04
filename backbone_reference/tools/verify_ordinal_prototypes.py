#!/usr/bin/env python
"""Fast CPU checks for training-only ordered quality prototypes."""

import inspect
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets import (
    AQADataset,
    get_train_quality_thresholds,
    read_labels,
)
from models.pamfn import PAMFN
from pipeline import build_dataset
from analyze_ordinal_prototypes import analyze_checkpoint


def _prototype_stub(weight, prototypes=None, axis=None):
    prototypes = (
        torch.eye(4)
        if prototypes is None else prototypes
    )
    axis = (
        torch.ones(4)
        if axis is None else axis
    )
    stub = SimpleNamespace(
        shared_rank_weight=0.0,
        shared_list_weight=0.0,
        ordinal_grade_weight=0.0,
        ordinal_prototype_weight=weight,
        ordinal_prototype_temperature=0.1,
        ordinal_prototype_order_margin=0.1,
        ordinal_prototype_order_scale=0.1,
        ordinal_thresholds=torch.tensor([0.25, 0.50, 0.75]),
        quality_prototypes=nn.Parameter(prototypes.clone()),
        quality_axis=nn.Parameter(axis.clone()),
        mse=nn.MSELoss(),
    )
    stub._ordinal_prototype_loss = (
        lambda embedding, target:
        PAMFN._ordinal_prototype_loss(stub, embedding, target)
    )
    stub._ordinal_prototype_loss_components = (
        lambda embedding, target:
        PAMFN._ordinal_prototype_loss_components(
            stub, embedding, target
        )
    )
    stub._ordinal_soft_targets = (
        lambda target, dtype:
        PAMFN._ordinal_soft_targets(stub, target, dtype)
    )
    stub.call_training_loss_components = (
        lambda prediction, label, **kwargs:
        PAMFN.call_training_loss_components(
            stub, prediction, label, **kwargs
        )
    )
    return stub


def main():
    dataset_source = inspect.getsource(AQADataset)
    loader_source = inspect.getsource(build_dataset)
    assert "normalized_full" not in dataset_source
    assert "criterion_targets" not in dataset_source
    assert "skeleton" not in dataset_source.lower()
    assert "np.random.randint" in dataset_source
    assert "generator=" not in loader_source
    assert "worker_init_fn=" not in loader_source
    assert "shuffle=True" in loader_source

    runner_path = os.path.join(
        ROOT, "tools", "run_ordinal_prototype_screen.sh"
    )
    with open(runner_path, "r", encoding="utf-8") as handle:
        runner_source = handle.read()
    assert 'PROTOTYPE_WEIGHT="0.0001"' in runner_source
    assert "PROTOTYPE_WEIGHT:-" not in runner_source
    assert "ACTIONS=(TES PCS Ball Clubs Hoop Ribbon)" in runner_source

    multiseed_path = os.path.join(
        ROOT, "tools", "run_ordinal_prototype_multiseed.sh"
    )
    with open(multiseed_path, "r", encoding="utf-8") as handle:
        multiseed_source = handle.read()
    assert 'PROTOTYPE_WEIGHT="0.0001"' in multiseed_source
    assert "PROTOTYPE_WEIGHT:-" not in multiseed_source
    assert "SEEDS=(1 2)" in multiseed_source
    assert "ACTIONS=(TES PCS Ball Clubs Hoop Ribbon)" in (
        multiseed_source
    )
    assert 'run_model "$seed" "$action" base' in multiseed_source
    assert 'run_model "$seed" "$action" prototype' in (
        multiseed_source
    )

    for action in (
            "TES", "PCS", "Ball", "Clubs", "Hoop", "Ribbon"):
        labels = np.asarray(
            [
                label
                for _, label in read_labels(action, is_train=True)
            ],
            dtype=np.float64,
        )
        expected = np.quantile(labels, (0.25, 0.50, 0.75))
        actual = np.asarray(get_train_quality_thresholds(action))
        assert np.allclose(actual, expected)
        bins = np.searchsorted(actual, labels, side="left")
        counts = np.bincount(bins, minlength=4)
        assert counts.min() >= int(labels.size * 0.20)

    aligned_embedding = torch.eye(4)
    reversed_embedding = torch.flip(aligned_embedding, dims=(0,))
    target = torch.tensor([0.10, 0.40, 0.60, 0.90])
    aligned_stub = _prototype_stub(0.0001)
    aligned_loss = PAMFN._ordinal_prototype_loss(
        aligned_stub, aligned_embedding, target
    )
    reversed_loss = PAMFN._ordinal_prototype_loss(
        aligned_stub, reversed_embedding, target
    )
    assert aligned_loss < reversed_loss

    train_embedding = torch.eye(4, requires_grad=True)
    train_stub = _prototype_stub(0.0001)
    prototype_loss = PAMFN._ordinal_prototype_loss(
        train_stub, train_embedding, target
    )
    assert torch.isfinite(prototype_loss)
    prototype_loss.backward()
    assert train_embedding.grad is not None
    assert train_stub.quality_prototypes.grad is not None
    assert train_stub.quality_axis.grad is not None

    prediction = torch.tensor([0.20, 0.40, 0.60, 0.80])
    label = torch.tensor([0.10, 0.40, 0.60, 0.90])
    base_loss = PAMFN.call_training_loss(
        _prototype_stub(0.0),
        prediction,
        label,
    )
    expected_mse = nn.functional.mse_loss(prediction, label)
    assert torch.equal(base_loss, expected_mse)
    ranked_loss = PAMFN.call_training_loss(
        _prototype_stub(0.0001),
        prediction,
        label,
        quality_embedding=aligned_embedding,
    )
    assert ranked_loss > base_loss

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = os.path.join(directory, "TES_prototype_seed0.pth")
        torch.save(
            {
                "quality_prototypes": torch.eye(4),
                "quality_axis": torch.tensor(
                    [-1.5, -0.5, 0.5, 1.5]
                ),
                "ordinal_thresholds": torch.tensor(
                    get_train_quality_thresholds("TES")
                ),
            },
            checkpoint,
        )
        report = analyze_checkpoint(checkpoint, "TES")
        assert sum(
            report[name]
            for name in (
                "count_poor",
                "count_fair",
                "count_good",
                "count_excellent",
            )
        ) == len(read_labels("TES", is_train=True))
        assert report["order_violations"] == 0
        assert not report["collapsed"]

    print("official dataset and DataLoader protocol: passed")
    print("train-only quality quartiles and occupancy: passed")
    print("ordered prototype loss and gradients: passed")
    print("prototype checkpoint diagnostics: passed")
    print("ordinal prototype verification: passed")


if __name__ == "__main__":
    main()
