#!/usr/bin/env python
"""Fast CPU checks for grouped ListMLE ranking supervision."""

import inspect
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets import AQADataset
from models.pamfn import PAMFN
from pipeline import build_dataset


def _loss(prediction, target, tie_range=0.02, temperature=0.05):
    return PAMFN._grouped_listmle_loss(
        None,
        prediction,
        target,
        tie_range=tie_range,
        temperature=temperature,
    )


def _training_stub(weight):
    stub = SimpleNamespace(
        shared_rank_weight=0.0,
        shared_list_weight=weight,
        shared_list_margin=0.02,
        shared_list_temperature=0.05,
        ordinal_prototype_weight=0.0,
        mse=nn.MSELoss(),
    )
    stub._grouped_listmle_loss = (
        lambda prediction, target, tie_range, temperature:
        PAMFN._grouped_listmle_loss(
            stub,
            prediction,
            target,
            tie_range=tie_range,
            temperature=temperature,
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
        ROOT, "tools", "run_shared_listwise_screen.sh"
    )
    with open(runner_path, "r", encoding="utf-8") as handle:
        runner_source = handle.read()
    assert 'LIST_WEIGHT="0.0001"' in runner_source
    assert "LIST_WEIGHT:-" not in runner_source
    assert "ACTIONS=(TES PCS Ball Clubs Hoop Ribbon)" in runner_source

    target = torch.tensor([0.10, 0.40, 0.90])
    ordered = torch.tensor([0.20, 0.50, 0.80])
    reversed_prediction = torch.tensor(
        [0.80, 0.50, 0.20], requires_grad=True
    )
    ordered_loss = _loss(ordered, target)
    reversed_loss = _loss(reversed_prediction, target)
    assert ordered_loss < reversed_loss

    reversed_loss.backward()
    assert reversed_prediction.grad is not None
    assert reversed_prediction.grad[0] > 0
    assert reversed_prediction.grad[-1] < 0

    tied_target = torch.tensor([0.90, 0.89, 0.20])
    tied_a = torch.tensor([0.20, 0.80, 0.10])
    tied_b = torch.tensor([0.80, 0.20, 0.10])
    assert torch.allclose(
        _loss(tied_a, tied_target),
        _loss(tied_b, tied_target),
    )

    all_tied_prediction = torch.tensor(
        [0.20, 0.50, 0.80], requires_grad=True
    )
    all_tied_target = torch.tensor([0.50, 0.51, 0.52])
    all_tied_loss = _loss(
        all_tied_prediction,
        all_tied_target,
        tie_range=0.02,
    )
    assert all_tied_loss.item() == 0.0
    all_tied_loss.backward()
    assert torch.equal(
        all_tied_prediction.grad,
        torch.zeros_like(all_tied_prediction),
    )

    prediction = torch.tensor([0.25, 0.50, 0.75])
    label = torch.tensor([0.20, 0.50, 0.80])
    base_loss = PAMFN.call_training_loss(
        _training_stub(0.0), prediction, label
    )
    expected_mse = nn.functional.mse_loss(prediction, label)
    assert torch.equal(base_loss, expected_mse)
    listwise_loss = PAMFN.call_training_loss(
        _training_stub(0.0001), prediction, label
    )
    assert listwise_loss > base_loss

    print("official dataset and DataLoader protocol: passed")
    print("grouped ListMLE ordering and tie handling: passed")
    print("shared List-wise ranking verification: passed")


if __name__ == "__main__":
    main()
