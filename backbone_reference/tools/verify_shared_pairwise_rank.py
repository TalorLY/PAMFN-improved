#!/usr/bin/env python
"""Fast CPU checks for the cue-free shared pairwise ranking objective."""

import os
import inspect
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.pamfn import PAMFN, _torch_load
from datasets import AQADataset
from pipeline import build_dataset, setup_seed


def _loss(prediction, target, margin=0.02, temperature=0.05):
    return PAMFN._pairwise_rank_loss(
        None,
        prediction,
        target,
        min_label_gap=margin,
        temperature=temperature,
    )


def _training_stub(weight):
    stub = SimpleNamespace(
        shared_rank_weight=weight,
        shared_rank_margin=0.02,
        shared_rank_temperature=0.05,
        shared_list_weight=0.0,
        ordinal_prototype_weight=0.0,
        mse=nn.MSELoss(),
    )
    stub._pairwise_rank_loss = (
        lambda prediction, target, min_label_gap, temperature:
        PAMFN._pairwise_rank_loss(
            stub,
            prediction,
            target,
            min_label_gap=min_label_gap,
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
        ROOT, "tools", "run_shared_rank_original_protocol.sh"
    )
    with open(runner_path, "r", encoding="utf-8") as handle:
        runner_source = handle.read()
    assert 'RANK_WEIGHT="${RANK_WEIGHT:-0.005}"' in runner_source

    diagnostic_path = os.path.join(
        ROOT, "tools", "run_tiny_rank_weight_diagnostic.sh"
    )
    with open(diagnostic_path, "r", encoding="utf-8") as handle:
        diagnostic_source = handle.read()
    assert 'RANK_WEIGHT="0.001"' in diagnostic_source
    assert "RANK_WEIGHT:-" not in diagnostic_source
    assert "ACTIONS=(TES PCS Ball Clubs Hoop Ribbon)" in (
        diagnostic_source
    )

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

    near_tie_prediction = torch.tensor(
        [0.20, 0.30], requires_grad=True
    )
    near_tie_target = torch.tensor([0.50, 0.51])
    near_tie_loss = _loss(
        near_tie_prediction, near_tie_target, margin=0.02
    )
    assert near_tie_loss.item() == 0.0
    near_tie_loss.backward()
    assert torch.equal(
        near_tie_prediction.grad,
        torch.zeros_like(near_tie_prediction),
    )

    prediction = torch.tensor([0.25, 0.75], requires_grad=True)
    label = torch.tensor([0.20, 0.80])
    base_loss = PAMFN.call_training_loss(
        _training_stub(0.0), prediction, label
    )
    expected_mse = nn.functional.mse_loss(prediction, label)
    assert torch.equal(base_loss, expected_mse)

    ranked_loss = PAMFN.call_training_loss(
        _training_stub(0.005), prediction, label
    )
    assert ranked_loss > base_loss
    tiny_ranked_loss = PAMFN.call_training_loss(
        _training_stub(0.001), prediction, label
    )
    assert base_loss < tiny_ranked_loss < ranked_loss

    branch_dir = os.path.join(ROOT, "pretrained_models", "feats2")
    required = [
        os.path.join(branch_dir, f"TES_{name}_{feature}.pth")
        for name, feature in (
            ("rgb", "UMT"),
            ("flow", "I3D"),
            ("audio", "MAST"),
        )
    ]
    official_checkpoint = os.path.join(
        branch_dir, "TES_multimodal.pth"
    )
    if all(os.path.isfile(path) for path in (
            required + [official_checkpoint])):
        common = {
            "model_dim": 256,
            "fc_drop": 0.3,
            "fc_r": 1,
            "feat_drop": 0.3,
            "K": 8,
            "ms_heads": 2,
            "cm_heads": 4,
            "ckpt_dir": branch_dir,
            "rgb_ckpt_name": "UMT",
            "flow_ckpt_name": "I3D",
            "audio_ckpt_name": "MAST",
            "dataset_name": "TES",
        }
        setup_seed(0)
        official_base = PAMFN(**common)
        setup_seed(0)
        pairwise_model = PAMFN(
            **common, shared_rank_weight=0.005
        )
        assert official_base.state_dict().keys() == (
            pairwise_model.state_dict().keys()
        )
        assert all(
            torch.equal(
                official_base.state_dict()[key],
                pairwise_model.state_dict()[key],
            )
            for key in official_base.state_dict()
        )
        del pairwise_model
        setup_seed(0)
        listwise_model = PAMFN(
            **common, shared_list_weight=0.0001
        )
        assert official_base.state_dict().keys() == (
            listwise_model.state_dict().keys()
        )
        assert all(
            torch.equal(
                official_base.state_dict()[key],
                listwise_model.state_dict()[key],
            )
            for key in official_base.state_dict()
        )
        del listwise_model
        setup_seed(0)
        prototype_model = PAMFN(
            **common,
            ordinal_prototype_weight=0.0001,
            ordinal_quality_thresholds=(0.4, 0.6, 0.8),
        )
        assert official_base.state_dict().keys() <= (
            prototype_model.state_dict().keys()
        )
        assert all(
            torch.equal(
                official_base.state_dict()[key],
                prototype_model.state_dict()[key],
            )
            for key in official_base.state_dict()
        )
        del prototype_model
        checkpoint = _torch_load(official_checkpoint)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        load_result = official_base.load_state_dict(
            checkpoint, strict=True
        )
        assert not load_result.missing_keys
        assert not load_result.unexpected_keys
        print("official checkpoint compatibility: passed")
    else:
        print(
            "official checkpoint compatibility: skipped "
            "(pretrained files absent)"
        )
    print("official dataset and DataLoader protocol: passed")
    print("shared pairwise ranking verification: passed")


if __name__ == "__main__":
    main()
