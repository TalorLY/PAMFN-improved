"""CPU checks for criterion-contrast cross-modal rank fusion."""

import csv
import os
import sys
import tempfile

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import configs.FISV_feats2 as fisv_config  # noqa: E402
import configs.RG_feats2 as rg_config  # noqa: E402
from datasets import AQADataset  # noqa: E402
from models.pamfn import PAMFN  # noqa: E402
from pipeline import (  # noqa: E402
    _save_state_dict,
    export_predictions,
    load_checkpoint,
)


MODES = ("rank_shared", "contrast_fusion")


def model_kwargs(action, mode):
    config = fisv_config if action in ("TES", "PCS") else rg_config
    current = config.multi_modality
    return {
        "model_dim": current["model_dim"],
        "fc_drop": current["fc_drop"],
        "fc_r": current["fc_r"],
        "feat_drop": current["feat_drop"],
        "K": current["K"],
        "ms_heads": current["ms_heads"],
        "cm_heads": current["cm_heads"],
        "ckpt_dir": current["ckpt_dir"],
        "rgb_ckpt_name": current["rgb_ckpt_name"],
        "flow_ckpt_name": current["flow_ckpt_name"],
        "audio_ckpt_name": current["audio_ckpt_name"],
        "dataset_name": action,
        "criterion_screen_mode": mode,
        "criterion_consistency_weight": 0.1,
        "criterion_rank_weight": 0.1,
        "criterion_contrast_weight": 1.0,
        "criterion_private_weight": 0.01,
        "criterion_penalty_weight": 0.1,
        "criterion_penalty_positive_weight": 4.0,
    }


def model_input(batch_size=4, length=18):
    torch.manual_seed(20260725)
    return {
        "V": torch.randn(batch_size, length, 1024),
        "F": torch.randn(batch_size, length, 1024),
        "A": torch.randn(batch_size, length, 768),
        "valid_mask": torch.ones(batch_size, length),
        "normalized_time": (
            (torch.arange(length).float() + 0.5) / length
        ).repeat(batch_size, 1),
    }


def parsed_targets(action):
    dataset = AQADataset.__new__(AQADataset)
    dataset.criterion_targets = True
    return dataset.build(action, True)


def verify_labels():
    tes = parsed_targets("TES")
    pcs = parsed_targets("PCS")
    assert tes == pcs
    assert len(tes) == 400 and len(tes[0][1]) == 2
    for action in ("Ball", "Clubs", "Hoop", "Ribbon"):
        rows = parsed_targets(action)
        assert len(rows) == 200 and len(rows[0][1]) == 4
        for _, (difficulty, execution, penalty, total) in rows:
            assert (
                abs(difficulty + execution - penalty - total)
                <= 0.0005
            )


def labels_for(action):
    if action in ("TES", "PCS"):
        return torch.tensor([
            [0.40, 0.30],
            [0.55, 0.50],
            [0.70, 0.58],
            [0.85, 0.75],
        ])
    difficulty = torch.tensor([0.30, 0.38, 0.47, 0.55])
    execution = torch.tensor([0.20, 0.28, 0.34, 0.40])
    penalty = torch.tensor([0.00, 0.01, 0.00, 0.02])
    total = difficulty + execution - penalty
    return torch.stack((difficulty, execution, penalty, total), dim=1)


def verify_action(action):
    inputs = model_input()
    labels = labels_for(action)
    target_count = labels.shape[1]
    models = {
        mode: PAMFN(**model_kwargs(action, mode))
        for mode in MODES
    }
    counts = {
        mode: sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        for mode, model in models.items()
    }
    assert len(set(counts.values())) == 1, counts

    for mode, model in models.items():
        model.eval()
        prediction, info = model(inputs)
        assert prediction.shape == (inputs["V"].shape[0], 1)
        assert info["criterion_predictions"].shape == (
            inputs["V"].shape[0], target_count
        )
        assert info["criterion_modality_weights_values"].shape == (
            inputs["V"].shape[0], target_count, 3
        )
        assert info["criterion_private_norm_values"].shape == (
            inputs["V"].shape[0], target_count
        )
        assert torch.isfinite(prediction).all()
        assert prediction.std() > 0
        loss = model.call_training_loss(prediction, labels, **info)
        assert torch.isfinite(loss)
        loss.backward()
        assert any(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if name.startswith("criterion_")
        )
        assert any(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if name.startswith("stage1.")
        )
        assert model.checkpoint_metadata()["schema_version"] == 11

    shared = models["rank_shared"]
    shared.eval()
    with torch.no_grad():
        _, shared_info = shared(inputs)
    expected_uniform = torch.full_like(
        shared_info["criterion_modality_weights_values"], 1 / 3
    )
    assert torch.allclose(
        shared_info["criterion_modality_weights_values"],
        expected_uniform,
    )

    contrast = models["contrast_fusion"]
    with torch.no_grad():
        contrast.criterion_modality_logits.copy_(torch.tensor(
            [
                [2.0, 0.0, -1.0],
                [-1.0, 2.0, 0.0],
            ] if target_count == 2 else [
                [2.0, 0.0, -1.0],
                [-1.0, 2.0, 0.0],
                [0.0, -1.0, 2.0],
                [1.0, 1.0, -1.0],
            ]
        ))
    contrast.eval()
    with torch.no_grad():
        prediction, info = contrast(inputs)
        uniform_prediction, uniform_info = contrast(
            inputs, criterion_control="uniform"
        )
    assert uniform_prediction.shape == prediction.shape
    assert torch.allclose(
        uniform_info["criterion_modality_weights_values"],
        expected_uniform,
    )
    assert not torch.allclose(
        info["criterion_modality_weights_values"],
        uniform_info["criterion_modality_weights_values"],
    )

    if action in ("TES", "PCS"):
        expected = torch.stack((
            info["criterion_common_prediction"]
            + info["criterion_contrast_prediction"],
            info["criterion_common_prediction"]
            - info["criterion_contrast_prediction"],
        ), dim=1).clamp(0, 1)
        assert torch.allclose(info["criterion_predictions"], expected)
    else:
        components = info["criterion_predictions"]
        expected = components[:, 0] + components[:, 1] - components[:, 2]
        assert torch.allclose(
            expected, info["criterion_composed_total"]
        )
        assert torch.allclose(
            prediction.reshape(-1), expected.clamp(0, 1)
        )
        assert (
            info["criterion_penalty_event_probability_values"].shape
            == (inputs["V"].shape[0],)
        )
        assert (info["criterion_penalty_amount_values"] >= 0).all()

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = os.path.join(directory, f"{action}.pth")
        _save_state_dict(contrast, checkpoint)
        restored = PAMFN(**model_kwargs(action, "contrast_fusion"))
        load_checkpoint(restored, checkpoint)
        restored.eval()
        with torch.no_grad():
            restored_prediction = restored(inputs)[0]
        assert torch.equal(restored_prediction, prediction)

        output = os.path.join(directory, f"{action}.csv")
        batch = dict(inputs)
        batch.update({
            "label": labels,
            "sample_id": ["a", "b", "c", "d"],
        })
        export_predictions(
            contrast, [batch], output, log_func=lambda _: None
        )
        with open(output, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 4
        expected_columns = {
            f"{contrast.criterion_names[0]}_label",
            f"{contrast.criterion_names[-1]}_prediction",
            f"{contrast.criterion_names[0]}_modality_weight_rgb",
            "criterion_shared_norm",
        }
        if action in ("TES", "PCS"):
            expected_columns.update({
                "common_quality_prediction",
                "criterion_contrast_prediction",
            })
        else:
            expected_columns.update({
                "penalty_event_probability",
                "penalty_amount_prediction",
                "composed_total_prediction",
                "direct_total_prediction",
                "total_consistency_gap",
            })
        assert expected_columns <= set(rows[0])
    return counts["rank_shared"]


def main():
    torch.set_num_threads(1)
    verify_labels()
    counts = {
        action: verify_action(action)
        for action in ("TES", "PCS", "Ball")
    }
    print("paired FIS-V and decomposed RG labels: passed")
    print(f"matched rank-shared/contrast parameters: {counts}")
    print("common/private fusion, ranking loss, and uniform control: passed")
    print("RG sparse penalty and total identity: passed")
    print("schema-11 strict checkpoint reload and export: passed")
    print("criterion contrast fusion verifier passed")


if __name__ == "__main__":
    main()
