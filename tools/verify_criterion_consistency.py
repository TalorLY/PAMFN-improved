"""CPU checks for the minimal criterion-conditioned score screen."""

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


def model_kwargs(action, mode):
    config = (
        fisv_config if action in ("TES", "PCS") else rg_config
    )
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
        "criterion_consistency_weight": 1.0,
    }


def model_input(batch_size=3, length=18):
    torch.manual_seed(23)
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


def verify_action(action):
    inputs = model_input()
    target_count = 2 if action in ("TES", "PCS") else 4
    labels = torch.rand(inputs["V"].shape[0], target_count)
    models = {
        mode: PAMFN(**model_kwargs(action, mode))
        for mode in ("shared", "conditioned")
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
        assert info["criterion_attention_center_values"].shape == (
            inputs["V"].shape[0], target_count
        )
        assert info["criterion_attention_entropy_values"].shape == (
            inputs["V"].shape[0], target_count
        )
        loss = model.call_training_loss(
            prediction, labels, **info
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert any(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if name.startswith("criterion_")
        )
        assert model.checkpoint_metadata()["schema_version"] == 10

    conditioned = models["conditioned"]
    conditioned.eval()
    with torch.no_grad():
        prediction, info = conditioned(inputs)
        uniform, uniform_info = conditioned(
            inputs, criterion_control="uniform"
        )
    assert uniform.shape == prediction.shape
    assert torch.allclose(
        uniform_info["criterion_attention_entropy_values"],
        torch.ones_like(
            uniform_info["criterion_attention_entropy_values"]
        ),
        atol=1e-5,
    )
    if action == "Ball":
        components = info["criterion_predictions"]
        expected = components[:, 0] + components[:, 1] - components[:, 2]
        assert torch.allclose(
            expected, info["criterion_composed_total"]
        )
        assert torch.allclose(
            prediction.reshape(-1), expected.clamp(0, 1)
        )

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = os.path.join(directory, f"{action}.pth")
        _save_state_dict(conditioned, checkpoint)
        restored = PAMFN(**model_kwargs(action, "conditioned"))
        load_checkpoint(restored, checkpoint)
        restored.eval()
        with torch.no_grad():
            restored_prediction = restored(inputs)[0]
        assert torch.equal(restored_prediction, prediction)

        output = os.path.join(directory, f"{action}.csv")
        batch = dict(inputs)
        batch.update({
            "label": labels,
            "sample_id": ["a", "b", "c"],
        })
        export_predictions(
            conditioned, [batch], output, log_func=lambda _: None
        )
        with open(output, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3
        expected_columns = {
            "criterion_attention_divergence",
            f"{conditioned.criterion_names[0]}_label",
            f"{conditioned.criterion_names[-1]}_prediction",
        }
        assert expected_columns <= set(rows[0])
        if action == "Ball":
            assert {
                "composed_total_prediction",
                "direct_total_prediction",
                "total_consistency_gap",
            } <= set(rows[0])
    return counts["shared"]


def main():
    torch.set_num_threads(1)
    verify_labels()
    counts = {
        action: verify_action(action)
        for action in ("TES", "PCS", "Ball")
    }
    print("joint FIS-V and decomposed RG labels: passed")
    print(f"matched shared/conditioned trainable parameters: {counts}")
    print("criterion forward, loss, uniform control, and export: passed")
    print("schema-10 strict checkpoint reload: passed")
    print("criterion consistency verifier passed")


if __name__ == "__main__":
    main()
