"""CPU regression checks for the protocol and three phase-rank branches."""

import csv
import os
import sys
import tempfile

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import configs.FISV_feats2 as config  # noqa: E402
from datasets import temporal_bin_average  # noqa: E402
from models.pamfn import PAMFN  # noqa: E402
from pipeline import (  # noqa: E402
    _save_state_dict,
    build_optimizer,
    export_predictions,
    load_checkpoint,
    load_initial_checkpoint,
    prepare_model_batch,
    state_dict_digest,
)


def model_kwargs(**overrides):
    current = config.multi_modality
    values = {
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
        "dataset_name": "TES",
    }
    values.update(overrides)
    return values


def model_input(batch_size=4, length=18):
    torch.manual_seed(37)
    return {
        "V": torch.randn(batch_size, length, 1024),
        "F": torch.randn(batch_size, length, 1024),
        "A": torch.randn(batch_size, length, 768),
        "valid_mask": torch.ones(batch_size, length),
        "normalized_time": (
            (torch.arange(length).float() + 0.5) / length
        ).repeat(batch_size, 1),
    }


def verify_protocol_batch():
    source = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    first = temporal_bin_average(source, 7)
    second = temporal_bin_average(source, 7)
    assert torch.equal(first, second)
    batch = {
        "V": torch.randn(2, 7, 4),
        "F": torch.randn(2, 7, 4),
        "A": torch.randn(2, 7, 3),
        "label": torch.tensor([0.1, 0.2]),
        "sample_id": ["a", "b"],
        "valid_mask": torch.ones(2, 7),
        "normalized_time": (
            (torch.arange(7).float() + 0.5) / 7
        ).repeat(2, 1),
    }
    prepared, label, sample_ids = prepare_model_batch(
        batch, torch.device("cpu")
    )
    assert prepared["valid_mask"].shape == (2, 7)
    assert prepared["normalized_time"].shape == (2, 7)
    assert label.shape == (2,)
    assert sample_ids == ["a", "b"]


def verify_mode(mode, checkpoint, inputs, base_prediction):
    options = {
        "anchored_residual": True,
        "phase_rank_mode": mode,
        "phase_count": 3,
        "phase_residual_scale": 0.05,
        "phase_rank_weight": 0.1,
        "phase_guard_weight": 1.0,
        "phase_zero_mean_weight": 1.0,
        "phase_low_margin": 0.02,
        "phase_guard_margin": 0.05,
        "phase_rank_temperature": 0.05,
        "phase_min_mass": 0.1,
        "phase_softness": 0.05,
        "residual_adapter_dim": 16,
    }
    model = PAMFN(**model_kwargs(**options))
    missing = load_initial_checkpoint(model, checkpoint)
    prefixes = model.fresh_adapter_prefixes()
    expected = {
        key for key in model.state_dict()
        if key.startswith(prefixes)
        and not key.endswith(".num_batches_tracked")
    }
    assert set(missing) == expected
    model.eval()
    with torch.no_grad():
        initial, initial_info = model(inputs)
    assert torch.equal(initial, base_prediction)
    assert torch.equal(
        initial_info["phase_residual_values"],
        torch.zeros_like(initial_info["phase_residual_values"]),
    )

    model.freeze_base()
    trainable = {
        name: parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith(prefixes) for name in trainable)
    trainable_count = sum(
        parameter.numel() for parameter in trainable.values()
    )
    frozen_digest = state_dict_digest(model, prefixes)
    optimizer = build_optimizer(
        model, optim="Adam", lr=0.01,
        model_name="pamfn_final", trainable_only=True
    )
    target = (
        base_prediction.detach().reshape(-1)
        + torch.tensor([-0.10, -0.03, 0.04, 0.11])
    ).clamp(0, 1)
    for _ in range(8):
        model.train()
        prediction, info = model(inputs)
        loss = model.call_training_loss(prediction, target, **info)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert state_dict_digest(model, prefixes) == frozen_digest
    assert all(not module.training for module in model._base_modules())

    model.eval()
    with torch.no_grad():
        trained, info = model(inputs)
    assert (trained - base_prediction).abs().max().item() > 0
    residual = info["phase_residual_values"]
    masses = info["phase_mass_values"]
    centers = info["phase_center_values"]
    assert residual.abs().max().item() <= 0.050001
    assert masses.shape == centers.shape == (4, 3)
    assert torch.allclose(
        masses.sum(dim=1), torch.ones(4), atol=1e-5
    )
    if mode in ("fixed", "ordered"):
        assert torch.all(centers[:, 1:] > centers[:, :-1])
    if mode == "ordered":
        with torch.no_grad():
            reverse = model(inputs, phase_control="reverse")[0]
            shuffle = model(inputs, phase_control="shuffle")[0]
        assert reverse.shape == trained.shape
        assert shuffle.shape == trained.shape

    assert model.checkpoint_metadata()["schema_version"] == 9
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, f"phase_{mode}.pth")
        _save_state_dict(model, path)
        restored = PAMFN(**model_kwargs(**options))
        load_checkpoint(restored, path)
        restored.eval()
        with torch.no_grad():
            restored_prediction = restored(inputs)[0]
        assert torch.equal(restored_prediction, trained)
        if mode == "ordered":
            csv_path = os.path.join(directory, "ordered.csv")
            batch = dict(inputs)
            batch.update({
                "label": target,
                "sample_id": ["a", "b", "c", "d"],
            })
            export_predictions(model, [batch], csv_path, log_func=lambda _: None)
            with open(csv_path, "r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == 4
            assert {
                "base_score", "phase_residual", "phase_entropy",
                "phase_mass_0", "phase_center_2",
            } <= set(rows[0])
    return trainable_count


def main():
    torch.set_num_threads(1)
    verify_protocol_batch()
    checkpoint = os.path.join(
        ROOT, "pretrained_models", "feats2", "TES_multimodal.pth"
    )
    inputs = model_input()
    base = PAMFN(**model_kwargs())
    load_checkpoint(base, checkpoint)
    base.eval()
    with torch.no_grad():
        base_prediction = base(inputs)[0]
    counts = {
        mode: verify_mode(mode, checkpoint, inputs, base_prediction)
        for mode in ("global", "fixed", "ordered")
    }
    assert len(set(counts.values())) == 1, counts
    print("normalized_full metadata protocol: passed")
    print(f"matched trainable parameters: {next(iter(counts.values()))}")
    print("global/fixed/ordered zero-init Base parity: passed")
    print("frozen Base and schema-9 strict reload: passed")
    print("phase rank adapter verification passed")


if __name__ == "__main__":
    main()
