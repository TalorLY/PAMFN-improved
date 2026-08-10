import os
import sys
import tempfile

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import configs.FISV_feats2 as config  # noqa: E402
from models.pamfn import PAMFN  # noqa: E402
from pipeline import (  # noqa: E402
    _save_state_dict,
    build_optimizer,
    load_checkpoint,
    load_initial_checkpoint,
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


def model_input(batch_size=4):
    torch.manual_seed(29)
    skeleton = torch.randn(batch_size, 16, 132)
    joints = skeleton.reshape(batch_size, 16, 33, 4)
    joints[..., 3] = torch.rand(batch_size, 16, 33)
    return {
        "V": torch.randn(batch_size, 16, 1024),
        "F": torch.randn(batch_size, 16, 1024),
        "A": torch.randn(batch_size, 16, 768),
        "S": skeleton,
    }


def expect_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def verify_cue(cue, checkpoint, base, inputs, base_prediction):
    options = {
        "anchored_residual": True,
        "temporal_feature_adapter": True,
        "temporal_feature_scale": 0.1,
        "temporal_feature_reg_weight": 0.01,
        "residual_adapter_dim": 16,
        "residual_rank_weight": 0.01,
        "default_skeleton_gate": 1.0,
    }
    gates = (1.0, 0.0)
    if cue == "motion":
        options["motion_saliency_proxy"] = True
    else:
        options["use_skeleton"] = True
        gates = (0.0, 1.0)

    model = PAMFN(**model_kwargs(**options))
    missing = load_initial_checkpoint(model, checkpoint)
    prefixes = model.fresh_adapter_prefixes()
    expected_missing = {
        key for key in model.state_dict()
        if key.startswith(prefixes)
        and not key.endswith(".num_batches_tracked")
    }
    assert set(missing) == expected_missing

    model.eval()
    with torch.no_grad():
        initial_off = model(inputs, force_gates=(0.0, 0.0))[0]
        initial_on = model(inputs, force_gates=gates)[0]
    assert torch.equal(initial_off, base_prediction)
    assert torch.equal(initial_on, base_prediction)

    model.freeze_base()
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith(prefixes) for name in trainable)
    frozen_digest = state_dict_digest(model, prefixes)
    optimizer = build_optimizer(
        model, optim="Adam", lr=0.01,
        model_name="pamfn_final", trainable_only=True
    )
    target = (base_prediction.detach().reshape(-1)
              + torch.tensor([-0.12, -0.04, 0.05, 0.13]))
    for _ in range(6):
        model.train()
        prediction, info = model(inputs, force_gates=gates)
        loss = model.call_training_loss(prediction, target, **info)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert state_dict_digest(model, prefixes) == frozen_digest
    assert all(not module.training for module in model._base_modules())

    model.eval()
    with torch.no_grad():
        trained_off = model(inputs, force_gates=(0.0, 0.0))[0]
        trained_on, info = model(inputs, force_gates=gates)
    off_error = (trained_off - base_prediction).abs().max().item()
    assert off_error <= 1e-6, (cue, off_error)
    assert (trained_on - trained_off).abs().max().item() > 0
    residual = info["temporal_feature_residual"]
    reliability = info["temporal_feature_reliability_values"]
    assert residual.shape[:2] == (4, config.multi_modality["model_dim"] * 3)
    assert residual.abs().max().item() <= 0.100001
    assert reliability.min().item() >= 0
    assert reliability.max().item() <= 1

    metadata = model.checkpoint_metadata()
    assert metadata["schema_version"] == 8
    assert metadata["temporal_feature_injection"] == "stage1_pre_pool"
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, f"temporal_{cue}.pth")
        _save_state_dict(model, path)
        restored = PAMFN(**model_kwargs(**options))
        load_checkpoint(restored, path)
        restored.eval()
        with torch.no_grad():
            restored_on = restored(inputs, force_gates=gates)[0]
        assert torch.equal(restored_on, trained_on)


def main():
    torch.set_num_threads(1)
    checkpoint = os.path.join(
        ROOT, "pretrained_models", "feats2", "TES_multimodal.pth"
    )
    inputs = model_input()
    base = PAMFN(**model_kwargs())
    load_checkpoint(base, checkpoint)
    base.eval()
    with torch.no_grad():
        base_prediction = base(inputs)[0]

    verify_cue("motion", checkpoint, base, inputs, base_prediction)
    verify_cue("skeleton", checkpoint, base, inputs, base_prediction)
    expect_value_error(lambda: PAMFN(**model_kwargs(
        motion_saliency_proxy=True,
        temporal_feature_adapter=True,
    )))
    expect_value_error(lambda: PAMFN(**model_kwargs(
        anchored_residual=True,
        motion_saliency_proxy=True,
        use_skeleton=True,
        temporal_feature_adapter=True,
    )))
    expect_value_error(lambda: PAMFN(**model_kwargs(
        anchored_residual=True,
        motion_saliency_proxy=True,
        temporal_feature_adapter=True,
        bias_separated_residual=True,
    )))
    print("Motion feature-level temporal expert: passed")
    print("Skeleton feature-level temporal expert: passed")
    print("gate-off Base parity (atol=1e-6) and frozen Base: passed")
    print("schema-8 strict checkpoint reload: passed")
    print("temporal feature adapter verification passed")


if __name__ == "__main__":
    main()
