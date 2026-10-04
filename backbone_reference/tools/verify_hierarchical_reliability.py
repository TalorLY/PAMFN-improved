"""Fast invariants for the task-prior x temporal-reliability probe."""

import os
import sys
import tempfile

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.pamfn import PAMFN  # noqa: E402
from pipeline import (  # noqa: E402
    _save_state_dict,
    build_optimizer,
    load_checkpoint,
    load_initial_checkpoint,
    state_dict_digest,
)
from verify_anchored_residual import model_input, model_kwargs  # noqa: E402


def hierarchical_kwargs(**overrides):
    values = model_kwargs(
        use_skeleton=True,
        skeleton_gate_max=0.02,
        motion_saliency_proxy=True,
        motion_saliency_alpha=0.015,
        default_skeleton_gate=1.0,
        anchored_residual=True,
        residual_adapter_dim=16,
        hierarchical_reliability_gate=True,
        reliability_quality_weight=1.0,
    )
    values.update(overrides)
    return values


def main():
    torch.set_num_threads(1)
    checkpoint = os.path.join(
        ROOT, "pretrained_models", "feats2", "TES_multimodal.pth"
    )
    inputs = model_input(batch_size=3)

    base = PAMFN(**model_kwargs())
    load_checkpoint(base, checkpoint)
    model = PAMFN(**hierarchical_kwargs())
    missing = load_initial_checkpoint(model, checkpoint)
    prefixes = model.fresh_adapter_prefixes()
    expected_missing = {
        name for name in model.state_dict()
        if name.startswith(prefixes)
        and not name.endswith(".num_batches_tracked")
    }
    assert set(missing) == expected_missing
    assert not any(
        name.startswith(("motion_encoder.", "motion_score_residual_head."))
        for name in model.state_dict()
    )
    assert model.checkpoint_metadata()["schema_version"] == 6

    base.eval()
    model.eval()
    with torch.no_grad():
        base_prediction = base(inputs)[0]
        off_prediction = model(inputs, force_gates=(0.0, 0.0))[0]
        prediction, info = model(inputs, force_gates=(0.75, 0.5))
    assert torch.equal(off_prediction, base_prediction)
    assert torch.allclose(
        info["motion_gate_values"],
        0.75 * info["motion_reliability_values"],
        atol=1e-7, rtol=0,
    )
    assert torch.allclose(
        info["skeleton_gate_values"],
        0.5 * info["skeleton_reliability_values"],
        atol=1e-7, rtol=0,
    )
    assert torch.isfinite(prediction).all()

    model.set_task_priors(0.75, 0.5)
    with torch.no_grad():
        default_prediction, default_info = model(inputs)
    assert torch.equal(default_prediction, prediction)
    assert torch.equal(
        default_info["motion_gate_values"], info["motion_gate_values"]
    )

    severities = (0.0, 0.25, 0.5, 0.75, 1.0)
    for cue, reliability_key, gate_key in (
            ("motion", "motion_reliability_values", "motion_gate_values"),
            ("skeleton", "skeleton_reliability_values",
             "skeleton_gate_values")):
        means = []
        with torch.no_grad():
            for severity in severities:
                _, corrupted = model(
                    inputs, cue_corruption=(cue, severity)
                )
                means.append(corrupted[reliability_key].mean().item())
                if severity == 1.0:
                    assert torch.equal(
                        corrupted[reliability_key],
                        torch.zeros_like(corrupted[reliability_key]),
                    )
                    assert torch.equal(
                        corrupted[gate_key],
                        torch.zeros_like(corrupted[gate_key]),
                    )
        assert all(
            left + 1e-7 >= right
            for left, right in zip(means, means[1:])
        ), (cue, means)

    model.freeze_base()
    frozen_digest = state_dict_digest(model, prefixes)
    model.train()
    prediction, train_info = model(inputs)
    target = base_prediction.detach() + torch.linspace(
        -0.05, 0.05, inputs["V"].shape[0]
    ).reshape(-1, 1)
    loss = model.call_training_loss(prediction, target, **train_info)
    optimizer = build_optimizer(
        model, optim="Adam", lr=0.01,
        model_name="pamfn_final", trainable_only=True,
    )
    optimizer.zero_grad()
    loss.backward()
    for name in (
            "skeleton_score_residual_head.1.weight",
            "motion_quality_head.0.weight",
            "skeleton_quality_head.0.weight"):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        assert gradient.abs().sum().item() > 0, name
    optimizer.step()
    assert state_dict_digest(model, prefixes) == frozen_digest

    model.eval()
    with torch.no_grad():
        saved_prediction = model(inputs)[0]
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "hierarchical.pth")
        _save_state_dict(model, path)
        restored = PAMFN(**hierarchical_kwargs())
        load_checkpoint(restored, path)
        restored.eval()
        with torch.no_grad():
            restored_prediction = restored(inputs)[0]
        assert torch.equal(saved_prediction, restored_prediction)

    print("hierarchical reliability verification: PASS")


if __name__ == "__main__":
    main()
