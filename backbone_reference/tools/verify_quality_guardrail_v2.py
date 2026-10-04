"""Fast invariants for task-prior x quality-guardrail v2."""

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


def guardrail_kwargs(**overrides):
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
        quality_guardrail_v2=True,
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
    model = PAMFN(**guardrail_kwargs())
    missing = load_initial_checkpoint(model, checkpoint)
    prefixes = model.fresh_adapter_prefixes()
    expected_missing = {
        name for name in model.state_dict()
        if name.startswith(prefixes)
        and not name.endswith(".num_batches_tracked")
    }
    assert set(missing) == expected_missing
    metadata = model.checkpoint_metadata()
    assert metadata["schema_version"] == 7
    assert metadata["hierarchical_architecture"].endswith("guardrail_v2")

    base.eval()
    model.eval()
    model.set_task_priors(0.25, 0.75)
    with torch.no_grad():
        base_prediction = base(inputs)[0]
        off_prediction = model(
            inputs, force_gates=(0.0, 0.0),
            disable_reliability=True,
        )[0]
        _, initial = model(inputs, disable_reliability=True)
    assert torch.equal(off_prediction, base_prediction)

    motion_reference = initial["motion_quality_raw_values"].min().item()
    skeleton_reference = initial["skeleton_quality_raw_values"].min().item()
    model.set_quality_references(motion_reference, skeleton_reference)
    with torch.no_grad():
        guard_prediction, guard = model(inputs)
        prior_prediction, prior = model(inputs, disable_reliability=True)
    assert torch.equal(guard_prediction, prior_prediction)
    assert torch.equal(
        prior["motion_gate_values"],
        torch.full_like(prior["motion_gate_values"], 0.25),
    )
    assert torch.equal(
        prior["skeleton_gate_values"],
        torch.full_like(prior["skeleton_gate_values"], 0.75),
    )
    for cue in ("motion", "skeleton"):
        assert torch.allclose(
            guard[f"{cue}_gate_values"],
            guard[f"{cue}_task_prior_values"]
            * guard[f"{cue}_reliability_values"],
            atol=1e-7, rtol=0,
        )
    with torch.no_grad():
        endpoint_zero = model(
            inputs, force_gates=(0.25, 0.0),
            disable_reliability=True,
        )[0]
        endpoint_one = model(
            inputs, force_gates=(0.25, 1.0),
            disable_reliability=True,
        )[0]
        direct = model(
            inputs, force_gates=(0.25, 0.37),
            disable_reliability=True,
        )[0]
    interpolated = endpoint_zero + 0.37 * (endpoint_one - endpoint_zero)
    assert torch.allclose(direct, interpolated, atol=2e-7, rtol=0)

    severities = (0.0, 0.25, 0.5, 0.75, 1.0)
    for cue in ("motion", "skeleton"):
        means = []
        with torch.no_grad():
            for severity in severities:
                _, info = model(inputs, cue_corruption=(cue, severity))
                means.append(info[f"{cue}_reliability_values"].mean().item())
                if severity == 1.0:
                    assert torch.equal(
                        info[f"{cue}_reliability_values"],
                        torch.zeros_like(
                            info[f"{cue}_reliability_values"]
                        ),
                    )
                    assert torch.equal(
                        info[f"{cue}_gate_values"],
                        torch.zeros_like(info[f"{cue}_gate_values"]),
                    )
        assert all(
            left + 1e-7 >= right
            for left, right in zip(means, means[1:])
        ), (cue, means)

    model.freeze_base()
    frozen_digest = state_dict_digest(model, prefixes)
    model.train()
    prediction, train_info = model(inputs)
    assert torch.equal(
        train_info["skeleton_gate_values"],
        torch.ones_like(train_info["skeleton_gate_values"]),
    )
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
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "quality_guardrail_v2.pth")
        _save_state_dict(model, path)
        restored = PAMFN(**guardrail_kwargs())
        load_checkpoint(restored, path)
        restored.eval()
        with torch.no_grad():
            expected = model(inputs)[0]
            actual = restored(inputs)[0]
        assert torch.equal(expected, actual)

    print("quality guardrail v2 verification: PASS")


if __name__ == "__main__":
    main()
