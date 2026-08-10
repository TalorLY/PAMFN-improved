import os
import sys
import tempfile

import torch
import torch.nn as nn


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
    solve_residual_logit_center,
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


def model_input(batch_size=2):
    torch.manual_seed(17)
    skeleton = torch.randn(batch_size, 16, 132)
    joints = skeleton.reshape(batch_size, 16, 33, 4)
    joints[..., 3] = torch.rand(batch_size, 16, 33)
    return {
        "V": torch.randn(batch_size, 16, 1024),
        "F": torch.randn(batch_size, 16, 1024),
        "A": torch.randn(batch_size, 16, 768),
        "S": skeleton,
    }


def expect_runtime_error(function):
    try:
        function()
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def expect_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def predictions(model, inputs):
    return {
        gates: model(inputs, force_gates=gates)[0]
        for gates in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
    }


def main():
    torch.set_num_threads(1)
    checkpoint = os.path.join(
        ROOT, "pretrained_models", "feats2", "TES_multimodal.pth"
    )

    base = PAMFN(**model_kwargs())
    load_checkpoint(base, checkpoint)
    anchored = PAMFN(**model_kwargs(
        use_skeleton=True,
        skeleton_gate_max=0.02,
        motion_saliency_proxy=True,
        motion_saliency_alpha=0.02,
        default_skeleton_gate=1.0,
        anchored_residual=True,
    ))
    missing = load_initial_checkpoint(anchored, checkpoint)
    adapter_prefixes = anchored.fresh_adapter_prefixes()
    expected_missing = {
        key for key in anchored.state_dict()
        if key.startswith(adapter_prefixes)
        and not key.endswith(".num_batches_tracked")
    }
    assert set(missing) == expected_missing

    inputs = model_input()
    base.eval()
    anchored.eval()
    with torch.no_grad():
        base_prediction = base(inputs)[0]
        initial = predictions(anchored, inputs)
    for prediction in initial.values():
        assert torch.equal(prediction, base_prediction)

    anchored.freeze_base()
    trainable_names = {
        name for name, parameter in anchored.named_parameters()
        if parameter.requires_grad
    }
    assert trainable_names
    assert all(name.startswith(adapter_prefixes) for name in trainable_names)
    frozen_digest = state_dict_digest(anchored, adapter_prefixes)

    anchored.train()
    assert anchored.training
    assert anchored.motion_encoder.training
    assert anchored.skeleton_encoder.training
    assert all(not module.training for module in anchored._base_modules())

    optimizer = build_optimizer(
        anchored,
        optim="Adam",
        lr=0.01,
        model_name="pamfn_final",
        trainable_only=True,
    )
    for _ in range(3):
        prediction, other_info = anchored(
            inputs, force_gates=(1.0, 1.0)
        )
        target = base_prediction.detach() + 0.1
        loss = anchored.call_loss(prediction, target, **other_info)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert state_dict_digest(anchored, adapter_prefixes) == frozen_digest
    anchored.train()
    with torch.no_grad():
        off_a = anchored(inputs, force_gates=(0.0, 0.0))[0]
        off_b = anchored(inputs, force_gates=(0.0, 0.0))[0]
    assert torch.equal(off_a, off_b)
    assert torch.equal(off_a, base_prediction)

    anchored.eval()
    with torch.no_grad():
        trained = predictions(anchored, inputs)
    p00 = trained[(0.0, 0.0)]
    motion = trained[(1.0, 0.0)] - p00
    skeleton = trained[(0.0, 1.0)] - p00
    combined = trained[(1.0, 1.0)] - p00
    assert motion.abs().max().item() > 0
    assert skeleton.abs().max().item() > 0
    assert motion.abs().max().item() <= 0.020001
    assert skeleton.abs().max().item() <= 0.020001
    identity_error = (combined - motion - skeleton).abs().max().item()
    assert identity_error <= 5e-7, identity_error

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "anchored.pth")
        _save_state_dict(anchored, path)
        restored = PAMFN(**model_kwargs(
            use_skeleton=True,
            skeleton_gate_max=0.02,
            motion_saliency_proxy=True,
            motion_saliency_alpha=0.02,
            default_skeleton_gate=1.0,
            anchored_residual=True,
        ))
        load_checkpoint(restored, path)
        restored.eval()
        with torch.no_grad():
            restored_prediction = restored(
                inputs, force_gates=(1.0, 1.0)
            )[0]
        assert torch.equal(
            restored_prediction, trained[(1.0, 1.0)]
        )
        restored.default_skeleton_gate = 0.5
        expect_runtime_error(lambda: load_checkpoint(restored, path))
        restored.default_skeleton_gate = 1.0
        raw_path = os.path.join(directory, "raw_state_dict.pth")
        torch.save(anchored.state_dict(), raw_path)
        expect_runtime_error(
            lambda: load_checkpoint(restored, raw_path)
        )
        expect_runtime_error(
            lambda: load_initial_checkpoint(restored, path)
        )

        bias_model = PAMFN(**model_kwargs(
            use_skeleton=True,
            skeleton_gate_max=0.02,
            default_skeleton_gate=1.0,
            anchored_residual=True,
            bias_separated_residual=True,
            residual_zero_mean_weight=1.0,
            residual_rank_weight=0.001,
        ))
        bias_missing = load_initial_checkpoint(bias_model, checkpoint)
        bias_fresh_prefixes = bias_model.fresh_adapter_prefixes()
        expected_bias_missing = {
            key for key in bias_model.state_dict()
            if key.startswith(bias_fresh_prefixes)
            and not key.endswith(".num_batches_tracked")
        }
        assert set(bias_missing) == expected_bias_missing
        assert bias_model.skeleton_score_residual_head[1].bias is None

        bias_model.eval()
        with torch.no_grad():
            zero_bias_prediction = bias_model(
                inputs, force_gates=(0.0, 0.0)
            )[0]
        assert torch.equal(zero_bias_prediction, base_prediction)

        shared_bias = -0.04
        bias_model.set_shared_score_bias(shared_bias)
        with torch.no_grad():
            shared_off = bias_model(
                inputs, force_gates=(0.0, 0.0)
            )[0]
            shared_on_initial = bias_model(
                inputs, force_gates=(0.0, 1.0)
            )[0]
        expected_shared = base_prediction + shared_bias
        assert torch.equal(shared_off, expected_shared)
        assert torch.equal(shared_on_initial, expected_shared)

        trainable_prefixes = bias_model.trainable_adapter_prefixes()
        fixed_digest = state_dict_digest(bias_model, trainable_prefixes)
        bias_model.set_shared_score_bias(shared_bias + 0.01)
        assert state_dict_digest(
            bias_model, trainable_prefixes
        ) != fixed_digest
        bias_model.set_shared_score_bias(shared_bias)
        assert state_dict_digest(
            bias_model, trainable_prefixes
        ) == fixed_digest

        bias_model.freeze_base()
        bias_optimizer = build_optimizer(
            bias_model,
            optim="Adam",
            lr=0.01,
            model_name="pamfn_final",
            trainable_only=True,
        )
        target = expected_shared.detach() + torch.tensor([
            [-0.01], [0.01]
        ])
        for _ in range(3):
            prediction, other_info = bias_model(
                inputs, force_gates=(0.0, 1.0)
            )
            loss = bias_model.call_training_loss(
                prediction, target, **other_info
            )
            bias_optimizer.zero_grad()
            loss.backward()
            bias_optimizer.step()
        assert state_dict_digest(
            bias_model, trainable_prefixes
        ) == fixed_digest

        bias_model.eval()
        with torch.no_grad():
            trained_off = bias_model(
                inputs, force_gates=(0.0, 0.0)
            )[0]
            trained_on = bias_model(
                inputs, force_gates=(0.0, 1.0)
            )[0]
        assert torch.equal(trained_off, expected_shared)
        skeleton_residual = trained_on - trained_off
        assert skeleton_residual.std().item() > 0
        assert skeleton_residual.abs().max().item() <= 0.020001

        mse = bias_model.call_loss(trained_on[:1], target[:1])
        expected_mse = torch.mean((trained_on[:1] - target[:1]) ** 2)
        assert torch.equal(mse, expected_mse)

        fake_base = torch.tensor([[0.2], [0.4], [0.6]])
        fake_residual = torch.tensor([[-0.01], [0.0], [0.01]])
        fake_target_residual = torch.tensor([[-0.02], [0.0], [0.02]])
        fake_target = (
            fake_base + shared_bias + fake_target_residual
        )
        fake_prediction = fake_base + shared_bias + fake_residual
        zero_motion = torch.zeros_like(fake_residual)
        centered_objective = bias_model.call_training_loss(
            fake_prediction,
            fake_target,
            base_score=fake_base,
            cue_residual=fake_residual,
            motion_score_correction=zero_motion,
            skeleton_score_correction=fake_residual,
        )
        shifted_residual = fake_residual + 0.005
        shifted_objective = bias_model.call_training_loss(
            fake_prediction + 0.005,
            fake_target,
            base_score=fake_base,
            cue_residual=shifted_residual,
            motion_score_correction=zero_motion,
            skeleton_score_correction=shifted_residual,
        )
        assert shifted_objective > centered_objective

        dual_bias_model = PAMFN(**model_kwargs(
            use_skeleton=True,
            motion_saliency_proxy=True,
            anchored_residual=True,
            bias_separated_residual=True,
        ))
        dual_prediction = torch.tensor([[0.2], [0.4]])
        dual_base = dual_prediction.clone()
        positive = torch.full_like(dual_prediction, 0.01)
        negative = -positive
        zero = torch.zeros_like(dual_prediction)
        zero_correction_loss = dual_bias_model.call_training_loss(
            dual_prediction,
            dual_prediction,
            base_score=dual_base,
            cue_residual=zero,
            motion_score_correction=zero,
            skeleton_score_correction=zero,
        )
        opposing_correction_loss = dual_bias_model.call_training_loss(
            dual_prediction,
            dual_prediction,
            base_score=dual_base,
            cue_residual=zero,
            motion_score_correction=positive,
            skeleton_score_correction=negative,
        )
        assert opposing_correction_loss > zero_correction_loss

        bias_path = os.path.join(directory, "bias_separated.pth")
        _save_state_dict(bias_model, bias_path)
        restored_bias = PAMFN(**model_kwargs(
            use_skeleton=True,
            skeleton_gate_max=0.02,
            default_skeleton_gate=1.0,
            anchored_residual=True,
            bias_separated_residual=True,
            residual_zero_mean_weight=1.0,
            residual_rank_weight=0.001,
        ))
        load_checkpoint(restored_bias, bias_path)
        restored_bias.eval()
        with torch.no_grad():
            restored_bias_prediction = restored_bias(
                inputs, force_gates=(0.0, 1.0)
            )[0]
        assert torch.equal(restored_bias_prediction, trained_on)
        expect_runtime_error(lambda: load_checkpoint(restored_bias, path))
        expect_runtime_error(lambda: load_checkpoint(restored, bias_path))
        expect_runtime_error(
            lambda: load_initial_checkpoint(restored_bias, bias_path)
        )

        asymmetric_logits = torch.tensor(
            [-3.0, -0.5, 0.2, 2.5, 7.0], dtype=torch.float64
        )
        solved_center = solve_residual_logit_center(asymmetric_logits)
        solved_mean = torch.tanh(
            asymmetric_logits - solved_center
        ).mean().abs().item()
        assert solved_mean <= 1e-10
        assert abs(solved_center - asymmetric_logits.mean().item()) > 0.1
        assert solve_residual_logit_center([2.25, 2.25]) == 2.25
        expect_value_error(lambda: solve_residual_logit_center([]))
        expect_value_error(
            lambda: solve_residual_logit_center([0.0, float("nan")])
        )
        expect_value_error(
            lambda: solve_residual_logit_center([0.0, float("inf")])
        )

        v2_model = PAMFN(**model_kwargs(
            use_skeleton=True,
            skeleton_gate_max=0.02,
            default_skeleton_gate=1.0,
            anchored_residual=True,
            bias_separated_residual=True,
            residual_v2=True,
            residual_adapter_dim=16,
            residual_zero_mean_weight=1.0,
            residual_rank_weight=0.0,
        ))
        load_initial_checkpoint(v2_model, checkpoint)
        active_modules = list(v2_model.skeleton_encoder.modules()) + list(
            v2_model.skeleton_score_residual_head.modules()
        )
        assert not any(
            isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.Dropout))
            for module in active_modules
        )
        assert v2_model.skeleton_score_residual_head[1].bias is None
        assert (
            "skeleton_score_residual_head.2.logit_center"
            in v2_model.state_dict()
        )
        v2_metadata = v2_model.checkpoint_metadata()
        assert v2_metadata["schema_version"] == 3
        assert v2_metadata["residual_v2"] is True
        assert v2_metadata["residual_adapter_dim"] == 16
        assert v2_metadata["residual_rank_weight"] == 0.0
        assert (
            v2_metadata["residual_architecture"]
            == "centered_no_norm_v2"
        )
        assert (
            v2_metadata["residual_center_method"]
            == "train_zero_mean_tanh_root"
        )

        v2_model.set_shared_score_bias(shared_bias)
        v2_model.freeze_base()
        v2_trainable_count = sum(
            parameter.numel() for parameter in v2_model.parameters()
            if parameter.requires_grad
        )
        assert v2_trainable_count == 3712
        v2_optimizer = build_optimizer(
            v2_model,
            optim="Adam",
            lr=0.01,
            model_name="pamfn_final",
            trainable_only=True,
        )
        for _ in range(3):
            v2_prediction, v2_info = v2_model(
                inputs, force_gates=(0.0, 1.0)
            )
            v2_loss = v2_model.call_training_loss(
                v2_prediction, target, **v2_info
            )
            v2_optimizer.zero_grad()
            v2_loss.backward()
            v2_optimizer.step()

        v2_model.eval()
        with torch.no_grad():
            v2_off = v2_model(inputs, force_gates=(0.0, 0.0))[0]
            _, v2_info = v2_model(inputs, force_gates=(0.0, 1.0))
        assert torch.equal(v2_off, expected_shared)
        raw_logits = v2_info["cue_residual_logit"].reshape(-1).double()
        fitted_center = solve_residual_logit_center(raw_logits)
        v2_model.set_residual_logit_center(fitted_center)
        stored_center = v2_model.residual_logit_center_value()
        normalized_mean = torch.tanh(
            raw_logits - stored_center
        ).mean().abs().item()
        assert normalized_mean <= 1e-6

        v2_model.eval()
        with torch.no_grad():
            v2_eval, v2_eval_info = v2_model(
                inputs, force_gates=(0.0, 1.0)
            )
        v2_model.train()
        with torch.no_grad():
            v2_train, v2_train_info = v2_model(
                inputs, force_gates=(0.0, 1.0)
            )
        assert torch.equal(v2_eval, v2_train)
        assert torch.equal(
            v2_eval_info["cue_residual"],
            v2_train_info["cue_residual"],
        )
        assert v2_eval_info["cue_residual"].abs().max().item() <= 0.020001

        v2_path = os.path.join(directory, "residual_v2.pth")
        _save_state_dict(v2_model, v2_path)
        restored_v2 = PAMFN(**model_kwargs(
            use_skeleton=True,
            skeleton_gate_max=0.02,
            default_skeleton_gate=1.0,
            anchored_residual=True,
            bias_separated_residual=True,
            residual_v2=True,
            residual_adapter_dim=16,
            residual_zero_mean_weight=1.0,
            residual_rank_weight=0.0,
        ))
        load_checkpoint(restored_v2, v2_path)
        restored_v2.eval()
        with torch.no_grad():
            restored_v2_prediction = restored_v2(
                inputs, force_gates=(0.0, 1.0)
            )[0]
        assert torch.equal(restored_v2_prediction, v2_eval)
        expect_runtime_error(
            lambda: load_checkpoint(restored_bias, v2_path)
        )
        expect_runtime_error(
            lambda: load_checkpoint(restored_v2, bias_path)
        )

        autogate = PAMFN(**model_kwargs(
            use_skeleton=True,
            skeleton_gate_max=0.02,
            motion_saliency_proxy=True,
            motion_saliency_alpha=0.02,
            anchored_residual=True,
            base_preserving_autogate=True,
            residual_adapter_dim=16,
            autogate_hidden_dim=8,
            autogate_no_regret_weight=1.0,
        ))
        load_initial_checkpoint(autogate, checkpoint)
        autogate.eval()
        with torch.no_grad():
            initial_auto, initial_info = autogate(inputs)
            initial_off = autogate(
                inputs, force_gates=(0.0, 0.0)
            )[0]
        assert torch.equal(initial_auto, base_prediction)
        assert torch.equal(initial_off, base_prediction)
        assert initial_info["motion_gate_values"].shape == (2,)
        assert initial_info["skeleton_gate_values"].shape == (2,)
        assert initial_info["base_gate_values"].shape == (2,)
        assert torch.allclose(
            initial_info["base_gate_values"],
            torch.full((2,), 0.5), atol=1e-6,
        )
        assert torch.allclose(
            initial_info["motion_gate_values"],
            torch.full((2,), 0.25), atol=1e-6,
        )
        assert torch.allclose(
            initial_info["base_gate_values"]
            + initial_info["motion_gate_values"]
            + initial_info["skeleton_gate_values"],
            torch.ones(2), atol=1e-6,
        )
        autogate.set_autogate_calibration(1.1, -0.05)
        calibrated_base = 1.1 * base_prediction - 0.05
        with torch.no_grad():
            calibrated_off = autogate(
                inputs, force_gates=(0.0, 0.0)
            )[0]
        assert torch.allclose(calibrated_off, calibrated_base, atol=1e-7)
        autogate.freeze_base()
        autogate_prefixes = autogate.trainable_adapter_prefixes()
        assert "cue_gate_head." in autogate_prefixes
        autogate_base_digest = state_dict_digest(
            autogate, autogate_prefixes
        )
        autogate_optimizer = build_optimizer(
            autogate,
            optim="Adam",
            lr=0.01,
            model_name="pamfn_final",
            trainable_only=True,
        )
        autogate.set_autogate_phase("experts")
        assert not any(
            parameter.requires_grad
            for name, parameter in autogate.named_parameters()
            if name.startswith("cue_gate_head.")
        )
        for _ in range(3):
            auto_prediction, auto_info = autogate(inputs)
            auto_loss = autogate.call_training_loss(
                auto_prediction, target, **auto_info
            )
            autogate_optimizer.zero_grad()
            auto_loss.backward()
            autogate_optimizer.step()
        expert_parameters = {
            name: parameter.detach().clone()
            for name, parameter in autogate.named_parameters()
            if name.startswith(("motion_encoder.", "skeleton_encoder.",
                                "motion_score_residual_head.",
                                "skeleton_score_residual_head."))
        }
        autogate.set_autogate_phase("gate")
        assert all(
            not parameter.requires_grad
            for name, parameter in autogate.named_parameters()
            if name in expert_parameters
        )
        assert all(
            parameter.grad is None
            for name, parameter in autogate.named_parameters()
            if name in expert_parameters
        )
        auto_prediction, auto_info = autogate(inputs)
        auto_loss = autogate.call_training_loss(
            auto_prediction, target, **auto_info
        )
        autogate_optimizer.zero_grad()
        auto_loss.backward()
        autogate_optimizer.step()
        assert all(
            torch.equal(parameter, expert_parameters[name])
            for name, parameter in autogate.named_parameters()
            if name in expert_parameters
        )
        assert state_dict_digest(
            autogate, autogate_prefixes
        ) == autogate_base_digest
        autogate.eval()
        with torch.no_grad():
            trained_auto = autogate(inputs)[0]
            trained_off = autogate(
                inputs, force_gates=(0.0, 0.0)
            )[0]
        assert torch.allclose(trained_off, calibrated_base, atol=1e-7)
        assert not torch.equal(trained_auto, calibrated_base)
        autogate_metadata = autogate.checkpoint_metadata()
        assert autogate_metadata["schema_version"] == 5
        assert autogate_metadata["base_preserving_autogate"] is True

        autogate_path = os.path.join(directory, "autogate.pth")
        _save_state_dict(autogate, autogate_path)
        restored_autogate = PAMFN(**model_kwargs(
            use_skeleton=True,
            skeleton_gate_max=0.02,
            motion_saliency_proxy=True,
            motion_saliency_alpha=0.02,
            anchored_residual=True,
            base_preserving_autogate=True,
            residual_adapter_dim=16,
            autogate_hidden_dim=8,
            autogate_no_regret_weight=1.0,
        ))
        load_checkpoint(restored_autogate, autogate_path)
        restored_autogate.eval()
        with torch.no_grad():
            restored_auto = restored_autogate(inputs)[0]
        assert torch.equal(restored_auto, trained_auto)

    print("fresh zero-initialized adapters: passed")
    print("forced (0,0) exactly equals Base: passed")
    print("Base BN/parameters frozen and deterministic: passed")
    print("optimizer contains adapter parameters only: passed")
    print("bounded signed score residuals: passed")
    print("dual residual additive identity: passed")
    print("strict anchored checkpoint reload: passed")
    print("anchored checkpoint metadata guard: passed")
    print("train-only shared bias and cue-off identity: passed")
    print("centered residual objective and per-cue mean penalty: passed")
    print("bias-separated checkpoint schema guard: passed")
    print("residual-v2 tanh-root center: passed")
    print("residual-v2 no-state train/eval parity: passed")
    print("residual-v2 schema3 checkpoint guard: passed")
    print("decoupled competitive AutoGate phases and reload: passed")
    print("anchored residual verification passed")


if __name__ == "__main__":
    main()
