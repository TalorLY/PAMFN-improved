import os
import runpy
import sys
import types

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from verify_dual_ru_v5 import (  # noqa: E402
    _assert_close,
    _common_kwargs,
    _load_full_model,
    _pipeline_paired_evaluation_test,
)
from verify_dual_ru_v7 import _endpoint_reliability_test  # noqa: E402


def _build_models():
    common = _common_kwargs()
    common.update({
        'cue_quality_zero_anchor': True,
        'cue_quality_zero_weight': 1.0,
    })
    checkpoint_dir = os.path.join(ROOT, 'pretrained_models', 'feats2')
    torch.manual_seed(0)
    fixed = _load_full_model(
        checkpoint_dir,
        **common,
        auto_gate=True,
        auto_gate_v2=True,
        auto_gate_skeleton_max=0.01,
        auto_gate_l1=0.0,
        cue_quality_aux_only=True,
    )
    torch.manual_seed(0)
    dual = _load_full_model(
        checkpoint_dir,
        **common,
        auto_gate=False,
        auto_gate_v2=False,
        auto_gate_skeleton_max=0.02,
        auto_gate_l1=0.0,
        cue_quality_aux_only=False,
        dual_ru_gate=True,
        dual_ru_v8=True,
        dual_ru_warmup_ratio=0.7,
        dual_ru_supervision_ratio=0.85,
        dual_ru_temperature=0.1,
        dual_ru_rank_weight=0.5,
        dual_ru_phase2_lr_scale=1.0,
        dual_ru_unidentifiable_weight=0.05,
        dual_ru_benefit_scale_min=1e-5,
        dual_ru_teacher_min_probe_coef=97.0,
        dual_ru_teacher_ema_decay=0.5,
    )
    fixed.set_cue_epoch(0, 20)
    dual.set_cue_epoch(0, 20)
    return fixed, dual


def _sample_batch():
    generator = torch.Generator().manual_seed(7)
    model_input = {
        'V': torch.randn(2, 70, 1024, generator=generator),
        'F': torch.randn(2, 70, 1024, generator=generator),
        'A': torch.randn(2, 70, 768, generator=generator),
        'S': torch.randn(2, 70, 132, generator=generator),
        'SQ': torch.ones(2),
    }
    label = torch.tensor([[0.35], [0.75]])
    return model_input, label


def _phase0_parity_test():
    fixed, dual = _build_models()
    model_input, label = _sample_batch()
    fixed.train()
    dual.train()

    torch.manual_seed(123)
    fixed_score, fixed_info = fixed(model_input)
    fixed_rng = torch.get_rng_state().clone()
    fixed_loss = fixed.call_loss(fixed_score, label, **fixed_info)

    torch.manual_seed(123)
    dual_score, dual_info = dual(model_input)
    dual_rng = torch.get_rng_state().clone()
    dual_loss = dual.call_loss(dual_score, label, **dual_info)

    _assert_close('v8 phase-0 score', fixed_score, dual_score)
    _assert_close('v8 phase-0 loss', fixed_loss, dual_loss)
    if not torch.equal(fixed_rng, dual_rng):
        raise AssertionError('v8 phase-0 RNG states differ')
    print('v8 phase-0 RNG state: identical')

    fixed.zero_grad()
    dual.zero_grad()
    fixed_loss.backward()
    dual_loss.backward()
    fixed_parameters = dict(fixed.named_parameters())
    dual_parameters = dict(dual.named_parameters())
    gradient_max = 0.0
    for name in sorted(set(fixed_parameters) & set(dual_parameters)):
        fixed_grad = fixed_parameters[name].grad
        dual_grad = dual_parameters[name].grad
        if fixed_grad is None and dual_grad is None:
            continue
        if fixed_grad is None or dual_grad is None:
            raise AssertionError(f'gradient presence differs for {name}')
        gradient_max = max(
            gradient_max,
            (fixed_grad - dual_grad).abs().max().item()
        )
    if gradient_max > 1e-6:
        raise AssertionError(f'v8 phase-0 gradient difference {gradient_max}')
    print(f'v8 phase-0 common gradients: max difference {gradient_max:.3e}')
    return dual, model_input, label


def _reset_benefit_scales(dual):
    for cue in ('motion', 'skeleton'):
        for objective in ('mse', 'rank'):
            getattr(
                dual,
                f'dual_{cue}_{objective}_benefit_scale_initialized'
            ).fill_(False)


def _baseline_centered_usefulness_test(dual):
    label = torch.tensor([[0.1], [0.3], [0.7], [0.9]])
    flat_score = torch.full((4, 1), 0.5)
    flat_kwargs = {
        f'_dual_joint_{name}_score': flat_score
        for name in ('none', 'motion', 'skeleton', 'both')
    }
    flat_kwargs.update({
        '_dual_motion_usefulness_values': torch.full((4, 1), 0.5),
        '_dual_skeleton_usefulness_values': torch.full((4, 1), 0.5),
    })
    _reset_benefit_scales(dual)
    _, flat_info = dual._dual_ru_v8_usefulness_loss(label, flat_kwargs)
    _assert_close(
        'v8 neutral motion target',
        flat_info['_dual_motion_usefulness_target_values'],
        torch.ones(4)
    )
    _assert_close(
        'v8 neutral skeleton target',
        flat_info['_dual_skeleton_usefulness_target_values'],
        torch.full((4,), 0.5)
    )
    for cue in ('motion', 'skeleton'):
        _assert_close(
            f'v8 neutral {cue} confidence',
            flat_info[f'_dual_{cue}_usefulness_target_weights'],
            torch.full((4,), 0.05)
        )
    print('v8 unidentifiable cues preserve fixed defaults: correct')

    motion_prediction = torch.full((4, 1), 0.5, requires_grad=True)
    skeleton_prediction = torch.full((4, 1), 0.5, requires_grad=True)
    kwargs = {
        '_dual_joint_none_score': torch.full((4, 1), 0.5),
        '_dual_joint_motion_score': label.clone(),
        '_dual_joint_skeleton_score': torch.tensor(
            [[0.9], [0.7], [0.3], [0.1]]
        ),
        '_dual_joint_both_score': label.clone(),
        '_dual_motion_usefulness_values': motion_prediction,
        '_dual_skeleton_usefulness_values': skeleton_prediction,
    }
    _reset_benefit_scales(dual)
    loss, info = dual._dual_ru_v8_usefulness_loss(label, kwargs)
    if info['motion_usefulness_target'].item() < 0.99:
        raise AssertionError('beneficial motion does not retain its default')
    if info['skeleton_usefulness_target'].item() >= 0.5:
        raise AssertionError('harmful skeleton does not close below default')
    loss.backward()
    if motion_prediction.grad.abs().sum().item() <= 0:
        raise AssertionError('motion usefulness has no supervision gradient')
    if skeleton_prediction.grad.abs().sum().item() <= 0:
        raise AssertionError('skeleton usefulness has no supervision gradient')
    print(
        'v8 baseline-centered directions: '
        f"motion {info['motion_usefulness_target'].item():.3f}, "
        f"skeleton {info['skeleton_usefulness_target'].item():.3f}"
    )


def _small_effect_and_scale_test(dual):
    label = torch.tensor([[0.1], [0.3], [0.7], [0.9]])
    base = torch.tensor([[0.2], [0.4], [0.6], [0.8]])
    away = base + torch.tensor([[0.001], [0.001], [-0.001], [-0.001]])
    kwargs = {
        '_dual_joint_none_score': base,
        '_dual_joint_motion_score': away,
        '_dual_joint_skeleton_score': base,
        '_dual_joint_both_score': away,
        '_dual_motion_usefulness_values': torch.full((4, 1), 0.95),
        '_dual_skeleton_usefulness_values': torch.full((4, 1), 0.5),
    }
    _reset_benefit_scales(dual)
    _, info = dual._dual_ru_v8_usefulness_loss(label, kwargs)
    if info['motion_usefulness_target'].item() >= 0.5:
        raise AssertionError('small harmful motion effect was treated as neutral')
    raw_benefit = abs(info['motion_usefulness_mse_benefit'].item())
    benefit_scale = info['motion_usefulness_mse_scale'].item()
    if abs(raw_benefit - benefit_scale) > 1e-5:
        raise AssertionError('motion benefit scale is not effect-relative')
    if info['dual_usefulness_rank_scale'].item() >= 0.1:
        raise AssertionError('rank scale still tracks base rank error')
    print(
        'v8 small motion effect remains identifiable: '
        f"target {info['motion_usefulness_target'].item():.3f}"
    )


def _local_counterfactual_probe_test(dual, model_input, label):
    calls = []
    original_forward = dual._forward_dual_ru

    def fake_forward(self, input_feats, force_gates=None):
        calls.append(tuple(float(value) for value in force_gates))
        value = sum(force_gates)
        return label.new_full(label.shape, value), {}

    dual._forward_dual_ru = types.MethodType(fake_forward, dual)
    try:
        dual.set_cue_epoch(14, 20)
        dual.train()
        targets = dual.build_dual_ru_counterfactual_targets(
            model_input, label
        )
    finally:
        dual._forward_dual_ru = original_forward
    expected_calls = {
        (0.5, 0.0), (1.0, 0.0), (0.5, 0.5), (1.0, 0.5)
    }
    if set(calls) != expected_calls:
        raise AssertionError(f'v8 local counterfactual probes differ: {calls}')
    if set(targets) != {
            f'_dual_joint_{name}_score'
            for name in ('none', 'motion', 'skeleton', 'both')}:
        raise AssertionError('v8 counterfactual target keys are incorrect')
    print('v8 local four-state counterfactual probes: correct')


def _teacher_ema_and_frozen_phase2_test(dual, model_input, label):
    dual.set_cue_epoch(0, 20)
    first = dual.fc[1].weight.detach().clone()
    if dual.consider_dual_teacher_snapshot(
            epoch=4, loss=0.6, coefficient=96.0, score_std=0.1):
        raise AssertionError('underfit v8 teacher snapshot was accepted')
    if not dual.consider_dual_teacher_snapshot(
            epoch=5, loss=0.5, coefficient=98.0, score_std=0.1):
        raise AssertionError('healthy v8 teacher snapshot was rejected')
    with torch.no_grad():
        dual.fc[1].weight.add_(2.0)
    second = dual.fc[1].weight.detach().clone()
    if not dual.consider_dual_teacher_snapshot(
            epoch=10, loss=0.4, coefficient=98.5, score_std=0.1):
        raise AssertionError('better v8 teacher snapshot was rejected')
    expected = 0.5 * first + 0.5 * second
    with torch.no_grad():
        dual.fc[1].weight.add_(3.0)
    dual.set_cue_epoch(14, 20)
    _assert_close('v8 teacher EMA restore', dual.fc[1].weight, expected)
    teacher_info = dual.dual_teacher_probe_info()
    if teacher_info['ema_updates'] != 2:
        raise AssertionError('v8 teacher EMA update count is incorrect')

    dual.set_cue_epoch(17, 20)
    dual.train()
    dual.prepare_dual_ru_train_mode()
    for name, parameter in dual.named_parameters():
        if dual._is_dual_teacher_core_name(name) and parameter.requires_grad:
            raise AssertionError(f'v8 phase-2 core is trainable: {name}')
    counterfactual = dual.build_dual_ru_counterfactual_targets(
        model_input, label
    )
    dual.prepare_dual_ru_train_mode()
    dual.zero_grad()
    score, info = dual(model_input)
    info.update(counterfactual)
    loss = dual.call_loss(score, label, **info)
    loss.backward()
    core_gradient = sum(
        parameter.grad.abs().sum().item()
        for name, parameter in dual.named_parameters()
        if dual._is_dual_teacher_core_name(name) and parameter.grad is not None
    )
    usefulness_gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in dual.dual_ru_usefulness_parameters()
        if parameter.grad is not None
    )
    if core_gradient != 0:
        raise AssertionError(f'v8 phase-2 core gradient is {core_gradient}')
    if usefulness_gradient <= 0:
        raise AssertionError('v8 phase-2 usefulness gradient is zero')
    print(
        'v8 EMA teacher and frozen phase-2 core: '
        f'{core_gradient:.3e}, {usefulness_gradient:.3e}'
    )


def _policy_configuration_test():
    captured = {}

    def capture_go(*args, **kwargs):
        captured['model'] = args[0]
        captured['dataset'] = args[3]
        captured['epochs'] = args[4]
        return 0.0, 0.0, 0.0, 0.0, 0.0

    pipeline_stub = types.ModuleType('pipeline')
    pipeline_stub.go = capture_go
    pipeline_stub.log_info = lambda *args, **kwargs: None
    pipeline_stub.test_model = lambda *args, **kwargs: None
    sys.modules['pipeline'] = pipeline_stub
    import main

    original_argv = sys.argv
    try:
        sys.argv = [
            'main.py', '--gpu', '0', '--feats', '2', '--action', 'TES',
            '--multi_modality', '--cue_policy', 'dual_ru_v8'
        ]
        args = main.parser_args()
        if not args.cue_quality_supervision or args.auto_gate:
            raise AssertionError('dual_ru_v8 gate policy is incorrect')
        if args.cue_corruption_max != 1.0:
            raise AssertionError('dual_ru_v8 corruption endpoint is disabled')

        runpy.run_path(os.path.join(ROOT, 'main.py'), run_name='__main__')
        model_conf = captured['model']['model_conf']
        if not model_conf['dual_ru_v8']:
            raise AssertionError('main does not build a v8 model')
        if model_conf['dual_ru_warmup_ratio'] != 0.7:
            raise AssertionError('v8 fixed scorer ratio is incorrect')
        if model_conf['dual_ru_supervision_ratio'] != 0.85:
            raise AssertionError('v8 supervision ratio is incorrect')
        if model_conf['dual_ru_rank_weight'] != 0.5:
            raise AssertionError('v8 rank utility is disabled')
        if model_conf['dual_ru_teacher_min_probe_coef'] != 97.0:
            raise AssertionError('v8 teacher health threshold is incorrect')
        if model_conf['dual_ru_teacher_ema_decay'] != 0.5:
            raise AssertionError('v8 teacher EMA decay is incorrect')
        if not captured['dataset']['build_train_probe']:
            raise AssertionError('v8 stable teacher probe is disabled')
        if captured['epochs'] != 172:
            raise AssertionError('v8 TES scorer budget is not preserved')
    finally:
        sys.argv = original_argv
    print('v8 policy configuration and scorer budget: correct')


if __name__ == '__main__':
    dual_model, sample_input, sample_label = _phase0_parity_test()
    _endpoint_reliability_test(dual_model)
    _baseline_centered_usefulness_test(dual_model)
    _small_effect_and_scale_test(dual_model)
    _local_counterfactual_probe_test(
        dual_model, sample_input, sample_label
    )
    _teacher_ema_and_frozen_phase2_test(
        dual_model, sample_input, sample_label
    )
    _pipeline_paired_evaluation_test()
    _policy_configuration_test()
    print('dual_ru_v8 verification passed')
