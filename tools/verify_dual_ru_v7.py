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
        dual_ru_v7=True,
        dual_ru_warmup_ratio=0.5,
        dual_ru_supervision_ratio=0.7,
        dual_ru_temperature=0.1,
        dual_ru_rank_weight=0.5,
        dual_ru_phase2_lr_scale=1.0,
        dual_ru_unidentifiable_weight=0.05,
    )
    fixed.set_cue_epoch(0, 10)
    dual.set_cue_epoch(0, 10)
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

    _assert_close('v7 phase-0 score', fixed_score, dual_score)
    _assert_close('v7 phase-0 loss', fixed_loss, dual_loss)
    if not torch.equal(fixed_rng, dual_rng):
        raise AssertionError('v7 phase-0 RNG states differ')
    print('v7 phase-0 RNG state: identical')

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
        raise AssertionError(f'v7 phase-0 gradient difference {gradient_max}')
    print(f'v7 phase-0 common gradients: max difference {gradient_max:.3e}')
    return dual, model_input, label


def _endpoint_reliability_test(dual):
    dual.eval()
    qualities = torch.tensor([0.2, 0.375, 0.55, 0.725, 0.9]).view(-1, 1, 1)
    zero = torch.tensor([0.2]).view(1, 1, 1)
    expected = torch.tensor([0.0, 0.2375, 0.475, 0.7125, 0.95]).view(-1, 1, 1)
    for cue in ('motion', 'skeleton'):
        getattr(dual, f'dual_{cue}_clean_ref').fill_(0.9)
        getattr(dual, f'dual_{cue}_zero_ref').fill_(0.2)
        getattr(dual, f'dual_{cue}_clean_initialized').fill_(True)
        getattr(dual, f'dual_{cue}_zero_initialized').fill_(True)
        reliability = dual._dual_ru_calibrated_reliability(
            qualities, cue, zero
        )
        _assert_close(f'{cue} endpoint reliability', reliability, expected)
        if not torch.all(reliability[1:] > reliability[:-1]):
            raise AssertionError(f'{cue} reliability is not monotonic')
    cue = torch.ones(1, 2, 20)
    retained = [
        dual._dual_ru_deterministic_corruption(cue, severity).mean().item()
        for severity in (0.25, 0.5, 0.75)
    ]
    _assert_close(
        'deterministic corruption retained ratios',
        torch.tensor(retained), torch.tensor([0.75, 0.5, 0.25])
    )
    print('zero-clean endpoint calibration and monotonicity: correct')


def _joint_usefulness_test(dual):
    label = torch.tensor([[0.1], [0.3], [0.7], [0.9]])
    motion_prediction = torch.full((4, 1), 0.5, requires_grad=True)
    skeleton_prediction = torch.full((4, 1), 0.5, requires_grad=True)
    kwargs = {
        '_dual_joint_none_score': torch.full((4, 1), 0.5),
        '_dual_joint_motion_score': label.clone(),
        '_dual_joint_skeleton_score': torch.tensor(
            [[0.9], [0.7], [0.3], [0.1]]
        ),
        '_dual_joint_both_score': torch.tensor(
            [[0.2], [0.35], [0.65], [0.8]]
        ),
        '_dual_motion_usefulness_values': motion_prediction,
        '_dual_skeleton_usefulness_values': skeleton_prediction,
    }
    loss, info = dual._dual_ru_v7_usefulness_loss(label, kwargs)
    motion_target = info['motion_usefulness_target'].item()
    skeleton_target = info['skeleton_usefulness_target'].item()
    if motion_target <= 0.5:
        raise AssertionError('beneficial motion target is not open')
    if skeleton_target >= 0.1:
        raise AssertionError('harmful skeleton target is not closed')
    if info['motion_usefulness_rank_delta'].item() <= 0:
        raise AssertionError('motion rank benefit was not detected')
    if info['skeleton_usefulness_rank_delta'].item() >= 0:
        raise AssertionError('skeleton rank harm was not detected')
    loss.backward()
    if motion_prediction.grad.abs().sum().item() <= 0:
        raise AssertionError('motion usefulness has no supervision gradient')
    if skeleton_prediction.grad.abs().sum().item() <= 0:
        raise AssertionError('skeleton usefulness has no supervision gradient')
    print(
        'joint marginal usefulness direction: '
        f'motion {motion_target:.3f}, skeleton {skeleton_target:.3f}'
    )

    flat_motion = torch.full((4, 1), 0.5, requires_grad=True)
    flat_skeleton = torch.full((4, 1), 0.5, requires_grad=True)
    flat_score = torch.full((4, 1), 0.5)
    flat_kwargs = {
        f'_dual_joint_{name}_score': flat_score
        for name in ('none', 'motion', 'skeleton', 'both')
    }
    flat_kwargs.update({
        '_dual_motion_usefulness_values': flat_motion,
        '_dual_skeleton_usefulness_values': flat_skeleton,
    })
    _, flat_info = dual._dual_ru_v7_usefulness_loss(label, flat_kwargs)
    for cue in ('motion', 'skeleton'):
        target = flat_info[f'_dual_{cue}_usefulness_target_values']
        weight = flat_info[f'_dual_{cue}_usefulness_target_weights']
        _assert_close(
            f'{cue} flat target', target, torch.zeros_like(target)
        )
        _assert_close(
            f'{cue} flat confidence', weight,
            torch.full_like(weight, 0.05)
        )
    print('unidentifiable joint cue closes gently: correct')

    dual.dual_usefulness_error_scale_initialized.fill_(False)
    dual.dual_usefulness_rank_scale_initialized.fill_(False)
    conflict_label = torch.tensor([[0.1], [0.2], [0.8], [0.9]])
    ranked_score = torch.tensor([[0.17], [0.27], [0.73], [0.83]])
    lower_mse_wrong_rank = torch.tensor(
        [[0.16], [0.14], [0.86], [0.84]]
    )
    conflict_kwargs = {
        '_dual_joint_none_score': ranked_score,
        '_dual_joint_motion_score': lower_mse_wrong_rank,
        '_dual_joint_skeleton_score': ranked_score,
        '_dual_joint_both_score': lower_mse_wrong_rank,
        '_dual_motion_usefulness_values': torch.full((4, 1), 0.5),
        '_dual_skeleton_usefulness_values': torch.full((4, 1), 0.5),
    }
    _, conflict_info = dual._dual_ru_v7_usefulness_loss(
        conflict_label, conflict_kwargs
    )
    if conflict_info['motion_usefulness_mse_benefit'].item() <= 0:
        raise AssertionError('rank-conflict fixture does not improve MSE')
    if conflict_info['motion_usefulness_rank_delta'].item() >= 0:
        raise AssertionError('rank-conflict fixture does not harm ranking')
    if conflict_info['motion_usefulness_target'].item() >= 0.1:
        raise AssertionError('ranking harm was ignored by usefulness target')
    print('MSE improvement with ranking harm closes the cue: correct')


def _frozen_phase2_test(dual, model_input, label):
    expected = dual.fc[1].weight.detach().clone()
    if not dual.consider_dual_teacher_snapshot(
            epoch=4, loss=0.5, coefficient=80.0, score_std=0.1):
        raise AssertionError('healthy v7 teacher snapshot was not selected')
    with torch.no_grad():
        dual.fc[1].weight.add_(1.0)
    dual.set_cue_epoch(5, 10)
    _assert_close('v7 stable teacher restore', dual.fc[1].weight, expected)
    dual.set_cue_epoch(7, 10)
    dual.train()
    dual.prepare_dual_ru_train_mode()
    if dual.fc.training or dual.skeleton_encoder.training:
        raise AssertionError('v7 phase-2 scoring modules are not in eval mode')
    if not dual.dual_motion_usefulness_head.training:
        raise AssertionError('v7 phase-2 usefulness head is not trainable')
    for name, parameter in dual.named_parameters():
        if dual._is_dual_teacher_core_name(name) and parameter.requires_grad:
            raise AssertionError(f'v7 phase-2 core parameter is trainable: {name}')

    counterfactual = dual.build_dual_ru_counterfactual_targets(
        model_input, label
    )
    expected_keys = {
        f'_dual_joint_{name}_score'
        for name in ('none', 'motion', 'skeleton', 'both')
    }
    if set(counterfactual) != expected_keys:
        raise AssertionError('v7 counterfactual states are incorrect')
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
        raise AssertionError(f'v7 phase-2 core gradient is {core_gradient}')
    if usefulness_gradient <= 0:
        raise AssertionError('v7 phase-2 usefulness gradient is zero')
    dual.eval()
    with torch.no_grad():
        _, eval_info = dual(model_input)
    for cue in ('motion', 'skeleton'):
        for suffix in ('25', '50', '75'):
            if f'{cue}_reliability_corrupt{suffix}' not in eval_info:
                raise AssertionError('v7 corruption diagnostic is missing')
    print(
        'v7 phase-2 frozen core and usefulness gradient: '
        f'{core_gradient:.3e}, {usefulness_gradient:.3e}'
    )


def _policy_configuration_test():
    captured = {}

    def capture_go(*args, **kwargs):
        captured['model'] = args[0]
        captured['scheduler'] = args[2]
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
            '--multi_modality', '--cue_policy', 'dual_ru_v7'
        ]
        args = main.parser_args()
        if not args.cue_quality_supervision or args.auto_gate:
            raise AssertionError('dual_ru_v7 gate policy is incorrect')
        if args.cue_corruption_max != 1.0:
            raise AssertionError('dual_ru_v7 corruption endpoint is disabled')

        runpy.run_path(os.path.join(ROOT, 'main.py'), run_name='__main__')
        model_conf = captured['model']['model_conf']
        if not model_conf['dual_ru_v7']:
            raise AssertionError('main does not build a v7 model')
        if model_conf['dual_ru_rank_weight'] != 0.5:
            raise AssertionError('v7 joint rank utility is disabled')
        if model_conf['dual_ru_phase2_lr_scale'] != 1.0:
            raise AssertionError('v7 usefulness-head LR scale is incorrect')
        if not captured['dataset']['build_train_probe']:
            raise AssertionError('v7 stable teacher probe is disabled')
        if captured['epochs'] != 120:
            raise AssertionError('v7 TES epoch schedule is incorrect')
    finally:
        sys.argv = original_argv
    print('v7 policy configuration: correct')


if __name__ == '__main__':
    dual_model, sample_input, sample_label = _phase0_parity_test()
    _endpoint_reliability_test(dual_model)
    _joint_usefulness_test(dual_model)
    _frozen_phase2_test(dual_model, sample_input, sample_label)
    _pipeline_paired_evaluation_test()
    _policy_configuration_test()
    print('dual_ru_v7 verification passed')
