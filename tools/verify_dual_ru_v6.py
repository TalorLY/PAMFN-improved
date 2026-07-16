import os
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
        dual_ru_v6=True,
        dual_ru_warmup_ratio=0.5,
        dual_ru_supervision_ratio=0.7,
        dual_ru_unidentifiable_weight=0.05,
        dual_ru_output_anchor_weight=0.1,
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

    _assert_close('v6 phase-0 score', fixed_score, dual_score)
    _assert_close('v6 phase-0 loss', fixed_loss, dual_loss)
    if not torch.equal(fixed_rng, dual_rng):
        raise AssertionError('v6 phase-0 RNG states differ')
    print('v6 phase-0 RNG state: identical')

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
        raise AssertionError(f'v6 phase-0 gradient difference {gradient_max}')
    print(f'v6 phase-0 common gradients: max difference {gradient_max:.3e}')
    return dual, model_input, label


def _teacher_snapshot_test(dual):
    parameter = dual.fc[1].weight
    expected = parameter.detach().clone()
    selected = dual.consider_dual_teacher_snapshot(
        epoch=4, loss=0.5, coefficient=80.0, score_std=0.1
    )
    if not selected:
        raise AssertionError('healthy teacher snapshot was not selected')
    with torch.no_grad():
        parameter.add_(1.0)
    if dual.consider_dual_teacher_snapshot(
            epoch=9, loss=0.4, coefficient=90.0, score_std=0.001):
        raise AssertionError('collapsed teacher snapshot was selected')
    dual.set_cue_epoch(5, 10)
    _assert_close('stable teacher restore', parameter, expected)
    info = dual.dual_teacher_probe_info()
    if not info['restored'] or info['epoch'] != 4:
        raise AssertionError('stable teacher metadata is incorrect')
    if next(dual.fc.parameters()).requires_grad:
        raise AssertionError('restored phase-1 teacher must be frozen')
    print('stable teacher selection and restore: correct')


def _zero_anchor_test(dual):
    zero_motion = torch.zeros(2, 256, 35)
    zero_skeleton = torch.zeros(2, 132, 70)
    motion_features = dual.motion_quality_features(zero_motion)
    skeleton_features = dual.skeleton_quality_features(zero_skeleton)
    _assert_close(
        'zero motion quality features',
        motion_features,
        torch.zeros_like(motion_features)
    )
    _assert_close(
        'zero skeleton quality features',
        skeleton_features,
        torch.zeros_like(skeleton_features)
    )
    zero_prediction = dual.skeleton_quality_head(skeleton_features)
    zero_loss = zero_prediction.square().mean()
    gradient = torch.autograd.grad(
        zero_loss, dual.skeleton_quality_head[2].bias,
        retain_graph=False
    )[0]
    if gradient.item() <= 0:
        raise AssertionError('zero anchor does not push reliability downward')
    print(f'zero reliability anchor gradient: {gradient.item():.3e}')


def _unidentifiable_target_test(dual):
    label = torch.tensor([[0.25], [0.75]])
    grid = torch.tensor(dual.dual_ru_gate_grid)
    prediction_motion = torch.full((2, 1), 0.5, requires_grad=True)
    prediction_skeleton = torch.full((2, 1), 0.5, requires_grad=True)
    flat_scores = torch.full((grid.numel(), 2, 1), 0.5)
    kwargs = {
        '_dual_gate_grid': grid,
        '_dual_motion_grid_scores': flat_scores,
        '_dual_skeleton_grid_scores': flat_scores,
        '_dual_motion_usefulness_values': prediction_motion,
        '_dual_skeleton_usefulness_values': prediction_skeleton,
    }
    loss, info = dual._dual_ru_v5_usefulness_loss(label, kwargs)
    for cue in ('motion', 'skeleton'):
        target = info[f'_dual_{cue}_usefulness_target_values']
        weight = info[f'_dual_{cue}_usefulness_target_weights']
        _assert_close(
            f'{cue} unidentifiable target', target, torch.zeros_like(target)
        )
        _assert_close(
            f'{cue} unidentifiable weight',
            weight,
            torch.full_like(weight, 0.05)
        )
    loss.backward()
    if prediction_motion.grad.abs().sum().item() <= 0:
        raise AssertionError('unidentifiable cue has no closing gradient')
    print('unidentifiable cue closing target: correct')


def _phase2_output_anchor_test(dual, model_input, label):
    dual.set_cue_epoch(7, 10)
    dual.train()
    with torch.no_grad():
        dual.dual_motion_usefulness_head[2].bias.add_(0.5)
    dual.zero_grad()
    score, info = dual(model_input)
    loss = dual.call_loss(score, label, **info)
    loss.backward()
    anchor_loss = dual.last_loss_info.get('dual_output_anchor_loss')
    if anchor_loss is None or anchor_loss.item() <= 0:
        raise AssertionError('phase-2 output anchor is inactive')
    gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in dual.dual_ru_usefulness_parameters()
        if parameter.grad is not None
    )
    if gradient <= 0:
        raise AssertionError('phase-2 usefulness gradient is zero')
    print(
        f'phase-2 output anchor loss: {anchor_loss.item():.3e}, '
        f'usefulness gradient: {gradient:.3e}'
    )


def _policy_configuration_test():
    pipeline_stub = types.ModuleType('pipeline')
    pipeline_stub.go = lambda *args, **kwargs: None
    pipeline_stub.log_info = lambda *args, **kwargs: None
    pipeline_stub.test_model = lambda *args, **kwargs: None
    sys.modules['pipeline'] = pipeline_stub
    import main

    original_argv = sys.argv
    try:
        sys.argv = [
            'main.py', '--gpu', '0', '--feats', '2', '--action', 'TES',
            '--multi_modality', '--cue_policy', 'dual_ru_v6'
        ]
        dual_args = main.parser_args()
        if not dual_args.cue_quality_supervision:
            raise AssertionError('dual_ru_v6 quality supervision is disabled')
        if dual_args.cue_corruption_max != 1.0 or dual_args.auto_gate:
            raise AssertionError('dual_ru_v6 policy defaults are incorrect')

        sys.argv = [
            'main.py', '--gpu', '0', '--feats', '2', '--action', 'TES',
            '--multi_modality', '--cue_policy', 'fixed_cue_v3'
        ]
        fixed_args = main.parser_args()
        if not fixed_args.auto_gate or not fixed_args.cue_quality_aux_only:
            raise AssertionError('fixed_cue_v3 is not a fixed auxiliary cue policy')
        if fixed_args.auto_gate_l1 != 0.0 or fixed_args.cue_corruption_max != 1.0:
            raise AssertionError('fixed_cue_v3 parity defaults are incorrect')
    finally:
        sys.argv = original_argv
    print('v6/fixed-v3 policy configuration: correct')


if __name__ == '__main__':
    dual_model, sample_input, sample_label = _phase0_parity_test()
    _zero_anchor_test(dual_model)
    _unidentifiable_target_test(dual_model)
    _teacher_snapshot_test(dual_model)
    _phase2_output_anchor_test(dual_model, sample_input, sample_label)
    _pipeline_paired_evaluation_test()
    _policy_configuration_test()
    print('dual_ru_v6 verification passed')
