import inspect
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
        dual_ru_v10=True,
        dual_ru_warmup_ratio=0.7,
        dual_ru_supervision_ratio=0.85,
        dual_ru_temperature=0.1,
        dual_ru_rank_weight=0.0,
        dual_ru_phase2_lr_scale=1.0,
        dual_ru_teacher_min_probe_coef=97.0,
        dual_ru_teacher_ema_decay=0.5,
        dual_ru_no_regret_weight=1.0,
        dual_ru_joint_rank_weight=0.2,
        dual_ru_joint_sample_weight=0.2,
        dual_ru_joint_min_gain=0.02,
        dual_ru_batch_rank_regret_weight=0.1,
        dual_ru_gate_grid=(0.0, 0.5, 1.0),
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

    _assert_close('v10 phase-0 score', fixed_score, dual_score)
    _assert_close('v10 phase-0 loss', fixed_loss, dual_loss)
    if not torch.equal(fixed_rng, dual_rng):
        raise AssertionError('v10 phase-0 RNG states differ')
    print('v10 phase-0 RNG state: identical')

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
        raise AssertionError(f'v10 phase-0 gradient difference {gradient_max}')
    print(f'v10 phase-0 common gradients: max difference {gradient_max:.3e}')
    return dual, model_input, label


def _endpoint_reliability_test(dual):
    dual.eval()
    qualities = torch.tensor([0.2, 0.375, 0.55, 0.725, 0.9]).view(-1, 1, 1)
    zero = torch.tensor([0.2]).view(1, 1, 1)
    expected = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).view(-1, 1, 1)
    for cue in ('motion', 'skeleton'):
        getattr(dual, f'dual_{cue}_clean_ref').fill_(0.9)
        getattr(dual, f'dual_{cue}_zero_ref').fill_(0.2)
        getattr(dual, f'dual_{cue}_clean_initialized').fill_(True)
        getattr(dual, f'dual_{cue}_zero_initialized').fill_(True)
        reliability = dual._dual_ru_calibrated_reliability(
            qualities, cue, zero
        )
        _assert_close(f'v10 {cue} endpoint reliability', reliability, expected)
        if not torch.all(reliability[1:] > reliability[:-1]):
            raise AssertionError(f'v10 {cue} reliability is not monotonic')
    print('v10 zero-clean reliability endpoints: correct')


def _joint_target_test(dual):
    grid = torch.tensor([0.0, 0.5, 1.0])
    gate_pairs = torch.cartesian_prod(grid, grid)
    label = torch.tensor([[0.15], [0.35], [0.65], [0.85]])
    joint_scores = []
    for motion_gate, skeleton_gate in gate_pairs:
        error = 0.4 * motion_gate + 0.4 * (1 - skeleton_gate)
        joint_scores.append(label + error)
    motion_prediction = torch.full((4, 1), 0.9, requires_grad=True)
    skeleton_prediction = torch.full((4, 1), 0.5, requires_grad=True)
    kwargs = {
        '_dual_joint_gate_pairs': gate_pairs,
        '_dual_joint_grid_scores': torch.stack(joint_scores),
        '_dual_motion_usefulness_values': motion_prediction,
        '_dual_skeleton_usefulness_values': skeleton_prediction,
    }
    loss, info = dual._dual_ru_v10_usefulness_loss(label, kwargs)
    motion_target = info['motion_usefulness_target'].item()
    skeleton_target = info['skeleton_usefulness_target'].item()
    if motion_target >= 0.25 or skeleton_target <= 0.75:
        raise AssertionError(
            'v10 joint target did not select low motion and high skeleton: '
            f'{motion_target:.3f}, {skeleton_target:.3f}'
        )
    _assert_close(
        'v10 joint best pair',
        torch.stack([
            info['dual_joint_best_motion_gate'],
            info['dual_joint_best_skeleton_gate']
        ]),
        torch.tensor([0.0, 1.0])
    )
    correction = info['dual_joint_sample_correction'].item()
    if correction > dual.dual_ru_joint_sample_weight + 1e-6:
        raise AssertionError('v10 sample correction exceeded its bound')
    loss.backward()
    if motion_prediction.grad.abs().sum().item() <= 0:
        raise AssertionError('v10 motion usefulness has no target gradient')
    if skeleton_prediction.grad.abs().sum().item() <= 0:
        raise AssertionError('v10 skeleton usefulness has no target gradient')

    flat_score = torch.full((9, 4, 1), 0.5)
    flat_kwargs = {
        '_dual_joint_gate_pairs': gate_pairs,
        '_dual_joint_grid_scores': flat_score,
        '_dual_motion_usefulness_values': torch.full((4, 1), 0.5),
        '_dual_skeleton_usefulness_values': torch.full((4, 1), 0.5),
    }
    _, flat_info = dual._dual_ru_v10_usefulness_loss(label, flat_kwargs)
    _assert_close(
        'v10 flat motion target',
        flat_info['_dual_motion_usefulness_target_values'],
        torch.ones(4)
    )
    _assert_close(
        'v10 flat skeleton target',
        flat_info['_dual_skeleton_usefulness_target_values'],
        torch.full((4,), 0.5)
    )
    _assert_close(
        'v10 flat global confidence',
        flat_info['dual_joint_global_confidence'],
        torch.tensor(0.0)
    )
    print('v10 joint global target, bounded correction, and safe tie: correct')


def _counterfactual_schedule_test(dual, model_input, label):
    calls = []
    original_forward = dual._forward_dual_ru

    def fake_forward(self, input_feats, force_gates=None, corruption=None):
        calls.append(tuple(float(value) for value in force_gates))
        return label.new_full(label.shape, sum(force_gates)), {}

    dual._forward_dual_ru = types.MethodType(fake_forward, dual)
    try:
        dual.set_cue_epoch(14, 20)
        dual.train()
        phase1_targets = dual.build_dual_ru_counterfactual_targets(
            model_input, label
        )
        phase1_calls = list(calls)
        calls.clear()
        dual.set_cue_epoch(19, 20)
        dual.train()
        phase2_targets = dual.build_dual_ru_counterfactual_targets(
            model_input, label
        )
        phase2_calls = list(calls)
    finally:
        dual._forward_dual_ru = original_forward

    expected_grid = {
        (motion, skeleton)
        for motion in (0.0, 0.5, 1.0)
        for skeleton in (0.0, 0.5, 1.0)
    }
    if set(phase1_calls) != expected_grid or len(phase1_calls) != 9:
        raise AssertionError(f'v10 phase-1 probes differ: {phase1_calls}')
    expected_phase1_keys = {
        '_dual_fixed_score', '_dual_joint_gate_pairs',
        '_dual_joint_grid_scores'
    }
    if set(phase1_targets) != expected_phase1_keys:
        raise AssertionError('v10 phase-1 target keys are incorrect')
    if phase2_calls != [(1.0, 0.5)]:
        raise AssertionError(f'v10 phase-2 replay differs: {phase2_calls}')
    if set(phase2_targets) != {'_dual_fixed_score'}:
        raise AssertionError('v10 phase-2 retained pseudo-label probes')
    print('v10 phase-1 joint grid and phase-2 fixed replay: correct')


def _phase2_no_regret_test(dual):
    dual.set_cue_epoch(19, 20)
    for name, parameter in dual.named_parameters():
        if dual._is_dual_teacher_core_name(name) and parameter.requires_grad:
            raise AssertionError(f'v10 phase-2 core is trainable: {name}')

    prediction = torch.tensor([[0.6], [0.4]], requires_grad=True)
    label = torch.tensor([[0.3], [0.7]])
    loss = dual.call_loss(
        prediction, label, _dual_fixed_score=label.clone()
    )
    if dual.last_loss_info['dual_no_regret_mse'].item() <= 0:
        raise AssertionError('v10 batch no-regret MSE penalty is inactive')
    if any('usefulness_target' in key for key in dual.last_loss_info):
        raise AssertionError('v10 phase-2 still applies pseudo-label targets')
    loss.backward()
    if prediction.grad.abs().sum().item() <= 0:
        raise AssertionError('v10 phase-2 score path has no gradient')
    print('v10 phase-2 batch no-regret without pseudo-labels: correct')


def _partial_force_test(dual, model_input):
    dual.set_cue_epoch(19, 20)
    dual.eval()
    with torch.no_grad():
        _, adaptive = dual(model_input)
        _, motion_only = dual(
            model_input, dual_force_gates=(None, 0.5)
        )
        _, skeleton_only = dual(
            model_input, dual_force_gates=(1.0, None)
        )
    _assert_close(
        'v10 partial motion adaptive gate',
        motion_only['_dual_motion_gate_values'],
        adaptive['_dual_motion_gate_values']
    )
    _assert_close(
        'v10 partial fixed skeleton gate',
        motion_only['_dual_skeleton_gate_values'],
        torch.full_like(motion_only['_dual_skeleton_gate_values'], 0.5)
    )
    _assert_close(
        'v10 partial fixed motion gate',
        skeleton_only['_dual_motion_gate_values'],
        torch.ones_like(skeleton_only['_dual_motion_gate_values'])
    )
    _assert_close(
        'v10 partial skeleton adaptive gate',
        skeleton_only['_dual_skeleton_gate_values'],
        adaptive['_dual_skeleton_gate_values']
    )
    print('v10 independent motion/skeleton force paths: correct')


def _final_diagnostics_test():
    import pipeline

    source = inspect.getsource(pipeline.train_loop)
    for token in (
            'Final clean joint gate grid',
            'Final adaptive component ablation',
            "('motion_adaptive', (None, 0.5))",
            "('skeleton_adaptive', (1.0, None))",
            'cue_fixed_coef'):
        if token not in source:
            raise AssertionError(f'v10 final diagnostics lack {token}')
    print('v10 joint-grid and isolated-cue final diagnostics: correct')


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
            '--multi_modality', '--cue_policy', 'dual_ru_v10'
        ]
        args = main.parser_args()
        if not args.cue_quality_supervision or args.auto_gate:
            raise AssertionError('dual_ru_v10 gate policy is incorrect')
        if args.cue_corruption_max != 1.0:
            raise AssertionError('dual_ru_v10 corruption endpoint is disabled')

        runpy.run_path(os.path.join(ROOT, 'main.py'), run_name='__main__')
        model_conf = captured['model']['model_conf']
        checks = {
            'dual_ru_v10': True,
            'dual_ru_warmup_ratio': 0.7,
            'dual_ru_supervision_ratio': 0.85,
            'dual_ru_usefulness_loss_weight': 1.0,
            'dual_ru_temperature': 0.1,
            'dual_ru_joint_rank_weight': 0.2,
            'dual_ru_joint_sample_weight': 0.2,
            'dual_ru_joint_min_gain': 0.02,
            'dual_ru_batch_rank_regret_weight': 0.1,
            'dual_ru_gate_grid': (0.0, 0.5, 1.0),
        }
        for key, expected in checks.items():
            if model_conf[key] != expected:
                raise AssertionError(
                    f'v10 policy {key}={model_conf[key]} expected {expected}'
                )
        if not captured['dataset']['build_train_probe']:
            raise AssertionError('v10 train-only teacher probe is disabled')
        if captured['epochs'] != 172:
            raise AssertionError('v10 TES scorer update budget is not preserved')
    finally:
        sys.argv = original_argv
    print('v10 policy configuration and scorer budget: correct')


if __name__ == '__main__':
    dual_model, sample_input, sample_label = _phase0_parity_test()
    _endpoint_reliability_test(dual_model)
    _joint_target_test(dual_model)
    _counterfactual_schedule_test(dual_model, sample_input, sample_label)
    _phase2_no_regret_test(dual_model)
    _partial_force_test(dual_model, sample_input)
    _pipeline_paired_evaluation_test()
    _final_diagnostics_test()
    _policy_configuration_test()
    print('dual_ru_v10 verification passed')
