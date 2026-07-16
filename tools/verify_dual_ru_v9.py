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
        dual_ru_v9=True,
        dual_ru_warmup_ratio=0.7,
        dual_ru_supervision_ratio=0.85,
        dual_ru_temperature=0.1,
        dual_ru_rank_weight=0.0,
        dual_ru_phase2_lr_scale=1.0,
        dual_ru_unidentifiable_weight=0.05,
        dual_ru_identifiability_abs=1e-6,
        dual_ru_identifiability_error_ratio=0.0,
        dual_ru_teacher_min_probe_coef=97.0,
        dual_ru_teacher_ema_decay=0.5,
        dual_ru_phase2_target_end_weight=0.1,
        dual_ru_no_regret_weight=1.0,
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

    _assert_close('v9 phase-0 score', fixed_score, dual_score)
    _assert_close('v9 phase-0 loss', fixed_loss, dual_loss)
    if not torch.equal(fixed_rng, dual_rng):
        raise AssertionError('v9 phase-0 RNG states differ')
    print('v9 phase-0 RNG state: identical')

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
        raise AssertionError(f'v9 phase-0 gradient difference {gradient_max}')
    print(f'v9 phase-0 common gradients: max difference {gradient_max:.3e}')
    return dual, model_input, label


def _three_point_target_test(dual):
    grid = torch.tensor([0.0, 0.5, 1.0])
    label = torch.tensor([[0.2], [0.4], [0.6], [0.8]])
    flat_score = torch.full((3, 4, 1), 0.5)
    flat_kwargs = {
        '_dual_gate_grid': grid,
        '_dual_motion_grid_scores': flat_score,
        '_dual_skeleton_grid_scores': flat_score,
        '_dual_motion_usefulness_values': torch.full((4, 1), 0.5),
        '_dual_skeleton_usefulness_values': torch.full((4, 1), 0.5),
    }
    _, flat_info = dual._dual_ru_v9_usefulness_loss(label, flat_kwargs)
    _assert_close(
        'v9 neutral motion target',
        flat_info['_dual_motion_usefulness_target_values'],
        torch.ones(4)
    )
    _assert_close(
        'v9 neutral skeleton target',
        flat_info['_dual_skeleton_usefulness_target_values'],
        torch.full((4,), 0.5)
    )
    for cue in ('motion', 'skeleton'):
        if flat_info[
                f'_dual_{cue}_usefulness_target_valid_values'].any():
            raise AssertionError(f'flat {cue} cue was marked identifiable')
        _assert_close(
            f'v9 neutral {cue} confidence',
            flat_info[f'_dual_{cue}_usefulness_target_weights'],
            torch.full((4,), 0.05)
        )

    motion_prediction = torch.full((4, 1), 0.9, requires_grad=True)
    skeleton_prediction = torch.full((4, 1), 0.5, requires_grad=True)
    kwargs = {
        '_dual_gate_grid': grid,
        '_dual_motion_grid_scores': torch.stack(
            [label - 0.2, label, label + 0.2], dim=0
        ),
        '_dual_skeleton_grid_scores': torch.stack(
            [label - 0.4, label - 0.2, label], dim=0
        ),
        '_dual_motion_usefulness_values': motion_prediction,
        '_dual_skeleton_usefulness_values': skeleton_prediction,
    }
    loss, info = dual._dual_ru_v9_usefulness_loss(label, kwargs)
    _assert_close(
        'v9 continuous motion target',
        info['_dual_motion_usefulness_target_values'],
        torch.full((4,), 0.5)
    )
    _assert_close(
        'v9 continuous skeleton target',
        info['_dual_skeleton_usefulness_target_values'],
        torch.ones(4)
    )
    loss.backward()
    if motion_prediction.grad.abs().sum().item() <= 0:
        raise AssertionError('v9 motion usefulness has no gradient')
    if skeleton_prediction.grad.abs().sum().item() <= 0:
        raise AssertionError('v9 skeleton usefulness has no gradient')

    tied_kwargs = dict(kwargs)
    tied_kwargs['_dual_motion_grid_scores'] = torch.stack(
        [label, label + 0.1, label], dim=0
    )
    tied_kwargs['_dual_skeleton_grid_scores'] = torch.stack(
        [label, label, label + 0.1], dim=0
    )
    _, tied_info = dual._dual_ru_v9_usefulness_loss(label, tied_kwargs)
    _assert_close(
        'v9 tied motion keeps default',
        tied_info['_dual_motion_usefulness_target_values'],
        torch.ones(4)
    )
    _assert_close(
        'v9 tied skeleton keeps default',
        tied_info['_dual_skeleton_usefulness_target_values'],
        torch.full((4,), 0.5)
    )
    print('v9 three-point continuous MSE targets and safe ties: correct')


def _counterfactual_probe_test(dual, model_input, label):
    calls = []
    original_forward = dual._forward_dual_ru

    def fake_forward(self, input_feats, force_gates=None, corruption=None):
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
        (0.0, 0.5), (0.5, 0.5), (1.0, 0.5),
        (1.0, 0.0), (1.0, 1.0)
    }
    if set(calls) != expected_calls or len(calls) != len(expected_calls):
        raise AssertionError(f'v9 counterfactual probes differ: {calls}')
    expected_keys = {
        '_dual_gate_grid', '_dual_motion_grid_scores',
        '_dual_skeleton_grid_scores', '_dual_fixed_score'
    }
    if set(targets) != expected_keys:
        raise AssertionError('v9 counterfactual target keys are incorrect')
    _assert_close(
        'v9 fixed replay score', targets['_dual_fixed_score'],
        label.new_full(label.shape, 1.5)
    )
    print('v9 isolated three-point counterfactual probes: correct')


def _teacher_candidate_test(dual):
    import pipeline

    dual.set_cue_epoch(0, 20)
    first = dual.fc[1].weight.detach().clone()
    if not dual.consider_dual_teacher_snapshot(
            epoch=5, loss=0.5, coefficient=98.0, score_std=0.1):
        raise AssertionError('healthy v9 teacher snapshot was rejected')
    with torch.no_grad():
        dual.fc[1].weight.add_(2.0)
    second = dual.fc[1].weight.detach().clone()
    if not dual.consider_dual_teacher_snapshot(
            epoch=10, loss=0.4, coefficient=98.5, score_std=0.1):
        raise AssertionError('better v9 teacher snapshot was rejected')
    expected_ema = 0.5 * first + 0.5 * second
    if dual.dual_teacher_candidate_names() != ('raw', 'ema'):
        raise AssertionError('v9 teacher candidates are incomplete')
    if not dual.restore_dual_teacher_candidate('raw'):
        raise AssertionError('v9 raw teacher cannot be restored')
    _assert_close('v9 raw teacher restore', dual.fc[1].weight, second)
    if not dual.restore_dual_teacher_candidate('ema'):
        raise AssertionError('v9 EMA teacher cannot be restored')
    _assert_close('v9 EMA teacher restore', dual.fc[1].weight, expected_ema)

    original_test_epoch = pipeline.test_epoch

    def fake_test_epoch(model, dataloader, **kwargs):
        candidate = model.dual_teacher_probe_info()['candidate']
        loss = 0.35 if candidate == 'ema' else 0.40
        coefficient = 98.6 if candidate == 'ema' else 98.4
        info = {}
        pipeline.set_info_meter(info, 'score_mean', 0.5)
        pipeline.set_info_meter(info, 'score_square', 0.26)
        return 0.0, loss, coefficient, info

    pipeline.test_epoch = fake_test_epoch
    try:
        selected = pipeline.select_dual_teacher_candidate(
            dual, object(), lambda message: None
        )
    finally:
        pipeline.test_epoch = original_test_epoch
    if selected != 'ema':
        raise AssertionError(f'v9 selected the wrong teacher: {selected}')
    _assert_close('v9 selected teacher restore', dual.fc[1].weight, expected_ema)
    print('v9 raw/EMA teacher revalidation and selection: correct')


def _phase2_no_regret_test(dual):
    dual.set_cue_epoch(19, 20)
    if abs(dual.dual_ru_target_weight() - 0.1) > 1e-6:
        raise AssertionError('v9 phase-2 target weight did not anneal to 0.1')
    for name, parameter in dual.named_parameters():
        if dual._is_dual_teacher_core_name(name) and parameter.requires_grad:
            raise AssertionError(f'v9 phase-2 core is trainable: {name}')

    prediction = torch.tensor([[0.4], [0.6]], requires_grad=True)
    label = torch.tensor([[0.3], [0.7]])
    motion_u = torch.full((2, 1), 0.8, requires_grad=True)
    skeleton_u = torch.full((2, 1), 0.4, requires_grad=True)
    grid_scores = torch.stack(
        [label - 0.1, label, label + 0.1], dim=0
    )
    loss = dual.call_loss(
        prediction,
        label,
        _dual_gate_grid=torch.tensor([0.0, 0.5, 1.0]),
        _dual_motion_grid_scores=grid_scores,
        _dual_skeleton_grid_scores=grid_scores,
        _dual_fixed_score=label.clone(),
        _dual_motion_usefulness_values=motion_u,
        _dual_skeleton_usefulness_values=skeleton_u,
    )
    if dual.last_loss_info['dual_no_regret_mse'].item() <= 0:
        raise AssertionError('v9 no-regret MSE penalty is inactive')
    if abs(dual.last_loss_info['dual_usefulness_weight'].item() - 0.1) > 1e-6:
        raise AssertionError('v9 call_loss uses the wrong target weight')
    loss.backward()
    if prediction.grad.abs().sum().item() <= 0:
        raise AssertionError('v9 no-regret path has no score gradient')
    if motion_u.grad.abs().sum().item() <= 0:
        raise AssertionError('v9 motion target path has no gradient')
    if skeleton_u.grad.abs().sum().item() <= 0:
        raise AssertionError('v9 skeleton target path has no gradient')
    print('v9 annealed targets, frozen scorer, and no-regret loss: correct')


def _internal_corruption_test(dual, model_input):
    calls = []
    original_corruption = dual._dual_ru_deterministic_corruption

    def record_corruption(cue, severity):
        calls.append((tuple(cue.shape), float(severity)))
        return cue

    dual._dual_ru_deterministic_corruption = record_corruption
    dual.eval()
    try:
        dual(
            model_input, dual_force_gates=(1.0, 0.5),
            dual_corruption=('motion', 0.5)
        )
        motion_calls = list(calls)
        calls.clear()
        dual(
            model_input, dual_force_gates=(1.0, 0.5),
            dual_corruption=('skeleton', 1.0)
        )
        skeleton_calls = list(calls)
    finally:
        dual._dual_ru_deterministic_corruption = original_corruption
    if len(motion_calls) != 1 or motion_calls[0][0][1] != 256:
        raise AssertionError(
            f'motion corruption did not target encoded cue: {motion_calls}'
        )
    if len(skeleton_calls) != 1 or skeleton_calls[0][0][1] != 132:
        raise AssertionError(
            f'skeleton corruption did not target skeleton cue: {skeleton_calls}'
        )
    print('v9 final corruption targets controlled cues only: correct')


def _corrupted_logging_test():
    import pipeline

    source = inspect.getsource(pipeline.train_loop)
    for token in (
            'Final corrupted paired evaluation', 'reliability',
            'usefulness', 'gate_norm', 'adaptive_coef', 'fixed_coef',
            'adaptive_loss', 'fixed_loss'):
        if token not in source:
            raise AssertionError(
                f'v9 corrupted evaluation does not log {token}'
            )
    print('v9 corrupted gate and paired score logging: correct')


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
            '--multi_modality', '--cue_policy', 'dual_ru_v9'
        ]
        args = main.parser_args()
        if not args.cue_quality_supervision or args.auto_gate:
            raise AssertionError('dual_ru_v9 gate policy is incorrect')
        if args.cue_corruption_max != 1.0:
            raise AssertionError('dual_ru_v9 corruption endpoint is disabled')

        runpy.run_path(os.path.join(ROOT, 'main.py'), run_name='__main__')
        model_conf = captured['model']['model_conf']
        checks = {
            'dual_ru_v9': True,
            'dual_ru_warmup_ratio': 0.7,
            'dual_ru_supervision_ratio': 0.85,
            'dual_ru_rank_weight': 0.0,
            'dual_ru_phase2_target_end_weight': 0.1,
            'dual_ru_no_regret_weight': 1.0,
            'dual_ru_identifiability_abs': 1e-6,
            'dual_ru_identifiability_error_ratio': 0.0,
            'dual_ru_gate_grid': (0.0, 0.5, 1.0),
        }
        for key, expected in checks.items():
            if model_conf[key] != expected:
                raise AssertionError(
                    f'v9 policy {key}={model_conf[key]} expected {expected}'
                )
        if not captured['dataset']['build_train_probe']:
            raise AssertionError('v9 train-only teacher probe is disabled')
        if captured['epochs'] != 172:
            raise AssertionError('v9 TES scorer update budget is not preserved')
    finally:
        sys.argv = original_argv
    print('v9 policy configuration and scorer budget: correct')


if __name__ == '__main__':
    dual_model, sample_input, sample_label = _phase0_parity_test()
    _endpoint_reliability_test(dual_model)
    _three_point_target_test(dual_model)
    _counterfactual_probe_test(dual_model, sample_input, sample_label)
    _pipeline_paired_evaluation_test()
    _teacher_candidate_test(dual_model)
    _phase2_no_regret_test(dual_model)
    _internal_corruption_test(dual_model, sample_input)
    _corrupted_logging_test()
    _policy_configuration_test()
    print('dual_ru_v9 verification passed')
