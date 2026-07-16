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
)


def _install_timm_fallback():
    try:
        import timm  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    timm = types.ModuleType('timm')
    scheduler = types.ModuleType('timm.scheduler')
    cosine = types.ModuleType('timm.scheduler.cosine_lr')
    step = types.ModuleType('timm.scheduler.step_lr')
    base = types.ModuleType('timm.scheduler.scheduler')

    class Scheduler:
        pass

    cosine.CosineLRScheduler = Scheduler
    step.StepLRScheduler = Scheduler
    base.Scheduler = Scheduler
    sys.modules.update({
        'timm': timm,
        'timm.scheduler': scheduler,
        'timm.scheduler.cosine_lr': cosine,
        'timm.scheduler.step_lr': step,
        'timm.scheduler.scheduler': base,
    })


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
        dual_ru_v11=True,
        dual_ru_warmup_ratio=0.7,
        dual_ru_supervision_ratio=0.85,
        dual_ru_temperature=0.1,
        dual_ru_rank_weight=0.0,
        dual_ru_phase2_lr_scale=1.0,
        dual_ru_teacher_min_probe_coef=97.0,
        dual_ru_teacher_ema_decay=0.5,
        dual_ru_no_regret_weight=1.0,
        dual_ru_joint_rank_weight=0.2,
        dual_ru_joint_sample_weight=0.0,
        dual_ru_joint_min_gain=0.02,
        dual_ru_batch_rank_regret_weight=0.1,
        dual_ru_gate_grid=(0.0, 0.5, 1.0),
    )
    fixed.set_cue_epoch(0, 20)
    dual.set_cue_epoch(0, 20)
    return fixed, dual


def _sample_batch():
    generator = torch.Generator().manual_seed(7)
    return {
        'V': torch.randn(4, 70, 1024, generator=generator),
        'F': torch.randn(4, 70, 1024, generator=generator),
        'A': torch.randn(4, 70, 768, generator=generator),
        'S': torch.randn(4, 70, 132, generator=generator),
        'SQ': torch.ones(4),
    }, torch.tensor([[0.2], [0.4], [0.6], [0.8]])


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

    _assert_close('v11 phase-0 score', fixed_score, dual_score)
    _assert_close('v11 phase-0 loss', fixed_loss, dual_loss)
    if not torch.equal(fixed_rng, dual_rng):
        raise AssertionError('v11 phase-0 RNG states differ')

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
        raise AssertionError(f'v11 phase-0 gradient difference {gradient_max}')
    print(f'v11 phase-0 parity: max gradient difference {gradient_max:.3e}')
    return dual, model_input, label


def _action_prior_parameter_test(dual):
    parameters = dual.dual_ru_usefulness_parameters()
    if len(parameters) != 2 or any(parameter.numel() != 1 for parameter in parameters):
        raise AssertionError('v11 must have exactly two scalar usefulness parameters')
    if not all('logit' in name for name, _ in dual._dual_usefulness_named_parameters()):
        raise AssertionError('v11 usefulness parameters are not global logits')

    dual.set_cue_epoch(14, 20)
    if not all(parameter.requires_grad for parameter in parameters):
        raise AssertionError('v11 action priors are frozen in phase 1')
    dual.set_cue_epoch(19, 20)
    if not all(parameter.requires_grad for parameter in parameters):
        raise AssertionError('v11 action priors are frozen in phase 2')
    for name, parameter in dual.named_parameters():
        if dual._is_dual_teacher_core_name(name) and parameter.requires_grad:
            raise AssertionError(f'v11 phase-2 scoring core is trainable: {name}')
    print('v11 parameterization: exactly two trainable action-level logits')


def _global_target_test(dual):
    grid = torch.tensor([0.0, 0.5, 1.0])
    gate_pairs = torch.cartesian_prod(grid, grid)
    label = torch.tensor([[0.15], [0.35], [0.65], [0.85]])
    joint_scores = []
    for motion_gate, skeleton_gate in gate_pairs:
        error = 0.4 * motion_gate + 0.4 * (1 - skeleton_gate)
        joint_scores.append(label + error)

    context = torch.zeros(label.shape[0], 1, 1)
    motion_prediction = dual.dual_motion_usefulness_head(context).squeeze(2)
    skeleton_prediction = dual.dual_skeleton_usefulness_head(context).squeeze(2)
    loss, info = dual._dual_ru_v10_usefulness_loss(label, {
        '_dual_joint_gate_pairs': gate_pairs,
        '_dual_joint_grid_scores': torch.stack(joint_scores),
        '_dual_motion_usefulness_values': motion_prediction,
        '_dual_skeleton_usefulness_values': skeleton_prediction,
    })

    for cue in ('motion', 'skeleton'):
        targets = info[f'_dual_{cue}_usefulness_target_values']
        if (targets - targets[0]).abs().max().item() > 1e-7:
            raise AssertionError(f'v11 {cue} target still contains sample correction')
    _assert_close(
        'v11 sample correction',
        info['dual_joint_sample_correction'],
        torch.tensor(0.0)
    )
    loss.backward()
    for cue, head in (
            ('motion', dual.dual_motion_usefulness_head),
            ('skeleton', dual.dual_skeleton_usefulness_head)):
        if head.logit.grad is None or head.logit.grad.abs().item() <= 0:
            raise AssertionError(f'v11 {cue} action prior has no gradient')
    print('v11 supervision: batch-global target with no sample correction')


def _hierarchical_gate_test(dual, model_input):
    dual.zero_grad()
    dual.set_cue_epoch(19, 20)
    dual.eval()
    with torch.no_grad():
        _, info = dual(model_input)
        for cue in ('motion', 'skeleton'):
            usefulness = info[f'_dual_{cue}_usefulness_values']
            if (usefulness - usefulness[0]).abs().max().item() > 1e-7:
                raise AssertionError(f'v11 {cue} usefulness varies by sample')
            expected_gate = (
                info[f'_dual_{cue}_reliability_values'] * usefulness
            )
            _assert_close(
                f'v11 {cue} prior x reliability gate',
                info[f'_dual_{cue}_gate_values'],
                expected_gate
            )

        for cue in ('motion', 'skeleton'):
            _, corrupt = dual(model_input, dual_corruption=(cue, 1.0))
            if corrupt[f'{cue}_reliability'].abs().item() > 1e-7:
                raise AssertionError(f'v11 fully corrupted {cue} reliability is nonzero')
            if corrupt[f'{cue}_gate_norm'].abs().item() > 1e-7:
                raise AssertionError(f'v11 fully corrupted {cue} gate is nonzero')
    print('v11 hierarchy: global prior x sample reliability, zero on full corruption')


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
        phase1 = dual.build_dual_ru_counterfactual_targets(model_input, label)
        phase1_calls = list(calls)
        calls.clear()
        dual.set_cue_epoch(19, 20)
        dual.train()
        phase2 = dual.build_dual_ru_counterfactual_targets(model_input, label)
        phase2_calls = list(calls)
    finally:
        dual._forward_dual_ru = original_forward

    expected_grid = {
        (motion, skeleton)
        for motion in (0.0, 0.5, 1.0)
        for skeleton in (0.0, 0.5, 1.0)
    }
    if set(phase1_calls) != expected_grid or len(phase1_calls) != 9:
        raise AssertionError('v11 phase-1 joint grid is incorrect')
    if set(phase1) != {
            '_dual_fixed_score', '_dual_joint_gate_pairs',
            '_dual_joint_grid_scores'}:
        raise AssertionError('v11 phase-1 targets are incorrect')
    if phase2_calls != [(1.0, 0.5)] or set(phase2) != {'_dual_fixed_score'}:
        raise AssertionError('v11 phase-2 should retain only the fixed replay')
    print('v11 schedule: phase-1 joint grid, phase-2 fixed replay only')


def _optimizer_test(dual):
    _install_timm_fallback()
    import pipeline

    optimizer = pipeline.build_optimizer(
        dual, 'AdamW', lr=1e-3, model_name='pamfn_final'
    )
    if len(optimizer.param_groups) != 2:
        raise AssertionError('v11 optimizer has no dedicated action-prior group')
    prior_group = optimizer.param_groups[1]
    if abs(prior_group['lr'] - 1e-3) > 1e-12:
        raise AssertionError('v11 action-prior LR is unexpectedly downscaled')
    if sum(parameter.numel() for parameter in prior_group['params']) != 2:
        raise AssertionError('v11 optimizer prior group contains non-scalar parameters')
    print('v11 optimizer: two scalar priors use the base learning rate')


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
    real_pipeline = sys.modules.get('pipeline')
    sys.modules['pipeline'] = pipeline_stub
    sys.modules.pop('main', None)
    original_argv = sys.argv
    try:
        sys.argv = [
            'main.py', '--gpu', '0', '--feats', '2', '--action', 'TES',
            '--multi_modality', '--cue_policy', 'dual_ru_v11'
        ]
        runpy.run_path(os.path.join(ROOT, 'main.py'), run_name='__main__')
        model_conf = captured['model']['model_conf']
        checks = {
            'dual_ru_v10': False,
            'dual_ru_v11': True,
            'dual_ru_warmup_ratio': 0.7,
            'dual_ru_supervision_ratio': 0.85,
            'dual_ru_joint_sample_weight': 0.0,
            'dual_ru_gate_grid': (0.0, 0.5, 1.0),
        }
        for key, expected in checks.items():
            if model_conf[key] != expected:
                raise AssertionError(
                    f'v11 policy {key}={model_conf[key]} expected {expected}'
                )
        if not captured['dataset']['build_train_probe']:
            raise AssertionError('v11 train-only teacher probe is disabled')
        if captured['epochs'] != 172:
            raise AssertionError('v11 TES scorer update budget is not preserved')
    finally:
        sys.argv = original_argv
        if real_pipeline is None:
            sys.modules.pop('pipeline', None)
        else:
            sys.modules['pipeline'] = real_pipeline
    print('v11 policy configuration and scorer budget: correct')


if __name__ == '__main__':
    dual_model, sample_input, sample_label = _phase0_parity_test()
    _action_prior_parameter_test(dual_model)
    _global_target_test(dual_model)
    _hierarchical_gate_test(dual_model, sample_input)
    _counterfactual_schedule_test(dual_model, sample_input, sample_label)
    _optimizer_test(dual_model)
    _policy_configuration_test()
    print('dual_ru_v11 verification passed')
