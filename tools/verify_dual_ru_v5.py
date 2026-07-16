import os
import inspect
import sys
import types

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _install_einops_fallback():
    try:
        import einops  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType('einops')

    def rearrange(x, pattern, **axes):
        if pattern == 'b t d -> b d t':
            return x.permute(0, 2, 1)
        if pattern == 'b c t -> (b t) 1 c':
            b, c, t = x.shape
            return x.permute(0, 2, 1).reshape(b * t, 1, c)
        if pattern == 'b c n t -> (b t) n c':
            b, c, n, t = x.shape
            return x.permute(0, 3, 2, 1).reshape(b * t, n, c)
        if pattern == '(b t) 1 c -> b c t':
            b = axes['b']
            bt, _, c = x.shape
            t = bt // b
            return x.reshape(b, t, c).permute(0, 2, 1)
        if pattern == 'b n t -> (b t) 1 n':
            b, n, t = x.shape
            return x.permute(0, 2, 1).reshape(b * t, 1, n)
        raise ValueError(f'Unsupported fallback pattern: {pattern}')

    module.rearrange = rearrange
    sys.modules['einops'] = module


_install_einops_fallback()

from models.pamfn import PAMFN  # noqa: E402


def _load_full_model(path, **kwargs):
    original_load = torch.load
    supports_weights_only = (
        'weights_only' in inspect.signature(original_load).parameters
    )

    def compatible_load(*args, **load_kwargs):
        if supports_weights_only:
            load_kwargs.setdefault('weights_only', False)
        return original_load(*args, **load_kwargs)

    torch.load = compatible_load
    try:
        model = PAMFN(ckpt_dir=path, **kwargs)
    finally:
        torch.load = original_load
    for module in model.modules():
        if isinstance(module, torch.nn.GELU) and not hasattr(module, 'approximate'):
            module.approximate = 'none'
    return model


def _common_kwargs():
    return {
        'model_dim': 256,
        'fc_drop': 0.3,
        'fc_r': 1,
        'feat_drop': 0.3,
        'K': 8,
        'ms_heads': 2,
        'cm_heads': 4,
        'rgb_ckpt_name': 'UMT',
        'flow_ckpt_name': 'I3D',
        'audio_ckpt_name': 'MAST',
        'dataset_name': 'TES',
        'use_skeleton': True,
        'skeleton_gate_max': 0.02,
        'motion_saliency_proxy': True,
        'motion_saliency_alpha': 0.015,
        'motion_saliency_mode': 'stage_residual',
        'motion_saliency_source': 'delta',
        'motion_saliency_positive_only': True,
        'auto_gate_motion_max': 0.015,
        'cue_quality_supervision': True,
        'cue_corruption_prob': 1.0,
        'cue_corruption_max': 1.0,
        'cue_quality_loss_weight': 1.0,
        'cue_quality_paired': True,
        'cue_quality_detach_gate': True,
        'cue_quality_pair_weight': 0.1,
        'dual_ru_task_rank_weight': 0.01,
    }


def _build_models():
    common = _common_kwargs()
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
        dual_ru_v5=True,
        dual_ru_warmup_ratio=0.5,
        dual_ru_supervision_ratio=0.7,
    )
    fixed.set_cue_epoch(0, 10)
    dual.set_cue_epoch(0, 10)
    return fixed, dual


def _assert_close(name, left, right, atol=1e-6):
    difference = (left - right).abs().max().item()
    if difference > atol:
        raise AssertionError(f'{name} max difference {difference} > {atol}')
    print(f'{name}: max difference {difference:.3e}')


def _phase0_parity_test():
    fixed, dual = _build_models()
    fixed_parameters = dict(fixed.named_parameters())
    dual_parameters = dict(dual.named_parameters())
    initial_max = 0.0
    for name in sorted(set(fixed_parameters) & set(dual_parameters)):
        initial_max = max(
            initial_max,
            (fixed_parameters[name] - dual_parameters[name]).abs().max().item()
        )
    if initial_max > 1e-6:
        raise AssertionError(f'initial parameter max difference {initial_max}')
    print(f'initial common parameters: max difference {initial_max:.3e}')

    generator = torch.Generator().manual_seed(7)
    model_input = {
        'V': torch.randn(2, 70, 1024, generator=generator),
        'F': torch.randn(2, 70, 1024, generator=generator),
        'A': torch.randn(2, 70, 768, generator=generator),
        'S': torch.randn(2, 70, 132, generator=generator),
        'SQ': torch.ones(2),
    }
    label = torch.tensor([[0.35], [0.75]])
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

    _assert_close('phase-0 score', fixed_score, dual_score)
    _assert_close('phase-0 loss', fixed_loss, dual_loss)
    if not torch.equal(fixed_rng, dual_rng):
        raise AssertionError('phase-0 RNG states differ')
    print('phase-0 RNG state: identical')

    fixed.zero_grad()
    dual.zero_grad()
    fixed_loss.backward()
    dual_loss.backward()
    gradient_max = 0.0
    for name in sorted(set(fixed_parameters) & set(dual_parameters)):
        fixed_grad = fixed_parameters[name].grad
        dual_grad = dual_parameters[name].grad
        if fixed_grad is None and dual_grad is None:
            continue
        if fixed_grad is None or dual_grad is None:
            raise AssertionError(f'gradient presence differs for {name}')
        gradient_max = max(
            gradient_max, (fixed_grad - dual_grad).abs().max().item()
        )
    if gradient_max > 1e-6:
        raise AssertionError(f'phase-0 gradient max difference {gradient_max}')
    print(f'phase-0 common gradients: max difference {gradient_max:.3e}')

    fixed_optimizer = torch.optim.AdamW(
        fixed.parameters(), lr=3e-4, weight_decay=1e-4
    )
    usefulness_parameters = dual.dual_ru_usefulness_parameters()
    usefulness_ids = {id(parameter) for parameter in usefulness_parameters}
    other_parameters = [
        parameter for parameter in dual.parameters()
        if id(parameter) not in usefulness_ids
    ]
    dual_optimizer = torch.optim.AdamW([
        {'params': other_parameters},
        {'params': usefulness_parameters, 'lr': 3e-5},
    ], lr=3e-4, weight_decay=1e-4)
    fixed_optimizer.step()
    dual_optimizer.step()
    update_max = 0.0
    for name in sorted(set(fixed_parameters) & set(dual_parameters)):
        update_max = max(
            update_max,
            (fixed_parameters[name] - dual_parameters[name]).abs().max().item()
        )
    if update_max > 1e-6:
        raise AssertionError(f'phase-0 update max difference {update_max}')
    print(f'phase-0 optimizer update: max difference {update_max:.3e}')
    return dual, model_input, label


def _continuous_target_test(dual):
    dual.dual_ru_target_rank_weight = 0.0
    label = torch.tensor([[0.2], [0.8]])
    grid = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    motion_scores = torch.tensor([
        [[0.2], [0.4]],
        [[0.3], [0.5]],
        [[0.4], [0.6]],
        [[0.5], [0.7]],
        [[0.6], [0.8]],
    ])
    skeleton_scores = torch.tensor([
        [[0.0], [0.7]],
        [[0.1], [0.8]],
        [[0.2], [0.9]],
        [[0.3], [1.0]],
        [[0.4], [1.1]],
    ])
    kwargs = {
        '_dual_gate_grid': grid,
        '_dual_motion_grid_scores': motion_scores,
        '_dual_skeleton_grid_scores': skeleton_scores,
        '_dual_motion_usefulness_values': torch.tensor([[0.5], [0.5]]),
        '_dual_skeleton_usefulness_values': torch.tensor([[0.5], [0.5]]),
    }
    _, info = dual._dual_ru_v5_usefulness_loss(label, kwargs)
    motion_target = info['_dual_motion_usefulness_target_values']
    skeleton_target = info['_dual_skeleton_usefulness_target_values']
    _assert_close('motion optimal gates', motion_target, torch.tensor([0.0, 1.0]))
    _assert_close(
        'skeleton optimal gates', skeleton_target, torch.tensor([0.5, 0.25])
    )
    print('continuous target selection: correct')


def _phase_and_closed_loop_test(dual, model_input, label):
    dual.set_cue_epoch(5, 10)
    dual.train()
    dual.prepare_dual_ru_train_mode()
    if dual.dual_ru_phase() != 1:
        raise AssertionError('epoch 5 should be frozen-teacher phase')
    if next(dual.fc.parameters()).requires_grad:
        raise AssertionError('scoring core must be frozen in phase 1')
    if dual.fc.training:
        raise AssertionError('scoring core must be in eval mode in phase 1')
    if not next(dual.dual_motion_usefulness_head.parameters()).requires_grad:
        raise AssertionError('usefulness head must train in phase 1')
    print('phase-1 frozen teacher: correct')

    original_threshold = dual.dual_ru_identifiability_abs
    original_error_ratio = dual.dual_ru_identifiability_error_ratio
    dual.dual_ru_identifiability_abs = 1e-6
    dual.dual_ru_identifiability_error_ratio = 0.0
    dual.zero_grad()
    counterfactual = dual.build_dual_ru_counterfactual_targets(
        model_input, label
    )
    expected_shape = (len(dual.dual_ru_gate_grid), label.shape[0], 1)
    for cue in ('motion', 'skeleton'):
        key = f'_dual_{cue}_grid_scores'
        if tuple(counterfactual[key].shape) != expected_shape:
            raise AssertionError(
                f'{key} shape {tuple(counterfactual[key].shape)} '
                f'!= {expected_shape}'
            )
    dual.prepare_dual_ru_train_mode()
    score, info = dual(model_input)
    info.update(counterfactual)
    usefulness_loss = dual.call_loss(score, label, **info)
    usefulness_loss.backward()
    supervised_grad = sum(
        parameter.grad.abs().sum().item()
        for parameter in dual.dual_ru_usefulness_parameters()
        if parameter.grad is not None
    )
    if supervised_grad <= 0:
        raise AssertionError('phase-1 usefulness supervision gradient is zero')
    if 'motion_usefulness_optimal_gate' not in dual.last_loss_info:
        raise AssertionError('continuous gate diagnostics are missing')
    print(f'phase-1 continuous supervision gradient: {supervised_grad:.3e}')
    dual.dual_ru_identifiability_abs = original_threshold
    dual.dual_ru_identifiability_error_ratio = original_error_ratio

    dual.set_cue_epoch(8, 10)
    dual.train()
    if dual.dual_ru_phase() != 2 or dual.dual_ru_gate_ramp() != 1.0:
        raise AssertionError('epoch 8 should have a fully adaptive gate')
    if not next(dual.fc.parameters()).requires_grad:
        raise AssertionError('scoring core must train in phase 2')
    if next(dual.motion_quality_head.parameters()).requires_grad:
        raise AssertionError('reliability head must be frozen in phase 2')
    if not next(dual.dual_motion_usefulness_head.parameters()).requires_grad:
        raise AssertionError('usefulness head must receive phase-2 task gradients')

    dual.zero_grad()
    torch.manual_seed(321)
    score, info = dual(model_input)
    loss = dual.call_loss(score, label, **info)
    loss.backward()
    usefulness_grad = sum(
        parameter.grad.abs().sum().item()
        for parameter in dual.dual_ru_usefulness_parameters()
        if parameter.grad is not None
    )
    if usefulness_grad <= 0:
        raise AssertionError('phase-2 usefulness task gradient is zero')
    print(f'phase-2 usefulness task gradient: {usefulness_grad:.3e}')

    dual.dual_motion_clean_ref.fill_(0.9)
    prediction = torch.tensor([[[0.9]], [[0.5]], [[0.01]]])
    reliability = dual._dual_ru_calibrated_reliability(
        prediction, 'motion'
    ).reshape(-1)
    if not (reliability[0] > reliability[1] > reliability[2]):
        raise AssertionError('v5 reliability is not monotonic')
    if reliability[-1] >= 0.05:
        raise AssertionError('severely corrupted reliability cannot approach zero')
    print(
        'smooth reliability: '
        + ', '.join(f'{value:.3f}' for value in reliability.tolist())
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
            '--multi_modality', '--cue_policy', 'dual_ru_v5'
        ]
        dual_args = main.parser_args()
        if not dual_args.cue_quality_supervision:
            raise AssertionError('dual_ru_v5 quality supervision is disabled')
        if dual_args.cue_corruption_max != 1.0 or dual_args.auto_gate:
            raise AssertionError('dual_ru_v5 policy defaults are incorrect')

        sys.argv = [
            'main.py', '--gpu', '0', '--feats', '2', '--action', 'TES',
            '--multi_modality', '--cue_policy', 'fixed_cue_v2'
        ]
        fixed_args = main.parser_args()
        if not fixed_args.auto_gate or not fixed_args.cue_quality_aux_only:
            raise AssertionError('fixed_cue_v2 is not using fixed auxiliary cues')
        if fixed_args.auto_gate_l1 != 0.0 or fixed_args.cue_corruption_max != 1.0:
            raise AssertionError('fixed_cue_v2 parity defaults are incorrect')
    finally:
        sys.argv = original_argv
    print('v5/fixed-v2 policy configuration: correct')


def _pipeline_paired_evaluation_test():
    timm = types.ModuleType('timm')
    timm_scheduler = types.ModuleType('timm.scheduler')
    cosine = types.ModuleType('timm.scheduler.cosine_lr')
    step = types.ModuleType('timm.scheduler.step_lr')
    scheduler = types.ModuleType('timm.scheduler.scheduler')

    class Scheduler:
        pass

    cosine.CosineLRScheduler = Scheduler
    step.StepLRScheduler = Scheduler
    scheduler.Scheduler = Scheduler
    sys.modules.update({
        'timm': timm,
        'timm.scheduler': timm_scheduler,
        'timm.scheduler.cosine_lr': cosine,
        'timm.scheduler.step_lr': step,
        'timm.scheduler.scheduler': scheduler,
    })
    import pipeline

    class DummyDualModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.received_gates = None

        def forward(self, input_feats, dual_force_gates=None):
            self.received_gates = dual_force_gates
            return input_feats['V'][:, 0, :1], {}

        def call_loss(self, prediction, label, **kwargs):
            return (prediction.reshape(-1) - label.reshape(-1)).square().mean()

    batch = (
        torch.tensor([[[0.2]], [[0.8]]]),
        torch.zeros(2, 1, 1),
        torch.zeros(2, 1, 1),
        torch.zeros(2, 1, 132),
        torch.tensor([[0.2], [0.8]]),
        ['a', 'b'],
    )
    model = DummyDualModel()
    original_cuda = torch.Tensor.cuda
    torch.Tensor.cuda = lambda tensor, *args, **kwargs: tensor
    try:
        _, loss, coefficient, _ = pipeline.test_epoch(
            model, [batch], dual_force_gates=(1.0, 0.5)
        )
    finally:
        torch.Tensor.cuda = original_cuda
    if model.received_gates != (1.0, 0.5):
        raise AssertionError('paired fixed gates were not forwarded to the model')
    if abs(loss) > 1e-6 or abs(coefficient - 100.0) > 1e-6:
        raise AssertionError('paired evaluation metrics are incorrect')
    print('final paired fixed-gate evaluation path: correct')


if __name__ == '__main__':
    dual_model, sample_input, sample_label = _phase0_parity_test()
    _continuous_target_test(dual_model)
    _phase_and_closed_loop_test(dual_model, sample_input, sample_label)
    _pipeline_paired_evaluation_test()
    _policy_configuration_test()
    print('dual_ru_v5 verification passed')
