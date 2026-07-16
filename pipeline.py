import copy

import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from datasets import AQADataset as Dset
from utils.utils import AverageMeter
from utils.lr_scheduler import build_scheduler


def build_dataset(dataset_name, feat_dir, rgb_feat, flow_feat, audio_feat,
                  batch_size=32, num_workers=2, segments=-1,
                  use_skeleton=False, skeleton_root='.', seed=0,
                  build_train_probe=False):
    train_dset = Dset(dataset_name=dataset_name, feat_dir=feat_dir,
                      rgb_feat=rgb_feat, flow_feat=flow_feat, audio_feat=audio_feat,
                      squeeze_rgb_feat="mean",
                      segments=segments, is_train=True,
                      use_skeleton=use_skeleton, skeleton_root=skeleton_root)
    test_dset = Dset(dataset_name=dataset_name, feat_dir=feat_dir,
                     rgb_feat=rgb_feat, flow_feat=flow_feat, audio_feat=audio_feat,
                     squeeze_rgb_feat="mean",
                     segments=segments, is_train=False,
                     use_skeleton=use_skeleton, skeleton_root=skeleton_root)
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_dloader = DataLoader(
        train_dset, batch_size=batch_size, shuffle=True,
        pin_memory=False, num_workers=num_workers,
        generator=train_generator
    )
    test_dloader = DataLoader(test_dset, batch_size=1, shuffle=False,
                              pin_memory=False, num_workers=num_workers)
    if build_train_probe:
        probe_dset = copy.copy(train_dset)
        probe_dset.is_train = False
        probe_dloader = DataLoader(
            probe_dset, batch_size=1, shuffle=False,
            pin_memory=False, num_workers=num_workers
        )
        return train_dloader, test_dloader, probe_dloader
    return train_dloader, test_dloader
 

def build_model(model_name, model_conf=None):
    if model_name == "pamfn_base":
        from models.pamfn import BaseModel
        return BaseModel(**model_conf)
    if model_name == "pamfn_final":
        from models.pamfn import PAMFN
        return PAMFN(**model_conf)
    raise ValueError(f"model: {model_name} is not supported!")


def is_dual_ru_version(model, minimum_version=1):
    return (
        getattr(model, 'dual_ru_gate', False) and
        getattr(model, 'dual_ru_version', 0) >= minimum_version
    )


def build_optimizer(model, optim, lr, momentum=0.9, weight_decay=None, reduce_fc_lr=False, model_name=None):
    if is_dual_ru_version(model):
        usefulness_params = model.dual_ru_usefulness_parameters()
        usefulness_ids = {id(parameter) for parameter in usefulness_params}
        other_params = [
            parameter for parameter in model.parameters()
            if id(parameter) not in usefulness_ids
        ]
        usefulness_lr_scale = (
            1.0 if getattr(model, 'dual_ru_v11', False) else 0.1
        )
        params_group = [
            {'params': other_params},
            {'params': usefulness_params, 'lr': lr * usefulness_lr_scale}
        ]
    elif reduce_fc_lr and model_name == 'pamfn_final':
        lr_scale = 0.1
        fc_params = list(map(id, model.fc.parameters()))
        other_params = filter(lambda p: id(p) not in fc_params, model.parameters())
        params_group = [
            {'params': model.fc.parameters(), 'lr': lr * lr_scale},
            {'params': other_params}
        ]
    else:
        params_group = model.parameters()
    if optim == 'Adam':
        optimizer = torch.optim.Adam(params_group, lr=lr)
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(params_group, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW(params_group, lr=lr)
    else:
        raise ValueError(f"optimizer: {optim} not found!")
    return optimizer


def apply_dual_phase_lr(model, optimizer, phase):
    if not is_dual_ru_version(model, 4):
        return 1.0
    scale = model.dual_ru_phase2_lr_scale if phase == 2 else 1.0
    for group in optimizer.param_groups:
        if '_dual_base_lr' not in group:
            group['_dual_base_lr'] = group['lr']
        group['lr'] = group['_dual_base_lr'] * scale
    return scale


def build_lr_scheduler(optimizer, scheduler, epoch_num, scheduler_steps,
                    lr_min=5e-6, warmup_lr_init=5e-6, decay_rate=0.1):
    return build_scheduler(optimizer, scheduler, epoch_num, scheduler_steps, decay_rate=decay_rate,
                    lr_min=lr_min, warmup_lr_init=warmup_lr_init)


def prepare_model_batch(batch):
    if len(batch) == 7:
        video, audio, flow, skeleton, skeleton_quality, label, _ = batch
    elif len(batch) == 6:
        video, audio, flow, skeleton, label, _ = batch
        skeleton_quality = None
    else:
        video, audio, flow, label, _ = batch
        skeleton, skeleton_quality = None, None

    model_input = {
        "V": video.cuda(),
        "F": flow.cuda(),
        "A": audio.cuda()
    }
    if skeleton is not None:
        model_input["S"] = skeleton.cuda()
        if skeleton_quality is not None:
            model_input["SQ"] = skeleton_quality.float().cuda()
    return model_input, label.float().cuda()


def spearman_coefficient(predictions, labels):
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    coefficient, _ = spearmanr(predictions, labels)
    return coefficient


def update_info_meters(info_meters, other_info):
    for key, value in other_info.items():
        if key.startswith('_'):
            continue
        if not torch.is_tensor(value):
            continue
        if value.numel() < 1:
            continue
        val = value.detach().float().mean().item()
        if key not in info_meters:
            info_meters[key] = AverageMeter()
        info_meters[key].update(val, 1)


def update_gate_quantiles(info_meters, gate_values, prefix=""):
    if not gate_values:
        return
    values = torch.cat(gate_values)
    for name, quantile in (("p05", 0.05), ("p50", 0.5), ("p95", 0.95)):
        key = f"{prefix}skeleton_usefulness_gate_{name}"
        info_meters[key] = AverageMeter()
        info_meters[key].update(torch.quantile(values, quantile).item(), 1)


DUAL_GATE_VALUE_KEYS = {
    'motion_reliability': '_dual_motion_reliability_values',
    'motion_usefulness': '_dual_motion_usefulness_values',
    'motion_gate_norm': '_dual_motion_gate_values',
    'skeleton_reliability': '_dual_skeleton_reliability_values',
    'skeleton_usefulness': '_dual_skeleton_usefulness_values',
    'skeleton_gate_norm': '_dual_skeleton_gate_values'
}


def collect_dual_gate_values(gate_values, other_info):
    for name, key in DUAL_GATE_VALUE_KEYS.items():
        if key in other_info:
            gate_values[name].append(
                other_info[key].detach().cpu().reshape(-1)
            )


def update_dual_gate_quantiles(info_meters, gate_values):
    for name, values in gate_values.items():
        if not values:
            continue
        values = torch.cat(values)
        for suffix, quantile in (('p05', 0.05), ('p50', 0.5), ('p95', 0.95)):
            key = f'{name}_{suffix}'
            info_meters[key] = AverageMeter()
            info_meters[key].update(torch.quantile(values, quantile).item(), 1)


def collect_dual_target_values(target_values, other_info):
    for cue in ('motion', 'skeleton'):
        prediction_key = f'_dual_{cue}_usefulness_prediction_values'
        target_key = f'_dual_{cue}_usefulness_target_values'
        weight_key = f'_dual_{cue}_usefulness_target_weights'
        valid_key = f'_dual_{cue}_usefulness_target_valid_values'
        if all(key in other_info for key in (
                prediction_key, target_key, weight_key)):
            target_values[cue]['prediction'].append(
                other_info[prediction_key].detach().cpu().reshape(-1)
            )
            target_values[cue]['target'].append(
                other_info[target_key].detach().cpu().reshape(-1)
            )
            target_values[cue]['weight'].append(
                other_info[weight_key].detach().cpu().reshape(-1)
            )
            if valid_key in other_info:
                target_values[cue]['valid'].append(
                    other_info[valid_key].detach().cpu().reshape(-1)
                )


def update_dual_target_diagnostics(info_meters, target_values):
    for cue, values in target_values.items():
        if not values['prediction']:
            continue
        prediction = torch.cat(values['prediction'])
        target = torch.cat(values['target'])
        active = torch.cat(values['weight']) > 0
        if values['valid']:
            active = torch.cat(values['valid']).bool()
        if active.sum() == 0:
            correlation = 0.0
            accuracy = 0.0
        else:
            active_prediction = prediction[active]
            active_target = target[active]
            pred_centered = active_prediction - active_prediction.mean()
            target_centered = active_target - active_target.mean()
            denominator = (
                pred_centered.square().sum() *
                target_centered.square().sum()
            ).sqrt().clamp_min(1e-6)
            correlation = (
                (pred_centered * target_centered).sum() / denominator
            ).item()
            accuracy = (
                (active_prediction >= 0.5) == (active_target >= 0.5)
            ).float().mean().item()
        for suffix, value in (
                ('target_corr_epoch', correlation),
                ('target_accuracy_epoch', accuracy)):
            key = f'{cue}_usefulness_{suffix}'
            info_meters[key] = AverageMeter()
            info_meters[key].update(value, 1)


def format_info_meters(info_meters, detailed=False):
    keys = ["skeleton_gate", "auto_motion_gate", "skeleton_quality",
            "auto_motion_gate_norm", "auto_skeleton_gate_norm",
            "motion_quality_pred", "skeleton_quality_pred",
            "motion_quality_reference", "skeleton_quality_reference",
            "motion_quality_gate_std", "motion_quality_gate_min",
            "motion_quality_gate_max", "skeleton_quality_gate_std",
            "skeleton_quality_gate_min", "skeleton_quality_gate_max",
            "motion_reliability", "skeleton_reliability",
            "motion_reliability_zero_pred", "skeleton_reliability_zero_pred",
            "motion_reliability_factor", "skeleton_reliability_factor",
            "motion_correction", "skeleton_correction", "skeleton_temporal_mean",
            "skeleton_late_gate", "auto_gate_loss",
            "cue_quality_clean_pred", "cue_quality_corrupt_pred",
            "cue_quality_target", "cue_quality_gap", "cue_quality_target_gap",
            "cue_quality_mae", "cue_quality_corr", "cue_quality_loss",
            "cue_quality_zero_pred",
            "skeleton_usefulness_gate", "skeleton_usefulness_target",
            "skeleton_usefulness_loss", "skeleton_usefulness_delta",
            "skeleton_usefulness_help_rate", "skeleton_usefulness_mask_rate",
            "skeleton_usefulness_relative_delta", "skeleton_usefulness_threshold",
            "skeleton_usefulness_active", "train_skeleton_usefulness_gate",
            "cue_usefulness_prior_loss",
            "skeleton_usefulness_gate_p05", "skeleton_usefulness_gate_p50",
            "skeleton_usefulness_gate_p95", "train_skeleton_usefulness_gate_p05",
            "train_skeleton_usefulness_gate_p50", "train_skeleton_usefulness_gate_p95",
            "dual_ru_phase", "dual_gate_ramp",
            "score_mean", "score_min", "score_max",
            "motion_action_prior", "skeleton_action_prior",
            "motion_usefulness", "skeleton_usefulness",
            "motion_gate_norm", "motion_gate", "skeleton_gate_norm",
            "motion_reliability_raw", "skeleton_reliability_raw",
            "motion_reliability_reference", "skeleton_reliability_reference",
            "motion_reliability_zero_reference",
            "skeleton_reliability_zero_reference",
            "motion_reliability_zero", "skeleton_reliability_zero",
            "motion_reliability_corrupt25", "motion_reliability_corrupt50",
            "motion_reliability_corrupt75",
            "skeleton_reliability_corrupt25",
            "skeleton_reliability_corrupt50",
            "skeleton_reliability_corrupt75",
            "motion_reliability_loss", "skeleton_reliability_loss",
            "motion_reliability_target", "skeleton_reliability_target",
            "motion_reliability_mae", "skeleton_reliability_mae",
            "motion_reliability_corr", "skeleton_reliability_corr",
            "motion_reliability_clean_pred", "skeleton_reliability_clean_pred",
            "motion_reliability_corrupt_pred", "skeleton_reliability_corrupt_pred",
            "motion_usefulness_target", "skeleton_usefulness_target",
            "motion_usefulness_loss", "skeleton_usefulness_loss",
            "motion_usefulness_delta", "skeleton_usefulness_delta",
            "motion_usefulness_mse_delta", "skeleton_usefulness_mse_delta",
            "motion_usefulness_mse_benefit",
            "skeleton_usefulness_mse_benefit",
            "motion_usefulness_mse_scale",
            "skeleton_usefulness_mse_scale",
            "motion_usefulness_score_effect",
            "skeleton_usefulness_score_effect",
            "motion_usefulness_effect_scale",
            "skeleton_usefulness_effect_scale",
            "motion_usefulness_effect_valid_rate",
            "skeleton_usefulness_effect_valid_rate",
            "motion_usefulness_directional_utility",
            "skeleton_usefulness_directional_utility",
            "motion_usefulness_rank_delta", "skeleton_usefulness_rank_delta",
            "motion_usefulness_rank_scale",
            "skeleton_usefulness_rank_scale",
            "motion_usefulness_help_rate", "skeleton_usefulness_help_rate",
            "motion_usefulness_mask_rate", "skeleton_usefulness_mask_rate",
            "motion_usefulness_target_std", "skeleton_usefulness_target_std",
            "motion_usefulness_target_corr", "skeleton_usefulness_target_corr",
            "motion_usefulness_target_corr_epoch",
            "skeleton_usefulness_target_corr_epoch",
            "motion_usefulness_target_accuracy_epoch",
            "skeleton_usefulness_target_accuracy_epoch",
            "motion_usefulness_harm_rate", "skeleton_usefulness_harm_rate",
            "motion_usefulness_neutral_rate", "skeleton_usefulness_neutral_rate",
            "motion_usefulness_confidence", "skeleton_usefulness_confidence",
            "motion_usefulness_objective_gain",
            "skeleton_usefulness_objective_gain",
            "motion_usefulness_grid_margin",
            "skeleton_usefulness_grid_margin",
            "motion_usefulness_optimal_gate",
            "skeleton_usefulness_optimal_gate",
            "motion_usefulness_joint_utility",
            "skeleton_usefulness_joint_utility",
            "motion_usefulness_context_agreement",
            "skeleton_usefulness_context_agreement",
            "motion_usefulness_default_gate",
            "skeleton_usefulness_default_gate",
            "dual_usefulness_error_scale", "dual_usefulness_rank_scale",
            "dual_usefulness_loss", "dual_usefulness_weight",
            "dual_no_regret_loss", "dual_no_regret_mse",
            "dual_no_regret_rank", "dual_no_regret_active_rate",
            "dual_no_regret_mse_delta", "dual_no_regret_rank_delta",
            "dual_joint_global_motion_target",
            "dual_joint_global_skeleton_target",
            "dual_joint_soft_motion_gate",
            "dual_joint_soft_skeleton_gate",
            "dual_joint_best_motion_gate",
            "dual_joint_best_skeleton_gate",
            "dual_joint_global_gain", "dual_joint_global_relative_gain",
            "dual_joint_global_confidence",
            "dual_joint_sample_confidence",
            "dual_joint_sample_correction",
            "dual_joint_default_mse", "dual_joint_best_mse",
            "dual_joint_default_rank", "dual_joint_best_rank",
            "dual_task_rank_loss", "dual_gate_anchor_loss",
            "dual_output_anchor_loss", "paired_fixed_coef",
            "paired_coef_delta", "paired_fixed_loss", "paired_loss_delta"]
    if not detailed:
        keys = [
            "dual_ru_phase", "dual_gate_ramp", "score_mean", "score_min",
            "score_max", "motion_action_prior", "skeleton_action_prior",
            "motion_reliability", "skeleton_reliability",
            "motion_usefulness", "skeleton_usefulness",
            "motion_gate_norm", "skeleton_gate_norm",
            "motion_reliability_loss", "skeleton_reliability_loss",
            "motion_usefulness_target", "skeleton_usefulness_target",
            "motion_usefulness_loss", "skeleton_usefulness_loss",
            "motion_usefulness_score_effect",
            "skeleton_usefulness_score_effect",
            "motion_usefulness_confidence", "skeleton_usefulness_confidence",
            "dual_joint_global_motion_target",
            "dual_joint_global_skeleton_target",
            "dual_joint_global_confidence", "dual_usefulness_loss",
            "dual_no_regret_loss", "paired_coef_delta", "paired_loss_delta"
        ]
    quantile_names = (
        DUAL_GATE_VALUE_KEYS if detailed else
        ('motion_reliability', 'skeleton_reliability',
         'motion_gate_norm', 'skeleton_gate_norm')
    )
    for name in quantile_names:
        keys.extend([
            f'{name}_p05', f'{name}_p50', f'{name}_p95'
        ])
    parts = []
    for key in dict.fromkeys(keys):
        if key in info_meters:
            parts.append(f"{key} {info_meters[key].avg:.4f}")
    for prefix in ("", "train_"):
        mean_key = f"{prefix}skeleton_usefulness_gate"
        square_key = f"{prefix}skeleton_usefulness_gate_square"
        if mean_key in info_meters and square_key in info_meters:
            mean = info_meters[mean_key].avg
            square = info_meters[square_key].avg
            std = max(square - mean * mean, 0) ** 0.5
            parts.append(f"{prefix}skeleton_usefulness_gate_std {std:.4f}")
    for name in quantile_names:
        square_key = f'{name}_square'
        if name in info_meters and square_key in info_meters:
            mean = info_meters[name].avg
            square = info_meters[square_key].avg
            std = max(square - mean * mean, 0) ** 0.5
            parts.append(f'{name}_std {std:.4f}')
    if 'score_mean' in info_meters and 'score_square' in info_meters:
        mean = info_meters['score_mean'].avg
        square = info_meters['score_square'].avg
        std = max(square - mean * mean, 0) ** 0.5
        parts.append(f'score_std {std:.4f}')
    return " ".join(parts)


def train_epoch(model, optimizer, lr_scheduler, dataloader, cur_epoch, clip_grad=False):
    losses = AverageMeter()
    info_meters = {}
    total_preds = []
    total_labels = []
    gate_values = []
    dual_gate_values = {name: [] for name in DUAL_GATE_VALUE_KEYS}
    dual_target_values = {
        cue: {'prediction': [], 'target': [], 'weight': [], 'valid': []}
        for cue in ('motion', 'skeleton')
    }
    tm = time.time()

    model.train()
    if hasattr(model, 'prepare_dual_ru_train_mode'):
        model.prepare_dual_ru_train_mode()

    for batch in dataloader:
        model_input, label = prepare_model_batch(batch)

        counterfactual_info = {}
        if (hasattr(model, 'dual_ru_needs_counterfactual') and
                model.dual_ru_needs_counterfactual()):
            counterfactual_info = model.build_dual_ru_counterfactual_targets(
                model_input, label
            )
            model.prepare_dual_ru_train_mode()
        preds, other_info = model(model_input)
        other_info.update(counterfactual_info)
        if "_skeleton_usefulness_gate_pred" in other_info:
            gate_values.append(
                other_info["_skeleton_usefulness_gate_pred"].detach().cpu().reshape(-1)
            )
        collect_dual_gate_values(dual_gate_values, other_info)
        loss = model.call_loss(preds, label, **other_info)
        if getattr(model, "last_loss_info", None):
            other_info.update(model.last_loss_info)
        collect_dual_target_values(dual_target_values, other_info)
        update_info_meters(info_meters, other_info)

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

        losses.update(loss.item(), len(label))
        total_preds.extend(preds.cpu().detach().numpy().tolist())
        total_labels.extend(label.cpu().detach().numpy().tolist())

    if lr_scheduler is not None:
        lr_scheduler.step(cur_epoch+1)

    tm = time.time() - tm
    coef = spearman_coefficient(total_preds, total_labels)
    update_gate_quantiles(info_meters, gate_values)
    update_dual_gate_quantiles(info_meters, dual_gate_values)
    update_dual_target_diagnostics(info_meters, dual_target_values)

    return tm, losses.avg * 100, coef * 100, info_meters


def test_epoch(model, dataloader, dual_force_gates=None,
               dual_corruption=None, collect_dual_diagnostics=None):
    losses = AverageMeter()
    info_meters = {}
    total_preds = []
    total_labels = []
    gate_values = []
    dual_gate_values = {name: [] for name in DUAL_GATE_VALUE_KEYS}
    tm = time.time()

    model.eval()
    previous_diagnostics = getattr(model, 'dual_ru_diagnostics', True)
    diagnostics_changed = (
        collect_dual_diagnostics is not None and is_dual_ru_version(model)
    )
    if diagnostics_changed:
        model.dual_ru_diagnostics = bool(collect_dual_diagnostics)

    with torch.no_grad():
        for batch in dataloader:
            model_input, label = prepare_model_batch(batch)
            if dual_corruption is not None:
                preds, other_info = model(
                    model_input, dual_force_gates=dual_force_gates,
                    dual_corruption=dual_corruption
                )
            elif dual_force_gates is not None:
                preds, other_info = model(
                    model_input, dual_force_gates=dual_force_gates
                )
            else:
                preds, other_info = model(model_input)
            if "_skeleton_usefulness_gate_pred" in other_info:
                gate_values.append(
                    other_info["_skeleton_usefulness_gate_pred"].detach().cpu().reshape(-1)
                )
            collect_dual_gate_values(dual_gate_values, other_info)
            loss = model.call_loss(preds, label, **other_info)
            if getattr(model, "last_loss_info", None):
                other_info.update(model.last_loss_info)
            update_info_meters(info_meters, other_info)

            losses.update(loss.item(), label.shape[0])
            total_preds.extend(preds.cpu().detach().numpy().tolist())
            total_labels.extend(label.cpu().detach().numpy().tolist())


    coef = spearman_coefficient(total_preds, total_labels)
    update_gate_quantiles(info_meters, gate_values)
    update_dual_gate_quantiles(info_meters, dual_gate_values)

    tm = time.time() - tm
    if diagnostics_changed:
        model.dual_ru_diagnostics = previous_diagnostics

    return tm, losses.avg * 100, coef * 100, info_meters


def info_meter_std(info_meters, mean_key, square_key):
    if mean_key not in info_meters or square_key not in info_meters:
        return 0.0
    mean = info_meters[mean_key].avg
    square = info_meters[square_key].avg
    return max(square - mean * mean, 0.0) ** 0.5


def set_info_meter(info_meters, key, value):
    meter = AverageMeter()
    meter.update(float(value), 1)
    info_meters[key] = meter


def select_dual_teacher_candidate(model, dataloader, log_func):
    if dataloader is None:
        raise RuntimeError('dual RU requires a train-probe dataloader')
    results = []
    for candidate in model.dual_teacher_candidate_names():
        if not model.restore_dual_teacher_candidate(candidate):
            continue
        _, loss, coefficient, info = test_epoch(
            model, dataloader, collect_dual_diagnostics=False
        )
        score_std = info_meter_std(info, 'score_mean', 'score_square')
        healthy = model.dual_teacher_metrics_healthy(
            loss, coefficient, score_std
        )
        results.append({
            'candidate': candidate,
            'loss': loss,
            'coefficient': coefficient,
            'score_std': score_std,
            'healthy': healthy
        })
        log_func(
            'Dual RU teacher candidate: '
            f'{candidate} loss {loss:.4f} coef {coefficient:.2f} '
            f'score_std {score_std:.4f} healthy {healthy}'
        )
    healthy_results = [result for result in results if result['healthy']]
    if not healthy_results:
        raise RuntimeError('dual RU has no healthy teacher candidate')
    selected = min(
        healthy_results,
        key=lambda result: (result['loss'], -result['coefficient'])
    )
    confirmed = model.confirm_dual_teacher_candidate(
        selected['candidate'], selected['loss'], selected['coefficient'],
        selected['score_std']
    )
    if not confirmed:
        raise RuntimeError('dual RU could not confirm its teacher candidate')
    return selected['candidate']


def train_loop(model, optimizer, lr_scheduler, train_dataloader, test_dataloader, epoch_num,
               save_base_model="", save_final_model="", save_moniter="",
               log_func=None, clip_grad=False, teacher_probe_dataloader=None):
    max_coef, max_avg_coef, min_loss, min_avg_loss = 0, 0, 10, 10
    coef_of_saved_model = 0
    _train_loss, _test_loss = [], []
    train_coefs, test_coefs = [], []
    best_checkpoint_coef = float('-inf')
    previous_dual_phase = None
    for cur_epoch in range(epoch_num):
        if hasattr(model, "set_cue_epoch"):
            model.set_cue_epoch(cur_epoch, epoch_num)
        if is_dual_ru_version(model):
            current_phase = model.dual_ru_phase()
            if current_phase != previous_dual_phase:
                version = model.dual_ru_version
                phase1_name = {
                    5: 'frozen-teacher continuous gate supervision',
                    6: 'frozen-teacher continuous gate supervision',
                    7: 'frozen-teacher joint marginal supervision',
                    8: 'baseline-centered usefulness supervision',
                    9: 'three-point MSE usefulness supervision',
                    10: 'joint grid global usefulness supervision',
                    11: 'joint grid action-prior supervision'
                }.get(version, 'counterfactual usefulness supervision')
                phase2_name = 'adaptive reliability x usefulness'
                if version >= 7:
                    phase2_name = 'adaptive gates with frozen scoring core'
                if version >= 9:
                    phase2_name = 'adaptive gates with paired no-regret'
                if version == 11:
                    phase2_name = 'action prior x sample reliability with no-regret'
                phase_names = {
                    0: 'fixed cue warmup',
                    1: phase1_name,
                    2: phase2_name
                }
                log_func(
                    f'Dual RU phase {current_phase}: '
                    f'{phase_names[current_phase]}'
                )
                lr_scale = apply_dual_phase_lr(
                    model, optimizer, current_phase
                )
                if version >= 4:
                    log_func(f'Dual RU optimizer LR scale: {lr_scale}')
                if current_phase == 1 and version >= 6:
                    if version >= 9:
                        select_dual_teacher_candidate(
                            model, teacher_probe_dataloader, log_func
                        )
                    teacher_info = model.dual_teacher_probe_info()
                    if not teacher_info['restored']:
                        raise RuntimeError(
                            'dual RU could not restore a healthy phase-0 teacher'
                        )
                    reset_count = model.reset_dual_teacher_optimizer_state(
                        optimizer
                    )
                    log_func(
                        'Dual RU stable teacher restored: '
                        f"epoch {teacher_info['epoch']} "
                        f"probe_loss {teacher_info['loss']:.4f} "
                        f"probe_coef {teacher_info['coefficient']:.2f} "
                        f"score_std {teacher_info['score_std']:.4f} "
                        f"ema_updates {teacher_info['ema_updates']} "
                        f"candidate {teacher_info['candidate']} "
                        f"optimizer_states_reset {reset_count}"
                    )
                previous_dual_phase = current_phase
        train_tm, train_loss, train_coef, train_info = train_epoch(model, optimizer, lr_scheduler, train_dataloader,
                                                                   cur_epoch, clip_grad=clip_grad)
        test_tm, test_loss, test_coef, test_info = test_epoch(
            model, test_dataloader, collect_dual_diagnostics=False
        )

        if (is_dual_ru_version(model, 6) and
                model.dual_ru_phase() == 0 and
                teacher_probe_dataloader is not None):
            warmup_end = max(1, int(round(
                epoch_num * model.dual_ru_warmup_ratio
            )))
            probe_due = ((cur_epoch + 1) % 5 == 0 or
                         cur_epoch + 1 == warmup_end)
            if probe_due:
                _, probe_loss, probe_coef, probe_info = test_epoch(
                    model, teacher_probe_dataloader,
                    collect_dual_diagnostics=False
                )
                probe_std = info_meter_std(
                    probe_info, 'score_mean', 'score_square'
                )
                selected = model.consider_dual_teacher_snapshot(
                    cur_epoch, probe_loss, probe_coef, probe_std
                )
                log_func(
                    'Dual RU teacher probe: '
                    f'epoch {cur_epoch} loss {probe_loss:.4f} '
                    f'coef {probe_coef:.2f} score_std {probe_std:.4f} '
                    f"selected {selected}"
                )

        if (is_dual_ru_version(model, 6) and
                model.dual_ru_phase() == 2):
            _, paired_fixed_loss, paired_fixed_coef, _ = test_epoch(
                model, test_dataloader, dual_force_gates=(1.0, 0.5),
                collect_dual_diagnostics=False
            )
            set_info_meter(test_info, 'paired_fixed_coef', paired_fixed_coef)
            set_info_meter(
                test_info, 'paired_coef_delta', test_coef - paired_fixed_coef
            )
            set_info_meter(test_info, 'paired_fixed_loss', paired_fixed_loss)
            set_info_meter(
                test_info, 'paired_loss_delta', test_loss - paired_fixed_loss
            )

        metric_active = (
            not is_dual_ru_version(model) or
            model.dual_ru_phase() == 2
        )
        train_coefs.append(train_coef)
        _train_loss.append(train_loss)
        if metric_active:
            test_coefs.append(test_coef)
            _test_loss.append(test_loss)
        if save_base_model != '' and metric_active:
            if save_moniter == "avg_coef" and np.mean(test_coefs[-10:]) > max_avg_coef: 
                torch.save(model, save_base_model)
                coef_of_saved_model = test_coef
            if save_moniter == "avg_loss" and np.mean(_test_loss[-10:]) < min_avg_loss:
                torch.save(model, save_base_model)
                coef_of_saved_model = test_coef

        if (save_final_model != "" and metric_active and
                test_coef > best_checkpoint_coef):
            parent = os.path.dirname(os.path.abspath(save_final_model))
            os.makedirs(parent, exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(), "cue_epoch": cur_epoch,
                 "cue_total_epochs": epoch_num},
                save_final_model
            )
            best_checkpoint_coef = test_coef
            coef_of_saved_model = test_coef

        rolling_avg_coef = (
            np.mean(test_coefs[-10:]) if test_coefs else test_coef
        )
        rolling_avg_loss = (
            np.mean(_test_loss[-10:]) if _test_loss else test_loss
        )
        if metric_active:
            max_coef = max(max_coef, test_coef)
            max_avg_coef = max(max_avg_coef, rolling_avg_coef)
            min_loss = min(min_loss, test_loss)
            min_avg_loss = min(min_avg_loss, rolling_avg_loss)
        for key in ["cue_quality_clean_pred", "cue_quality_corrupt_pred",
                    "cue_quality_target", "cue_quality_gap",
                    "cue_quality_target_gap", "cue_quality_mae",
                    "cue_quality_corr", "cue_quality_loss",
                    "cue_quality_zero_pred",
                    "skeleton_usefulness_target",
                    "skeleton_usefulness_loss", "skeleton_usefulness_delta",
                    "skeleton_usefulness_help_rate",
                    "skeleton_usefulness_mask_rate",
                    "skeleton_usefulness_relative_delta",
                    "skeleton_usefulness_threshold",
                    "skeleton_usefulness_active",
                    "motion_reliability_loss", "skeleton_reliability_loss",
                    "motion_reliability_target", "skeleton_reliability_target",
                    "motion_reliability_mae", "skeleton_reliability_mae",
                    "motion_reliability_corr", "skeleton_reliability_corr",
                    "motion_reliability_clean_pred",
                    "skeleton_reliability_clean_pred",
                    "motion_reliability_corrupt_pred",
                    "skeleton_reliability_corrupt_pred",
                    "motion_reliability_zero_pred",
                    "skeleton_reliability_zero_pred",
                    "motion_usefulness_target", "skeleton_usefulness_target",
                    "motion_usefulness_loss", "skeleton_usefulness_loss",
                     "motion_usefulness_delta", "skeleton_usefulness_delta",
                     "motion_usefulness_mse_delta",
                     "skeleton_usefulness_mse_delta",
                     "motion_usefulness_mse_benefit",
                     "skeleton_usefulness_mse_benefit",
                     "motion_usefulness_mse_scale",
                     "skeleton_usefulness_mse_scale",
                     "motion_usefulness_score_effect",
                     "skeleton_usefulness_score_effect",
                     "motion_usefulness_effect_scale",
                     "skeleton_usefulness_effect_scale",
                     "motion_usefulness_effect_valid_rate",
                     "skeleton_usefulness_effect_valid_rate",
                     "motion_usefulness_directional_utility",
                     "skeleton_usefulness_directional_utility",
                     "motion_usefulness_rank_delta",
                     "skeleton_usefulness_rank_delta",
                     "motion_usefulness_rank_scale",
                     "skeleton_usefulness_rank_scale",
                     "motion_usefulness_help_rate",
                     "skeleton_usefulness_help_rate",
                     "motion_usefulness_mask_rate",
                     "skeleton_usefulness_mask_rate",
                     "motion_usefulness_target_std",
                     "skeleton_usefulness_target_std",
                     "motion_usefulness_target_corr",
                     "skeleton_usefulness_target_corr",
                     "motion_usefulness_target_corr_epoch",
                     "skeleton_usefulness_target_corr_epoch",
                     "motion_usefulness_target_accuracy_epoch",
                     "skeleton_usefulness_target_accuracy_epoch",
                     "motion_usefulness_harm_rate",
                     "skeleton_usefulness_harm_rate",
                     "motion_usefulness_neutral_rate",
                     "skeleton_usefulness_neutral_rate",
                      "motion_usefulness_confidence",
                      "skeleton_usefulness_confidence",
                      "motion_usefulness_objective_gain",
                      "skeleton_usefulness_objective_gain",
                      "motion_usefulness_grid_margin",
                      "skeleton_usefulness_grid_margin",
                      "motion_usefulness_optimal_gate",
                      "skeleton_usefulness_optimal_gate",
                      "motion_usefulness_joint_utility",
                      "skeleton_usefulness_joint_utility",
                      "motion_usefulness_context_agreement",
                      "skeleton_usefulness_context_agreement",
                      "motion_usefulness_default_gate",
                      "skeleton_usefulness_default_gate",
                      "dual_usefulness_error_scale",
                      "dual_usefulness_rank_scale",
                      "dual_usefulness_loss", "dual_usefulness_weight",
                      "dual_no_regret_loss", "dual_no_regret_mse",
                      "dual_no_regret_rank", "dual_no_regret_active_rate",
                      "dual_no_regret_mse_delta",
                      "dual_no_regret_rank_delta",
                      "dual_joint_global_motion_target",
                      "dual_joint_global_skeleton_target",
                      "dual_joint_soft_motion_gate",
                      "dual_joint_soft_skeleton_gate",
                      "dual_joint_best_motion_gate",
                      "dual_joint_best_skeleton_gate",
                      "dual_joint_global_gain",
                      "dual_joint_global_relative_gain",
                      "dual_joint_global_confidence",
                      "dual_joint_sample_confidence",
                      "dual_joint_sample_correction",
                      "dual_joint_default_mse", "dual_joint_best_mse",
                      "dual_joint_default_rank", "dual_joint_best_rank",
                      "dual_task_rank_loss", "dual_gate_anchor_loss",
                      "dual_output_anchor_loss"]:
            if key in train_info:
                test_info[key] = train_info[key]
        for key in ["skeleton_usefulness_gate", "skeleton_usefulness_gate_square"]:
            if key in train_info:
                test_info[f"train_{key}"] = train_info[key]
        for key in ["skeleton_usefulness_gate_p05", "skeleton_usefulness_gate_p50",
                    "skeleton_usefulness_gate_p95"]:
            if key in train_info:
                test_info[f"train_{key}"] = train_info[key]
        info_text = format_info_meters(test_info)
        if info_text:
            info_text = f"\t {info_text}"
        log_func('Epoch[{0}/{1}] \t'
                 'Time: {train_tm:.1f}/{test_tm:.1f} \t'
                 'Loss {train_loss:.4f}/{test_loss:.4f} \t'
                 'Avg_loss {avg_loss:.2f}/{min_avg_loss:.2f}\t'
                 'Coef {train_coef:.2f}/{test_coef:.2f} \t'
                 'Avg_coef {avg_coef:.2f}/{max_avg_coef:.2f} \t'
                 'BestCoef {max_coef:.2f}\t BestLoss {min_loss:.2f}\t {info_text}'.format(
            cur_epoch, epoch_num, train_tm=train_tm, test_tm=test_tm,
            train_loss=train_loss, test_loss=test_loss,
            avg_loss=rolling_avg_loss, min_avg_loss=min_avg_loss,
            train_coef=train_coef, test_coef=test_coef,
            avg_coef=rolling_avg_coef, max_avg_coef=max_avg_coef,
            max_coef=max_coef, min_loss=min_loss, info_text=info_text
        ))
    if getattr(model, 'dual_ru_gate', False):
        if save_final_model != "":
            stem, extension = os.path.splitext(save_final_model)
            final_epoch_path = f'{stem}_last{extension}'
            torch.save(
                {"state_dict": model.state_dict(), "cue_epoch": epoch_num - 1,
                 "cue_total_epochs": epoch_num},
                final_epoch_path
            )
        _, _, _, train_eval_info = test_epoch(
            model, train_dataloader, collect_dual_diagnostics=True
        )
        train_eval_text = format_info_meters(train_eval_info, detailed=True)
        if train_eval_text:
            log_func(f'Final train-eval gates: {train_eval_text}')
        if is_dual_ru_version(model, 5):
            _, fixed_loss, fixed_coef, fixed_info = test_epoch(
                model, test_dataloader, dual_force_gates=(1.0, 0.5),
                collect_dual_diagnostics=False
            )
            fixed_text = format_info_meters(fixed_info, detailed=True)
            log_func(
                'Final paired fixed gates: '
                f'Coef {fixed_coef:.2f} Loss {fixed_loss:.4f}'
            )
            log_func(
                'Final paired adaptive-minus-fixed: '
                f'Coef {test_coef - fixed_coef:+.2f} '
                f'Loss {test_loss - fixed_loss:+.4f}'
            )
            if fixed_text:
                log_func(f'Final paired fixed gate stats: {fixed_text}')
        if is_dual_ru_version(model, 10):
            for motion_gate in model.dual_ru_gate_grid:
                for skeleton_gate in model.dual_ru_gate_grid:
                    _, grid_loss, grid_coef, _ = test_epoch(
                        model, test_dataloader,
                        dual_force_gates=(motion_gate, skeleton_gate),
                        collect_dual_diagnostics=False
                    )
                    log_func(
                        'Final clean joint gate grid: '
                        f'motion {motion_gate:.2f} '
                        f'skeleton {skeleton_gate:.2f} '
                        f'coef {grid_coef:.2f} loss {grid_loss:.4f} '
                        f'delta_coef {grid_coef - fixed_coef:+.2f} '
                        f'delta_loss {grid_loss - fixed_loss:+.4f}'
                    )
            component_modes = (
                ('motion_adaptive', (None, 0.5)),
                ('skeleton_adaptive', (1.0, None))
            )
            for mode, force_gates in component_modes:
                _, component_loss, component_coef, component_info = test_epoch(
                    model, test_dataloader, dual_force_gates=force_gates,
                    collect_dual_diagnostics=False
                )
                motion_gate = component_info['motion_gate_norm'].avg
                skeleton_gate = component_info['skeleton_gate_norm'].avg
                log_func(
                    'Final adaptive component ablation: '
                    f'mode {mode} motion_gate {motion_gate:.4f} '
                    f'skeleton_gate {skeleton_gate:.4f} '
                    f'coef {component_coef:.2f} loss {component_loss:.4f} '
                    f'delta_coef {component_coef - fixed_coef:+.2f} '
                    f'delta_loss {component_loss - fixed_loss:+.4f}'
                )
        if is_dual_ru_version(model, 9):
            for cue in ('motion', 'skeleton'):
                for severity in (0.5, 1.0):
                    _, corrupt_loss, corrupt_coef, corrupt_info = test_epoch(
                        model, test_dataloader,
                        dual_corruption=(cue, severity),
                        collect_dual_diagnostics=False
                    )
                    force_gates = (1.0, 0.5)
                    if is_dual_ru_version(model, 10):
                        force_gates = (
                            (1.0, None) if cue == 'motion' else
                            (None, 0.5)
                        )
                    _, corrupt_fixed_loss, corrupt_fixed_coef, _ = test_epoch(
                        model, test_dataloader,
                        dual_force_gates=force_gates,
                        dual_corruption=(cue, severity),
                        collect_dual_diagnostics=False
                    )
                    reliability_key = f'{cue}_reliability'
                    reliability = corrupt_info[reliability_key].avg
                    usefulness = corrupt_info[f'{cue}_usefulness'].avg
                    gate_norm = corrupt_info[f'{cue}_gate_norm'].avg
                    log_func(
                        'Final corrupted paired evaluation: '
                        f'cue {cue} severity {severity:.2f} '
                        f'reliability {reliability:.4f} '
                        f'usefulness {usefulness:.4f} '
                        f'gate_norm {gate_norm:.4f} '
                        f'adaptive_coef {corrupt_coef:.2f} '
                        f'cue_fixed_coef {corrupt_fixed_coef:.2f} '
                        f'delta_coef {corrupt_coef - corrupt_fixed_coef:+.2f} '
                        f'adaptive_loss {corrupt_loss:.4f} '
                        f'cue_fixed_loss {corrupt_fixed_loss:.4f} '
                        f'delta_loss {corrupt_loss - corrupt_fixed_loss:+.4f}'
                    )
    return max_avg_coef, max_coef, min_avg_loss, min_loss, coef_of_saved_model


def go(model_kwargs, optimizer_kwargs, lr_scheduler_kwargs, dataset_kwargs, epoch_num,
         save_base_model="", save_final_model="", save_moniter="",
         log_func=None, seed=0, clip_grad=False):
    setup_seed(seed)
    dataloaders = build_dataset(**dataset_kwargs)
    if len(dataloaders) == 3:
        train_dataloader, test_dataloader, teacher_probe_dataloader = dataloaders
    else:
        train_dataloader, test_dataloader = dataloaders
        teacher_probe_dataloader = None

    model = build_model(**model_kwargs)
    optimizer = build_optimizer(model, model_name=model_kwargs["model_name"], **optimizer_kwargs)
    lr_scheduler = build_lr_scheduler(optimizer, **lr_scheduler_kwargs)

    model.cuda()
    rets = train_loop(model, optimizer, lr_scheduler, train_dataloader, test_dataloader, epoch_num=epoch_num, 
            save_base_model=save_base_model, save_moniter=save_moniter, save_final_model=save_final_model, 
            log_func=log_func, clip_grad=clip_grad,
            teacher_probe_dataloader=teacher_probe_dataloader)
    log_func(f"best avg coef: {rets[0]/100:.4f}\tbest coef:{rets[1]/100:.4f}\t"
             f"min avg loss: {rets[2]/100:.6f}\tmin loss:{rets[3]/100:.6f}\t"
             f"coef of saved model: {rets[4]/100:.4f}")
    return rets


def test_model(model_kwargs, dataset_kwargs, ckpt_path="", log_func=None, seed=0):
    setup_seed(seed)
    dataloaders = build_dataset(**dataset_kwargs)
    test_dataloader = dataloaders[1]

    model = build_model(**model_kwargs)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        cue_epoch = checkpoint.get("cue_epoch", 0)
        cue_total_epochs = checkpoint.get("cue_total_epochs", cue_epoch + 1)
    else:
        state_dict = checkpoint
        cue_epoch = getattr(model, "cue_usefulness_warmup_epochs", 0)
        cue_total_epochs = max(cue_epoch + 1, 1)
    model.load_state_dict(state_dict)
    if hasattr(model, "set_cue_epoch"):
        model.set_cue_epoch(cue_epoch, cue_total_epochs)
    
    model.cuda()

    rets = test_epoch(
        model, test_dataloader, collect_dual_diagnostics=True
    )

    info_text = format_info_meters(rets[3], detailed=True)
    if info_text:
        info_text = f"\t{info_text}"
    log_func(f"Coef:{rets[2] / 100:.4f}\tLoss: {rets[1] / 100:.6f}{info_text}")
    

def print_flops(model_kwargs):
    from thop import clever_format, profile
    model = build_model(**model_kwargs)
    data_in = {
        "V": torch.zeros([1, 70, 1024]),
        "F": torch.zeros([1, 70, 1024]),
        "A": torch.zeros([1, 70, 768])
    }
    if model_kwargs.get("model_conf", {}).get("use_skeleton", False):
        data_in["S"] = torch.zeros([1, 70, 132])
    macs, params = profile(model, inputs=(data_in,))
    macs, params = clever_format([macs, params], "%.3f")
    return macs, params


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  


def log_info(info, logger, print_log):
    if logger is not None:
        logger.info(info)
    elif print_log is not False:
        print(info)
