import os
import argparse
import sys
import numpy as np
from functools import partial

RELIABILITY_POLICIES = tuple(
    f'reliability_aware_v{version}' for version in range(1, 4)
)
QUALITY_POLICIES = tuple(
    f'quality_supervised_v{version}' for version in range(1, 11)
)
FIXED_CUE_POLICIES = tuple(
    f'fixed_cue_v{version}' for version in range(1, 4)
)
DUAL_RU_POLICIES = tuple(
    f'dual_ru_v{version}' for version in range(1, 12)
)
DUAL_RU_V5_PLUS = DUAL_RU_POLICIES[4:]
DUAL_RU_V6_PLUS = DUAL_RU_POLICIES[5:]
DUAL_RU_V7_PLUS = DUAL_RU_POLICIES[6:]
DUAL_RU_V8_PLUS = DUAL_RU_POLICIES[7:]
DUAL_RU_V9_PLUS = DUAL_RU_POLICIES[8:]
DUAL_RU_ZERO_RANK_POLICIES = (
    DUAL_RU_POLICIES[2:6] + DUAL_RU_POLICIES[8:]
)
ACTIONS = ('TES', 'PCS', 'Ball', 'Clubs', 'Hoop', 'Ribbon')


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', required=True)

    parser.add_argument('--feats', type=int, default=1, required=True)
    parser.add_argument('--action', type=str, default='Ball', required=True,
                        choices=ACTIONS)
    parser.add_argument('--multi_modality', action='store_true', default=False)
    parser.add_argument('--modality', type=str, default='V')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cue_policy', type=str, default='none',
                        choices=(
                            ('none', 'action_aware_v1') +
                            RELIABILITY_POLICIES + QUALITY_POLICIES +
                            FIXED_CUE_POLICIES + DUAL_RU_POLICIES
                        ))
    parser.add_argument('--skeleton_mode', type=str, default='auto',
                        choices=['auto', 'off', 'residual', 'temporal'])
    parser.add_argument('--use_skeleton', dest='use_skeleton', action='store_true', default=None)
    parser.add_argument('--no_skeleton', dest='use_skeleton', action='store_false')
    parser.add_argument('--skeleton_root', type=str, default='.')
    parser.add_argument('--skeleton_gate_init', type=float, default=-2.0)
    parser.add_argument('--skeleton_gate_max', type=float, default=None)
    parser.add_argument('--skeleton_alpha', type=float, default=-1.0)
    parser.add_argument('--skeleton_temporal_weight', action='store_true', default=False)
    parser.add_argument('--skeleton_temporal_beta', type=float, default=0.02)
    parser.add_argument('--skeleton_confidence_gate', action='store_true', default=False)
    parser.add_argument('--skeleton_late_fusion', action='store_true', default=False)
    parser.add_argument('--skeleton_late_gamma', type=float, default=0.01)
    parser.add_argument('--motion_saliency_proxy', dest='motion_saliency_proxy',
                        action='store_true', default=None)
    parser.add_argument('--no_motion_saliency_proxy', dest='motion_saliency_proxy',
                        action='store_false')
    parser.add_argument('--motion_saliency_alpha', type=float, default=None)
    parser.add_argument('--motion_saliency_mode', type=str, default='stage_residual',
                        choices=['stage_residual', 'pool'])
    parser.add_argument('--motion_saliency_source', type=str, default=None,
                        choices=['energy', 'delta'])
    parser.add_argument('--motion_saliency_positive_only', dest='motion_saliency_positive_only',
                        action='store_true', default=None)
    parser.add_argument('--no_motion_saliency_positive_only', dest='motion_saliency_positive_only',
                        action='store_false')
    parser.add_argument('--auto_gate', action='store_true', default=False)
    parser.add_argument('--auto_gate_v2', action='store_true', default=False)
    parser.add_argument('--auto_gate_init', type=float, default=-2.0)
    parser.add_argument('--auto_gate_skeleton_max', type=float, default=None)
    parser.add_argument('--auto_gate_motion_max', type=float, default=None)
    parser.add_argument('--auto_gate_l1', type=float, default=0.0)
    parser.add_argument('--auto_gate_budget', type=float, default=-1.0)
    parser.add_argument('--auto_gate_budget_weight', type=float, default=0.0)
    parser.add_argument('--auto_gate_motion_gamma', type=float, default=0.5)
    parser.add_argument('--auto_gate_skeleton_gamma', type=float, default=0.5)
    parser.add_argument('--auto_gate_no_downscale', action='store_true', default=False)
    parser.add_argument('--cue_reliability', action='store_true', default=False)
    parser.add_argument('--cue_reliability_mode', type=str, default='hard',
                        choices=['hard', 'soft', 'calibrated'])
    parser.add_argument('--cue_reliability_floor', type=float, default=0.75)
    parser.add_argument('--cue_reliability_span', type=float, default=0.5)
    parser.add_argument('--cue_reliability_momentum', type=float, default=0.1)
    parser.add_argument('--cue_reliability_min_std', type=float, default=0.05)
    parser.add_argument('--cue_quality_supervision', action='store_true', default=False)
    parser.add_argument('--cue_corruption_prob', type=float, default=0.5)
    parser.add_argument('--cue_corruption_max', type=float, default=0.6)
    parser.add_argument('--cue_quality_loss_weight', type=float, default=0.1)
    parser.add_argument('--cue_quality_paired', action='store_true', default=False)
    parser.add_argument('--cue_quality_detach_gate', action='store_true', default=False)
    parser.add_argument('--cue_quality_pair_weight', type=float, default=0.0)
    parser.add_argument('--cue_quality_aux_only', action='store_true', default=False)
    parser.add_argument('--cue_quality_normalize_gate', action='store_true', default=False)
    parser.add_argument('--cue_quality_gate_mode', type=str, default='raw',
                        choices=['raw', 'ema_ratio', 'calibrated'])
    parser.add_argument('--cue_quality_gate_floor', type=float, default=0.5)
    parser.add_argument('--cue_quality_gate_ceiling', type=float, default=1.0)
    parser.add_argument('--cue_quality_gate_up_span', type=float, default=0.0)
    parser.add_argument('--cue_quality_gate_down_span', type=float, default=0.5)
    parser.add_argument('--cue_quality_gate_margin', type=float, default=0.0)
    parser.add_argument('--cue_quality_gate_min_std', type=float, default=0.03)
    parser.add_argument('--cue_quality_ema_momentum', type=float, default=0.1)
    parser.add_argument('--cue_usefulness_supervision', action='store_true', default=False)
    parser.add_argument('--cue_usefulness_loss_weight', type=float, default=0.05)
    parser.add_argument('--cue_usefulness_margin', type=float, default=0.005)
    parser.add_argument('--cue_usefulness_span', type=float, default=0.5)
    parser.add_argument('--cue_usefulness_binary_target', action='store_true', default=False)
    parser.add_argument('--cue_usefulness_downscale_only', action='store_true', default=False)
    parser.add_argument('--cue_usefulness_gate_init', type=float, default=0.0)
    parser.add_argument('--cue_usefulness_relative_target', action='store_true', default=False)
    parser.add_argument('--cue_usefulness_quantile', type=float, default=0.0)
    parser.add_argument('--cue_usefulness_warmup_epochs', type=int, default=0)
    parser.add_argument('--cue_usefulness_quality_prior', action='store_true', default=False)
    parser.add_argument('--cue_usefulness_end_to_end', action='store_true', default=False)
    parser.add_argument('--cue_usefulness_prior_weight', type=float, default=0.0)
    parser.add_argument('--feat_dir', type=str, default='./data/features')
    parser.add_argument('--ckpt_path', type=str, default='')
    
    parser.add_argument('--save_base_model', type=str, default="")
    parser.add_argument('--save_final_model', type=str, default="")
    
    parser.add_argument('--test', action='store_true', default=False)
    
    parser.add_argument('--print_flops', action='store_true', default=False)

    args = parser.parse_args()
    apply_cue_policy(args)
    if args.cue_policy in DUAL_RU_POLICIES and not args.multi_modality:
        parser.error('dual_ru policies require --multi_modality')
    if args.auto_gate_v2:
        args.auto_gate = True
    return args


def arg_was_set(name):
    flag = f"--{name}"
    if name == 'use_skeleton':
        return '--use_skeleton' in sys.argv or '--no_skeleton' in sys.argv
    if name == 'motion_saliency_proxy':
        return '--motion_saliency_proxy' in sys.argv or '--no_motion_saliency_proxy' in sys.argv
    if name == 'motion_saliency_positive_only':
        return '--motion_saliency_positive_only' in sys.argv or '--no_motion_saliency_positive_only' in sys.argv
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv)


def set_if_unset(args, name, value):
    if not arg_was_set(name):
        setattr(args, name, value)


def apply_cue_policy(args):
    if args.cue_policy == 'none':
        return
    if args.cue_policy in DUAL_RU_POLICIES:
        set_if_unset(args, 'auto_gate', False)
        set_if_unset(args, 'auto_gate_v2', False)
        set_if_unset(args, 'use_skeleton', True)
        set_if_unset(args, 'motion_saliency_proxy', True)
        set_if_unset(args, 'skeleton_mode', 'residual')
        set_if_unset(args, 'motion_saliency_alpha', 0.015)
        set_if_unset(args, 'motion_saliency_source', 'delta')
        set_if_unset(args, 'motion_saliency_positive_only', True)
        set_if_unset(args, 'skeleton_gate_max', 0.02)
        set_if_unset(args, 'auto_gate_motion_max', 0.015)
        set_if_unset(args, 'auto_gate_skeleton_max', 0.02)
        set_if_unset(args, 'auto_gate_l1', 0.0)
        set_if_unset(args, 'cue_reliability', False)
        set_if_unset(args, 'cue_quality_supervision', True)
        set_if_unset(args, 'cue_corruption_prob', 1.0)
        set_if_unset(args, 'cue_corruption_max', 0.6)
        set_if_unset(args, 'cue_quality_loss_weight', 1.0)
        set_if_unset(args, 'cue_quality_paired', True)
        set_if_unset(args, 'cue_quality_detach_gate', True)
        set_if_unset(args, 'cue_quality_pair_weight', 0.1)
        set_if_unset(args, 'cue_quality_aux_only', False)
        set_if_unset(args, 'cue_usefulness_supervision', False)
        if args.cue_policy in DUAL_RU_V5_PLUS:
            set_if_unset(args, 'cue_corruption_max', 1.0)
        return
    if args.cue_policy in QUALITY_POLICIES + FIXED_CUE_POLICIES:
        set_if_unset(args, 'auto_gate_v2', True)
        set_if_unset(args, 'use_skeleton', True)
        set_if_unset(args, 'motion_saliency_proxy', True)
        set_if_unset(args, 'skeleton_mode', 'residual')
        set_if_unset(args, 'motion_saliency_alpha', 0.015)
        set_if_unset(args, 'auto_gate_motion_max', 0.015)
        set_if_unset(args, 'auto_gate_skeleton_max', 0.01)
        set_if_unset(args, 'auto_gate_l1', 0.01)
        set_if_unset(args, 'cue_reliability', False)
        set_if_unset(args, 'cue_quality_supervision', True)
        set_if_unset(args, 'cue_corruption_prob', 0.5)
        set_if_unset(args, 'cue_corruption_max', 0.6)
        set_if_unset(args, 'cue_quality_loss_weight', 0.1)
        if args.cue_policy in QUALITY_POLICIES[1:] + FIXED_CUE_POLICIES:
            set_if_unset(args, 'cue_corruption_prob', 1.0)
            set_if_unset(args, 'cue_quality_loss_weight', 1.0)
            set_if_unset(args, 'cue_quality_paired', True)
            set_if_unset(args, 'cue_quality_detach_gate', True)
            set_if_unset(args, 'cue_quality_pair_weight', 0.1)
        if args.cue_policy == 'quality_supervised_v3':
            set_if_unset(args, 'cue_quality_normalize_gate', True)
            set_if_unset(args, 'cue_quality_gate_mode', 'ema_ratio')
            set_if_unset(args, 'cue_quality_gate_floor', 0.5)
            set_if_unset(args, 'cue_quality_gate_ceiling', 1.0)
            set_if_unset(args, 'cue_quality_ema_momentum', 0.1)
        elif args.cue_policy == 'quality_supervised_v4':
            set_if_unset(args, 'cue_quality_normalize_gate', True)
            set_if_unset(args, 'cue_quality_gate_mode', 'calibrated')
            set_if_unset(args, 'cue_quality_gate_floor', 0.85)
            set_if_unset(args, 'cue_quality_gate_ceiling', 1.08)
            set_if_unset(args, 'cue_quality_gate_up_span', 0.06)
            set_if_unset(args, 'cue_quality_gate_down_span', 0.15)
            set_if_unset(args, 'cue_quality_gate_margin', 0.25)
            set_if_unset(args, 'cue_quality_gate_min_std', 0.02)
            set_if_unset(args, 'cue_quality_ema_momentum', 0.1)
        elif args.cue_policy == 'quality_supervised_v5':
            set_if_unset(args, 'cue_quality_aux_only', True)
            set_if_unset(args, 'cue_quality_normalize_gate', False)
        elif args.cue_policy == 'quality_supervised_v6':
            set_if_unset(args, 'cue_quality_loss_weight', 0.2)
            set_if_unset(args, 'cue_quality_aux_only', False)
            set_if_unset(args, 'cue_quality_normalize_gate', False)
            set_if_unset(args, 'cue_usefulness_supervision', True)
            set_if_unset(args, 'cue_usefulness_loss_weight', 0.05)
            set_if_unset(args, 'cue_usefulness_margin', 0.005)
            set_if_unset(args, 'cue_usefulness_span', 0.5)
        elif args.cue_policy == 'quality_supervised_v7':
            set_if_unset(args, 'cue_quality_loss_weight', 0.1)
            set_if_unset(args, 'cue_quality_aux_only', False)
            set_if_unset(args, 'cue_quality_normalize_gate', False)
            set_if_unset(args, 'cue_usefulness_supervision', True)
            set_if_unset(args, 'cue_usefulness_loss_weight', 0.2)
            set_if_unset(args, 'cue_usefulness_margin', 0.0002)
            set_if_unset(args, 'cue_usefulness_span', 0.5)
            set_if_unset(args, 'cue_usefulness_binary_target', True)
            set_if_unset(args, 'cue_usefulness_downscale_only', True)
            set_if_unset(args, 'cue_usefulness_gate_init', 4.0)
        elif args.cue_policy == 'quality_supervised_v8':
            set_if_unset(args, 'cue_quality_loss_weight', 1.0)
            set_if_unset(args, 'cue_quality_aux_only', False)
            set_if_unset(args, 'cue_quality_normalize_gate', True)
            set_if_unset(args, 'cue_quality_gate_mode', 'ema_ratio')
            set_if_unset(args, 'cue_quality_gate_floor', 0.5)
            set_if_unset(args, 'cue_quality_gate_ceiling', 1.0)
            set_if_unset(args, 'cue_quality_ema_momentum', 0.1)
            set_if_unset(args, 'cue_usefulness_supervision', True)
            set_if_unset(args, 'cue_usefulness_loss_weight', 0.1)
            set_if_unset(args, 'cue_usefulness_span', 0.3)
            set_if_unset(args, 'cue_usefulness_binary_target', True)
            set_if_unset(args, 'cue_usefulness_downscale_only', True)
            set_if_unset(args, 'cue_usefulness_gate_init', 2.0)
            set_if_unset(args, 'cue_usefulness_relative_target', True)
            set_if_unset(args, 'cue_usefulness_quantile', 0.6)
            set_if_unset(args, 'cue_usefulness_warmup_epochs', 15)
            set_if_unset(args, 'cue_usefulness_quality_prior', True)
        elif args.cue_policy == 'quality_supervised_v9':
            set_if_unset(args, 'cue_quality_loss_weight', 1.0)
            set_if_unset(args, 'cue_quality_aux_only', False)
            set_if_unset(args, 'cue_quality_normalize_gate', True)
            set_if_unset(args, 'cue_quality_gate_mode', 'ema_ratio')
            set_if_unset(args, 'cue_quality_gate_floor', 0.5)
            set_if_unset(args, 'cue_quality_gate_ceiling', 1.0)
            set_if_unset(args, 'cue_quality_ema_momentum', 0.1)
            set_if_unset(args, 'cue_usefulness_supervision', True)
            set_if_unset(args, 'cue_usefulness_loss_weight', 0.0)
            set_if_unset(args, 'cue_usefulness_span', 0.2)
            set_if_unset(args, 'cue_usefulness_downscale_only', True)
            set_if_unset(args, 'cue_usefulness_gate_init', 4.0)
            set_if_unset(args, 'cue_usefulness_warmup_epochs', 30)
            set_if_unset(args, 'cue_usefulness_quality_prior', True)
            set_if_unset(args, 'cue_usefulness_end_to_end', True)
            set_if_unset(args, 'cue_usefulness_prior_weight', 0.001)
        elif args.cue_policy in FIXED_CUE_POLICIES:
            set_if_unset(args, 'cue_quality_aux_only', True)
            set_if_unset(args, 'cue_quality_normalize_gate', False)
            set_if_unset(args, 'motion_saliency_source', 'delta')
            set_if_unset(args, 'motion_saliency_positive_only', True)
            if args.cue_policy in FIXED_CUE_POLICIES[1:]:
                set_if_unset(args, 'auto_gate_l1', 0.0)
                set_if_unset(args, 'cue_corruption_max', 1.0)
        elif args.cue_policy == 'quality_supervised_v10':
            set_if_unset(args, 'cue_quality_aux_only', True)
            set_if_unset(args, 'cue_quality_normalize_gate', False)
            set_if_unset(args, 'cue_usefulness_supervision', True)
            set_if_unset(args, 'cue_usefulness_loss_weight', 0.0)
            set_if_unset(args, 'cue_usefulness_span', 0.2)
            set_if_unset(args, 'cue_usefulness_downscale_only', True)
            set_if_unset(args, 'cue_usefulness_gate_init', 2.0)
            set_if_unset(args, 'cue_usefulness_warmup_epochs', 30)
            set_if_unset(args, 'cue_usefulness_quality_prior', False)
            set_if_unset(args, 'cue_usefulness_end_to_end', True)
            set_if_unset(args, 'cue_usefulness_prior_weight', 0.0)
            set_if_unset(args, 'motion_saliency_source', 'delta')
            set_if_unset(args, 'motion_saliency_positive_only', True)
        return
    if args.cue_policy in RELIABILITY_POLICIES:
        set_if_unset(args, 'auto_gate_v2', True)
        set_if_unset(args, 'use_skeleton', True)
        set_if_unset(args, 'motion_saliency_proxy', True)
        set_if_unset(args, 'skeleton_mode', 'residual')
        set_if_unset(args, 'motion_saliency_alpha', 0.015)
        set_if_unset(args, 'auto_gate_motion_max', 0.015)
        set_if_unset(args, 'auto_gate_skeleton_max', 0.01)
        set_if_unset(args, 'auto_gate_motion_gamma', 0.15)
        set_if_unset(args, 'auto_gate_skeleton_gamma', 0.15)
        set_if_unset(args, 'auto_gate_l1', 0.001)
        set_if_unset(args, 'cue_reliability', True)
        if args.cue_policy == 'reliability_aware_v2':
            set_if_unset(args, 'cue_reliability_mode', 'soft')
            set_if_unset(args, 'cue_reliability_floor', 0.75)
            set_if_unset(args, 'cue_reliability_span', 0.5)
        elif args.cue_policy == 'reliability_aware_v3':
            set_if_unset(args, 'cue_reliability_mode', 'calibrated')
            set_if_unset(args, 'cue_reliability_span', 0.25)
            set_if_unset(args, 'cue_reliability_momentum', 0.1)
            set_if_unset(args, 'cue_reliability_min_std', 0.05)
            set_if_unset(args, 'auto_gate_l1', 0.01)
        return
    if args.cue_policy != 'action_aware_v1':
        raise ValueError(f"cue_policy: {args.cue_policy} is not supported!")

    if args.action == 'TES':
        set_if_unset(args, 'auto_gate_v2', True)
        set_if_unset(args, 'use_skeleton', False)
        set_if_unset(args, 'motion_saliency_proxy', True)
        set_if_unset(args, 'auto_gate_motion_gamma', 0.25)
        set_if_unset(args, 'auto_gate_l1', 0.0)
        set_if_unset(args, 'auto_gate_no_downscale', True)
    elif args.action == 'PCS':
        set_if_unset(args, 'auto_gate', False)
        set_if_unset(args, 'auto_gate_v2', False)
        set_if_unset(args, 'use_skeleton', True)
        set_if_unset(args, 'motion_saliency_proxy', False)
        set_if_unset(args, 'skeleton_mode', 'residual')
        set_if_unset(args, 'skeleton_gate_max', 0.02)
    elif args.action == 'Ball':
        set_if_unset(args, 'auto_gate', False)
        set_if_unset(args, 'auto_gate_v2', False)
        set_if_unset(args, 'use_skeleton', True)
        set_if_unset(args, 'motion_saliency_proxy', True)
        set_if_unset(args, 'skeleton_mode', 'residual')
        set_if_unset(args, 'skeleton_gate_max', 0.003)
    elif args.action == 'Clubs':
        set_if_unset(args, 'auto_gate', False)
        set_if_unset(args, 'auto_gate_v2', False)
        set_if_unset(args, 'use_skeleton', True)
        set_if_unset(args, 'motion_saliency_proxy', True)
        set_if_unset(args, 'skeleton_mode', 'residual')
        set_if_unset(args, 'skeleton_gate_max', 0.003)
    elif args.action == 'Hoop':
        set_if_unset(args, 'auto_gate', False)
        set_if_unset(args, 'auto_gate_v2', False)
        set_if_unset(args, 'use_skeleton', True)
        set_if_unset(args, 'motion_saliency_proxy', True)
        set_if_unset(args, 'skeleton_mode', 'residual')
        set_if_unset(args, 'skeleton_gate_max', 0.002)
    elif args.action == 'Ribbon':
        set_if_unset(args, 'auto_gate_v2', True)
        set_if_unset(args, 'use_skeleton', True)
        set_if_unset(args, 'motion_saliency_proxy', True)
        set_if_unset(args, 'auto_gate_skeleton_max', 0.003)
        set_if_unset(args, 'auto_gate_motion_gamma', 0.15)
        set_if_unset(args, 'auto_gate_skeleton_gamma', 0.15)
        set_if_unset(args, 'auto_gate_l1', 0.001)


def resolve_skeleton_config(args):
    mode = args.skeleton_mode
    temporal_beta = args.skeleton_temporal_beta
    gate_max = args.skeleton_gate_max
    if args.auto_gate and not args.auto_gate_v2 and args.use_skeleton is not False:
        mode = 'residual'
    elif args.use_skeleton is False:
        mode = 'off'
    elif args.use_skeleton is True and mode == 'auto':
        mode = 'residual'
    if args.skeleton_temporal_weight and mode != 'off':
        mode = 'temporal'

    if mode == 'auto':
        if args.action == 'PCS':
            mode = 'residual'
        elif args.action == 'TES':
            mode = 'temporal'
            temporal_beta = 0.03
        else:
            mode = 'off'
    if gate_max is None:
        gate_max = 0.01 if args.action == 'TES' else 0.02

    return {
        "mode": mode,
        "use_skeleton": mode != 'off',
        "skeleton_temporal_weight": mode == 'temporal',
        "skeleton_temporal_beta": temporal_beta,
        "skeleton_gate_max": gate_max
    }


def resolve_motion_saliency_config(args):
    if args.motion_saliency_proxy is not None:
        enabled = args.motion_saliency_proxy
    elif args.auto_gate:
        enabled = True
    else:
        enabled = args.action in ["Ball", "Clubs", "Hoop", "Ribbon"]
    alpha = args.motion_saliency_alpha
    if alpha is None:
        alpha = 0.02 if args.action in ["Hoop", "Ribbon"] else 0.01
    source = args.motion_saliency_source
    if source is None:
        source = "energy" if args.action == "Ribbon" else "delta"
    positive_only = args.motion_saliency_positive_only
    if positive_only is None:
        positive_only = args.action != "Hoop"
    return {
        "motion_saliency_proxy": enabled,
        "motion_saliency_alpha": alpha,
        "motion_saliency_mode": args.motion_saliency_mode,
        "motion_saliency_source": source,
        "motion_saliency_positive_only": positive_only
    }


def log_run_configuration(log_func, args, model_config, skeleton_config,
                          motion_config, motion_max, skeleton_max,
                          epoch_num, configured_epoch_num):
    log_func(f"Cue policy: {args.cue_policy}")
    log_func("Fusion evaluation: deterministic argmax")
    if epoch_num != configured_epoch_num:
        log_func(
            f"Epoch schedule override: {configured_epoch_num} -> {epoch_num}"
        )
    log_func(f"Skeleton mode: {skeleton_config['mode']}")
    if skeleton_config["skeleton_temporal_weight"]:
        log_func(
            f"Skeleton temporal beta: "
            f"{skeleton_config['skeleton_temporal_beta']}"
        )
    elif skeleton_config["use_skeleton"]:
        log_func(
            f"Skeleton gate max: {skeleton_config['skeleton_gate_max']}"
        )
    log_func(
        "Motion saliency: "
        f"{'on' if motion_config['motion_saliency_proxy'] else 'off'}"
    )
    if motion_config["motion_saliency_proxy"]:
        log_func(
            "Motion cue: "
            f"alpha {motion_config['motion_saliency_alpha']} "
            f"mode {motion_config['motion_saliency_mode']} "
            f"source {motion_config['motion_saliency_source']} "
            f"positive_only {motion_config['motion_saliency_positive_only']}"
        )

    if args.cue_policy in DUAL_RU_POLICIES:
        config = model_config["model_conf"]
        version = int(args.cue_policy.rsplit('_v', 1)[1])
        warmup = config["dual_ru_warmup_ratio"]
        supervision = config["dual_ru_supervision_ratio"]
        log_func(f"AutoGate: dual reliability x usefulness (v{version})")
        log_func(
            "Dual RU schedule: "
            f"fixed {warmup:.0%}, supervision {supervision - warmup:.0%}, "
            f"adaptive {1 - supervision:.0%}"
        )
        gate_scope = (
            "action prior x sample reliability"
            if version == 11 else "sample usefulness x sample reliability"
        )
        log_func(
            f"Dual RU gate: {gate_scope}; "
            f"motion_max {motion_max}; skeleton_max {skeleton_max}"
        )
        log_func(
            "Dual RU supervision: "
            f"temperature {config['dual_ru_temperature']}; "
            f"task_rank {config['dual_ru_task_rank_weight']}; "
            f"joint_rank {config['dual_ru_joint_rank_weight']}; "
            f"sample_correction {config['dual_ru_joint_sample_weight']}"
        )
        log_func(
            "Dual RU grid: " +
            ", ".join(f"{value:g}" for value in config["dual_ru_gate_grid"])
        )
        if version >= 6:
            log_func(
                "Dual RU teacher: train-probe every 5 epochs; "
                f"min_coef {config['dual_ru_teacher_min_probe_coef']}; "
                f"min_score_std {config['dual_ru_teacher_min_score_std']}; "
                f"EMA {config['dual_ru_teacher_ema_decay']}"
            )
        log_func(
            "Dual RU diagnostics: compact per epoch; full gate grid and "
            "corruption checks at final evaluation"
        )
        return

    if args.cue_quality_supervision:
        auto_gate_name = 'quality-supervised'
    elif args.auto_gate_v2:
        auto_gate_name = 'v2'
    elif args.auto_gate:
        auto_gate_name = 'v1'
    else:
        auto_gate_name = 'off'
    log_func(f"AutoGate: {auto_gate_name}")
    if args.auto_gate:
        log_func(
            f"AutoGate limits: motion {motion_max}; skeleton {skeleton_max}; "
            f"L1 {args.auto_gate_l1}"
        )
    if args.cue_quality_supervision:
        log_func(
            "Cue quality: "
            f"corruption_prob {args.cue_corruption_prob}; "
            f"corruption_max {args.cue_corruption_max}; "
            f"loss_weight {args.cue_quality_loss_weight}; "
            f"aux_only {args.cue_quality_aux_only}"
        )
    if args.cue_usefulness_supervision:
        log_func(
            "Cue usefulness: "
            f"loss_weight {args.cue_usefulness_loss_weight}; "
            f"warmup {args.cue_usefulness_warmup_epochs}; "
            f"end_to_end {args.cue_usefulness_end_to_end}"
        )
    if args.cue_reliability:
        log_func(
            f"Cue reliability: mode {args.cue_reliability_mode}; "
            f"span {args.cue_reliability_span}"
        )


if __name__ == '__main__':
    args = parser_args()
    from pipeline import go, log_info, test_model

    if args.gpu != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    if args.action in ["Ball", "Clubs", "Hoop", "Ribbon"]:
        if args.feats == 1:
            import configs.RG_feats1 as conf
        elif args.feats == 2:
            import configs.RG_feats2 as conf
        else:
            raise ValueError(f"feats-{args.feats} is not supported!")
    elif args.action in ["TES", "PCS"]:
        if args.feats == 1:
            import configs.FISV_feats1 as conf
        elif args.feats == 2:
            import configs.FISV_feats2 as conf
        else:
            raise ValueError(f"feats-{args.feats} is not supported!")


    if args.multi_modality:
        cur_conf = conf.multi_modality
    else:
        cur_conf = conf.single_modality
    epoch_schedule = dict(cur_conf["epoch"])
    if (args.cue_policy in DUAL_RU_POLICIES + FIXED_CUE_POLICIES and
            args.multi_modality and
            args.feats == 2):
        dual_ru_epochs = {
            'TES': 120,
            'PCS': 120,
            'Ball': 300,
            'Clubs': 120,
            'Hoop': 200,
            'Ribbon': 120
        }
        if args.cue_policy in DUAL_RU_V8_PLUS:
            epoch_schedule[args.action] = int(np.ceil(
                dual_ru_epochs[args.action] / 0.70
            ))
        else:
            epoch_schedule[args.action] = dual_ru_epochs[args.action]
    if (args.cue_policy in QUALITY_POLICIES[3:] + FIXED_CUE_POLICIES and
            args.multi_modality and
            args.action == 'Ribbon'):
        epoch_schedule[args.action] = max(epoch_schedule[args.action], 100)
    skeleton_config = resolve_skeleton_config(args)
    motion_saliency_config = resolve_motion_saliency_config(args)
    auto_gate_skeleton_max = args.auto_gate_skeleton_max
    if auto_gate_skeleton_max is None:
        auto_gate_skeleton_max = skeleton_config["skeleton_gate_max"]
    auto_gate_motion_max = args.auto_gate_motion_max
    if auto_gate_motion_max is None:
        auto_gate_motion_max = motion_saliency_config["motion_saliency_alpha"]

    dataset_config = {
        "dataset_name": args.action,
        "feat_dir": args.feat_dir,
        "rgb_feat": conf.features["V"],
        "flow_feat": conf.features["F"],
        "audio_feat": conf.features["A"],
        "batch_size": cur_conf["batch_size"],
        "num_workers": cur_conf["num_workers"],
        "seed": args.seed,
        "use_skeleton": skeleton_config["use_skeleton"],
        "skeleton_root": args.skeleton_root,
        "build_train_probe": args.cue_policy in DUAL_RU_V6_PLUS,
    }

    if not args.multi_modality:
        model_config = {
            "model_name": "pamfn_base",
            "model_conf": {
                "in_dim": cur_conf["in_dim"][args.modality],
                "model_dim": cur_conf["model_dim"],
                "drop_rate": cur_conf["drop_rate"],
                "modality": args.modality
            }
        }
    else:
        model_config = {
            "model_name": "pamfn_final",
            "model_conf": {
                "model_dim": cur_conf["model_dim"],
                "fc_drop": cur_conf["fc_drop"],
                "fc_r": cur_conf["fc_r"],
                "feat_drop": cur_conf["feat_drop"],
                "K": cur_conf["K"],
                "ms_heads": cur_conf["ms_heads"],
                "cm_heads": cur_conf["cm_heads"],
                "ckpt_dir": cur_conf["ckpt_dir"], 
                "rgb_ckpt_name": cur_conf["rgb_ckpt_name"], 
                "flow_ckpt_name": cur_conf["flow_ckpt_name"],
                "audio_ckpt_name": cur_conf["audio_ckpt_name"],
                "dataset_name": args.action,
                "use_skeleton": skeleton_config["use_skeleton"],
                "skeleton_gate_init": args.skeleton_gate_init,
                "skeleton_gate_max": skeleton_config["skeleton_gate_max"],
                "skeleton_alpha": args.skeleton_alpha,
                "skeleton_temporal_weight": skeleton_config["skeleton_temporal_weight"],
                "skeleton_temporal_beta": skeleton_config["skeleton_temporal_beta"],
                "skeleton_confidence_gate": args.skeleton_confidence_gate,
                "skeleton_late_fusion": args.skeleton_late_fusion,
                "skeleton_late_gamma": args.skeleton_late_gamma,
                "motion_saliency_proxy": motion_saliency_config["motion_saliency_proxy"],
                "motion_saliency_alpha": motion_saliency_config["motion_saliency_alpha"],
                "motion_saliency_mode": motion_saliency_config["motion_saliency_mode"],
                "motion_saliency_source": motion_saliency_config["motion_saliency_source"],
                "motion_saliency_positive_only": motion_saliency_config["motion_saliency_positive_only"],
                "auto_gate": args.auto_gate,
                "auto_gate_init": args.auto_gate_init,
                "auto_gate_skeleton_max": auto_gate_skeleton_max,
                "auto_gate_motion_max": auto_gate_motion_max,
                "auto_gate_v2": args.auto_gate_v2,
                "auto_gate_l1": args.auto_gate_l1,
                "auto_gate_budget": args.auto_gate_budget,
                "auto_gate_budget_weight": args.auto_gate_budget_weight,
                "auto_gate_motion_gamma": args.auto_gate_motion_gamma,
                "auto_gate_skeleton_gamma": args.auto_gate_skeleton_gamma,
                "auto_gate_no_downscale": args.auto_gate_no_downscale,
                "cue_reliability": args.cue_reliability,
                "cue_reliability_mode": args.cue_reliability_mode,
                "cue_reliability_floor": args.cue_reliability_floor,
                "cue_reliability_span": args.cue_reliability_span,
                "cue_reliability_momentum": args.cue_reliability_momentum,
                "cue_reliability_min_std": args.cue_reliability_min_std,
                "cue_quality_supervision": args.cue_quality_supervision,
                "cue_corruption_prob": args.cue_corruption_prob,
                "cue_corruption_max": args.cue_corruption_max,
                "cue_quality_loss_weight": args.cue_quality_loss_weight,
                "cue_quality_paired": args.cue_quality_paired,
                "cue_quality_detach_gate": args.cue_quality_detach_gate,
                "cue_quality_pair_weight": args.cue_quality_pair_weight,
                "cue_quality_zero_anchor": (
                    args.cue_policy in DUAL_RU_V6_PLUS + ('fixed_cue_v3',)
                ),
                "cue_quality_zero_weight": 1.0,
                "cue_quality_aux_only": args.cue_quality_aux_only,
                "cue_quality_normalize_gate": args.cue_quality_normalize_gate,
                "cue_quality_gate_mode": args.cue_quality_gate_mode,
                "cue_quality_gate_floor": args.cue_quality_gate_floor,
                "cue_quality_gate_ceiling": args.cue_quality_gate_ceiling,
                "cue_quality_gate_up_span": args.cue_quality_gate_up_span,
                "cue_quality_gate_down_span": args.cue_quality_gate_down_span,
                "cue_quality_gate_margin": args.cue_quality_gate_margin,
                "cue_quality_gate_min_std": args.cue_quality_gate_min_std,
                "cue_quality_ema_momentum": args.cue_quality_ema_momentum,
                "cue_usefulness_supervision": args.cue_usefulness_supervision,
                "cue_usefulness_loss_weight": args.cue_usefulness_loss_weight,
                "cue_usefulness_margin": args.cue_usefulness_margin,
                "cue_usefulness_span": args.cue_usefulness_span,
                "cue_usefulness_binary_target": args.cue_usefulness_binary_target,
                "cue_usefulness_downscale_only": args.cue_usefulness_downscale_only,
                "cue_usefulness_gate_init": args.cue_usefulness_gate_init,
                "cue_usefulness_relative_target": args.cue_usefulness_relative_target,
                "cue_usefulness_quantile": args.cue_usefulness_quantile,
                "cue_usefulness_warmup_epochs": args.cue_usefulness_warmup_epochs,
                "cue_usefulness_quality_prior": args.cue_usefulness_quality_prior,
                "cue_usefulness_end_to_end": args.cue_usefulness_end_to_end,
                "cue_usefulness_prior_weight": args.cue_usefulness_prior_weight,
                "dual_ru_gate": args.cue_policy in DUAL_RU_POLICIES,
                "dual_ru_warmup_ratio": (
                    0.70 if args.cue_policy in DUAL_RU_V8_PLUS else (
                    0.50 if args.cue_policy in DUAL_RU_V5_PLUS[:3] else 0.25)
                ),
                "dual_ru_supervision_ratio": (
                    0.85 if args.cue_policy in DUAL_RU_V8_PLUS else (
                    0.70 if args.cue_policy in DUAL_RU_V5_PLUS[:3] else 0.50)
                ),
                "dual_ru_usefulness_loss_weight": (
                    1.0 if args.cue_policy in DUAL_RU_V5_PLUS else 0.2
                ),
                "dual_ru_temperature": (
                    0.1 if args.cue_policy in DUAL_RU_V7_PLUS else 0.25
                ),
                **{
                    f"dual_ru_v{version}":
                        args.cue_policy == f'dual_ru_v{version}'
                    for version in range(2, 12)
                },
                "dual_ru_deadzone": 0.02,
                "dual_ru_rank_weight": (
                    0.0 if args.cue_policy in DUAL_RU_ZERO_RANK_POLICIES
                    else 0.5
                ),
                "dual_ru_rank_temperature": 0.1,
                "dual_ru_reliability_momentum": 0.1,
                "dual_ru_error_scale_momentum": 0.1,
                "dual_ru_error_scale_min": 0.001,
                "dual_ru_benefit_scale_min": 0.00001,
                "dual_ru_effect_scale_momentum": 0.1,
                "dual_ru_effect_scale_min": 0.0001,
                "dual_ru_effect_valid_ratio": 0.1,
                "dual_ru_gate_ramp_ratio": 0.2,
                "dual_ru_phase2_lr_scale": (
                    1.0 if args.cue_policy in DUAL_RU_V7_PLUS else 0.1
                ),
                "dual_ru_target_rank_weight": 0.0,
                "dual_ru_task_rank_weight": (
                    0.01 if args.cue_policy in (
                        DUAL_RU_V5_PLUS + ('fixed_cue_v2', 'fixed_cue_v3')
                    ) else 0.0
                ),
                "dual_ru_gate_anchor_weight": 0.01,
                "dual_ru_output_anchor_weight": 0.1,
                "dual_ru_identifiability_abs": (
                    0.000001 if args.cue_policy == 'dual_ru_v9' else 0.0001
                ),
                "dual_ru_identifiability_error_ratio": (
                    0.0 if args.cue_policy == 'dual_ru_v9' else 0.001
                ),
                "dual_ru_unidentifiable_weight": 0.05,
                "dual_ru_teacher_min_score_std": 0.01,
                "dual_ru_teacher_min_probe_coef": 97.0,
                "dual_ru_teacher_ema_decay": 0.5,
                "dual_ru_phase2_target_end_weight": 0.1,
                "dual_ru_no_regret_weight": 1.0,
                "dual_ru_joint_rank_weight": 0.2,
                "dual_ru_joint_sample_weight": (
                    0.0 if args.cue_policy == 'dual_ru_v11' else 0.2
                ),
                "dual_ru_joint_min_gain": 0.02,
                "dual_ru_batch_rank_regret_weight": 0.1,
                "dual_ru_gate_grid": (
                    (0.0, 0.5, 1.0)
                    if args.cue_policy in DUAL_RU_V9_PLUS else
                    (0.0, 0.25, 0.5, 0.75, 1.0)
                )
            }
        }

    optimizer_config = {
        "optim": cur_conf["optim"],
        "lr": cur_conf["lr"],
        "weight_decay": cur_conf["weight_decay"],
        "reduce_fc_lr": cur_conf["reduce_fc_lr"]
    }

    lr_scheduler = {
        "scheduler": cur_conf["lr_scheduler"],
        "epoch_num": epoch_schedule[args.action],
        "scheduler_steps": epoch_schedule,
        "lr_min": cur_conf["lr_min"],
        "warmup_lr_init": cur_conf["warmup_lr_init"],
        "decay_rate": 0.1
    }

    if args.print_flops:
        from pipeline import print_flops
        macs, params = print_flops(model_config)
        print(macs)
        print(params)
        exit()

    log_func = partial(log_info, logger=None, print_log=True)
    log_run_configuration(
        log_func, args, model_config, skeleton_config,
        motion_saliency_config, auto_gate_motion_max,
        auto_gate_skeleton_max, epoch_schedule[args.action],
        cur_conf["epoch"][args.action]
    )
    if args.test:
        if args.ckpt_path == "":
            ckpt_path = os.path.join(cur_conf["ckpt_dir"], args.action+"_multimodal.pth")
        else:
            ckpt_path = args.ckpt_path
            
        test_model(model_kwargs=model_config, dataset_kwargs=dataset_config,
                   ckpt_path=ckpt_path, log_func=log_func, seed=args.seed)
    else:
        ret = go(model_config, optimizer_config, lr_scheduler, dataset_config, epoch_schedule[args.action],
            save_base_model=args.save_base_model, save_moniter=cur_conf["save_moniter"], save_final_model=args.save_final_model,
            log_func=log_func, seed=args.seed, clip_grad=cur_conf["clip_grad"])
        print("%.4f\t%.4f\t%.4f\t%.4f" % (ret[0], ret[1], ret[2], ret[3]))
