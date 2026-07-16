import copy
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange


def smart_fn(fn, feat_list, func_list=False):
    if func_list is False:
        return [fn(x) for x in feat_list]
    return [fn[idx](x) for idx, x in enumerate(feat_list)]


class BaseConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.GELU()
        )

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class ActionLevelGate(nn.Module):
    def __init__(self, initial_value=0.5):
        super().__init__()
        initial_value = torch.tensor(float(initial_value)).clamp(1e-4, 1 - 1e-4)
        self.logit = nn.Parameter(torch.logit(initial_value))

    def forward(self, context):
        gate = torch.sigmoid(self.logit).reshape(1, 1, 1)
        return gate.expand(context.shape[0], 1, 1)


class BaseModel(nn.Module):
    def __init__(self, in_dim, model_dim, drop_rate, modality="V"):
        super().__init__()
        dim = model_dim
        self.modality = modality
        self.embedding = nn.Sequential(
            nn.Conv1d(in_dim, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )
        self.stage1 = BaseConvBlock(dim)
        self.stage2 = BaseConvBlock(dim)
        self.stage3 = BaseConvBlock(dim)
        self.pool = nn.AvgPool1d(2, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        # self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Conv1d(dim, 1, 1),
            nn.Sigmoid()
        )
        self.mse = nn.MSELoss()

    def forward(self, feats):
        x = feats[self.modality]
        if len(x.shape) == 4:
            x = x.mean(dim=2)
        x = rearrange(x, 'b t d -> b d t').contiguous()
        x = self.embedding(x)

        x1 = self.stage1(x)
        x1 = self.pool(x1)

        x2 = self.stage2(x1)
        x2 = self.pool(x2)

        x3 = self.stage3(x2)
        x3 = self.gap(x3)

        score = self.fc(x3).squeeze(dim=2)
        return score, {'feats': [x, x1, x2, x3]}

    def call_loss(self, pred, label, **kwargs):
        return self.mse(pred.squeeze(), label.squeeze())


class Multi_Head_Attention(nn.Module):
    def __init__(self, num_heads, dim_model, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        self.num_heads = num_heads
        assert dim_model % num_heads == 0
        self.dim_head = dim_model // self.num_heads
        self.linear_layers = nn.ModuleList([nn.Linear(dim_model, dim_model) for _ in range(3)])
        self.drop = nn.Dropout(p=dropout)
        # Retain checkpoint keys from the original implementation without
        # optimizing a layer that is not part of the forward path.
        self.norm = nn.LayerNorm(dim_model)
        self.norm.requires_grad_(False)

    def self_attention(self, Q, K, V, mask=None, dropout=None):
        scores = -1 * torch.matmul(Q, K.transpose(-2, -1)) \
                 / np.sqrt(Q.size(-1))
        if mask is not None:
            # scores = scores.masked_fill(mask == 0, -1e9)
            mask = mask.unsqueeze(dim=1).repeat([1, self.num_heads, 1, 1])
            # scores = torch.mul(scores, mask)  # mask
            scores = scores + mask  # mask
        attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        context = torch.matmul(attn, V)
        return context, attn

    def forward(self, input_Q, input_K, input_V, mask=None):
        b = input_V.size(0)
        Q, K, V = [l(x).view(b, -1, self.num_heads, self.dim_head).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (input_Q, input_K, input_V))]
        x, _ = self.self_attention(Q, K, V, mask=mask, dropout=self.drop)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.num_heads * self.dim_head)
        return x


class MS_Fusion(nn.Module):
    def __init__(self, dim, num_heads=1, dropout=0.1):
        super().__init__()
        self.fusion_s = Multi_Head_Attention(num_heads=num_heads, dim_model=dim, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
        )

    def forward(self, s, c):
        q = rearrange(self.proj(c), 'b c t -> (b t) 1 c')
        s = torch.stack(s, dim=2)
        s = rearrange(s, 'b c n t -> (b t) n c')
        ns = self.fusion_s(q, s, s)
        ns = rearrange(ns, '(b t) 1 c -> b c t', b=c.shape[0])
        return ns


class CM_Fusion(nn.Module):
    def __init__(self, dim, K,  num_heads=1,  gap=False, dropout=0.1,):
        super().__init__()
        self.K = K
        self.proj1 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Conv1d(dim, self.K, 1),
        ) for _ in range(3)
        ])
        self.ffn = nn.ModuleList([nn.Sequential(
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True)
        ) for _ in range(self.K)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1) if gap else nn.AvgPool1d(2, 2)

        self.fusion_f = Multi_Head_Attention(num_heads=num_heads, dim_model=dim, dropout=dropout)
        self.proj2 = nn.Sequential(
            nn.Conv1d(2 * dim, dim, 1)
        )

        self.policy = nn.Sequential(
            nn.Conv1d(3*dim, K, 1)
        )

    def gen_mask(self, fusion_kind):
        b, k, t = fusion_kind.shape
        mask = torch.zeros_like(fusion_kind, device=fusion_kind.device)
        # b k t
        idx = torch.arange(0, k, device=fusion_kind.device).repeat([b, 1, 1]).permute([0, 2, 1]).repeat([1, 1, t])
        pos_mask = idx > torch.mul(fusion_kind, idx).sum(dim=1, keepdim=True) # b 1 t
        mask[pos_mask] = -1e9
        mask = mask + fusion_kind - fusion_kind.detach() # b k t
        return mask

    def gen_fusion_kind(self, f, tau):
        # B 3C T -> B N T
        logit = self.pool(self.policy(torch.cat(f, dim=1)))
        if self.training:
            fusion_kind = F.gumbel_softmax(logit, tau, hard=True, dim=1)
        else:
            index = logit.argmax(dim=1, keepdim=True)
            fusion_kind = torch.zeros_like(logit).scatter_(1, index, 1.0)
        mask = self.gen_mask(fusion_kind)
        return fusion_kind, mask

    def forward(self, f, c, s, tau):
        att = torch.softmax(torch.stack(smart_fn(self.proj1, f, True), dim=1), dim=1)
        att = [_ for _ in att.split(1, dim=2)]
        nf = [torch.mul(_, torch.stack(f, dim=1)).sum(dim=1) for _ in att]
        nf = torch.stack([self.pool(self.ffn[idx](_) + _) for idx, _ in enumerate(nf)], dim=2)
        nf = rearrange(nf, 'b c n t -> (b t) n c')

        q = self.proj2(torch.cat([c, s], dim=1))
        q = rearrange(q, 'b c t -> (b t) 1 c')
        q = -1 * q
        fusion_kind, mask = self.gen_fusion_kind(f, tau)
        mask = rearrange(mask, 'b n t -> (b t) 1 n')
        nf = self.fusion_f(q, nf, nf, mask=mask)
        nf = rearrange(nf, '(b t) 1 c -> b c t', b=c.shape[0])
        return nf, fusion_kind


class FusionBlock(nn.Module):
    def __init__(self, dim, K, ms_heads, cm_heads, gap=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim*3, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True)
        )
        self.short_cut = nn.Sequential(
            nn.Conv1d(dim*3, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True)
        )
        self.s_fusion = MS_Fusion(dim, ms_heads)
        self.f_fusion = CM_Fusion(dim, K, cm_heads, gap)

    def forward(self, c, f, s, tau):
        c = self.conv(c) + self.short_cut(c)
        s = self.s_fusion(s, c)
        f, action = self.f_fusion(f, c, s, tau)
        c = torch.cat([c, s, f], dim=1)
        return c, action

class PAMFN(nn.Module):
    def __init__(self,
                 model_dim, fc_drop, fc_r, feat_drop, K,
                 ms_heads, cm_heads,
                 ckpt_dir, rgb_ckpt_name, flow_ckpt_name, audio_ckpt_name,
                 dataset_name, use_skeleton=False, skeleton_dim=132,
                 skeleton_gate_init=-2.0, skeleton_gate_max=0.02,
                 skeleton_alpha=-1.0, skeleton_temporal_weight=False,
                 skeleton_temporal_beta=0.02, skeleton_confidence_gate=False,
                 skeleton_late_fusion=False, skeleton_late_gamma=0.01,
                 motion_saliency_proxy=False, motion_saliency_alpha=0.02,
                 motion_saliency_mode='stage_residual',
                 motion_saliency_source='delta',
                 motion_saliency_positive_only=True,
                 auto_gate=False, auto_gate_init=-2.0,
                 auto_gate_skeleton_max=0.01,
                 auto_gate_motion_max=0.01,
                 auto_gate_v2=False, auto_gate_l1=0.0,
                 auto_gate_budget=-1.0, auto_gate_budget_weight=0.0,
                 auto_gate_motion_gamma=0.5,
                 auto_gate_skeleton_gamma=0.5,
                 auto_gate_no_downscale=False,
                 cue_reliability=False,
                 cue_reliability_mode='hard',
                 cue_reliability_floor=0.75,
                 cue_reliability_span=0.5,
                 cue_reliability_momentum=0.1,
                 cue_reliability_min_std=0.05,
                 cue_quality_supervision=False,
                 cue_corruption_prob=0.5,
                 cue_corruption_max=0.6,
                 cue_quality_loss_weight=0.1,
                 cue_quality_paired=False,
                 cue_quality_detach_gate=False,
                 cue_quality_pair_weight=0.0,
                 cue_quality_zero_anchor=False,
                 cue_quality_zero_weight=1.0,
                 cue_quality_aux_only=False,
                 cue_quality_normalize_gate=False,
                 cue_quality_gate_mode='raw',
                 cue_quality_gate_floor=0.5,
                 cue_quality_gate_ceiling=1.0,
                 cue_quality_gate_up_span=0.0,
                 cue_quality_gate_down_span=0.5,
                 cue_quality_gate_margin=0.0,
                 cue_quality_gate_min_std=0.03,
                 cue_quality_ema_momentum=0.1,
                 cue_usefulness_supervision=False,
                 cue_usefulness_loss_weight=0.05,
                 cue_usefulness_margin=0.005,
                 cue_usefulness_span=0.5,
                 cue_usefulness_binary_target=False,
                 cue_usefulness_downscale_only=False,
                 cue_usefulness_gate_init=0.0,
                 cue_usefulness_relative_target=False,
                 cue_usefulness_quantile=0.0,
                 cue_usefulness_warmup_epochs=0,
                 cue_usefulness_quality_prior=False,
                 cue_usefulness_end_to_end=False,
                 cue_usefulness_prior_weight=0.0,
                 dual_ru_gate=False,
                 dual_ru_warmup_ratio=0.25,
                 dual_ru_supervision_ratio=0.50,
                 dual_ru_usefulness_loss_weight=0.2,
                 dual_ru_temperature=0.25,
                 dual_ru_v2=False,
                 dual_ru_v3=False,
                 dual_ru_v4=False,
                 dual_ru_v5=False,
                 dual_ru_v6=False,
                 dual_ru_v7=False,
                 dual_ru_v8=False,
                 dual_ru_v9=False,
                 dual_ru_v10=False,
                 dual_ru_v11=False,
                 dual_ru_deadzone=0.02,
                 dual_ru_rank_weight=0.5,
                 dual_ru_rank_temperature=0.1,
                 dual_ru_reliability_momentum=0.1,
                 dual_ru_error_scale_momentum=0.1,
                 dual_ru_error_scale_min=0.001,
                 dual_ru_benefit_scale_min=1e-5,
                 dual_ru_effect_scale_momentum=0.1,
                 dual_ru_effect_scale_min=0.0001,
                 dual_ru_effect_valid_ratio=0.1,
                 dual_ru_gate_ramp_ratio=0.2,
                 dual_ru_phase2_lr_scale=0.1,
                 dual_ru_target_rank_weight=0.0,
                 dual_ru_task_rank_weight=0.0,
                 dual_ru_gate_anchor_weight=0.01,
                 dual_ru_output_anchor_weight=0.1,
                 dual_ru_identifiability_abs=0.0001,
                 dual_ru_identifiability_error_ratio=0.001,
                 dual_ru_unidentifiable_weight=0.05,
                 dual_ru_teacher_min_score_std=0.01,
                 dual_ru_teacher_min_probe_coef=97.0,
                 dual_ru_teacher_ema_decay=0.5,
                 dual_ru_phase2_target_end_weight=0.1,
                 dual_ru_no_regret_weight=1.0,
                 dual_ru_joint_rank_weight=0.2,
                 dual_ru_joint_sample_weight=0.2,
                 dual_ru_joint_min_gain=0.02,
                 dual_ru_batch_rank_regret_weight=0.1,
                 dual_ru_gate_grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
        super().__init__()
        self.use_skeleton = use_skeleton
        self.skeleton_alpha = skeleton_alpha
        self.skeleton_gate_max = skeleton_gate_max
        self.skeleton_temporal_weight = skeleton_temporal_weight
        self.skeleton_temporal_beta = skeleton_temporal_beta
        self.skeleton_confidence_gate = skeleton_confidence_gate
        self.skeleton_late_fusion = skeleton_late_fusion
        self.skeleton_late_gamma = skeleton_late_gamma
        self.motion_saliency_proxy = motion_saliency_proxy
        self.motion_saliency_alpha = motion_saliency_alpha
        self.motion_saliency_mode = motion_saliency_mode
        self.motion_saliency_source = motion_saliency_source
        self.motion_saliency_positive_only = motion_saliency_positive_only
        self.auto_gate = auto_gate
        self.auto_gate_v2 = auto_gate_v2
        self.auto_gate_skeleton_max = auto_gate_skeleton_max
        self.auto_gate_motion_max = auto_gate_motion_max
        self.auto_gate_l1 = auto_gate_l1
        self.auto_gate_budget = auto_gate_budget
        self.auto_gate_budget_weight = auto_gate_budget_weight
        self.auto_gate_motion_gamma = auto_gate_motion_gamma
        self.auto_gate_skeleton_gamma = auto_gate_skeleton_gamma
        self.auto_gate_no_downscale = auto_gate_no_downscale
        self.cue_reliability = cue_reliability
        self.cue_reliability_mode = cue_reliability_mode
        self.cue_reliability_floor = cue_reliability_floor
        self.cue_reliability_span = cue_reliability_span
        self.cue_reliability_momentum = cue_reliability_momentum
        self.cue_reliability_min_std = cue_reliability_min_std
        self.cue_quality_supervision = cue_quality_supervision
        self.cue_corruption_prob = cue_corruption_prob
        self.cue_corruption_max = cue_corruption_max
        self.cue_quality_loss_weight = cue_quality_loss_weight
        self.cue_quality_paired = cue_quality_paired
        self.cue_quality_detach_gate = cue_quality_detach_gate
        self.cue_quality_pair_weight = cue_quality_pair_weight
        self.cue_quality_zero_anchor = cue_quality_zero_anchor
        self.cue_quality_zero_weight = cue_quality_zero_weight
        self.cue_quality_aux_only = cue_quality_aux_only
        self.cue_quality_normalize_gate = cue_quality_normalize_gate
        self.cue_quality_gate_mode = cue_quality_gate_mode
        self.cue_quality_gate_floor = cue_quality_gate_floor
        self.cue_quality_gate_ceiling = cue_quality_gate_ceiling
        self.cue_quality_gate_up_span = cue_quality_gate_up_span
        self.cue_quality_gate_down_span = cue_quality_gate_down_span
        self.cue_quality_gate_margin = cue_quality_gate_margin
        self.cue_quality_gate_min_std = cue_quality_gate_min_std
        self.cue_quality_ema_momentum = cue_quality_ema_momentum
        self.cue_usefulness_supervision = cue_usefulness_supervision
        self.cue_usefulness_loss_weight = cue_usefulness_loss_weight
        self.cue_usefulness_margin = cue_usefulness_margin
        self.cue_usefulness_span = cue_usefulness_span
        self.cue_usefulness_binary_target = cue_usefulness_binary_target
        self.cue_usefulness_downscale_only = cue_usefulness_downscale_only
        self.cue_usefulness_gate_init = cue_usefulness_gate_init
        self.cue_usefulness_relative_target = cue_usefulness_relative_target
        self.cue_usefulness_quantile = cue_usefulness_quantile
        self.cue_usefulness_warmup_epochs = cue_usefulness_warmup_epochs
        self.cue_usefulness_quality_prior = cue_usefulness_quality_prior
        self.cue_usefulness_end_to_end = cue_usefulness_end_to_end
        self.cue_usefulness_prior_weight = cue_usefulness_prior_weight
        self.dual_ru_gate = dual_ru_gate
        self.dual_ru_warmup_ratio = dual_ru_warmup_ratio
        self.dual_ru_supervision_ratio = dual_ru_supervision_ratio
        self.dual_ru_usefulness_loss_weight = dual_ru_usefulness_loss_weight
        self.dual_ru_temperature = dual_ru_temperature
        self.dual_ru_v2 = dual_ru_v2
        self.dual_ru_v3 = dual_ru_v3
        self.dual_ru_v4 = dual_ru_v4
        self.dual_ru_v5 = dual_ru_v5
        self.dual_ru_v6 = dual_ru_v6
        self.dual_ru_v7 = dual_ru_v7
        self.dual_ru_v8 = dual_ru_v8
        self.dual_ru_v9 = dual_ru_v9
        self.dual_ru_v10 = dual_ru_v10
        self.dual_ru_v11 = dual_ru_v11
        if sum((self.dual_ru_v2, self.dual_ru_v3, self.dual_ru_v4,
                self.dual_ru_v5, self.dual_ru_v6, self.dual_ru_v7,
                self.dual_ru_v8, self.dual_ru_v9,
                self.dual_ru_v10, self.dual_ru_v11)) > 1:
            raise ValueError('dual_ru version flags are mutually exclusive')
        self.dual_ru_version = 11 if self.dual_ru_v11 else (
            10 if self.dual_ru_v10 else (
                9 if self.dual_ru_v9 else (
                    8 if self.dual_ru_v8 else (
                        7 if self.dual_ru_v7 else (
                            6 if self.dual_ru_v6 else (
                                5 if self.dual_ru_v5 else (
                                    4 if self.dual_ru_v4 else (
                                        3 if self.dual_ru_v3 else (
                                            2 if self.dual_ru_v2 else 1
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        self.dual_ru_deadzone = dual_ru_deadzone
        self.dual_ru_rank_weight = dual_ru_rank_weight
        self.dual_ru_rank_temperature = dual_ru_rank_temperature
        self.dual_ru_reliability_momentum = dual_ru_reliability_momentum
        self.dual_ru_error_scale_momentum = dual_ru_error_scale_momentum
        self.dual_ru_error_scale_min = dual_ru_error_scale_min
        self.dual_ru_benefit_scale_min = dual_ru_benefit_scale_min
        self.dual_ru_effect_scale_momentum = dual_ru_effect_scale_momentum
        self.dual_ru_effect_scale_min = dual_ru_effect_scale_min
        self.dual_ru_effect_valid_ratio = dual_ru_effect_valid_ratio
        self.dual_ru_gate_ramp_ratio = dual_ru_gate_ramp_ratio
        self.dual_ru_phase2_lr_scale = dual_ru_phase2_lr_scale
        self.dual_ru_target_rank_weight = dual_ru_target_rank_weight
        self.dual_ru_task_rank_weight = dual_ru_task_rank_weight
        self.dual_ru_gate_anchor_weight = dual_ru_gate_anchor_weight
        self.dual_ru_output_anchor_weight = dual_ru_output_anchor_weight
        self.dual_ru_identifiability_abs = dual_ru_identifiability_abs
        self.dual_ru_identifiability_error_ratio = (
            dual_ru_identifiability_error_ratio
        )
        self.dual_ru_unidentifiable_weight = dual_ru_unidentifiable_weight
        self.dual_ru_teacher_min_score_std = dual_ru_teacher_min_score_std
        self.dual_ru_teacher_min_probe_coef = dual_ru_teacher_min_probe_coef
        self.dual_ru_teacher_ema_decay = dual_ru_teacher_ema_decay
        self.dual_ru_phase2_target_end_weight = (
            dual_ru_phase2_target_end_weight
        )
        self.dual_ru_no_regret_weight = dual_ru_no_regret_weight
        self.dual_ru_joint_rank_weight = dual_ru_joint_rank_weight
        self.dual_ru_joint_sample_weight = dual_ru_joint_sample_weight
        self.dual_ru_joint_min_gain = dual_ru_joint_min_gain
        self.dual_ru_batch_rank_regret_weight = (
            dual_ru_batch_rank_regret_weight
        )
        self.dual_ru_gate_grid = tuple(float(value) for value in dual_ru_gate_grid)
        if not 0 <= self.dual_ru_rank_weight <= 1:
            raise ValueError('dual_ru_rank_weight must be in [0, 1]')
        if self.dual_ru_deadzone < 0:
            raise ValueError('dual_ru_deadzone must be non-negative')
        if self.dual_ru_rank_temperature <= 0:
            raise ValueError('dual_ru_rank_temperature must be positive')
        if not 0 < self.dual_ru_reliability_momentum <= 1:
            raise ValueError('dual_ru_reliability_momentum must be in (0, 1]')
        if not 0 < self.dual_ru_error_scale_momentum <= 1:
            raise ValueError('dual_ru_error_scale_momentum must be in (0, 1]')
        if self.dual_ru_error_scale_min <= 0:
            raise ValueError('dual_ru_error_scale_min must be positive')
        if self.dual_ru_benefit_scale_min <= 0:
            raise ValueError('dual_ru_benefit_scale_min must be positive')
        if not 0 < self.dual_ru_effect_scale_momentum <= 1:
            raise ValueError('dual_ru_effect_scale_momentum must be in (0, 1]')
        if self.dual_ru_effect_scale_min <= 0:
            raise ValueError('dual_ru_effect_scale_min must be positive')
        if not 0 < self.dual_ru_effect_valid_ratio <= 1:
            raise ValueError('dual_ru_effect_valid_ratio must be in (0, 1]')
        if not 0 < self.dual_ru_gate_ramp_ratio <= 1:
            raise ValueError('dual_ru_gate_ramp_ratio must be in (0, 1]')
        if not 0 < self.dual_ru_phase2_lr_scale <= 1:
            raise ValueError('dual_ru_phase2_lr_scale must be in (0, 1]')
        if not 0 <= self.dual_ru_target_rank_weight <= 1:
            raise ValueError('dual_ru_target_rank_weight must be in [0, 1]')
        if not 0 <= self.dual_ru_task_rank_weight <= 1:
            raise ValueError('dual_ru_task_rank_weight must be in [0, 1]')
        if self.dual_ru_gate_anchor_weight < 0:
            raise ValueError('dual_ru_gate_anchor_weight must be non-negative')
        if self.dual_ru_output_anchor_weight < 0:
            raise ValueError('dual_ru_output_anchor_weight must be non-negative')
        if self.dual_ru_identifiability_abs <= 0:
            raise ValueError('dual_ru_identifiability_abs must be positive')
        if self.dual_ru_identifiability_error_ratio < 0:
            raise ValueError(
                'dual_ru_identifiability_error_ratio must be non-negative'
            )
        if not 0 <= self.dual_ru_unidentifiable_weight <= 1:
            raise ValueError('dual_ru_unidentifiable_weight must be in [0, 1]')
        if self.dual_ru_teacher_min_score_std <= 0:
            raise ValueError('dual_ru_teacher_min_score_std must be positive')
        if not -100 <= self.dual_ru_teacher_min_probe_coef <= 100:
            raise ValueError(
                'dual_ru_teacher_min_probe_coef must be in [-100, 100]'
            )
        if not 0 <= self.dual_ru_teacher_ema_decay < 1:
            raise ValueError('dual_ru_teacher_ema_decay must be in [0, 1)')
        if not 0 <= self.dual_ru_phase2_target_end_weight <= 1:
            raise ValueError(
                'dual_ru_phase2_target_end_weight must be in [0, 1]'
            )
        if self.dual_ru_no_regret_weight < 0:
            raise ValueError('dual_ru_no_regret_weight must be non-negative')
        if not 0 <= self.dual_ru_joint_rank_weight <= 1:
            raise ValueError('dual_ru_joint_rank_weight must be in [0, 1]')
        if not 0 <= self.dual_ru_joint_sample_weight <= 1:
            raise ValueError('dual_ru_joint_sample_weight must be in [0, 1]')
        if self.dual_ru_joint_min_gain <= 0:
            raise ValueError('dual_ru_joint_min_gain must be positive')
        if self.dual_ru_batch_rank_regret_weight < 0:
            raise ValueError(
                'dual_ru_batch_rank_regret_weight must be non-negative'
            )
        if self.cue_quality_zero_weight < 0:
            raise ValueError('cue_quality_zero_weight must be non-negative')
        if (len(self.dual_ru_gate_grid) < 3 or
                tuple(sorted(self.dual_ru_gate_grid)) != self.dual_ru_gate_grid or
                self.dual_ru_gate_grid[0] != 0.0 or
                self.dual_ru_gate_grid[-1] != 1.0 or
                0.5 not in self.dual_ru_gate_grid):
            raise ValueError(
                'dual_ru_gate_grid must be sorted and include 0, 0.5, and 1'
            )
        self.cue_current_epoch = 0
        self.cue_total_epochs = 1
        self.last_loss_info = {}
        if self.dual_ru_gate and self.dual_ru_version >= 2:
            for cue in ('motion', 'skeleton'):
                self.register_buffer(
                    f'dual_{cue}_clean_ref', torch.tensor(1.0)
                )
                self.register_buffer(
                    f'dual_{cue}_clean_initialized',
                    torch.tensor(False, dtype=torch.bool)
                )
                if (self.dual_ru_v7 or self.dual_ru_v8 or self.dual_ru_v9 or
                        self.dual_ru_v10 or self.dual_ru_v11):
                    self.register_buffer(
                        f'dual_{cue}_zero_ref', torch.tensor(0.0)
                    )
                    self.register_buffer(
                        f'dual_{cue}_zero_initialized',
                        torch.tensor(False, dtype=torch.bool)
                    )
        if self.dual_ru_gate and (
                self.dual_ru_v3 or self.dual_ru_v7 or self.dual_ru_v8 or
                self.dual_ru_v9 or self.dual_ru_v10 or self.dual_ru_v11):
            self.register_buffer(
                'dual_usefulness_error_scale',
                torch.tensor(self.dual_ru_error_scale_min)
            )
            self.register_buffer(
                'dual_usefulness_error_scale_initialized',
                torch.tensor(False, dtype=torch.bool)
            )
        if self.dual_ru_gate and self.dual_ru_v7:
            self.register_buffer(
                'dual_usefulness_rank_scale', torch.tensor(1.0)
            )
            self.register_buffer(
                'dual_usefulness_rank_scale_initialized',
                torch.tensor(False, dtype=torch.bool)
            )
        if self.dual_ru_gate and self.dual_ru_v8:
            for cue in ('motion', 'skeleton'):
                for objective in ('mse', 'rank'):
                    self.register_buffer(
                        f'dual_{cue}_{objective}_benefit_scale',
                        torch.tensor(self.dual_ru_benefit_scale_min)
                    )
                    self.register_buffer(
                        f'dual_{cue}_{objective}_benefit_scale_initialized',
                        torch.tensor(False, dtype=torch.bool)
                    )
        if self.dual_ru_gate and self.dual_ru_v4:
            for cue in ('motion', 'skeleton'):
                self.register_buffer(
                    f'dual_{cue}_effect_scale',
                    torch.tensor(self.dual_ru_effect_scale_min)
                )
                self.register_buffer(
                    f'dual_{cue}_effect_scale_initialized',
                    torch.tensor(False, dtype=torch.bool)
                )
        if self.cue_quality_normalize_gate:
            for cue in ('motion', 'skeleton'):
                self.register_buffer(f'{cue}_clean_quality_ref', torch.tensor(1.0))
                self.register_buffer(f'{cue}_clean_quality_var', torch.tensor(1.0))
                self.register_buffer(
                    f'{cue}_clean_quality_initialized',
                    torch.tensor(False, dtype=torch.bool)
                )
        if self.cue_reliability_mode == 'calibrated':
            for cue in ('motion', 'skeleton'):
                self.register_buffer(f'{cue}_reliability_mean', torch.tensor(0.0))
                self.register_buffer(f'{cue}_reliability_var', torch.tensor(1.0))
                self.register_buffer(
                    f'{cue}_reliability_initialized',
                    torch.tensor(False, dtype=torch.bool)
                )
        self.model_r = torch.load(osp.join(ckpt_dir, f"{dataset_name}_rgb_{rgb_ckpt_name}.pth"), map_location='cpu')
        self.model_f = torch.load(osp.join(ckpt_dir, f"{dataset_name}_flow_{flow_ckpt_name}.pth"), map_location='cpu')
        self.model_a = torch.load(osp.join(ckpt_dir, f"{dataset_name}_audio_{audio_ckpt_name}.pth"), map_location='cpu')

        self.c = nn.Parameter(torch.zeros([model_dim*3]), requires_grad=False)
        self.stage1 = FusionBlock(model_dim, K, ms_heads, cm_heads)
        self.stage2 = FusionBlock(model_dim, K, ms_heads, cm_heads)
        self.stage3 = FusionBlock(model_dim, K, ms_heads, cm_heads, True) 
        self.pool = nn.AvgPool1d(2, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(feat_drop)

        self.fc = nn.Sequential(
            nn.Dropout(fc_drop),
            nn.Conv1d(model_dim * 3, model_dim//fc_r, 1),
            nn.BatchNorm1d(model_dim//fc_r),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(model_dim//fc_r, 1, 1),
            nn.Sigmoid()
        )
        if self.use_skeleton:
            self.skeleton_encoder = nn.Sequential(
                nn.Conv1d(skeleton_dim, model_dim, 1),
                nn.BatchNorm1d(model_dim),
                nn.ReLU(True),
                BaseConvBlock(model_dim),
                nn.AdaptiveAvgPool1d(1)
            )
            self.skeleton_score_residual_head = nn.Sequential(
                nn.Dropout(fc_drop),
                nn.Conv1d(model_dim, 1, 1),
                nn.Tanh()
            )
            nn.init.zeros_(self.skeleton_score_residual_head[1].weight)
            nn.init.zeros_(self.skeleton_score_residual_head[1].bias)
            if self.skeleton_alpha < 0:
                self.skeleton_gate = nn.Parameter(
                    torch.ones([]) * skeleton_gate_init,
                    requires_grad=not self.dual_ru_gate
                )
            if self.skeleton_temporal_weight:
                self.skeleton_temporal_head = nn.Sequential(
                    nn.Conv1d(skeleton_dim, model_dim // 2, 1),
                    nn.BatchNorm1d(model_dim // 2),
                    nn.ReLU(True),
                    nn.Conv1d(model_dim // 2, 1, 1)
                )
                nn.init.zeros_(self.skeleton_temporal_head[3].weight)
                nn.init.zeros_(self.skeleton_temporal_head[3].bias)
            if self.skeleton_late_fusion:
                self.skeleton_late_proj = nn.Sequential(
                    nn.Conv1d(model_dim, model_dim * 3, 1),
                    nn.BatchNorm1d(model_dim * 3),
                    nn.ReLU(True)
                )
                self.skeleton_late_gate = nn.Sequential(
                    nn.Conv1d(model_dim * 6, model_dim * 3, 1),
                    nn.Sigmoid()
                )
                self.skeleton_late_head = nn.Sequential(
                    nn.Dropout(fc_drop),
                    nn.Conv1d(model_dim * 3, 1, 1),
                    nn.Tanh()
                )
                nn.init.zeros_(self.skeleton_late_head[1].weight)
                nn.init.zeros_(self.skeleton_late_head[1].bias)
            if self.auto_gate and not self.cue_quality_supervision:
                self.auto_skeleton_gate_head = nn.Conv1d(model_dim * 4, 1, 1)
                nn.init.zeros_(self.auto_skeleton_gate_head.weight)
                nn.init.constant_(self.auto_skeleton_gate_head.bias, 0.0 if self.auto_gate_v2 else auto_gate_init)
            if self.auto_gate and self.auto_gate_v2 and self.cue_usefulness_supervision:
                self.skeleton_usefulness_gate_head = nn.Conv1d(model_dim * 4, 1, 1)
                nn.init.zeros_(self.skeleton_usefulness_gate_head.weight)
                nn.init.constant_(
                    self.skeleton_usefulness_gate_head.bias,
                    self.cue_usefulness_gate_init
                )
        if self.auto_gate and self.auto_gate_v2 and not self.cue_quality_supervision:
            self.auto_motion_stage_gate_head = nn.Conv1d(model_dim * 3, 1, 1)
            nn.init.zeros_(self.auto_motion_stage_gate_head.weight)
            nn.init.zeros_(self.auto_motion_stage_gate_head.bias)
        if self.cue_quality_supervision:
            self.motion_quality_head = nn.Sequential(
                nn.Conv1d(3, 16, 1),
                nn.ReLU(True),
                nn.Conv1d(16, 1, 1),
                nn.Sigmoid()
            )
            nn.init.zeros_(self.motion_quality_head[2].weight)
            nn.init.constant_(self.motion_quality_head[2].bias, 2.0)
            if self.use_skeleton:
                self.skeleton_quality_head = nn.Sequential(
                    nn.Conv1d(3, 16, 1),
                    nn.ReLU(True),
                    nn.Conv1d(16, 1, 1),
                    nn.Sigmoid()
                )
                nn.init.zeros_(self.skeleton_quality_head[2].weight)
                nn.init.constant_(self.skeleton_quality_head[2].bias, 2.0)
        if self.dual_ru_gate:
            if not self.use_skeleton or not self.cue_quality_supervision:
                raise ValueError(
                    "dual_ru_gate requires skeleton and cue quality supervision"
                )
            # Auxiliary-head shape must not perturb data/dropout RNG streams.
            with torch.random.fork_rng(devices=[], enabled=True):
                if self.dual_ru_v11:
                    self.dual_motion_usefulness_head = ActionLevelGate(0.5)
                    self.dual_skeleton_usefulness_head = ActionLevelGate(0.5)
                else:
                    motion_usefulness_dim = model_dim * (
                        4 if self.dual_ru_version >= 3 else 3
                    )
                    self.dual_motion_usefulness_head = nn.Sequential(
                        nn.Conv1d(motion_usefulness_dim, model_dim, 1),
                        nn.ReLU(True),
                        nn.Conv1d(model_dim, 1, 1),
                        nn.Sigmoid()
                    )
                    self.dual_skeleton_usefulness_head = nn.Sequential(
                        nn.Conv1d(model_dim * 4, model_dim, 1),
                        nn.ReLU(True),
                        nn.Conv1d(model_dim, 1, 1),
                        nn.Sigmoid()
                    )
                    nn.init.zeros_(self.dual_motion_usefulness_head[2].weight)
                    if (self.dual_ru_v8 or self.dual_ru_v9 or
                            self.dual_ru_v10):
                        nn.init.constant_(
                            self.dual_motion_usefulness_head[2].bias, 2.944439
                        )
                    elif self.dual_ru_v6 or self.dual_ru_v7:
                        nn.init.zeros_(self.dual_motion_usefulness_head[2].bias)
                    elif self.dual_ru_version >= 2:
                        nn.init.constant_(
                            self.dual_motion_usefulness_head[2].bias, 2.1972246
                        )
                    else:
                        nn.init.zeros_(self.dual_motion_usefulness_head[2].bias)
                    nn.init.zeros_(self.dual_skeleton_usefulness_head[2].weight)
                    nn.init.zeros_(self.dual_skeleton_usefulness_head[2].bias)
                if self.dual_ru_v6:
                    self.dual_motion_usefulness_anchor_head = copy.deepcopy(
                        self.dual_motion_usefulness_head
                    )
                    self.dual_skeleton_usefulness_anchor_head = copy.deepcopy(
                        self.dual_skeleton_usefulness_head
                    )
                    for module in (
                            self.dual_motion_usefulness_anchor_head,
                            self.dual_skeleton_usefulness_anchor_head):
                        module.eval()
                        for parameter in module.parameters():
                            parameter.requires_grad = False
                    self.register_buffer(
                        'dual_output_anchor_initialized',
                        torch.tensor(False, dtype=torch.bool)
                    )
        if self.auto_gate and not self.auto_gate_v2:
            self.motion_score_residual_head = nn.Sequential(
                nn.Dropout(fc_drop),
                nn.Conv1d(model_dim * 3, 1, 1),
                nn.Tanh()
            )
            self.auto_motion_gate_head = nn.Conv1d(model_dim * 6, 1, 1)
            nn.init.zeros_(self.motion_score_residual_head[1].weight)
            nn.init.zeros_(self.motion_score_residual_head[1].bias)
            nn.init.zeros_(self.auto_motion_gate_head.weight)
            nn.init.constant_(self.auto_motion_gate_head.bias, auto_gate_init)
        self.tau = nn.Parameter(torch.ones([]) * 10)
        self.mse = nn.MSELoss()
        self._dual_ru_phase_configured = None
        self._dual_usefulness_anchor = None
        self._dual_teacher_state = None
        self._dual_teacher_raw_state = None
        self._dual_teacher_ema_state = None
        self._dual_teacher_ema_updates = 0
        self._dual_teacher_probe_loss = float('inf')
        self._dual_teacher_probe_coef = float('-inf')
        self._dual_teacher_probe_std = 0.0
        self._dual_teacher_probe_epoch = -1
        self._dual_teacher_restored = False
        self._dual_teacher_candidate = 'none'
        self.dual_ru_diagnostics = True
        self._base_requires_grad = {
            id(parameter): parameter.requires_grad for parameter in self.parameters()
        }

    def motion_saliency_delta(self, flow_feat, target_length):
        flow_feat = flow_feat.detach()
        if self.motion_saliency_source == 'delta':
            if flow_feat.shape[2] > 1:
                delta = flow_feat[:, :, 1:] - flow_feat[:, :, :-1]
                energy = delta.abs().mean(dim=1, keepdim=True)
                energy = F.pad(energy, (1, 0), mode='replicate')
            else:
                energy = flow_feat.new_ones(flow_feat.shape[0], 1, flow_feat.shape[2])
        else:
            energy = flow_feat.pow(2).mean(dim=1, keepdim=True)
        if energy.shape[2] > 2:
            energy = F.avg_pool1d(energy, kernel_size=3, stride=1, padding=1)
        saliency = energy / (energy.mean(dim=2, keepdim=True) + 1e-6)
        saliency = saliency.clamp(0.5, 2.0)
        if saliency.shape[2] != target_length:
            saliency = F.interpolate(saliency, size=target_length, mode='linear', align_corners=False)
        if self.motion_saliency_positive_only:
            saliency_delta = F.relu(saliency - 1)
        else:
            saliency_delta = saliency - 1
        return saliency_delta

    def set_cue_epoch(self, epoch, total_epochs=None):
        self.cue_current_epoch = epoch
        if total_epochs is not None:
            self.cue_total_epochs = max(int(total_epochs), 1)
        if self.dual_ru_gate:
            self._configure_dual_ru_phase(self.dual_ru_phase())

    @staticmethod
    def _is_dual_teacher_core_name(name):
        if name in ('c', 'tau'):
            return True
        prefixes = (
            'model_r.', 'model_f.', 'model_a.',
            'stage1.', 'stage2.', 'stage3.', 'fc.',
            'skeleton_encoder.', 'skeleton_score_residual_head.'
        )
        return name.startswith(prefixes)

    def consider_dual_teacher_snapshot(self, epoch, loss, coefficient,
                                       score_std):
        if not (self.dual_ru_v6 or self.dual_ru_v7 or
                self.dual_ru_v8 or self.dual_ru_v9 or
                self.dual_ru_v10 or self.dual_ru_v11) or \
                self.dual_ru_phase() != 0:
            return False
        if not self.dual_teacher_metrics_healthy(
                loss, coefficient, score_std):
            return False
        better = loss < self._dual_teacher_probe_loss - 1e-8
        tied = abs(loss - self._dual_teacher_probe_loss) <= 1e-8
        better = better or (tied and coefficient > self._dual_teacher_probe_coef)
        if not better:
            return False
        current_state = {
            name: value.detach().cpu().clone()
            for name, value in self.state_dict().items()
            if self._is_dual_teacher_core_name(name)
        }
        if (self.dual_ru_v8 or self.dual_ru_v9 or self.dual_ru_v10 or
                self.dual_ru_v11):
            if self.dual_ru_v9 or self.dual_ru_v10 or self.dual_ru_v11:
                self._dual_teacher_raw_state = {
                    name: value.clone() for name, value in current_state.items()
                }
            if self._dual_teacher_ema_state is None:
                self._dual_teacher_ema_state = current_state
            else:
                decay = self.dual_ru_teacher_ema_decay
                for name, current in current_state.items():
                    averaged = self._dual_teacher_ema_state[name]
                    if torch.is_floating_point(averaged):
                        averaged.mul_(decay).add_(current, alpha=1 - decay)
                    else:
                        averaged.copy_(current)
            self._dual_teacher_ema_updates += 1
            self._dual_teacher_state = {
                name: value.clone()
                for name, value in self._dual_teacher_ema_state.items()
            }
            self._dual_teacher_candidate = 'ema'
        else:
            self._dual_teacher_state = current_state
            self._dual_teacher_candidate = 'raw'
        self._dual_teacher_probe_loss = float(loss)
        self._dual_teacher_probe_coef = float(coefficient)
        self._dual_teacher_probe_std = float(score_std)
        self._dual_teacher_probe_epoch = int(epoch)
        return True

    def dual_teacher_metrics_healthy(self, loss, coefficient, score_std):
        values = (loss, coefficient, score_std)
        healthy = all(np.isfinite(value) for value in values)
        healthy = healthy and score_std >= self.dual_ru_teacher_min_score_std
        if (self.dual_ru_v8 or self.dual_ru_v9 or self.dual_ru_v10 or
                self.dual_ru_v11):
            healthy = (
                healthy and
                coefficient >= self.dual_ru_teacher_min_probe_coef
            )
        return healthy

    def dual_teacher_candidate_names(self):
        if not (self.dual_ru_v9 or self.dual_ru_v10 or self.dual_ru_v11):
            return ()
        names = []
        if self._dual_teacher_raw_state:
            names.append('raw')
        if self._dual_teacher_ema_state:
            names.append('ema')
        return tuple(names)

    def restore_dual_teacher_candidate(self, candidate):
        states = {
            'raw': self._dual_teacher_raw_state,
            'ema': self._dual_teacher_ema_state
        }
        if candidate not in states or not states[candidate]:
            self._dual_teacher_restored = False
            return False
        self._dual_teacher_state = {
            name: value.clone() for name, value in states[candidate].items()
        }
        self._dual_teacher_candidate = candidate
        return self._restore_dual_teacher_snapshot()

    def confirm_dual_teacher_candidate(self, candidate, loss, coefficient,
                                       score_std):
        if not self.dual_teacher_metrics_healthy(
                loss, coefficient, score_std):
            return False
        if not self.restore_dual_teacher_candidate(candidate):
            return False
        self._dual_teacher_probe_loss = float(loss)
        self._dual_teacher_probe_coef = float(coefficient)
        self._dual_teacher_probe_std = float(score_std)
        return True

    def _restore_dual_teacher_snapshot(self):
        if not self._dual_teacher_state:
            self._dual_teacher_restored = False
            return False
        current_state = self.state_dict()
        with torch.no_grad():
            for name, saved in self._dual_teacher_state.items():
                current = current_state[name]
                current.copy_(saved.to(device=current.device, dtype=current.dtype))
        self._dual_teacher_restored = True
        return True

    def reset_dual_teacher_optimizer_state(self, optimizer):
        if not (self.dual_ru_v6 or self.dual_ru_v7 or
                self.dual_ru_v8 or self.dual_ru_v9 or
                self.dual_ru_v10 or self.dual_ru_v11) or \
                not self._dual_teacher_restored:
            return 0
        reset = 0
        for name, parameter in self.named_parameters():
            if self._is_dual_teacher_core_name(name):
                if parameter in optimizer.state:
                    optimizer.state.pop(parameter)
                    reset += 1
        return reset

    def dual_teacher_probe_info(self):
        return {
            'epoch': self._dual_teacher_probe_epoch,
            'loss': self._dual_teacher_probe_loss,
            'coefficient': self._dual_teacher_probe_coef,
            'score_std': self._dual_teacher_probe_std,
            'ema_updates': self._dual_teacher_ema_updates,
            'candidate': self._dual_teacher_candidate,
            'restored': self._dual_teacher_restored
        }

    def dual_ru_phase(self):
        warmup_end = max(1, int(round(
            self.cue_total_epochs * self.dual_ru_warmup_ratio
        )))
        supervision_end = max(warmup_end + 1, int(round(
            self.cue_total_epochs * self.dual_ru_supervision_ratio
        )))
        if self.cue_current_epoch < warmup_end:
            return 0
        if self.cue_current_epoch < supervision_end:
            return 1
        return 2

    def dual_ru_gate_ramp(self):
        if self.dual_ru_phase() < 2:
            return 0.0
        if not (self.dual_ru_v4 or self.dual_ru_v5 or self.dual_ru_v6 or
                self.dual_ru_v7 or self.dual_ru_v8 or self.dual_ru_v9 or
                self.dual_ru_v10 or self.dual_ru_v11):
            return 1.0
        adaptive_start = max(1, int(round(
            self.cue_total_epochs * self.dual_ru_supervision_ratio
        )))
        adaptive_epochs = max(self.cue_total_epochs - adaptive_start, 1)
        ramp_epochs = max(1, int(round(
            adaptive_epochs * self.dual_ru_gate_ramp_ratio
        )))
        elapsed = max(self.cue_current_epoch - adaptive_start, 0)
        return min(elapsed / ramp_epochs, 1.0)

    def dual_ru_phase2_progress(self):
        if self.dual_ru_phase() < 2:
            return 0.0
        adaptive_start = max(1, int(round(
            self.cue_total_epochs * self.dual_ru_supervision_ratio
        )))
        adaptive_epochs = max(self.cue_total_epochs - adaptive_start - 1, 1)
        elapsed = max(self.cue_current_epoch - adaptive_start, 0)
        return min(elapsed / adaptive_epochs, 1.0)

    def dual_ru_target_weight(self):
        if not self.dual_ru_v9 or self.dual_ru_phase() < 2:
            return 1.0
        progress = self.dual_ru_phase2_progress()
        end_weight = self.dual_ru_phase2_target_end_weight
        return 1.0 + progress * (end_weight - 1.0)

    def _configure_dual_ru_phase(self, phase):
        if phase == self._dual_ru_phase_configured:
            return
        if phase == 1 and (
                self.dual_ru_v6 or self.dual_ru_v7 or self.dual_ru_v8 or
                self.dual_ru_v9 or self.dual_ru_v10 or self.dual_ru_v11):
            self._restore_dual_teacher_snapshot()
        for parameter in self.parameters():
            parameter.requires_grad = self._base_requires_grad[id(parameter)]
        quality_modules = [self.motion_quality_head, self.skeleton_quality_head]
        usefulness_modules = [
            self.dual_motion_usefulness_head,
            self.dual_skeleton_usefulness_head
        ]
        if phase == 1 and (self.dual_ru_version == 1 or self.dual_ru_v5 or
                           self.dual_ru_v6 or self.dual_ru_v7 or
                           self.dual_ru_v8 or self.dual_ru_v9 or
                           self.dual_ru_v10 or self.dual_ru_v11):
            for parameter in self.parameters():
                parameter.requires_grad = False
            for module in quality_modules + usefulness_modules:
                for parameter in module.parameters():
                    parameter.requires_grad = True
        elif phase == 2:
            if (self.dual_ru_v7 or self.dual_ru_v8 or self.dual_ru_v9 or
                    self.dual_ru_v10 or self.dual_ru_v11):
                for parameter in self.parameters():
                    parameter.requires_grad = False
                for module in usefulness_modules:
                    for parameter in module.parameters():
                        parameter.requires_grad = True
            else:
                if self.dual_ru_v5 and self._dual_usefulness_anchor is None:
                    self._dual_usefulness_anchor = {
                        name: parameter.detach().clone()
                        for name, parameter in self._dual_usefulness_named_parameters()
                    }
                if self.dual_ru_v6:
                    if not self.dual_output_anchor_initialized.item():
                        self.dual_motion_usefulness_anchor_head.load_state_dict(
                            self.dual_motion_usefulness_head.state_dict()
                        )
                        self.dual_skeleton_usefulness_anchor_head.load_state_dict(
                            self.dual_skeleton_usefulness_head.state_dict()
                        )
                        self.dual_output_anchor_initialized.fill_(True)
                frozen_modules = quality_modules
                if not (self.dual_ru_v5 or self.dual_ru_v6):
                    frozen_modules = frozen_modules + usefulness_modules
                for module in frozen_modules:
                    for parameter in module.parameters():
                        parameter.requires_grad = False
        self._dual_ru_phase_configured = phase

    def dual_ru_needs_counterfactual(self):
        if (self.dual_ru_v7 or self.dual_ru_v8 or self.dual_ru_v9 or
                self.dual_ru_v10 or self.dual_ru_v11):
            return (
                self.dual_ru_gate and self.training and
                self.dual_ru_phase() in (1, 2)
            )
        return self.dual_ru_gate and self.training and self.dual_ru_phase() == 1

    def prepare_dual_ru_train_mode(self):
        phase = self.dual_ru_phase()
        if not self.dual_ru_gate:
            return
        if (self.dual_ru_v7 or self.dual_ru_v8 or self.dual_ru_v9 or
                self.dual_ru_v10 or self.dual_ru_v11):
            if phase not in (1, 2):
                return
        elif phase != 1:
            return
        if (self.dual_ru_version >= 3 and
                not (self.dual_ru_v5 or self.dual_ru_v6 or self.dual_ru_v7 or
                     self.dual_ru_v8 or self.dual_ru_v9 or
                     self.dual_ru_v10 or self.dual_ru_v11)):
            return
        frozen_modules = [
            self.model_r, self.model_f, self.model_a,
            self.stage1, self.stage2, self.stage3, self.drop, self.fc,
            self.skeleton_encoder, self.skeleton_score_residual_head
        ]
        for module in frozen_modules:
            module.eval()
        if phase == 1:
            self.motion_quality_head.train()
            self.skeleton_quality_head.train()
        else:
            self.motion_quality_head.eval()
            self.skeleton_quality_head.eval()
        self.dual_motion_usefulness_head.train()
        self.dual_skeleton_usefulness_head.train()
        if self.dual_ru_v6:
            self.dual_motion_usefulness_anchor_head.eval()
            self.dual_skeleton_usefulness_anchor_head.eval()

    def dual_ru_usefulness_parameters(self):
        if not self.dual_ru_gate:
            return []
        return list(self.dual_motion_usefulness_head.parameters()) + list(
            self.dual_skeleton_usefulness_head.parameters()
        )

    def _dual_usefulness_named_parameters(self):
        for prefix, module in (
                ('motion', self.dual_motion_usefulness_head),
                ('skeleton', self.dual_skeleton_usefulness_head)):
            for name, parameter in module.named_parameters():
                yield f'{prefix}.{name}', parameter

    def _dual_ru_anchor_loss(self):
        if not self._dual_usefulness_anchor:
            return self.tau.new_tensor(0.0)
        terms = []
        for name, parameter in self._dual_usefulness_named_parameters():
            anchor = self._dual_usefulness_anchor[name].to(parameter.device)
            terms.append((parameter - anchor).square().mean())
        return torch.stack(terms).mean()

    def motion_saliency_weight(self, flow_feat, target_length):
        saliency_delta = self.motion_saliency_delta(flow_feat, target_length)
        return 1 + self.motion_saliency_alpha * saliency_delta

    def apply_motion_saliency_residual(self, x, flow_feat, motion_alpha=None):
        saliency_delta = self.motion_saliency_delta(flow_feat, x.shape[2])
        if motion_alpha is None:
            motion_alpha = self.motion_saliency_alpha
        return x + motion_alpha * saliency_delta * x

    def auto_gate_delta_effect(self, delta):
        if self.auto_gate_no_downscale:
            return F.relu(delta)
        return delta

    def motion_reliability(self, flow_feat, target_length):
        saliency_delta = self.motion_saliency_delta(flow_feat, target_length)
        reliability = saliency_delta.abs().mean(dim=2, keepdim=True)
        return reliability.clamp(0, 1)

    def skeleton_reliability(self, skeleton):
        if skeleton.shape[1] >= 4:
            joints = skeleton.view(skeleton.shape[0], skeleton.shape[1] // 4, 4, skeleton.shape[2])
            conf = joints[:, :, 3, :].clamp(0, 1)
            quality = conf.mean(dim=(1, 2), keepdim=True)
            coords = joints[:, :, :3, :].reshape(skeleton.shape[0], -1, skeleton.shape[2])
        else:
            quality = skeleton.new_ones(skeleton.shape[0], 1, 1)
            coords = skeleton
        if coords.shape[2] > 1:
            jitter = (coords[:, :, 1:] - coords[:, :, :-1]).abs().mean(dim=(1, 2), keepdim=True)
            scale = coords.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            smoothness = torch.exp(-(jitter / scale).clamp(0, 5))
        else:
            smoothness = skeleton.new_ones(skeleton.shape[0], 1, 1)
        return (quality * smoothness).clamp(0, 1)

    def corrupt_motion_cue(self, motion):
        batch_size, _, length = motion.shape
        apply = (torch.rand(batch_size, 1, 1, device=motion.device) <
                 self.cue_corruption_prob).to(motion.dtype)
        severity = torch.rand(batch_size, 1, 1, device=motion.device)
        severity = severity * self.cue_corruption_max * apply
        keep = (torch.rand(batch_size, 1, length, device=motion.device) >= severity)
        keep = keep.to(motion.dtype)
        return motion * keep, keep.mean(dim=2, keepdim=True)

    def corrupt_skeleton_cue(self, skeleton):
        batch_size, channels, length = skeleton.shape
        apply = (torch.rand(batch_size, 1, 1, device=skeleton.device) <
                 self.cue_corruption_prob).to(skeleton.dtype)
        severity = torch.rand(batch_size, 1, 1, device=skeleton.device)
        severity = severity * self.cue_corruption_max * apply
        if channels % 4 == 0:
            joints = skeleton.view(batch_size, channels // 4, 4, length)
            keep = torch.rand(batch_size, channels // 4, 1, length,
                              device=skeleton.device) >= severity.unsqueeze(1)
            keep = keep.to(skeleton.dtype)
            corrupted = (joints * keep).view_as(skeleton)
            target = keep.mean(dim=(1, 3)).unsqueeze(2)
        else:
            keep = torch.rand(batch_size, 1, length, device=skeleton.device) >= severity
            keep = keep.to(skeleton.dtype)
            corrupted = skeleton * keep
            target = keep.mean(dim=2, keepdim=True)
        return corrupted, target

    def motion_quality_features(self, motion):
        motion = motion.detach()
        magnitude = motion.abs().mean(dim=1, keepdim=True)
        active = (magnitude > 1e-6).to(motion.dtype).mean(dim=2, keepdim=True)
        if motion.shape[2] > 1:
            delta = (motion[:, :, 1:] - motion[:, :, :-1]).abs()
            delta = delta.mean(dim=(1, 2), keepdim=True)
            scale = magnitude.mean(dim=2, keepdim=True).clamp_min(1e-6)
            smoothness = torch.exp(-(delta / scale).clamp(0, 5))
        else:
            smoothness = motion.new_ones(motion.shape[0], 1, 1)
        if self.cue_quality_zero_anchor:
            smoothness = smoothness * active
        dynamics = self.motion_reliability(motion, motion.shape[2])
        return torch.cat([active, smoothness, dynamics], dim=1)

    def skeleton_quality_features(self, skeleton):
        skeleton = skeleton.detach()
        batch_size, channels, length = skeleton.shape
        if channels % 4 == 0:
            joints = skeleton.view(batch_size, channels // 4, 4, length)
            confidence = joints[:, :, 3, :].clamp(0, 1).mean(
                dim=(1, 2), keepdim=True
            )
            coords = joints[:, :, :3, :].reshape(batch_size, -1, length)
        else:
            confidence = skeleton.new_ones(batch_size, 1, 1)
            coords = skeleton
        magnitude = coords.abs().mean(dim=1, keepdim=True)
        active = (magnitude > 1e-6).to(skeleton.dtype).mean(dim=2, keepdim=True)
        if length > 1:
            jitter = (coords[:, :, 1:] - coords[:, :, :-1]).abs()
            jitter = jitter.mean(dim=(1, 2), keepdim=True)
            scale = magnitude.mean(dim=2, keepdim=True).clamp_min(1e-6)
            smoothness = torch.exp(-(jitter / scale).clamp(0, 5))
        else:
            smoothness = skeleton.new_ones(batch_size, 1, 1)
        if self.cue_quality_zero_anchor:
            smoothness = smoothness * active
        return torch.cat([active, confidence, smoothness], dim=1)

    @staticmethod
    def _dual_ru_deterministic_corruption(cue, severity):
        length = cue.shape[2]
        drop_count = int(round(length * float(severity)))
        positions = torch.arange(length, device=cue.device)
        order = (positions * 37) % length
        keep = (order >= drop_count).to(cue.dtype).view(1, 1, length)
        return cue * keep

    def normalize_clean_quality_gate(self, quality, cue):
        reference = getattr(self, f'{cue}_clean_quality_ref')
        variance = getattr(self, f'{cue}_clean_quality_var')
        initialized = getattr(self, f'{cue}_clean_quality_initialized')
        if self.training:
            batch_reference = quality.detach().mean()
            batch_variance = quality.detach().var(unbiased=False)
            if not initialized.item():
                reference.copy_(batch_reference)
                variance.copy_(batch_variance.clamp_min(self.cue_quality_gate_min_std ** 2))
                initialized.fill_(True)
            else:
                reference.lerp_(batch_reference, self.cue_quality_ema_momentum)
                variance.lerp_(
                    batch_variance.clamp_min(self.cue_quality_gate_min_std ** 2),
                    self.cue_quality_ema_momentum
                )
        if self.cue_quality_gate_mode == 'calibrated':
            std = variance.detach().sqrt().clamp_min(self.cue_quality_gate_min_std)
            normalized = (quality - reference.detach()) / std
            excess = (normalized.abs() - self.cue_quality_gate_margin).clamp_min(0)
            adjusted = normalized.sign() * excess
            positive = 1 + self.cue_quality_gate_up_span * torch.tanh(adjusted)
            negative = 1 + self.cue_quality_gate_down_span * torch.tanh(adjusted)
            gate = torch.where(adjusted >= 0, positive, negative)
            return gate.clamp(
                self.cue_quality_gate_floor,
                self.cue_quality_gate_ceiling
            ), reference
        if self.cue_quality_gate_mode == 'ema_ratio':
            normalized = quality / reference.detach().clamp_min(1e-6)
            return normalized.clamp(
                self.cue_quality_gate_floor,
                self.cue_quality_gate_ceiling
            ), reference
        return quality, reference

    def calibrated_reliability_factor(self, reliability, cue):
        running_mean = getattr(self, f'{cue}_reliability_mean')
        running_var = getattr(self, f'{cue}_reliability_var')
        initialized = getattr(self, f'{cue}_reliability_initialized')
        if self.training:
            batch_mean = reliability.detach().mean()
            batch_var = reliability.detach().var(unbiased=False)
            if not initialized.item():
                running_mean.copy_(batch_mean)
                running_var.copy_(batch_var)
                initialized.fill_(True)
            else:
                momentum = self.cue_reliability_momentum
                running_mean.lerp_(batch_mean, momentum)
                running_var.lerp_(batch_var, momentum)
        std = running_var.sqrt().clamp_min(self.cue_reliability_min_std)
        normalized = (reliability - running_mean) / std
        return 1 + self.cue_reliability_span * torch.tanh(normalized)

    def cue_reliability_factor(self, reliability, cue):
        if self.cue_reliability_mode == 'calibrated':
            return self.calibrated_reliability_factor(reliability, cue)
        if self.cue_reliability_mode == 'soft':
            return self.cue_reliability_floor + self.cue_reliability_span * reliability
        return reliability

    def motion_saliency_pool(self, x, flow_feat):
        weight = self.motion_saliency_weight(flow_feat, x.shape[2])
        weighted_sum = (x * weight).sum(dim=2, keepdim=True)
        normalizer = weight.sum(dim=2, keepdim=True).clamp_min(1e-6)
        return weighted_sum / normalizer

    @staticmethod
    def _dual_ru_context(x):
        pooled = F.adaptive_avg_pool1d(x.detach(), 1).squeeze(dim=2)
        return F.layer_norm(pooled, (pooled.shape[1],)).unsqueeze(dim=2)

    @staticmethod
    def _dual_ru_correlation(prediction, target):
        prediction = prediction.reshape(-1)
        target = target.reshape(-1)
        pred_centered = prediction - prediction.mean()
        target_centered = target - target.mean()
        denominator = (
            pred_centered.square().sum() * target_centered.square().sum()
        ).sqrt().clamp_min(1e-6)
        return (pred_centered * target_centered).sum() / denominator

    def _dual_ru_calibrated_reliability(self, prediction, cue,
                                        zero_prediction=None):
        if self.dual_ru_version < 2:
            return prediction
        reference = getattr(self, f'dual_{cue}_clean_ref')
        initialized = getattr(self, f'dual_{cue}_clean_initialized')
        if self.training and self.dual_ru_phase() < 2:
            batch_reference = prediction.detach().mean()
            if not initialized.item():
                reference.copy_(batch_reference)
                initialized.fill_(True)
            else:
                reference.lerp_(
                    batch_reference, self.dual_ru_reliability_momentum
                )
        if (self.dual_ru_v7 or self.dual_ru_v8 or self.dual_ru_v9 or
                self.dual_ru_v10 or self.dual_ru_v11):
            zero_reference = getattr(self, f'dual_{cue}_zero_ref')
            zero_initialized = getattr(
                self, f'dual_{cue}_zero_initialized'
            )
            if zero_prediction is not None:
                batch_zero = zero_prediction.detach().mean()
                if not zero_initialized.item():
                    zero_reference.copy_(batch_zero)
                    zero_initialized.fill_(True)
                elif self.training and self.dual_ru_phase() < 2:
                    zero_reference.lerp_(
                        batch_zero, self.dual_ru_reliability_momentum
                    )
                zero_anchor = zero_prediction.detach()
            else:
                zero_anchor = zero_reference.detach()
            span = (
                reference.detach() - zero_reference.detach()
            ).clamp_min(1e-4)
            clean_ceiling = 1.0 if (
                self.dual_ru_v10 or self.dual_ru_v11
            ) else 0.95
            return clean_ceiling * (
                (prediction - zero_anchor) / span
            ).clamp(0, 1)
        if self.dual_ru_v5 or self.dual_ru_v6:
            eps = 1e-4
            prediction_logit = torch.logit(prediction.clamp(eps, 1 - eps))
            reference_logit = torch.logit(
                reference.detach().clamp(eps, 1 - eps)
            )
            clean_logit = torch.logit(prediction.new_tensor(0.95))
            return torch.sigmoid(
                prediction_logit - reference_logit + clean_logit
            )
        return (prediction / reference.clamp_min(1e-4)).clamp(0, 1)

    def _dual_ru_pairwise_rank_error(self, score, label):
        score = score.reshape(-1)
        label = label.reshape(-1)
        label_delta = label[:, None] - label[None, :]
        valid = label_delta.abs() > 1e-6
        signed_margin = label_delta.sign() * (
            score[:, None] - score[None, :]
        )
        pair_error = F.softplus(
            -signed_margin / self.dual_ru_rank_temperature
        )
        valid_weight = valid.to(pair_error.dtype)
        return (pair_error * valid_weight).sum(dim=1) / (
            valid_weight.sum(dim=1).clamp_min(1.0)
        )

    def _dual_ru_shared_error_scale(self, full_error):
        batch_scale = full_error.detach().mean().clamp_min(
            self.dual_ru_error_scale_min
        )
        initialized = self.dual_usefulness_error_scale_initialized
        if not initialized.item():
            self.dual_usefulness_error_scale.copy_(batch_scale)
            initialized.fill_(True)
        else:
            self.dual_usefulness_error_scale.lerp_(
                batch_scale, self.dual_ru_error_scale_momentum
            )
        return self.dual_usefulness_error_scale.detach().clamp_min(
            self.dual_ru_error_scale_min
        )

    def _dual_ru_shared_rank_scale(self, rank_error):
        batch_scale = rank_error.detach().mean().clamp_min(1e-3)
        initialized = self.dual_usefulness_rank_scale_initialized
        if not initialized.item():
            self.dual_usefulness_rank_scale.copy_(batch_scale)
            initialized.fill_(True)
        else:
            self.dual_usefulness_rank_scale.lerp_(
                batch_scale, self.dual_ru_error_scale_momentum
            )
        return self.dual_usefulness_rank_scale.detach().clamp_min(1e-3)

    def _dual_ru_benefit_scale(self, cue, objective, benefit):
        scale = getattr(self, f'dual_{cue}_{objective}_benefit_scale')
        initialized = getattr(
            self, f'dual_{cue}_{objective}_benefit_scale_initialized'
        )
        batch_scale = benefit.detach().abs().mean().clamp_min(
            self.dual_ru_benefit_scale_min
        )
        if not initialized.item():
            scale.copy_(batch_scale)
            initialized.fill_(True)
        else:
            scale.lerp_(batch_scale, self.dual_ru_error_scale_momentum)
        return scale.detach().clamp_min(self.dual_ru_benefit_scale_min)

    def _dual_ru_effect_scale(self, cue, score_effect):
        scale = getattr(self, f'dual_{cue}_effect_scale')
        initialized = getattr(
            self, f'dual_{cue}_effect_scale_initialized'
        )
        batch_scale = score_effect.detach().mean().clamp_min(
            self.dual_ru_effect_scale_min
        )
        if not initialized.item():
            scale.copy_(batch_scale)
            initialized.fill_(True)
        else:
            scale.lerp_(batch_scale, self.dual_ru_effect_scale_momentum)
        return scale.detach().clamp_min(self.dual_ru_effect_scale_min)

    @staticmethod
    def _dual_ru_masked_correlation(prediction, target, mask):
        prediction = prediction.reshape(-1)
        target = target.reshape(-1)
        weight = (mask.reshape(-1) > 0).to(prediction.dtype)
        count = weight.sum().clamp_min(1.0)
        pred_mean = (prediction * weight).sum() / count
        target_mean = (target * weight).sum() / count
        pred_centered = (prediction - pred_mean) * weight
        target_centered = (target - target_mean) * weight
        denominator = (
            pred_centered.square().sum() * target_centered.square().sum()
        ).sqrt().clamp_min(1e-6)
        return (pred_centered * target_centered).sum() / denominator

    def _dual_ru_quality_loss(self, motion_source, skeleton, motion_clean_pred,
                              skeleton_clean_pred, other_info):
        if not self.training or self.dual_ru_phase() == 2:
            return
        corrupted_motion, motion_target = self.corrupt_motion_cue(motion_source)
        corrupted_skeleton, skeleton_target = self.corrupt_skeleton_cue(skeleton)
        motion_corrupt_pred = self.motion_quality_head(
            self.motion_quality_features(corrupted_motion)
        )
        skeleton_corrupt_pred = self.skeleton_quality_head(
            self.skeleton_quality_features(corrupted_skeleton)
        )
        zero_predictions = {'motion': None, 'skeleton': None}
        if self.cue_quality_zero_anchor:
            zero_predictions = {
                'motion': self.motion_quality_head(
                    self.motion_quality_features(torch.zeros_like(motion_source))
                ),
                'skeleton': self.skeleton_quality_head(
                    self.skeleton_quality_features(torch.zeros_like(skeleton))
                )
            }

        cue_losses = []
        for cue, clean_pred, corrupt_pred, target in (
                ('motion', motion_clean_pred, motion_corrupt_pred, motion_target),
                 ('skeleton', skeleton_clean_pred, skeleton_corrupt_pred,
                  skeleton_target)):
            clean_loss = F.mse_loss(clean_pred, torch.ones_like(clean_pred))
            corrupt_loss = F.mse_loss(corrupt_pred, target)
            pair_loss = F.mse_loss(clean_pred - corrupt_pred, 1 - target)
            denominator = 2.0
            cue_loss = clean_loss + corrupt_loss
            zero_pred = zero_predictions[cue]
            if zero_pred is not None:
                zero_loss = F.mse_loss(zero_pred, torch.zeros_like(zero_pred))
                cue_loss = cue_loss + self.cue_quality_zero_weight * zero_loss
                denominator += self.cue_quality_zero_weight
                other_info[f'{cue}_reliability_zero_pred'] = zero_pred.mean()
            cue_loss = cue_loss / denominator
            cue_loss = cue_loss + self.cue_quality_pair_weight * pair_loss
            cue_losses.append(cue_loss)
            other_info[f'{cue}_reliability_loss'] = cue_loss.detach()
            other_info[f'{cue}_reliability_target'] = target.mean()
            other_info[f'{cue}_reliability_mae'] = (
                corrupt_pred - target
            ).abs().mean()
            other_info[f'{cue}_reliability_corr'] = self._dual_ru_correlation(
                corrupt_pred, target
            )
            other_info[f'{cue}_reliability_clean_pred'] = clean_pred.mean()
            other_info[f'{cue}_reliability_corrupt_pred'] = corrupt_pred.mean()
        other_info['cue_quality_loss'] = (
            self.cue_quality_loss_weight * torch.stack(cue_losses).mean()
        )

    def _forward_dual_ru(self, input_feats, force_gates=None,
                         corruption=None):
        if 'S' not in input_feats:
            raise ValueError('dual_ru_gate requires skeleton input')
        skeleton = rearrange(
            input_feats['S'], 'b t d -> b d t'
        ).contiguous()

        _, feat_r = self.model_r(input_feats)
        _, feat_a = self.model_a(input_feats)
        _, feat_f = self.model_f(input_feats)
        motion_source = feat_f['feats'][0]
        if corruption is not None:
            cue, severity = corruption
            if cue not in ('motion', 'skeleton'):
                raise ValueError(f'unknown dual RU corruption cue: {cue}')
            if not 0 <= float(severity) <= 1:
                raise ValueError('dual RU corruption severity must be in [0, 1]')
            if cue == 'motion':
                motion_source = self._dual_ru_deterministic_corruption(
                    motion_source, severity
                )
            else:
                skeleton = self._dual_ru_deterministic_corruption(
                    skeleton, severity
                )

        motion_quality_pred = self.motion_quality_head(
            self.motion_quality_features(motion_source)
        )
        motion_zero_pred = None
        skeleton_zero_pred = None
        if (self.dual_ru_v7 or self.dual_ru_v8 or self.dual_ru_v9 or
                self.dual_ru_v10 or self.dual_ru_v11):
            motion_zero_pred = self.motion_quality_head(
                self.motion_quality_features(torch.zeros_like(motion_source))
            )
            skeleton_zero_pred = self.skeleton_quality_head(
                self.skeleton_quality_features(torch.zeros_like(skeleton))
            )
        motion_reliability = self._dual_ru_calibrated_reliability(
            motion_quality_pred, 'motion', motion_zero_pred
        )
        skeleton_quality_pred = self.skeleton_quality_head(
            self.skeleton_quality_features(skeleton)
        )
        skeleton_reliability = self._dual_ru_calibrated_reliability(
            skeleton_quality_pred, 'skeleton', skeleton_zero_pred
        )
        if 'SQ' in input_feats:
            external_quality = input_feats['SQ'].reshape(
                skeleton.shape[0], -1
            ).mean(dim=1, keepdim=True).unsqueeze(dim=2).clamp(0, 1)
            skeleton_reliability = skeleton_reliability * external_quality

        early_quality_info = {}
        if (self.dual_ru_v5 or self.dual_ru_v6 or self.dual_ru_v7 or
                self.dual_ru_v8 or self.dual_ru_v9 or self.dual_ru_v10 or
                self.dual_ru_v11):
            # Match the fixed-cue RNG order before scoring dropout.
            self._dual_ru_quality_loss(
                motion_source, skeleton, motion_quality_pred,
                skeleton_quality_pred, early_quality_info
            )

        feats = [[self.drop(value.detach()) for value in feat_r['feats']],
                 [self.drop(value.detach()) for value in feat_f['feats']],
                 [self.drop(value) for value in feat_a['feats']]]
        x = self.c.repeat([
            feats[0][0].shape[0], feats[0][0].shape[2] // 2, 1
        ]).permute([0, 2, 1])

        x1, action1 = self.stage1(
            x, [value[0] for value in feats], [value[1] for value in feats],
            self.tau
        )
        if self.dual_ru_v11:
            motion_usefulness = self.dual_motion_usefulness_head(x1)
        else:
            motion_context = self._dual_ru_context(x1)
            if self.dual_ru_version >= 3:
                motion_context = torch.cat([
                    motion_context, self._dual_ru_context(motion_source)
                ], dim=1)
            motion_usefulness = self.dual_motion_usefulness_head(motion_context)
        phase = self.dual_ru_phase()
        motion_usefulness_anchor = None
        if self.dual_ru_v6 and phase == 2:
            with torch.no_grad():
                motion_usefulness_anchor = (
                    self.dual_motion_usefulness_anchor_head(motion_context)
                )
        gate_ramp = self.dual_ru_gate_ramp()
        if force_gates is not None and force_gates[0] is not None:
            motion_gate_norm = x1.new_full(
                (x1.shape[0], 1, 1), float(force_gates[0])
            )
        elif phase < 2:
            motion_gate_norm = x1.new_ones((x1.shape[0], 1, 1))
        else:
            if (self.dual_ru_v5 or self.dual_ru_v6 or self.dual_ru_v7 or
                    self.dual_ru_v8 or self.dual_ru_v9 or
                    self.dual_ru_v10 or self.dual_ru_v11):
                adaptive_motion_gate = (
                    motion_reliability.detach() * motion_usefulness
                ).clamp(0, 1)
            else:
                adaptive_motion_gate = (
                    motion_reliability.detach() * motion_usefulness.detach()
                ).clamp(0, 1)
            if (self.dual_ru_v4 or self.dual_ru_v5 or self.dual_ru_v6 or
                    self.dual_ru_v7 or self.dual_ru_v8 or self.dual_ru_v9 or
                    self.dual_ru_v10 or self.dual_ru_v11):
                motion_gate_norm = 1 + gate_ramp * (
                    adaptive_motion_gate - 1
                )
            else:
                motion_gate_norm = adaptive_motion_gate
        motion_gate = self.auto_gate_motion_max * motion_gate_norm
        motion_delta1 = self.motion_saliency_delta(motion_source, x1.shape[2])
        motion_correction1 = motion_gate * motion_delta1
        x1 = x1 + motion_correction1 * x1
        x1 = self.pool(x1)

        x2, action2 = self.stage2(
            x1, [value[1] for value in feats], [value[2] for value in feats],
            self.tau
        )
        motion_delta2 = self.motion_saliency_delta(motion_source, x2.shape[2])
        motion_correction2 = motion_gate * motion_delta2
        x2 = x2 + motion_correction2 * x2
        x2 = self.gap(x2)

        x3, action3 = self.stage3(
            x2, [value[2] for value in feats], [value[3] for value in feats],
            self.tau
        )
        score = self.fc(x3).squeeze(dim=2)

        skeleton_feat = self.skeleton_encoder(skeleton)
        if self.dual_ru_v11:
            skeleton_usefulness = self.dual_skeleton_usefulness_head(
                skeleton_feat
            )
        else:
            skeleton_context = torch.cat([
                self._dual_ru_context(x3), self._dual_ru_context(skeleton_feat)
            ], dim=1)
            skeleton_usefulness = self.dual_skeleton_usefulness_head(
                skeleton_context
            )
        skeleton_usefulness_anchor = None
        if self.dual_ru_v6 and phase == 2:
            with torch.no_grad():
                skeleton_usefulness_anchor = (
                    self.dual_skeleton_usefulness_anchor_head(skeleton_context)
                )
        if force_gates is not None and force_gates[1] is not None:
            skeleton_gate_norm = score.new_full(
                (score.shape[0], 1, 1), float(force_gates[1])
            )
        elif phase < 2:
            skeleton_gate_norm = score.new_full(
                (score.shape[0], 1, 1), 0.5
            )
        else:
            if (self.dual_ru_v5 or self.dual_ru_v6 or self.dual_ru_v7 or
                    self.dual_ru_v8 or self.dual_ru_v9 or
                    self.dual_ru_v10 or self.dual_ru_v11):
                adaptive_skeleton_gate = (
                    skeleton_reliability.detach() * skeleton_usefulness
                ).clamp(0, 1)
            else:
                adaptive_skeleton_gate = (
                    skeleton_reliability.detach() * skeleton_usefulness.detach()
                ).clamp(0, 1)
            if (self.dual_ru_v4 or self.dual_ru_v5 or self.dual_ru_v6 or
                    self.dual_ru_v7 or self.dual_ru_v8 or self.dual_ru_v9 or
                    self.dual_ru_v10 or self.dual_ru_v11):
                skeleton_gate_norm = 0.5 + gate_ramp * (
                    adaptive_skeleton_gate - 0.5
                )
            else:
                skeleton_gate_norm = adaptive_skeleton_gate
        skeleton_gate = (
            self.auto_gate_skeleton_max * skeleton_gate_norm.squeeze(dim=2)
        )
        skeleton_residual = self.skeleton_score_residual_head(
            skeleton_feat
        ).squeeze(dim=2)
        skeleton_correction = skeleton_gate * skeleton_residual
        score = score + skeleton_correction

        motion_correction = 0.5 * (
            motion_correction1.abs().mean() + motion_correction2.abs().mean()
        )
        other_info = early_quality_info
        other_info.update({
            'action': [action1, action2, action3],
            'dual_ru_phase': score.new_tensor(float(phase)),
            'dual_gate_ramp': score.new_tensor(float(gate_ramp)),
            'score_mean': score.mean(),
            'score_square': score.square().mean(),
            'score_min': score.min(),
            'score_max': score.max(),
            'motion_reliability': motion_reliability.mean(),
            'motion_reliability_square': motion_reliability.square().mean(),
            'motion_usefulness': motion_usefulness.mean(),
            'motion_usefulness_square': motion_usefulness.square().mean(),
            'motion_gate_norm': motion_gate_norm.mean(),
            'motion_gate_norm_square': motion_gate_norm.square().mean(),
            'motion_gate': motion_gate.mean(),
            'motion_correction': motion_correction,
            'skeleton_reliability': skeleton_reliability.mean(),
            'skeleton_reliability_square': skeleton_reliability.square().mean(),
            'skeleton_usefulness': skeleton_usefulness.mean(),
            'skeleton_usefulness_square': skeleton_usefulness.square().mean(),
            'skeleton_gate_norm': skeleton_gate_norm.mean(),
            'skeleton_gate_norm_square': skeleton_gate_norm.square().mean(),
            'skeleton_gate': skeleton_gate.mean(),
            'skeleton_correction': skeleton_correction.abs().mean(),
            '_dual_motion_reliability_values': motion_reliability.squeeze(dim=2),
            '_dual_motion_usefulness_values': motion_usefulness.squeeze(dim=2),
            '_dual_motion_gate_values': motion_gate_norm.squeeze(dim=2),
            '_dual_skeleton_reliability_values': skeleton_reliability.squeeze(dim=2),
            '_dual_skeleton_usefulness_values': skeleton_usefulness.squeeze(dim=2),
            '_dual_skeleton_gate_values': skeleton_gate_norm.squeeze(dim=2)
        })
        if self.dual_ru_v11:
            other_info.update({
                'motion_action_prior': motion_usefulness.mean(),
                'skeleton_action_prior': skeleton_usefulness.mean()
            })
        if self.dual_ru_version >= 2:
            other_info.update({
                'motion_reliability_raw': motion_quality_pred.mean(),
                'skeleton_reliability_raw': skeleton_quality_pred.mean(),
                'motion_reliability_reference':
                    self.dual_motion_clean_ref.detach(),
                'skeleton_reliability_reference':
                    self.dual_skeleton_clean_ref.detach()
            })
            if (self.dual_ru_v7 or self.dual_ru_v8 or self.dual_ru_v9 or
                    self.dual_ru_v10 or self.dual_ru_v11):
                other_info.update({
                    'motion_reliability_zero_reference':
                        self.dual_motion_zero_ref.detach(),
                    'skeleton_reliability_zero_reference':
                        self.dual_skeleton_zero_ref.detach()
                })
        if motion_usefulness_anchor is not None:
            other_info.update({
                '_dual_motion_usefulness_anchor_values':
                    motion_usefulness_anchor.squeeze(dim=2),
                '_dual_skeleton_usefulness_anchor_values':
                    skeleton_usefulness_anchor.squeeze(dim=2)
            })
        if ((self.dual_ru_v5 or self.dual_ru_v6 or self.dual_ru_v7 or
             self.dual_ru_v8 or self.dual_ru_v9 or self.dual_ru_v10 or
             self.dual_ru_v11) and
                not self.training and force_gates is None and
                getattr(self, 'dual_ru_diagnostics', True)):
            zero_motion_pred = motion_zero_pred
            zero_skeleton_pred = skeleton_zero_pred
            if zero_motion_pred is None:
                zero_motion_pred = self.motion_quality_head(
                    self.motion_quality_features(torch.zeros_like(motion_source))
                )
                zero_skeleton_pred = self.skeleton_quality_head(
                    self.skeleton_quality_features(torch.zeros_like(skeleton))
                )
            other_info.update({
                'motion_reliability_zero': (
                    self._dual_ru_calibrated_reliability(
                        zero_motion_pred, 'motion', zero_motion_pred
                    ).mean()
                ),
                'skeleton_reliability_zero': (
                    self._dual_ru_calibrated_reliability(
                        zero_skeleton_pred, 'skeleton', zero_skeleton_pred
                    ).mean()
                )
            })
            if (self.dual_ru_v7 or self.dual_ru_v8 or self.dual_ru_v9 or
                    self.dual_ru_v10 or self.dual_ru_v11):
                for suffix, severity in (
                        ('25', 0.25), ('50', 0.50), ('75', 0.75)):
                    corrupt_motion = self._dual_ru_deterministic_corruption(
                        motion_source, severity
                    )
                    corrupt_skeleton = self._dual_ru_deterministic_corruption(
                        skeleton, severity
                    )
                    corrupt_motion_pred = self.motion_quality_head(
                        self.motion_quality_features(corrupt_motion)
                    )
                    corrupt_skeleton_pred = self.skeleton_quality_head(
                        self.skeleton_quality_features(corrupt_skeleton)
                    )
                    other_info.update({
                        f'motion_reliability_corrupt{suffix}':
                            self._dual_ru_calibrated_reliability(
                                corrupt_motion_pred, 'motion',
                                zero_motion_pred
                            ).mean(),
                        f'skeleton_reliability_corrupt{suffix}':
                            self._dual_ru_calibrated_reliability(
                                corrupt_skeleton_pred, 'skeleton',
                                zero_skeleton_pred
                            ).mean()
                    })
        if not (self.dual_ru_v5 or self.dual_ru_v6 or self.dual_ru_v7 or
                self.dual_ru_v8 or self.dual_ru_v9 or self.dual_ru_v10 or
                self.dual_ru_v11):
            self._dual_ru_quality_loss(
                motion_source, skeleton, motion_quality_pred,
                skeleton_quality_pred, other_info
            )
        return score, other_info

    def build_dual_ru_counterfactual_targets(self, input_feats, label):
        if not self.dual_ru_needs_counterfactual():
            return {}
        device = label.device
        devices = [device.index] if device.type == 'cuda' else []
        was_training = self.training
        try:
            with torch.random.fork_rng(devices=devices, enabled=True):
                self.eval()
                cpu_state = torch.get_rng_state()
                cuda_state = (
                    torch.cuda.get_rng_state(device) if device.type == 'cuda'
                    else None
                )
                replay_cache = {}

                # Keep identical RNG state so only the forced cue changes.
                def replay(gates):
                    cache_key = tuple(float(value) for value in gates)
                    if cache_key in replay_cache:
                        return replay_cache[cache_key]
                    torch.set_rng_state(cpu_state)
                    if cuda_state is not None:
                        torch.cuda.set_rng_state(cuda_state, device)
                    result = self._forward_dual_ru(
                        input_feats, force_gates=gates
                    )[0].detach()
                    replay_cache[cache_key] = result
                    return result

                with torch.no_grad():
                    if self.dual_ru_v10 or self.dual_ru_v11:
                        if self.dual_ru_phase() == 1:
                            gate_grid = tuple(self.dual_ru_gate_grid)
                            joint_gate_pairs = tuple(
                                (motion_gate, skeleton_gate)
                                for motion_gate in gate_grid
                                for skeleton_gate in gate_grid
                            )
                            joint_grid_scores = torch.stack([
                                replay(gates) for gates in joint_gate_pairs
                            ], dim=0)
                        fixed_score = replay((1.0, 0.5))
                    elif self.dual_ru_v9:
                        gate_grid = tuple(self.dual_ru_gate_grid)
                        motion_grid_scores = torch.stack([
                            replay((gate, 0.5)) for gate in gate_grid
                        ], dim=0)
                        skeleton_grid_scores = torch.stack([
                            replay((1.0, gate)) for gate in gate_grid
                        ], dim=0)
                    elif self.dual_ru_v8:
                        joint_scores = {
                            'none': replay((0.5, 0.0)),
                            'motion': replay((1.0, 0.0)),
                            'skeleton': replay((0.5, 0.5)),
                            'both': replay((1.0, 0.5))
                        }
                    elif self.dual_ru_v7:
                        joint_scores = {
                            'none': replay((0.0, 0.0)),
                            'motion': replay((1.0, 0.0)),
                            'skeleton': replay((0.0, 0.5)),
                            'both': replay((1.0, 0.5))
                        }
                    elif self.dual_ru_v5 or self.dual_ru_v6:
                        gate_grid = tuple(self.dual_ru_gate_grid)
                        motion_grid_scores = torch.stack([
                            replay((gate, 0.5)) for gate in gate_grid
                        ], dim=0)
                        skeleton_grid_scores = torch.stack([
                            replay((1.0, gate)) for gate in gate_grid
                        ], dim=0)
                    elif self.dual_ru_version >= 3:
                        full_score = replay((1.0, 0.5))
                    if not (self.dual_ru_v5 or self.dual_ru_v6 or
                            self.dual_ru_v7 or self.dual_ru_v8 or
                            self.dual_ru_v9 or self.dual_ru_v10 or
                            self.dual_ru_v11):
                        motion_off_score = replay((0.0, 0.5))
                        skeleton_off_score = replay((1.0, 0.0))
        finally:
            self.train(was_training)
        if self.dual_ru_v10 or self.dual_ru_v11:
            counterfactual_info = {'_dual_fixed_score': fixed_score}
            if self.dual_ru_phase() == 1:
                counterfactual_info.update({
                    '_dual_joint_gate_pairs': label.new_tensor(
                        joint_gate_pairs
                    ),
                    '_dual_joint_grid_scores': joint_grid_scores
                })
            return counterfactual_info
        if self.dual_ru_v9:
            return {
                '_dual_gate_grid': label.new_tensor(self.dual_ru_gate_grid),
                '_dual_motion_grid_scores': motion_grid_scores,
                '_dual_skeleton_grid_scores': skeleton_grid_scores,
                '_dual_fixed_score': replay_cache[(1.0, 0.5)]
            }
        if self.dual_ru_v7 or self.dual_ru_v8:
            return {
                f'_dual_joint_{name}_score': score
                for name, score in joint_scores.items()
            }
        if self.dual_ru_v5 or self.dual_ru_v6:
            return {
                '_dual_gate_grid': label.new_tensor(self.dual_ru_gate_grid),
                '_dual_motion_grid_scores': motion_grid_scores,
                '_dual_skeleton_grid_scores': skeleton_grid_scores
            }
        counterfactual_info = {
            '_dual_motion_off_score': motion_off_score,
            '_dual_skeleton_off_score': skeleton_off_score
        }
        if self.dual_ru_version >= 3:
            counterfactual_info['_dual_full_score'] = full_score
        return counterfactual_info

    def forward(self, input_feats, dual_force_gates=None,
                dual_corruption=None):
        if self.dual_ru_gate:
            return self._forward_dual_ru(
                input_feats, dual_force_gates, dual_corruption
            )
        skeleton = None
        skeleton_temporal_mean = None
        if self.use_skeleton and "S" in input_feats:
            skeleton = rearrange(input_feats["S"], 'b t d -> b d t').contiguous()

        _, feat_r = self.model_r(input_feats)
        _, feat_a = self.model_a(input_feats)
        _, feat_f = self.model_f(input_feats)
        motion_source = feat_f['feats'][0] if self.motion_saliency_proxy else None
        motion_quality_target = None
        skeleton_quality_target = None
        motion_quality_pred = None
        skeleton_quality_pred = None
        motion_quality_corrupt_pred = None
        skeleton_quality_corrupt_pred = None
        motion_quality_zero_pred = None
        skeleton_quality_zero_pred = None
        if self.cue_quality_supervision:
            if motion_source is not None:
                motion_quality_pred = self.motion_quality_head(
                    self.motion_quality_features(motion_source)
                )
                if self.training:
                    corrupted_motion, motion_quality_target = self.corrupt_motion_cue(
                        motion_source
                    )
                    motion_quality_corrupt_pred = self.motion_quality_head(
                        self.motion_quality_features(corrupted_motion)
                    )
                    if not self.cue_quality_paired:
                        motion_source = corrupted_motion
                        motion_quality_pred = motion_quality_corrupt_pred
            if skeleton is not None:
                skeleton_quality_pred = self.skeleton_quality_head(
                    self.skeleton_quality_features(skeleton)
                )
                if self.training:
                    corrupted_skeleton, skeleton_quality_target = self.corrupt_skeleton_cue(
                        skeleton
                    )
                    skeleton_quality_corrupt_pred = self.skeleton_quality_head(
                        self.skeleton_quality_features(corrupted_skeleton)
                    )
                    if not self.cue_quality_paired:
                        skeleton = corrupted_skeleton
                        skeleton_quality_pred = skeleton_quality_corrupt_pred
            if self.training and self.cue_quality_zero_anchor:
                if motion_source is not None:
                    motion_quality_zero_pred = self.motion_quality_head(
                        self.motion_quality_features(torch.zeros_like(motion_source))
                    )
                if skeleton is not None:
                    skeleton_quality_zero_pred = self.skeleton_quality_head(
                        self.skeleton_quality_features(torch.zeros_like(skeleton))
                    )
        motion_gate_quality = motion_quality_pred
        skeleton_gate_quality = skeleton_quality_pred
        motion_quality_reference = None
        skeleton_quality_reference = None
        motion_quality_gate_stats = None
        skeleton_quality_gate_stats = None
        if self.cue_quality_normalize_gate:
            if motion_gate_quality is not None:
                motion_gate_quality, motion_quality_reference = (
                    self.normalize_clean_quality_gate(motion_gate_quality, 'motion')
                )
            if skeleton_gate_quality is not None:
                skeleton_gate_quality, skeleton_quality_reference = (
                    self.normalize_clean_quality_gate(skeleton_gate_quality, 'skeleton')
                )
        if self.cue_quality_aux_only:
            if motion_gate_quality is not None:
                motion_gate_quality = torch.ones_like(motion_gate_quality)
            if skeleton_gate_quality is not None:
                skeleton_gate_quality = torch.ones_like(skeleton_gate_quality)
        if self.cue_usefulness_supervision and motion_gate_quality is not None:
            motion_gate_quality = torch.ones_like(motion_gate_quality)
        if motion_gate_quality is not None:
            motion_quality_gate_stats = (
                motion_gate_quality.detach().std(unbiased=False),
                motion_gate_quality.detach().min(),
                motion_gate_quality.detach().max()
            )
        if skeleton_gate_quality is not None:
            skeleton_quality_gate_stats = (
                skeleton_gate_quality.detach().std(unbiased=False),
                skeleton_gate_quality.detach().min(),
                skeleton_gate_quality.detach().max()
            )
        if self.cue_quality_detach_gate:
            if motion_gate_quality is not None:
                motion_gate_quality = motion_gate_quality.detach()
            if skeleton_gate_quality is not None:
                skeleton_gate_quality = skeleton_gate_quality.detach()
                
        feats = [[self.drop(_.detach()) for _ in feat_r['feats']],
                 [self.drop(_.detach()) for _ in feat_f['feats']],
                 [self.drop(_) for _ in feat_a['feats']]]

        x = self.c.repeat([feats[0][0].shape[0], feats[0][0].shape[2]//2, 1]).permute([0, 2, 1])
        gate_terms = []

        x1, action1 = self.stage1(x, [_[0] for _ in feats], [_[1] for _ in feats], self.tau)
        auto_motion_gate = None
        auto_motion_gate_norm = None
        auto_motion_gate_logs = []
        auto_motion_gate_norm_logs = []
        motion_reliability_logs = []
        motion_reliability_factor_logs = []
        motion_correction_logs = []
        if motion_source is not None and self.auto_gate and self.auto_gate_v2:
            if self.cue_quality_supervision:
                auto_motion_gate_norm = motion_gate_quality
                auto_motion_gate_effect = None
            else:
                auto_motion_gate_delta = torch.tanh(self.auto_motion_stage_gate_head(x1))
                auto_motion_gate_effect = self.auto_gate_delta_effect(auto_motion_gate_delta)
                auto_motion_gate_norm = 1 + self.auto_gate_motion_gamma * auto_motion_gate_effect
            auto_motion_gate = self.auto_gate_motion_max * auto_motion_gate_norm
            if self.cue_reliability:
                motion_rel = self.motion_reliability(motion_source, x1.shape[2])
                motion_rel_factor = self.cue_reliability_factor(motion_rel, 'motion')
                auto_motion_gate = auto_motion_gate * motion_rel_factor
                motion_reliability_logs.append(motion_rel.mean())
                motion_reliability_factor_logs.append(motion_rel_factor.mean())
            if self.cue_reliability_mode == 'calibrated' or self.cue_quality_supervision:
                saliency_delta = self.motion_saliency_delta(motion_source, x1.shape[2])
                motion_correction = auto_motion_gate * saliency_delta
                x1 = x1 + motion_correction * x1
                gate_terms.append(motion_correction.abs().mean(dim=(1, 2)))
                motion_correction_logs.append(motion_correction.abs().mean())
            else:
                x1 = self.apply_motion_saliency_residual(x1, motion_source, auto_motion_gate)
                gate_terms.append(auto_motion_gate_effect.abs().mean(dim=(1, 2)))
            auto_motion_gate_logs.append(auto_motion_gate.mean())
            auto_motion_gate_norm_logs.append(auto_motion_gate_norm.mean())
        elif motion_source is not None and not self.auto_gate and self.motion_saliency_mode == 'stage_residual':
            x1 = self.apply_motion_saliency_residual(x1, motion_source)
        x1 = self.pool(x1)

        x2, action2 = self.stage2(x1, [_[1] for _ in feats], [_[2] for _ in feats], self.tau)
        if motion_source is not None and self.auto_gate and self.auto_gate_v2:
            if self.cue_quality_supervision:
                auto_motion_gate_norm = motion_gate_quality
                auto_motion_gate_effect = None
            else:
                auto_motion_gate_delta = torch.tanh(self.auto_motion_stage_gate_head(x2))
                auto_motion_gate_effect = self.auto_gate_delta_effect(auto_motion_gate_delta)
                auto_motion_gate_norm = 1 + self.auto_gate_motion_gamma * auto_motion_gate_effect
            auto_motion_gate = self.auto_gate_motion_max * auto_motion_gate_norm
            if self.cue_reliability:
                motion_rel = self.motion_reliability(motion_source, x2.shape[2])
                motion_rel_factor = self.cue_reliability_factor(motion_rel, 'motion')
                auto_motion_gate = auto_motion_gate * motion_rel_factor
                motion_reliability_logs.append(motion_rel.mean())
                motion_reliability_factor_logs.append(motion_rel_factor.mean())
            if self.cue_reliability_mode == 'calibrated' or self.cue_quality_supervision:
                saliency_delta = self.motion_saliency_delta(motion_source, x2.shape[2])
                motion_correction = auto_motion_gate * saliency_delta
                x2 = x2 + motion_correction * x2
                gate_terms.append(motion_correction.abs().mean(dim=(1, 2)))
                motion_correction_logs.append(motion_correction.abs().mean())
            else:
                x2 = self.apply_motion_saliency_residual(x2, motion_source, auto_motion_gate)
                gate_terms.append(auto_motion_gate_effect.abs().mean(dim=(1, 2)))
            auto_motion_gate_logs.append(auto_motion_gate.mean())
            auto_motion_gate_norm_logs.append(auto_motion_gate_norm.mean())
        elif motion_source is not None and not self.auto_gate and self.motion_saliency_mode == 'stage_residual':
            x2 = self.apply_motion_saliency_residual(x2, motion_source)
        if self.use_skeleton and self.skeleton_temporal_weight and skeleton is not None:
            temporal_weight = torch.sigmoid(self.skeleton_temporal_head(skeleton)) - 0.5
            temporal_weight = F.interpolate(temporal_weight, size=x2.shape[2], mode='linear', align_corners=False)
            x2 = x2 * (1 + self.skeleton_temporal_beta * temporal_weight)
            skeleton_temporal_mean = temporal_weight.mean()
        motion_context = None
        if motion_source is not None and self.auto_gate and not self.auto_gate_v2:
            motion_context = self.motion_saliency_pool(x2, motion_source)
            x2 = self.gap(x2)
        elif motion_source is not None and self.motion_saliency_mode == 'pool':
            x2 = self.motion_saliency_pool(x2, motion_source)
        else:
            x2 = self.gap(x2)

        x3, action3 = self.stage3(x2, [_[2] for _ in feats], [_[3] for _ in feats], self.tau)

        score = self.fc(x3).squeeze(dim=2)
        other_info = {'action': [action1, action2, action3]}
        if motion_quality_pred is not None:
            other_info["motion_quality_pred"] = motion_quality_pred.mean()
        if skeleton_quality_pred is not None:
            other_info["skeleton_quality_pred"] = skeleton_quality_pred.mean()
        if motion_quality_reference is not None:
            other_info["motion_quality_reference"] = motion_quality_reference
        if skeleton_quality_reference is not None:
            other_info["skeleton_quality_reference"] = skeleton_quality_reference
        if motion_quality_gate_stats is not None:
            other_info["motion_quality_gate_std"] = motion_quality_gate_stats[0]
            other_info["motion_quality_gate_min"] = motion_quality_gate_stats[1]
            other_info["motion_quality_gate_max"] = motion_quality_gate_stats[2]
        if skeleton_quality_gate_stats is not None:
            other_info["skeleton_quality_gate_std"] = skeleton_quality_gate_stats[0]
            other_info["skeleton_quality_gate_min"] = skeleton_quality_gate_stats[1]
            other_info["skeleton_quality_gate_max"] = skeleton_quality_gate_stats[2]
        if self.auto_gate and motion_context is not None:
            motion_residual = self.motion_score_residual_head(motion_context).squeeze(dim=2)
            motion_gate_norm = torch.sigmoid(self.auto_motion_gate_head(torch.cat([x3, motion_context], dim=1)))
            motion_gate = self.auto_gate_motion_max * motion_gate_norm.squeeze(dim=2)
            score = score + motion_gate * motion_residual
            other_info["auto_motion_gate"] = motion_gate.mean()
            other_info["auto_motion_gate_norm"] = motion_gate_norm.mean()
            other_info["motion_residual"] = motion_residual.mean()
            gate_terms.append(motion_gate_norm.squeeze(dim=2).mean(dim=1))
        elif auto_motion_gate_logs:
            other_info["auto_motion_gate"] = torch.stack(auto_motion_gate_logs).mean()
            other_info["auto_motion_gate_norm"] = torch.stack(auto_motion_gate_norm_logs).mean()
            if motion_reliability_logs:
                other_info["motion_reliability"] = torch.stack(motion_reliability_logs).mean()
            if motion_reliability_factor_logs:
                other_info["motion_reliability_factor"] = torch.stack(motion_reliability_factor_logs).mean()
            if motion_correction_logs:
                other_info["motion_correction"] = torch.stack(motion_correction_logs).mean()
        if skeleton is not None:
            skeleton_rel = None
            skeleton_rel_factor = None
            skeleton_usefulness_gate_pred = None
            skeleton_score_before_cue = None
            if self.cue_reliability:
                skeleton_rel = self.skeleton_reliability(skeleton)
                skeleton_rel_factor = self.cue_reliability_factor(skeleton_rel, 'skeleton')
                other_info["skeleton_reliability"] = skeleton_rel.mean()
                other_info["skeleton_reliability_factor"] = skeleton_rel_factor.mean()
            skeleton_feat = self.skeleton_encoder(skeleton)
            skeleton_residual = self.skeleton_score_residual_head(skeleton_feat).squeeze(dim=2)
            if self.auto_gate:
                usefulness_active = (
                    self.cue_usefulness_supervision and
                    self.cue_current_epoch >= self.cue_usefulness_warmup_epochs
                )
                if usefulness_active:
                    skeleton_usefulness_gate_pred = torch.sigmoid(
                        self.skeleton_usefulness_gate_head(
                            torch.cat([x3.detach(), skeleton_feat.detach()], dim=1)
                        )
                    )
                    skeleton_usefulness_gate_value = skeleton_usefulness_gate_pred
                    if not self.cue_usefulness_end_to_end:
                        skeleton_usefulness_gate_value = skeleton_usefulness_gate_value.detach()
                    if self.cue_usefulness_downscale_only:
                        skeleton_gate_norm = (
                            1 - self.cue_usefulness_span *
                            (1 - skeleton_usefulness_gate_value)
                        )
                    else:
                        skeleton_gate_norm = (
                            1 + self.cue_usefulness_span *
                            (2 * skeleton_usefulness_gate_value - 1)
                        )
                    if (self.cue_usefulness_quality_prior and
                            skeleton_gate_quality is not None):
                        skeleton_gate_norm = skeleton_gate_norm * skeleton_gate_quality
                    skeleton_gate_effect = None
                elif self.cue_quality_supervision:
                    skeleton_gate_norm = skeleton_gate_quality
                    skeleton_gate_effect = None
                else:
                    skeleton_gate_delta = torch.tanh(
                        self.auto_skeleton_gate_head(torch.cat([x3, skeleton_feat], dim=1))
                    )
                    skeleton_gate_effect = self.auto_gate_delta_effect(skeleton_gate_delta)
                    skeleton_gate_norm = 1 + self.auto_gate_skeleton_gamma * skeleton_gate_effect
                skeleton_gate = self.auto_gate_skeleton_max * skeleton_gate_norm.squeeze(dim=2)
                if skeleton_rel_factor is not None:
                    skeleton_gate = skeleton_gate * skeleton_rel_factor.squeeze(dim=2)
                other_info["auto_skeleton_gate_norm"] = skeleton_gate_norm.mean()
                if self.cue_usefulness_supervision:
                    other_info["skeleton_usefulness_active"] = skeleton_gate_norm.new_tensor(
                        float(usefulness_active)
                    )
                if skeleton_usefulness_gate_pred is not None:
                    other_info["skeleton_usefulness_gate"] = skeleton_usefulness_gate_pred.mean()
                    other_info["skeleton_usefulness_gate_square"] = (
                        skeleton_usefulness_gate_pred.square().mean()
                    )
                    if self.cue_usefulness_end_to_end:
                        other_info["cue_usefulness_prior_loss"] = (
                            self.cue_usefulness_prior_weight *
                            (1 - skeleton_usefulness_gate_pred).square().mean()
                        )
                if (self.cue_reliability_mode != 'calibrated' and
                        not self.cue_quality_supervision and
                        not self.cue_usefulness_supervision):
                    gate_terms.append(skeleton_gate_effect.squeeze(dim=2).abs().mean(dim=1))
            elif self.skeleton_alpha >= 0:
                skeleton_gate = skeleton_residual.new_tensor(self.skeleton_alpha)
            else:
                skeleton_gate = self.skeleton_gate_max * torch.sigmoid(self.skeleton_gate)
                if skeleton_rel_factor is not None:
                    skeleton_gate = skeleton_gate * skeleton_rel_factor.squeeze(dim=2)
            if self.skeleton_confidence_gate and "SQ" in input_feats:
                skeleton_quality = input_feats["SQ"].view(-1, 1).clamp(0, 1)
                skeleton_gate = skeleton_gate * skeleton_quality
                other_info["skeleton_quality"] = skeleton_quality.mean()
            skeleton_correction = skeleton_gate * skeleton_residual
            if skeleton_usefulness_gate_pred is not None:
                skeleton_score_before_cue = score.detach()
            score = score + skeleton_correction
            other_info["skeleton_gate"] = skeleton_gate
            other_info["skeleton_correction"] = skeleton_correction.abs().mean()
            if skeleton_usefulness_gate_pred is not None:
                other_info["_skeleton_usefulness_gate_pred"] = (
                    skeleton_usefulness_gate_pred.squeeze(dim=2)
                )
                other_info["_skeleton_score_before_cue"] = skeleton_score_before_cue
            if (self.auto_gate and
                    (self.cue_reliability_mode == 'calibrated' or
                     self.cue_quality_supervision)):
                gate_terms.append(skeleton_correction.abs().mean(dim=1))
            if skeleton_temporal_mean is not None:
                other_info["skeleton_temporal_mean"] = skeleton_temporal_mean
            if self.skeleton_late_fusion:
                skeleton_late_feat = self.skeleton_late_proj(skeleton_feat)
                late_gate = self.skeleton_late_gate(torch.cat([x3, skeleton_late_feat], dim=1))
                late_residual = self.skeleton_late_head(x3 + late_gate * skeleton_late_feat).squeeze(dim=2)
                score = score + self.skeleton_late_gamma * late_residual
                other_info["skeleton_late_gate"] = late_gate.mean()
        if self.auto_gate and gate_terms:
            gate_stack = torch.stack(gate_terms, dim=0)
            gate_loss = score.new_tensor(0.0)
            if self.auto_gate_l1 > 0:
                gate_loss = gate_loss + self.auto_gate_l1 * gate_stack.mean()
            if self.auto_gate_budget >= 0 and self.auto_gate_budget_weight > 0:
                gate_loss = gate_loss + self.auto_gate_budget_weight * F.relu(
                    gate_stack.sum(dim=0) - self.auto_gate_budget
                ).mean()
            other_info["auto_gate_loss"] = gate_loss
        if self.training and self.cue_quality_supervision:
            quality_losses = []
            zero_losses = []
            pair_losses = []
            clean_predictions = []
            corrupt_predictions = []
            quality_predictions = []
            quality_targets = []
            if motion_quality_target is not None:
                supervised_pred = motion_quality_corrupt_pred
                if self.cue_quality_paired:
                    quality_losses.append(F.mse_loss(
                        motion_quality_pred, torch.ones_like(motion_quality_pred)
                    ))
                    pair_losses.append(F.mse_loss(
                        motion_quality_pred - supervised_pred,
                        1 - motion_quality_target
                    ))
                    clean_predictions.append(motion_quality_pred.mean())
                    corrupt_predictions.append(supervised_pred.mean())
                quality_losses.append(F.mse_loss(supervised_pred, motion_quality_target))
                if motion_quality_zero_pred is not None:
                    zero_losses.append(F.mse_loss(
                        motion_quality_zero_pred,
                        torch.zeros_like(motion_quality_zero_pred)
                    ))
                quality_predictions.append(supervised_pred.reshape(-1))
                quality_targets.append(motion_quality_target.reshape(-1))
            if skeleton_quality_target is not None:
                supervised_pred = skeleton_quality_corrupt_pred
                if self.cue_quality_paired:
                    quality_losses.append(F.mse_loss(
                        skeleton_quality_pred, torch.ones_like(skeleton_quality_pred)
                    ))
                    pair_losses.append(F.mse_loss(
                        skeleton_quality_pred - supervised_pred,
                        1 - skeleton_quality_target
                    ))
                    clean_predictions.append(skeleton_quality_pred.mean())
                    corrupt_predictions.append(supervised_pred.mean())
                quality_losses.append(F.mse_loss(supervised_pred, skeleton_quality_target))
                if skeleton_quality_zero_pred is not None:
                    zero_losses.append(F.mse_loss(
                        skeleton_quality_zero_pred,
                        torch.zeros_like(skeleton_quality_zero_pred)
                    ))
                quality_predictions.append(supervised_pred.reshape(-1))
                quality_targets.append(skeleton_quality_target.reshape(-1))
            if quality_losses:
                quality_loss_sum = torch.stack(quality_losses).sum()
                quality_loss_denominator = float(len(quality_losses))
                if zero_losses:
                    quality_loss_sum = quality_loss_sum + (
                        self.cue_quality_zero_weight *
                        torch.stack(zero_losses).sum()
                    )
                    quality_loss_denominator += (
                        self.cue_quality_zero_weight * len(zero_losses)
                    )
                    zero_mean = torch.stack([
                        prediction.mean() for prediction in (
                            motion_quality_zero_pred,
                            skeleton_quality_zero_pred
                        ) if prediction is not None
                    ]).mean()
                    other_info['cue_quality_zero_pred'] = zero_mean
                quality_loss = quality_loss_sum / quality_loss_denominator
                if pair_losses:
                    quality_loss = quality_loss + self.cue_quality_pair_weight * torch.stack(
                        pair_losses
                    ).mean()
                other_info["cue_quality_loss"] = self.cue_quality_loss_weight * quality_loss
                predictions = torch.cat(quality_predictions)
                targets = torch.cat(quality_targets)
                other_info["cue_quality_target"] = targets.mean()
                other_info["cue_quality_mae"] = (predictions - targets).abs().mean()
                pred_centered = predictions - predictions.mean()
                target_centered = targets - targets.mean()
                denominator = (
                    pred_centered.square().sum() * target_centered.square().sum()
                ).sqrt().clamp_min(1e-6)
                other_info["cue_quality_corr"] = (
                    pred_centered * target_centered
                ).sum() / denominator
                if clean_predictions:
                    clean_mean = torch.stack(clean_predictions).mean()
                    corrupt_mean = torch.stack(corrupt_predictions).mean()
                    other_info["cue_quality_clean_pred"] = clean_mean
                    other_info["cue_quality_corrupt_pred"] = corrupt_mean
                    other_info["cue_quality_gap"] = clean_mean - corrupt_mean
                    other_info["cue_quality_target_gap"] = 1 - targets.mean()
        return score, other_info

    def _dual_ru_v10_usefulness_loss(self, label, kwargs):
        label_value = label.reshape(-1)
        gate_pairs = kwargs['_dual_joint_gate_pairs'].to(
            device=label_value.device, dtype=label_value.dtype
        ).reshape(-1, 2)
        scores = kwargs['_dual_joint_grid_scores'].reshape(
            gate_pairs.shape[0], -1
        ).detach()
        default_pair = gate_pairs.new_tensor((1.0, 0.5))
        default_distance = (gate_pairs - default_pair).abs().sum(dim=1)
        default_index = int(default_distance.argmin().item())
        if default_distance[default_index].item() > 1e-6:
            raise ValueError('dual RU v10 grid must include gates (1.0, 0.5)')

        mse_errors = (scores - label_value.unsqueeze(0)).square()
        rank_errors = torch.stack([
            self._dual_ru_pairwise_rank_error(score, label_value)
            for score in scores
        ], dim=0)
        batch_mse = mse_errors.mean(dim=1)
        batch_rank = rank_errors.mean(dim=1)
        mse_scale = batch_mse[default_index].clamp_min(
            self.dual_ru_error_scale_min
        )
        rank_scale = batch_rank[default_index].clamp_min(1e-3)
        rank_weight = self.dual_ru_joint_rank_weight
        batch_objective = (
            (1 - rank_weight) * batch_mse / mse_scale +
            rank_weight * batch_rank / rank_scale
        )
        global_weights = torch.softmax(
            -(batch_objective - batch_objective.min()) /
            self.dual_ru_temperature,
            dim=0
        )
        global_soft_pair = (
            global_weights.unsqueeze(1) * gate_pairs
        ).sum(dim=0)
        best_index = int(batch_objective.argmin().item())
        global_best_pair = gate_pairs[best_index]
        global_gain = (
            batch_objective[default_index] - batch_objective[best_index]
        ).clamp_min(0)
        global_relative_gain = global_gain / batch_objective[
            default_index
        ].abs().clamp_min(1e-6)
        global_confidence = (
            global_relative_gain / self.dual_ru_joint_min_gain
        ).clamp(0, 1)
        global_target = default_pair + global_confidence * (
            global_soft_pair - default_pair
        )

        if self.dual_ru_v11:
            sample_confidence = label_value.new_zeros(label_value.shape)
            target_pairs = global_target.unsqueeze(0).expand(
                label_value.shape[0], -1
            ).detach()
            target_weights = label_value.new_ones(label_value.shape)
        else:
            sample_mse_scale = (
                mse_errors[default_index] + 0.1 * mse_scale
            ).clamp_min(self.dual_ru_error_scale_min)
            sample_rank_scale = (
                rank_errors[default_index] + 0.1 * rank_scale
            ).clamp_min(1e-3)
            sample_objective = (
                (1 - rank_weight) * mse_errors /
                sample_mse_scale.unsqueeze(0) +
                rank_weight * rank_errors / sample_rank_scale.unsqueeze(0)
            )
            sample_weights = torch.softmax(
                -(sample_objective - sample_objective.min(
                    dim=0, keepdim=True
                ).values) / self.dual_ru_temperature,
                dim=0
            )
            sample_soft_pairs = torch.einsum(
                'kb,kc->bc', sample_weights, gate_pairs
            )
            sample_best = sample_objective.min(dim=0).values
            sample_gain = (
                sample_objective[default_index] - sample_best
            ).clamp_min(0)
            sample_relative_gain = sample_gain / sample_objective[
                default_index
            ].abs().clamp_min(1e-6)
            sample_confidence = (
                sample_relative_gain / self.dual_ru_joint_min_gain
            ).clamp(0, 1)
            correction_confidence = sample_confidence * (
                0.25 + 0.75 * global_confidence
            )
            target_pairs = global_target.unsqueeze(0) + (
                self.dual_ru_joint_sample_weight *
                correction_confidence.unsqueeze(1) *
                (sample_soft_pairs - global_target.unsqueeze(0))
            )
            target_pairs = target_pairs.clamp(0, 1).detach()
            target_weights = (
                0.5 + 0.5 * correction_confidence
            ).detach()
        target_valid = torch.ones_like(target_weights, dtype=torch.bool)

        sorted_objective = batch_objective.sort().values
        grid_margin = sorted_objective[1] - sorted_objective[0]
        objective_gain = batch_objective[default_index] - batch_objective[
            best_index
        ]
        loss_info = {
            'dual_joint_global_motion_target': global_target[0],
            'dual_joint_global_skeleton_target': global_target[1],
            'dual_joint_soft_motion_gate': global_soft_pair[0],
            'dual_joint_soft_skeleton_gate': global_soft_pair[1],
            'dual_joint_best_motion_gate': global_best_pair[0],
            'dual_joint_best_skeleton_gate': global_best_pair[1],
            'dual_joint_global_gain': global_gain,
            'dual_joint_global_relative_gain': global_relative_gain,
            'dual_joint_global_confidence': global_confidence,
            'dual_joint_sample_confidence': sample_confidence.mean(),
            'dual_joint_sample_correction': (
                target_pairs - global_target.unsqueeze(0)
            ).abs().mean(),
            'dual_joint_default_mse': batch_mse[default_index],
            'dual_joint_best_mse': batch_mse[best_index],
            'dual_joint_default_rank': batch_rank[default_index],
            'dual_joint_best_rank': batch_rank[best_index]
        }
        cue_losses = []
        grid = torch.unique(gate_pairs[:, 0])
        for cue_index, cue in enumerate(('motion', 'skeleton')):
            prediction = kwargs[
                f'_dual_{cue}_usefulness_values'
            ].reshape(-1)
            target = target_pairs[:, cue_index]
            cue_loss = (
                (prediction - target).square() * target_weights
            ).sum() / target_weights.sum().clamp_min(1e-6)
            cue_losses.append(cue_loss)

            other_index = 1 - cue_index
            context_effects = []
            for other_gate in grid:
                context_mask = (
                    gate_pairs[:, other_index] - other_gate
                ).abs() < 1e-6
                context_scores = scores[context_mask]
                context_effects.append(
                    context_scores.max(dim=0).values -
                    context_scores.min(dim=0).values
                )
            score_effect = torch.stack(context_effects).mean()
            default_gate = default_pair[cue_index]
            target_mean = target.mean()
            target_std = target.std(unbiased=False)
            help_mask = target > default_gate + 1e-6
            harm_mask = target < default_gate - 1e-6
            neutral_mask = ~(help_mask | harm_mask)
            loss_info.update({
                f'{cue}_usefulness_target': target_mean,
                f'{cue}_usefulness_target_std': target_std,
                f'{cue}_usefulness_loss': cue_loss.detach(),
                f'{cue}_usefulness_delta': objective_gain,
                f'{cue}_usefulness_mse_delta': (
                    batch_mse[default_index] - batch_mse[best_index]
                ),
                f'{cue}_usefulness_mse_benefit': (
                    batch_mse[default_index] - batch_mse[best_index]
                ),
                f'{cue}_usefulness_score_effect': score_effect,
                f'{cue}_usefulness_effect_scale': (
                    scores.new_tensor(self.dual_ru_joint_min_gain)
                ),
                f'{cue}_usefulness_effect_valid_rate': (
                    global_confidence > 0
                ).float(),
                f'{cue}_usefulness_directional_utility': (
                    target - default_gate
                ).mean(),
                f'{cue}_usefulness_rank_delta': (
                    batch_rank[default_index] - batch_rank[best_index]
                ),
                f'{cue}_usefulness_help_rate': help_mask.float().mean(),
                f'{cue}_usefulness_harm_rate': harm_mask.float().mean(),
                f'{cue}_usefulness_neutral_rate': neutral_mask.float().mean(),
                f'{cue}_usefulness_mask_rate': (
                    global_confidence > 0
                ).float(),
                f'{cue}_usefulness_confidence': target_weights.mean(),
                f'{cue}_usefulness_objective_gain': objective_gain,
                f'{cue}_usefulness_grid_margin': grid_margin,
                f'{cue}_usefulness_optimal_gate': target_mean,
                f'{cue}_usefulness_joint_utility': (
                    target - default_gate
                ).mean(),
                f'{cue}_usefulness_default_gate': default_gate,
                f'{cue}_usefulness_target_corr': (
                    self._dual_ru_masked_correlation(
                        prediction.detach(), target, target_valid
                    )
                ),
                f'_dual_{cue}_usefulness_prediction_values':
                    prediction.detach(),
                f'_dual_{cue}_usefulness_target_values': target,
                f'_dual_{cue}_usefulness_target_weights': target_weights,
                f'_dual_{cue}_usefulness_target_valid_values': target_valid
            })

        combined_loss = torch.stack(cue_losses).mean()
        loss_info['dual_usefulness_loss'] = combined_loss.detach()
        return combined_loss, loss_info

    def _dual_ru_v10_no_regret_loss(self, prediction, label, fixed_score):
        prediction = prediction.reshape(-1)
        label_value = label.reshape(-1)
        fixed_score = fixed_score.reshape(-1).detach()
        adaptive_mse = (prediction - label_value).square().mean()
        fixed_mse = (fixed_score - label_value).square().mean()
        mse_delta = adaptive_mse - fixed_mse
        mse_regret = F.relu(mse_delta)
        adaptive_rank = self._dual_ru_pairwise_rank_error(
            prediction, label_value
        ).mean()
        fixed_rank = self._dual_ru_pairwise_rank_error(
            fixed_score, label_value
        ).mean().detach()
        rank_delta = adaptive_rank - fixed_rank
        rank_regret = F.relu(rank_delta)
        no_regret_loss = (
            mse_regret + self.dual_ru_batch_rank_regret_weight * rank_regret
        )
        active = ((mse_delta > 0) | (rank_delta > 0)).float()
        loss_info = {
            'dual_no_regret_loss': no_regret_loss.detach(),
            'dual_no_regret_mse': mse_regret.detach(),
            'dual_no_regret_rank': rank_regret.detach(),
            'dual_no_regret_active_rate': active.detach(),
            'dual_no_regret_mse_delta': mse_delta.detach(),
            'dual_no_regret_rank_delta': rank_delta.detach()
        }
        return no_regret_loss, loss_info

    def _dual_ru_v9_usefulness_loss(self, label, kwargs):
        label_value = label.reshape(-1)
        grid = kwargs['_dual_gate_grid'].to(label_value.device)
        default_gates = {'motion': 1.0, 'skeleton': 0.5}
        cue_losses = []
        loss_info = {}

        for cue in ('motion', 'skeleton'):
            prediction = kwargs[
                f'_dual_{cue}_usefulness_values'
            ].reshape(-1)
            scores = kwargs[
                f'_dual_{cue}_grid_scores'
            ].reshape(grid.numel(), -1).detach()
            mse_errors = (scores - label_value.unsqueeze(0)).square()
            rank_errors = torch.stack([
                self._dual_ru_pairwise_rank_error(score, label_value)
                for score in scores
            ], dim=0)

            batch_size = label_value.numel()
            grid_candidates = grid.unsqueeze(1).expand(-1, batch_size)
            score_delta = scores[1:] - scores[:-1]
            safe_delta = torch.where(
                score_delta.abs() > 1e-8,
                score_delta,
                torch.ones_like(score_delta)
            )
            fraction = (
                (label_value.unsqueeze(0) - scores[:-1]) / safe_delta
            ).clamp(0, 1)
            segment_gates = (
                grid[:-1].unsqueeze(1) + fraction *
                (grid[1:] - grid[:-1]).unsqueeze(1)
            )
            segment_scores = scores[:-1] + fraction * score_delta
            candidate_gates = torch.cat(
                [grid_candidates, segment_gates], dim=0
            )
            candidate_mse = torch.cat([
                mse_errors,
                (segment_scores - label_value.unsqueeze(0)).square()
            ], dim=0)
            best_index = candidate_mse.argmin(dim=0)
            target = candidate_gates.gather(
                0, best_index.unsqueeze(0)
            ).squeeze(0)
            best_mse = candidate_mse.gather(
                0, best_index.unsqueeze(0)
            ).squeeze(0)

            default_gate = default_gates[cue]
            default_index = int(
                torch.argmin((grid - default_gate).abs()).item()
            )
            default_mse = mse_errors[default_index]
            default_rank = rank_errors[default_index]
            discrete_best = mse_errors.argmin(dim=0)
            best_rank = rank_errors.gather(
                0, discrete_best.unsqueeze(0)
            ).squeeze(0)
            objective_gain = (default_mse - best_mse).clamp_min(0)

            score_range = scores.max(dim=0).values - scores.min(dim=0).values
            effect_threshold = (
                self.dual_ru_identifiability_abs +
                self.dual_ru_identifiability_error_ratio *
                (scores[default_index] - label_value).abs()
            )
            valid = score_range >= effect_threshold
            improves_default = objective_gain > 1e-12
            sorted_mse = mse_errors.sort(dim=0).values
            decision_margin = (sorted_mse[1] - sorted_mse[0]).clamp_min(0)
            effect_strength = (
                (score_range - effect_threshold).clamp_min(0) /
                score_range.clamp_min(self.dual_ru_identifiability_abs)
            ).clamp(0, 1)
            gain_strength = (
                objective_gain /
                (default_mse.abs() + self.dual_ru_identifiability_abs)
            )
            gain_strength = (
                gain_strength / max(self.dual_ru_temperature, 1e-6)
            ).clamp(0, 1)
            valid_confidence = 0.1 + 0.9 * (
                effect_strength * gain_strength
            ).sqrt()

            default = prediction.new_full(prediction.shape, default_gate)
            target = torch.where(improves_default, target, default)
            target = torch.where(valid, target, default).detach()
            confidence = torch.where(
                valid,
                valid_confidence,
                valid_confidence.new_full(
                    valid_confidence.shape,
                    self.dual_ru_unidentifiable_weight
                )
            ).detach()
            cue_loss = (
                (prediction - target).square() * confidence
            ).sum() / confidence.sum().clamp_min(1e-6)
            cue_losses.append(cue_loss)

            valid_weight = valid.to(target.dtype)
            valid_sum = valid_weight.sum()
            valid_count = valid_sum.clamp_min(1.0)
            valid_target_mean = (
                (target * valid_weight).sum() / valid_count
            )
            target_mean = torch.where(
                valid_sum > 0,
                valid_target_mean,
                target.new_tensor(default_gate)
            )
            target_std = (
                ((target - target_mean).square() * valid_weight).sum() /
                valid_count
            ).sqrt()
            rank_benefit = default_rank - best_rank
            help_mask = (target > default_gate + 1e-6) & valid
            harm_mask = (target < default_gate - 1e-6) & valid
            neutral_mask = valid & ~(help_mask | harm_mask)
            loss_info.update({
                f'{cue}_usefulness_target': target_mean,
                f'{cue}_usefulness_target_std': target_std,
                f'{cue}_usefulness_loss': cue_loss.detach(),
                f'{cue}_usefulness_delta': objective_gain.mean(),
                f'{cue}_usefulness_mse_delta': objective_gain.mean(),
                f'{cue}_usefulness_mse_benefit': objective_gain.mean(),
                f'{cue}_usefulness_score_effect': score_range.mean(),
                f'{cue}_usefulness_effect_scale': effect_threshold.mean(),
                f'{cue}_usefulness_effect_valid_rate': valid.float().mean(),
                f'{cue}_usefulness_directional_utility': (
                    ((target - default_gate) * valid_weight).sum() /
                    valid_count
                ),
                f'{cue}_usefulness_rank_delta': (
                    (rank_benefit * valid_weight).sum() / valid_count
                ),
                f'{cue}_usefulness_help_rate': (
                    help_mask.float().sum() / valid_count
                ),
                f'{cue}_usefulness_harm_rate': (
                    harm_mask.float().sum() / valid_count
                ),
                f'{cue}_usefulness_neutral_rate': (
                    neutral_mask.float().sum() / valid_count
                ),
                f'{cue}_usefulness_mask_rate': valid.float().mean(),
                f'{cue}_usefulness_confidence': confidence.mean(),
                f'{cue}_usefulness_objective_gain': (
                    (objective_gain * valid_weight).sum() / valid_count
                ),
                f'{cue}_usefulness_grid_margin': (
                    (decision_margin * valid_weight).sum() / valid_count
                ),
                f'{cue}_usefulness_optimal_gate': target_mean,
                f'{cue}_usefulness_joint_utility': (
                    target - default_gate
                ).mean(),
                f'{cue}_usefulness_default_gate': default.mean(),
                f'{cue}_usefulness_target_corr': (
                    self._dual_ru_masked_correlation(
                        prediction.detach(), target, valid
                    )
                ),
                f'_dual_{cue}_usefulness_prediction_values':
                    prediction.detach(),
                f'_dual_{cue}_usefulness_target_values': target,
                f'_dual_{cue}_usefulness_target_weights': confidence,
                f'_dual_{cue}_usefulness_target_valid_values': valid.detach()
            })

        combined_loss = torch.stack(cue_losses).mean()
        loss_info['dual_usefulness_loss'] = combined_loss.detach()
        return combined_loss, loss_info

    def _dual_ru_v8_usefulness_loss(self, label, kwargs):
        label_value = label.reshape(-1)
        state_names = ('none', 'motion', 'skeleton', 'both')
        scores = {
            name: kwargs[f'_dual_joint_{name}_score'].reshape(-1).detach()
            for name in state_names
        }
        mse_errors = {
            name: (score - label_value).square()
            for name, score in scores.items()
        }
        rank_errors = {
            name: self._dual_ru_pairwise_rank_error(score, label_value)
            for name, score in scores.items()
        }
        cue_pairs = {
            'motion': (('none', 'motion'), ('skeleton', 'both')),
            'skeleton': (('none', 'skeleton'), ('motion', 'both'))
        }
        default_gates = {'motion': 1.0, 'skeleton': 0.5}
        cue_losses = []
        mse_scales = []
        rank_scales = []
        loss_info = {}

        for cue, pairs in cue_pairs.items():
            prediction = kwargs[
                f'_dual_{cue}_usefulness_values'
            ].reshape(-1)
            mse_contexts = []
            rank_contexts = []
            score_effects = []
            for before, after in pairs:
                mse_contexts.append(
                    mse_errors[before] - mse_errors[after]
                )
                rank_contexts.append(
                    rank_errors[before] - rank_errors[after]
                )
                score_effects.append(
                    (scores[after] - scores[before]).abs()
                )

            mse_contexts = torch.stack(mse_contexts)
            rank_contexts = torch.stack(rank_contexts)
            mse_scale = self._dual_ru_benefit_scale(
                cue, 'mse', mse_contexts
            )
            rank_scale = self._dual_ru_benefit_scale(
                cue, 'rank', rank_contexts
            )
            mse_scales.append(mse_scale)
            rank_scales.append(rank_scale)
            normalized_mse = (mse_contexts / mse_scale).clamp(-4, 4)
            normalized_rank = (rank_contexts / rank_scale).clamp(-4, 4)
            utility_contexts = (
                (1 - self.dual_ru_rank_weight) * normalized_mse +
                self.dual_ru_rank_weight * normalized_rank
            )
            utility = utility_contexts.mean(dim=0)
            context_agreement = (
                (utility_contexts[0] >= 0) ==
                (utility_contexts[1] >= 0)
            ).to(utility.dtype)
            score_effect = torch.stack(score_effects).mean(dim=0)
            effect_threshold = (
                self.dual_ru_identifiability_abs +
                self.dual_ru_identifiability_error_ratio *
                (scores['both'] - label_value).abs()
            )
            valid = score_effect >= effect_threshold

            evidence = F.relu(utility.abs() - self.dual_ru_deadzone)
            strength = evidence / (
                evidence + self.dual_ru_temperature
            )
            default = prediction.new_full(
                prediction.shape, default_gates[cue]
            )
            positive_target = default + (1 - default) * strength
            negative_target = default * (1 - strength)
            target = torch.where(
                utility > self.dual_ru_deadzone,
                positive_target,
                torch.where(
                    utility < -self.dual_ru_deadzone,
                    negative_target,
                    default
                )
            )
            target = torch.where(valid, target, default)
            confidence = (
                0.1 + 0.9 * strength
            ) * (0.25 + 0.75 * context_agreement)
            confidence = torch.where(
                valid,
                confidence,
                confidence.new_full(
                    confidence.shape, self.dual_ru_unidentifiable_weight
                )
            )

            target = target.detach()
            confidence = confidence.detach()
            cue_loss = (
                (prediction - target).square() * confidence
            ).sum() / confidence.sum().clamp_min(1e-6)
            cue_losses.append(cue_loss)

            target_mean = target.mean()
            target_std = (target - target_mean).square().mean().sqrt()
            mse_benefit = mse_contexts.mean(dim=0)
            rank_benefit = rank_contexts.mean(dim=0)
            help_mask = (utility > self.dual_ru_deadzone) & valid
            harm_mask = (utility < -self.dual_ru_deadzone) & valid
            neutral_mask = ~(help_mask | harm_mask)
            loss_info.update({
                f'{cue}_usefulness_target': target_mean,
                f'{cue}_usefulness_target_std': target_std,
                f'{cue}_usefulness_loss': cue_loss.detach(),
                f'{cue}_usefulness_delta': F.relu(utility).mean(),
                f'{cue}_usefulness_mse_delta': mse_benefit.mean(),
                f'{cue}_usefulness_mse_benefit': mse_benefit.mean(),
                f'{cue}_usefulness_mse_scale': mse_scale,
                f'{cue}_usefulness_score_effect': score_effect.mean(),
                f'{cue}_usefulness_effect_scale': effect_threshold.mean(),
                f'{cue}_usefulness_effect_valid_rate': valid.float().mean(),
                f'{cue}_usefulness_directional_utility': utility.mean(),
                f'{cue}_usefulness_rank_delta': rank_benefit.mean(),
                f'{cue}_usefulness_rank_scale': rank_scale,
                f'{cue}_usefulness_help_rate': help_mask.float().mean(),
                f'{cue}_usefulness_harm_rate': harm_mask.float().mean(),
                f'{cue}_usefulness_neutral_rate': neutral_mask.float().mean(),
                f'{cue}_usefulness_mask_rate': valid.float().mean(),
                f'{cue}_usefulness_confidence': confidence.mean(),
                f'{cue}_usefulness_objective_gain': F.relu(utility).mean(),
                f'{cue}_usefulness_grid_margin': utility.abs().mean(),
                f'{cue}_usefulness_optimal_gate': target_mean,
                f'{cue}_usefulness_joint_utility': utility.mean(),
                f'{cue}_usefulness_context_agreement':
                    context_agreement.mean(),
                f'{cue}_usefulness_default_gate': default.mean(),
                f'{cue}_usefulness_target_corr': (
                    self._dual_ru_masked_correlation(
                        prediction.detach(), target, confidence
                    )
                ),
                f'_dual_{cue}_usefulness_prediction_values':
                    prediction.detach(),
                f'_dual_{cue}_usefulness_target_values': target,
                f'_dual_{cue}_usefulness_target_weights': confidence
            })

        combined_loss = torch.stack(cue_losses).mean()
        loss_info.update({
            'dual_usefulness_error_scale': torch.stack(mse_scales).mean(),
            'dual_usefulness_rank_scale': torch.stack(rank_scales).mean(),
            'dual_usefulness_loss': combined_loss.detach()
        })
        return combined_loss, loss_info

    def _dual_ru_v7_usefulness_loss(self, label, kwargs):
        label_value = label.reshape(-1)
        state_names = ('none', 'motion', 'skeleton', 'both')
        scores = {
            name: kwargs[f'_dual_joint_{name}_score'].reshape(-1).detach()
            for name in state_names
        }
        mse_errors = {
            name: (score - label_value).square()
            for name, score in scores.items()
        }
        rank_errors = {
            name: self._dual_ru_pairwise_rank_error(score, label_value)
            for name, score in scores.items()
        }
        mse_scale = self._dual_ru_shared_error_scale(
            torch.stack(tuple(mse_errors.values())).mean(dim=0)
        )
        rank_scale = self._dual_ru_shared_rank_scale(
            torch.stack(tuple(rank_errors.values())).mean(dim=0)
        )
        cue_pairs = {
            'motion': (('none', 'motion'), ('skeleton', 'both')),
            'skeleton': (('none', 'skeleton'), ('motion', 'both'))
        }
        cue_losses = []
        loss_info = {
            'dual_usefulness_error_scale': mse_scale,
            'dual_usefulness_rank_scale': rank_scale
        }

        for cue, pairs in cue_pairs.items():
            prediction = kwargs[
                f'_dual_{cue}_usefulness_values'
            ].reshape(-1)
            mse_contexts = []
            rank_contexts = []
            utility_contexts = []
            score_effects = []
            for before, after in pairs:
                mse_benefit = mse_errors[before] - mse_errors[after]
                rank_benefit = rank_errors[before] - rank_errors[after]
                mse_contexts.append(mse_benefit)
                rank_contexts.append(rank_benefit)
                utility_contexts.append(
                    (1 - self.dual_ru_rank_weight) *
                    (mse_benefit / mse_scale) +
                    self.dual_ru_rank_weight *
                    (rank_benefit / rank_scale)
                )
                score_effects.append((scores[after] - scores[before]).abs())

            utility_contexts = torch.stack(utility_contexts)
            utility = utility_contexts.mean(dim=0)
            context_agreement = (
                (utility_contexts[0] >= 0) ==
                (utility_contexts[1] >= 0)
            ).to(utility.dtype)
            score_effect = torch.stack(score_effects).mean(dim=0)
            effect_threshold = (
                self.dual_ru_identifiability_abs +
                self.dual_ru_identifiability_error_ratio *
                (scores['both'] - label_value).abs()
            )
            valid = score_effect >= effect_threshold

            positive_utility = F.relu(utility - self.dual_ru_deadzone)
            target = positive_utility / (
                positive_utility + self.dual_ru_temperature
            )
            magnitude = utility.abs() / (
                utility.abs() + self.dual_ru_temperature
            )
            confidence = (
                0.1 + 0.9 * magnitude
            ) * (0.25 + 0.75 * context_agreement)
            confidence = torch.where(
                valid,
                confidence,
                confidence.new_full(
                    confidence.shape, self.dual_ru_unidentifiable_weight
                )
            )

            target = target.detach()
            confidence = confidence.detach()
            cue_loss = (
                (prediction - target).square() * confidence
            ).sum() / confidence.sum().clamp_min(1e-6)
            cue_losses.append(cue_loss)

            target_mean = target.mean()
            target_std = (target - target_mean).square().mean().sqrt()
            mse_benefit = torch.stack(mse_contexts).mean(dim=0)
            rank_benefit = torch.stack(rank_contexts).mean(dim=0)
            help_mask = (utility > self.dual_ru_deadzone) & valid
            harm_mask = (utility < -self.dual_ru_deadzone) & valid
            neutral_mask = ~(help_mask | harm_mask)
            loss_info.update({
                f'{cue}_usefulness_target': target_mean,
                f'{cue}_usefulness_target_std': target_std,
                f'{cue}_usefulness_loss': cue_loss.detach(),
                f'{cue}_usefulness_delta': F.relu(utility).mean(),
                f'{cue}_usefulness_mse_delta': mse_benefit.mean(),
                f'{cue}_usefulness_mse_benefit': mse_benefit.mean(),
                f'{cue}_usefulness_score_effect': score_effect.mean(),
                f'{cue}_usefulness_effect_scale': effect_threshold.mean(),
                f'{cue}_usefulness_effect_valid_rate': valid.float().mean(),
                f'{cue}_usefulness_directional_utility': utility.mean(),
                f'{cue}_usefulness_rank_delta': rank_benefit.mean(),
                f'{cue}_usefulness_help_rate': help_mask.float().mean(),
                f'{cue}_usefulness_harm_rate': harm_mask.float().mean(),
                f'{cue}_usefulness_neutral_rate': neutral_mask.float().mean(),
                f'{cue}_usefulness_mask_rate': valid.float().mean(),
                f'{cue}_usefulness_confidence': confidence.mean(),
                f'{cue}_usefulness_objective_gain': F.relu(utility).mean(),
                f'{cue}_usefulness_grid_margin': utility.abs().mean(),
                f'{cue}_usefulness_optimal_gate': target_mean,
                f'{cue}_usefulness_joint_utility': utility.mean(),
                f'{cue}_usefulness_context_agreement':
                    context_agreement.mean(),
                f'{cue}_usefulness_target_corr': (
                    self._dual_ru_masked_correlation(
                        prediction.detach(), target, confidence
                    )
                ),
                f'_dual_{cue}_usefulness_prediction_values':
                    prediction.detach(),
                f'_dual_{cue}_usefulness_target_values': target,
                f'_dual_{cue}_usefulness_target_weights': confidence
            })

        combined_loss = torch.stack(cue_losses).mean()
        loss_info['dual_usefulness_loss'] = combined_loss.detach()
        return combined_loss, loss_info

    def _dual_ru_v5_usefulness_loss(self, label, kwargs):
        label_value = label.reshape(-1)
        grid = kwargs['_dual_gate_grid'].to(label_value.device)
        cue_losses = []
        loss_info = {}
        default_gates = {'motion': 1.0, 'skeleton': 0.5}

        for cue in ('motion', 'skeleton'):
            prediction = kwargs[
                f'_dual_{cue}_usefulness_values'
            ].reshape(-1)
            scores = kwargs[
                f'_dual_{cue}_grid_scores'
            ].reshape(grid.numel(), -1).detach()
            mse_errors = (scores - label_value.unsqueeze(0)).square()
            if self.dual_ru_target_rank_weight > 0:
                rank_errors = torch.stack([
                    self._dual_ru_pairwise_rank_error(score, label_value)
                    for score in scores
                ], dim=0)
            else:
                rank_errors = torch.zeros_like(mse_errors)
            objective = (
                mse_errors + self.dual_ru_target_rank_weight * rank_errors
            )

            best_rank = None
            if self.dual_ru_target_rank_weight == 0:
                batch_size = label_value.numel()
                grid_candidates = grid.unsqueeze(1).expand(-1, batch_size)
                score_delta = scores[1:] - scores[:-1]
                safe_delta = torch.where(
                    score_delta.abs() > 1e-8,
                    score_delta,
                    torch.ones_like(score_delta)
                )
                fraction = (
                    (label_value.unsqueeze(0) - scores[:-1]) / safe_delta
                ).clamp(0, 1)
                segment_gates = (
                    grid[:-1].unsqueeze(1) + fraction *
                    (grid[1:] - grid[:-1]).unsqueeze(1)
                )
                segment_scores = scores[:-1] + fraction * score_delta
                candidate_gates = torch.cat([
                    grid_candidates, segment_gates
                ], dim=0)
                candidate_mse = torch.cat([
                    mse_errors,
                    (segment_scores - label_value.unsqueeze(0)).square()
                ], dim=0)
                best_index = candidate_mse.argmin(dim=0)
                target = candidate_gates.gather(
                    0, best_index.unsqueeze(0)
                ).squeeze(0)
                best_objective = candidate_mse.gather(
                    0, best_index.unsqueeze(0)
                ).squeeze(0)
                best_mse = best_objective
                selection_objective = candidate_mse
            else:
                best_index = objective.argmin(dim=0)
                target = grid[best_index]
                best_objective = objective.gather(
                    0, best_index.unsqueeze(0)
                ).squeeze(0)
                best_mse = mse_errors.gather(
                    0, best_index.unsqueeze(0)
                ).squeeze(0)
                best_rank = rank_errors.gather(
                    0, best_index.unsqueeze(0)
                ).squeeze(0)
                selection_objective = objective

            default_gate = default_gates[cue]
            default_index = int(
                torch.argmin((grid - default_gate).abs()).item()
            )
            default_objective = objective[default_index]
            default_mse = mse_errors[default_index]
            default_rank = rank_errors[default_index]
            if best_rank is None:
                best_rank = default_rank
            objective_gain = (
                default_objective - best_objective
            ).clamp_min(0)

            margin_objective = (
                objective if self.dual_ru_v6 else selection_objective
            )
            sorted_objective = margin_objective.sort(dim=0).values
            decision_margin = (
                sorted_objective[1] - sorted_objective[0]
            ).clamp_min(0)
            score_range = scores.max(dim=0).values - scores.min(dim=0).values
            effect_threshold = (
                self.dual_ru_identifiability_abs +
                self.dual_ru_identifiability_error_ratio *
                (scores[default_index] - label_value).abs()
            )
            valid = score_range >= effect_threshold
            margin_ratio = decision_margin / (
                default_objective.abs() + self.dual_ru_identifiability_abs
            )
            valid_confidence = (
                0.1 + 0.9 * (
                    margin_ratio / max(self.dual_ru_temperature, 1e-6)
                ).clamp(0, 1)
            )
            if self.dual_ru_v6:
                target = torch.where(valid, target, torch.zeros_like(target))
                confidence = torch.where(
                    valid,
                    valid_confidence,
                    valid_confidence.new_full(
                        valid_confidence.shape,
                        self.dual_ru_unidentifiable_weight
                    )
                )
            else:
                confidence = valid.to(prediction.dtype) * valid_confidence

            target = target.detach()
            confidence = confidence.detach()
            per_sample_loss = (prediction - target).square()
            cue_loss = (per_sample_loss * confidence).sum() / (
                confidence.sum().clamp_min(1e-6)
            )
            cue_losses.append(cue_loss)

            active = confidence > 0
            active_weight = active.to(target.dtype)
            active_count = active_weight.sum().clamp_min(1.0)
            target_mean = (target * active_weight).sum() / active_count
            target_std = (
                ((target - target_mean).square() * active_weight).sum() /
                active_count
            ).sqrt()
            mse_benefit = default_mse - best_mse
            rank_benefit = default_rank - best_rank

            loss_info.update({
                f'{cue}_usefulness_target': target_mean,
                f'{cue}_usefulness_target_std': target_std,
                f'{cue}_usefulness_loss': cue_loss.detach(),
                f'{cue}_usefulness_delta': objective_gain.mean(),
                f'{cue}_usefulness_mse_delta': mse_benefit.mean(),
                f'{cue}_usefulness_mse_benefit': mse_benefit.mean(),
                f'{cue}_usefulness_score_effect': score_range.mean(),
                f'{cue}_usefulness_effect_scale': effect_threshold.mean(),
                f'{cue}_usefulness_effect_valid_rate': valid.float().mean(),
                f'{cue}_usefulness_directional_utility': (
                    target - default_gate
                ).mean(),
                f'{cue}_usefulness_rank_delta': rank_benefit.mean(),
                f'{cue}_usefulness_help_rate': (
                    (target > default_gate + 1e-6) & valid
                ).float().mean(),
                f'{cue}_usefulness_harm_rate': (
                    (target < default_gate - 1e-6) & valid
                ).float().mean(),
                f'{cue}_usefulness_neutral_rate': (
                    (~valid) | ((target - default_gate).abs() <= 1e-6)
                ).float().mean(),
                f'{cue}_usefulness_mask_rate': valid.float().mean(),
                f'{cue}_usefulness_confidence': confidence.mean(),
                f'{cue}_usefulness_target_corr': (
                    self._dual_ru_masked_correlation(
                        prediction.detach(), target, confidence
                    )
                ),
                f'{cue}_usefulness_objective_gain': objective_gain.mean(),
                f'{cue}_usefulness_grid_margin': decision_margin.mean(),
                f'{cue}_usefulness_optimal_gate': target_mean,
                f'_dual_{cue}_usefulness_prediction_values': prediction.detach(),
                f'_dual_{cue}_usefulness_target_values': target,
                f'_dual_{cue}_usefulness_target_weights': confidence
            })

        combined_loss = torch.stack(cue_losses).mean()
        loss_info['dual_usefulness_loss'] = combined_loss.detach()
        return combined_loss, loss_info

    def call_loss(self, pred, label, **kwargs):
        self.last_loss_info = {}
        loss = self.mse(pred.squeeze(), label.squeeze())
        if self.training and self.dual_ru_task_rank_weight > 0:
            task_rank_loss = self._dual_ru_pairwise_rank_error(
                pred.reshape(-1), label.reshape(-1)
            ).mean()
            loss = loss + self.dual_ru_task_rank_weight * task_rank_loss
            self.last_loss_info['dual_task_rank_loss'] = (
                task_rank_loss.detach()
            )
        if self.dual_ru_gate:
            if "cue_quality_loss" in kwargs:
                loss = loss + kwargs["cue_quality_loss"]
            if self.dual_ru_v10 or self.dual_ru_v11:
                joint_keys = [
                    '_dual_joint_gate_pairs', '_dual_joint_grid_scores'
                ]
                if (self.training and self.dual_ru_phase() == 1 and
                        all(key in kwargs for key in joint_keys)):
                    usefulness_loss, loss_info = (
                        self._dual_ru_v10_usefulness_loss(label, kwargs)
                    )
                    loss = loss + (
                        self.dual_ru_usefulness_loss_weight *
                        usefulness_loss
                    )
                    loss_info['dual_usefulness_weight'] = pred.new_tensor(1.0)
                    self.last_loss_info.update(loss_info)
                elif (self.training and self.dual_ru_phase() == 2 and
                      '_dual_fixed_score' in kwargs):
                    no_regret_loss, loss_info = (
                        self._dual_ru_v10_no_regret_loss(
                            pred, label, kwargs['_dual_fixed_score']
                        )
                    )
                    loss = loss + (
                        self.dual_ru_no_regret_weight * no_regret_loss
                    )
                    self.last_loss_info.update(loss_info)
                return loss
            if self.dual_ru_v9:
                grid_keys = [
                    '_dual_gate_grid', '_dual_motion_grid_scores',
                    '_dual_skeleton_grid_scores', '_dual_fixed_score'
                ]
                if self.training and all(key in kwargs for key in grid_keys):
                    usefulness_loss, loss_info = (
                        self._dual_ru_v9_usefulness_loss(label, kwargs)
                    )
                    target_weight = self.dual_ru_target_weight()
                    loss = loss + (
                        self.dual_ru_usefulness_loss_weight * target_weight *
                        usefulness_loss
                    )
                    loss_info['dual_usefulness_weight'] = pred.new_tensor(
                        target_weight
                    )
                    if self.dual_ru_phase() == 2:
                        prediction = pred.reshape(-1)
                        label_value = label.reshape(-1)
                        fixed_score = kwargs['_dual_fixed_score'].reshape(
                            -1
                        ).detach()
                        adaptive_mse = (prediction - label_value).square()
                        fixed_mse = (fixed_score - label_value).square()
                        mse_regret_values = adaptive_mse - fixed_mse
                        mse_regret = F.relu(mse_regret_values).mean()
                        adaptive_rank = self._dual_ru_pairwise_rank_error(
                            prediction, label_value
                        )
                        fixed_rank = self._dual_ru_pairwise_rank_error(
                            fixed_score, label_value
                        ).detach()
                        rank_regret_values = adaptive_rank - fixed_rank
                        rank_regret = F.relu(rank_regret_values).mean()
                        no_regret_loss = (
                            mse_regret + self.dual_ru_task_rank_weight *
                            rank_regret
                        )
                        loss = loss + (
                            self.dual_ru_no_regret_weight * no_regret_loss
                        )
                        active = (
                            (mse_regret_values > 0) |
                            (rank_regret_values > 0)
                        ).float().mean()
                        loss_info.update({
                            'dual_no_regret_loss': no_regret_loss.detach(),
                            'dual_no_regret_mse': mse_regret.detach(),
                            'dual_no_regret_rank': rank_regret.detach(),
                            'dual_no_regret_active_rate': active.detach()
                        })
                    self.last_loss_info.update(loss_info)
                return loss
            if self.dual_ru_v7 or self.dual_ru_v8:
                joint_keys = [
                    f'_dual_joint_{name}_score'
                    for name in ('none', 'motion', 'skeleton', 'both')
                ]
                if self.training and all(key in kwargs for key in joint_keys):
                    if self.dual_ru_v8:
                        usefulness_loss, loss_info = (
                            self._dual_ru_v8_usefulness_loss(label, kwargs)
                        )
                    else:
                        usefulness_loss, loss_info = (
                            self._dual_ru_v7_usefulness_loss(label, kwargs)
                        )
                    loss = loss + (
                        self.dual_ru_usefulness_loss_weight * usefulness_loss
                    )
                    self.last_loss_info.update(loss_info)
                return loss
            if self.dual_ru_v5 or self.dual_ru_v6:
                grid_keys = [
                    '_dual_gate_grid', '_dual_motion_grid_scores',
                    '_dual_skeleton_grid_scores'
                ]
                if self.training and all(key in kwargs for key in grid_keys):
                    usefulness_loss, loss_info = (
                        self._dual_ru_v5_usefulness_loss(label, kwargs)
                    )
                    loss = loss + (
                        self.dual_ru_usefulness_loss_weight * usefulness_loss
                    )
                    self.last_loss_info.update(loss_info)
                elif self.training and self.dual_ru_phase() == 2:
                    if self.dual_ru_v6:
                        anchor_terms = []
                        for cue in ('motion', 'skeleton'):
                            value_key = f'_dual_{cue}_usefulness_values'
                            anchor_key = f'_dual_{cue}_usefulness_anchor_values'
                            if value_key in kwargs and anchor_key in kwargs:
                                anchor_terms.append(F.mse_loss(
                                    kwargs[value_key], kwargs[anchor_key]
                                ))
                        if anchor_terms:
                            anchor_loss = torch.stack(anchor_terms).mean()
                            loss = loss + (
                                self.dual_ru_output_anchor_weight * anchor_loss
                            )
                            self.last_loss_info[
                                'dual_output_anchor_loss'
                            ] = anchor_loss.detach()
                    else:
                        anchor_loss = self._dual_ru_anchor_loss()
                        loss = loss + (
                            self.dual_ru_gate_anchor_weight * anchor_loss
                        )
                        self.last_loss_info['dual_gate_anchor_loss'] = (
                            anchor_loss.detach()
                        )
                return loss
            required = [
                '_dual_motion_off_score', '_dual_skeleton_off_score'
            ]
            if self.dual_ru_version >= 3:
                required.append('_dual_full_score')
            if self.training and all(key in kwargs for key in required):
                usefulness_losses = []
                loss_info = {}
                label_value = label.reshape(-1)
                full_score = (
                    kwargs['_dual_full_score'].detach().reshape(-1)
                    if self.dual_ru_version >= 3
                    else pred.detach().reshape(-1)
                )
                full_error = (
                    full_score - label_value
                ).square()
                full_rank_error = self._dual_ru_pairwise_rank_error(
                    full_score, label_value
                )
                if self.dual_ru_v3:
                    error_scale = self._dual_ru_shared_error_scale(full_error)
                    loss_info['dual_usefulness_error_scale'] = error_scale
                for cue in ('motion', 'skeleton'):
                    prediction = kwargs[
                        f'_dual_{cue}_usefulness_values'
                    ].reshape(-1)
                    off_score = kwargs[
                        f'_dual_{cue}_off_score'
                    ].reshape(-1)
                    off_error = (off_score - label_value).square()
                    mse_benefit = off_error - full_error
                    valid_target = torch.ones_like(
                        mse_benefit, dtype=torch.bool
                    )
                    effect_scale = None
                    score_effect = (full_score - off_score).abs()
                    if self.dual_ru_v4:
                        effect_scale = self._dual_ru_effect_scale(
                            cue, score_effect
                        )
                        effect_threshold = (
                            effect_scale * self.dual_ru_effect_valid_ratio
                        ).clamp_min(self.dual_ru_effect_scale_min)
                        valid_target = score_effect >= effect_threshold
                        relative_delta = mse_benefit / score_effect.clamp_min(
                            effect_threshold
                        )
                    elif self.dual_ru_v3:
                        relative_delta = (
                            mse_benefit
                        ) / error_scale
                    else:
                        relative_delta = mse_benefit / (
                            0.5 * (off_error + full_error) + 1e-6
                        )
                    temperature = max(self.dual_ru_temperature, 1e-6)
                    if self.dual_ru_version >= 2:
                        off_rank_error = self._dual_ru_pairwise_rank_error(
                            off_score, label_value
                        )
                        rank_delta = (
                            off_rank_error - full_rank_error
                        ) / (
                            0.5 * (off_rank_error + full_rank_error) + 1e-6
                        )
                        if self.dual_ru_version >= 3:
                            combined_delta = relative_delta
                        else:
                            combined_delta = (
                                (1 - self.dual_ru_rank_weight) * relative_delta +
                                self.dual_ru_rank_weight * rank_delta
                            )
                        target = (
                            (combined_delta > self.dual_ru_deadzone) &
                            valid_target
                        ).to(prediction.dtype)
                        confidence = (
                            (combined_delta.abs() - self.dual_ru_deadzone) /
                            temperature
                        ).clamp(0, 1) * valid_target.to(prediction.dtype)
                    else:
                        rank_delta = relative_delta.new_zeros(
                            relative_delta.shape
                        )
                        combined_delta = relative_delta
                        # V1 intentionally maps no measured benefit toward zero.
                        target = 1 - torch.exp(
                            -F.relu(relative_delta) / temperature
                        )
                        confidence = 0.25 + 0.75 * (
                            relative_delta.abs() / temperature
                        ).clamp(max=1.0)
                    target = target.detach()
                    confidence = confidence.detach()
                    relative_delta = relative_delta.detach()
                    rank_delta = rank_delta.detach()
                    combined_delta = combined_delta.detach()
                    per_sample_loss = (prediction - target).square()
                    usefulness_loss = (
                        (per_sample_loss * confidence).sum() /
                        confidence.sum().clamp_min(1e-6)
                    )
                    usefulness_losses.append(usefulness_loss)
                    active = confidence > 0
                    active_weight = active.to(target.dtype)
                    active_count = active_weight.sum().clamp_min(1.0)
                    loss_info[f'{cue}_usefulness_target'] = (
                        target * active_weight
                    ).sum() / active_count
                    active_target_mean = loss_info[
                        f'{cue}_usefulness_target'
                    ]
                    loss_info[f'{cue}_usefulness_target_std'] = (
                        ((target - active_target_mean).square() * active_weight)
                        .sum() / active_count
                    ).sqrt()
                    loss_info[f'{cue}_usefulness_loss'] = usefulness_loss.detach()
                    loss_info[f'{cue}_usefulness_delta'] = combined_delta.mean()
                    loss_info[f'{cue}_usefulness_mse_delta'] = (
                        relative_delta.mean()
                    )
                    loss_info[f'{cue}_usefulness_mse_benefit'] = (
                        mse_benefit.mean()
                    )
                    loss_info[f'{cue}_usefulness_score_effect'] = (
                        score_effect.mean()
                    )
                    if self.dual_ru_v4:
                        loss_info[f'{cue}_usefulness_effect_scale'] = (
                            effect_scale
                        )
                        loss_info[f'{cue}_usefulness_effect_valid_rate'] = (
                            valid_target.float().mean()
                        )
                        loss_info[
                            f'{cue}_usefulness_directional_utility'
                        ] = relative_delta.mean()
                    loss_info[f'{cue}_usefulness_rank_delta'] = (
                        rank_delta.mean()
                    )
                    help_threshold = (
                        self.dual_ru_deadzone
                        if self.dual_ru_version >= 2 else 0.0
                    )
                    loss_info[f'{cue}_usefulness_help_rate'] = (
                        (combined_delta > help_threshold) & valid_target
                    ).float().mean()
                    loss_info[f'{cue}_usefulness_mask_rate'] = (
                        active.float().mean()
                    )
                    loss_info[f'{cue}_usefulness_harm_rate'] = (
                        (combined_delta < -help_threshold) & valid_target
                    ).float().mean()
                    loss_info[f'{cue}_usefulness_neutral_rate'] = (
                        (~active).float().mean()
                    )
                    loss_info[f'{cue}_usefulness_confidence'] = confidence.mean()
                    loss_info[f'{cue}_usefulness_target_corr'] = (
                        self._dual_ru_masked_correlation(
                            prediction.detach(), target, confidence
                        )
                    )
                    loss_info[
                        f'_dual_{cue}_usefulness_prediction_values'
                    ] = prediction.detach()
                    loss_info[
                        f'_dual_{cue}_usefulness_target_values'
                    ] = target
                    loss_info[
                        f'_dual_{cue}_usefulness_target_weights'
                    ] = confidence
                combined_usefulness_loss = torch.stack(
                    usefulness_losses
                ).mean()
                loss = loss + (
                    self.dual_ru_usefulness_loss_weight *
                    combined_usefulness_loss
                )
                loss_info['dual_usefulness_loss'] = (
                    combined_usefulness_loss.detach()
                )
                self.last_loss_info = loss_info
            return loss
        if self.training and "auto_gate_loss" in kwargs:
            loss = loss + kwargs["auto_gate_loss"]
        if "cue_quality_loss" in kwargs:
            loss = loss + kwargs["cue_quality_loss"]
        if self.training and "cue_usefulness_prior_loss" in kwargs:
            loss = loss + kwargs["cue_usefulness_prior_loss"]
        usefulness_active = self.cue_current_epoch >= self.cue_usefulness_warmup_epochs
        if (self.cue_usefulness_supervision and not self.cue_usefulness_end_to_end and
                usefulness_active and
                "_skeleton_usefulness_gate_pred" in kwargs and
                "_skeleton_score_before_cue" in kwargs):
            gate_pred = kwargs["_skeleton_usefulness_gate_pred"].reshape(-1)
            label_value = label.reshape(-1)
            final_error = (pred.detach().reshape(-1) - label_value).square()
            before_error = (
                kwargs["_skeleton_score_before_cue"].reshape(-1) - label_value
            ).square()
            delta = before_error - final_error
            relative_delta = None
            threshold = delta.new_tensor(0.0)
            if self.cue_usefulness_binary_target:
                if self.cue_usefulness_relative_target:
                    relative_delta = delta / (
                        0.5 * (before_error + final_error) + 1e-6
                    )
                    if self.cue_usefulness_quantile > 0:
                        threshold = torch.quantile(
                            relative_delta.detach().abs(),
                            self.cue_usefulness_quantile
                        )
                    else:
                        threshold = relative_delta.new_tensor(self.cue_usefulness_margin)
                    useful_mask = relative_delta.abs() > threshold
                else:
                    relative_delta = None
                    threshold = delta.new_tensor(self.cue_usefulness_margin)
                    useful_mask = delta.abs() > threshold
                target = (delta > 0).float().detach()
                per_sample_loss = (gate_pred - target).square()
                if useful_mask.any():
                    usefulness_loss = per_sample_loss[useful_mask].mean()
                    target_log = target[useful_mask].mean()
                else:
                    usefulness_loss = per_sample_loss.new_tensor(0.0)
                    target_log = per_sample_loss.new_tensor(0.5)
                mask_rate = useful_mask.float().mean()
            else:
                target = torch.sigmoid(
                    delta / max(self.cue_usefulness_margin, 1e-6)
                ).detach()
                usefulness_loss = F.mse_loss(gate_pred, target)
                target_log = target.mean()
                mask_rate = target.new_ones([])
            if self.training:
                loss = loss + self.cue_usefulness_loss_weight * usefulness_loss
            self.last_loss_info = {
                "skeleton_usefulness_target": target_log,
                "skeleton_usefulness_loss": usefulness_loss.detach(),
                "skeleton_usefulness_delta": delta.mean(),
                "skeleton_usefulness_help_rate": (delta > 0).float().mean(),
                "skeleton_usefulness_mask_rate": mask_rate,
                "skeleton_usefulness_relative_delta": (
                    relative_delta.mean() if relative_delta is not None else delta.new_tensor(0.0)
                ),
                "skeleton_usefulness_threshold": threshold
            }
        return loss
