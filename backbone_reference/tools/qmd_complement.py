"""Training-only bias separation and low-capacity complementary correction."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.score_qmd import MeanSegmentMask, rng_state
from tools.qmd_residual import anchor_view, preserved_rng, gradient_stats
from tools.qmd_complement_config import RECIPES, LR, MEAN_PENALTY, RAMP_EPOCHS, branch_config


class ComplementBranch(nn.Module):
    def __init__(self, tag, input_dim=768, seed=0):
        super().__init__()
        self.config = branch_config(tag, input_dim)
        dim, hidden = input_dim, self.config['hidden']
        # Private CPU initialization does not reseed CUDA or change loader RNG.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed+20260915)
            self.net = (nn.Linear(dim,1,bias=False) if not hidden else
                        nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,1,bias=False)))
            nn.init.zeros_(self.net.weight if not hidden else self.net[-1].weight)
        self.register_buffer('feature_mean',torch.zeros(dim))
        self.register_buffer('feature_scale',torch.ones(dim))
        self.register_buffer('global_bias',torch.zeros(()))
        self.register_buffer('calibrated',torch.tensor(False))

    def fit_training_statistics(self, cache):
        if bool(self.calibrated): raise RuntimeError('training statistics are immutable once fitted')
        base, feat, y = [v.detach() for v in cache]
        if base.numel()<2 or feat.shape!=(base.numel(),self.config['input_dim']) or y.numel()!=base.numel():
            raise ValueError('invalid training-only calibration cache')
        if not all(torch.isfinite(v).all() for v in (base,feat,y)):
            raise ValueError('nonfinite training statistics')
        # Float64 reductions; persistent buffers remain float32 for old GPUs.
        mean=feat.double().mean(0)
        scale=feat.double().std(0,unbiased=False).clamp_min(self.config['std_floor'])
        bias=(y.double().reshape(-1)-base.double().reshape(-1)).mean()
        self.feature_mean.copy_(mean);self.feature_scale.copy_(scale);self.global_bias.copy_(bias)
        self.calibrated.fill_(True)
        return dict(source_split='train',samples=base.numel(),passes=1,
            global_bias=float(self.global_bias),mean_label=float(y.double().mean()),
            mean_anchor=float(base.double().mean()),feature_dim=feat.shape[1],
            feature_scale_min=float(self.feature_scale.min()),feature_scale_max=float(self.feature_scale.max()))

    def forward(self, features):
        if not bool(self.calibrated): raise RuntimeError('fit training statistics before correction')
        x=((features.detach()-self.feature_mean)/self.feature_scale).clamp(-self.config['feature_clip'],self.config['feature_clip'])
        return self.config['bound']*torch.tanh(self.net(x).reshape(-1))

    def score(self, base, feature):
        return base.detach().reshape(-1)+self.global_bias+self(feature)

    def evaluation_score(self, base, feature):
        # Adding a global offset in float32 can merge tiny legacy score gaps.
        # Double accumulation preserves the ranking of the bias-only anchor.
        residual=self(feature).double()
        return base.detach().reshape(-1).double()+self.global_bias.double()+residual


def centered_supervision(corrected, label, residual):
    centered=F.mse_loss(corrected-corrected.mean(),label.reshape(-1)-label.mean())
    zero_mean=residual.mean().square()
    return centered+MEAN_PENALTY*zero_mean,centered,zero_mean


def teacher_target(kind, raw_full, corrected_full):
    if kind=='raw': return raw_full.detach()
    if kind=='corrected': return corrected_full.detach()
    raise ValueError('unknown teacher target')


class ComplementObjective:
    def __init__(self, seed, tag, smoke=False):
        self.recipe=RECIPES[tag]
        self.mask=MeanSegmentMask(seed,.15)
        self.ramp_epochs=1 if smoke else RAMP_EPOCHS
        self.beta=None;self.calibration=None

    def losses(self, anchor, branch, inputs, label, lengths, epoch):
        before=(rng_state(),np.random.get_state())
        base,features=anchor_view(anchor,inputs)
        residual=branch(features)
        corrected=base+branch.global_bias+residual
        gt,centered,mean_term=centered_supervision(corrected,label,residual)
        # All six arms perform the same paired mask draw, even GT-only controls.
        masked,info=self.mask(inputs,lengths)
        with preserved_rng(before):
            masked_base,masked_feature=anchor_view(anchor,masked)
        masked_score=branch.score(masked_base,masked_feature)
        target=teacher_target(self.recipe['target'],base,corrected)
        kd=F.smooth_l1_loss(masked_score,target)
        stats=gradient_stats(gt,kd,branch.parameters())
        scale=min(1.,(epoch+1)/self.ramp_epochs)
        # Fixed first-batch norm calibration: same across matched KD weights.
        if self.beta is None:
            raw=.075*stats['gt_norm']/stats['kd_norm'] if stats['kd_norm']>1e-12 else 0.
            self.beta=min(1.,raw)
            self.calibration=dict(epoch=epoch+1,beta=self.beta,uncapped_beta=raw,
                target_ratio=.075,beta_cap=1.,scope='all_branch_parameters_first_batch',
                gt_norm=stats['gt_norm'],kd_norm=stats['kd_norm'],
                achieved_ratio=self.recipe['weight']*self.beta*stats['kd_norm']/stats['gt_norm'] if stats['gt_norm']>1e-12 else 0.)
        coefficient=self.recipe['weight']*scale*self.beta
        total=gt+coefficient*kd
        report=dict(gt=float(gt.detach()),centered_mse=float(centered.detach()),residual_mean_penalty=float(mean_term.detach()),
            raw_mse=float(F.mse_loss(corrected.detach(),label.reshape(-1))),kd=float(kd.detach()),
            active=True,scale=scale,coefficient=coefficient,weighted_kd=coefficient*float(kd.detach()),
            modality=info['modality'],anchor_disagreement=float(F.smooth_l1_loss(masked_base,base)),
            residual_mean=float(residual.detach().mean()),residual_std=float(residual.detach().std(unbiased=False)),
            residual_saturation=float((residual.detach().abs()>=.95*branch.config['bound']).float().mean()),
            **stats)
        return total,report
