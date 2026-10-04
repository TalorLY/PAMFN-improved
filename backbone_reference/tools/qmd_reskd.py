"""Keep Ordered fixed; constrain correction energy and distill residuals only."""
import math
import numpy as np
import torch
from torch.nn import functional as F

from models.score_qmd import MeanSegmentMask,rng_state
from tools.qmd_residual import anchor_view,preserved_rng,gradient_stats
from tools.qmd_complement import ComplementBranch,centered_supervision
from tools.qmd_reskd_config import (RECIPES,V13_GT,RAMP_EPOCHS,TARGET_RATIO,BETA_CAP,GRAD_EPS,
    calibration_contract)


class ResidualKDBranch(ComplementBranch):
    def __init__(self,tag,input_dim=768,seed=0):
        if tag not in RECIPES:raise ValueError('unknown v14 recipe')
        # Exactly the retained v13 MLP16 architecture and private initialization.
        super().__init__(V13_GT,input_dim,seed)


def distillation_loss(kind,base,masked_base,residual,masked_residual,bias):
    if kind=='score':
        return F.smooth_l1_loss(masked_base.detach()+bias+masked_residual,
                               (base.detach()+bias+residual).detach())
    if kind=='residual':
        # No frozen-anchor mask drift enters this objective, even as a target.
        return F.smooth_l1_loss(masked_residual,residual.detach())
    raise ValueError('unknown distillation space')


def calibrated_beta(gt_norm,kd_norm):
    if not all(math.isfinite(v) and v>=0 for v in (gt_norm,kd_norm)):
        raise FloatingPointError('invalid calibration gradient norm')
    if min(gt_norm,kd_norm)<=GRAD_EPS:return 0.
    return min(BETA_CAP,TARGET_RATIO*gt_norm/kd_norm)


class ResidualKDObjective:
    def __init__(self,seed,tag,smoke=False):
        self.recipe=RECIPES[tag];self.mask=MeanSegmentMask(seed,.15)
        self.ramp_epochs=1 if smoke else RAMP_EPOCHS
        self.steps=0;self.zero=0;self.capped=0;self.beta_sum=0.
        self.beta_min=float('inf');self.beta_max=0.;self.max_ratio=0.;self.first_positive=None

    def losses(self,anchor,branch,inputs,label,lengths,epoch):
        before=(rng_state(),np.random.get_state())
        base,features=anchor_view(anchor,inputs)
        residual=branch(features);corrected=base+branch.global_bias+residual
        gt,centered,mean_term=centered_supervision(corrected,label,residual)
        energy=residual.square().mean()
        supervised=gt+self.recipe['energy']*energy
        # All arms draw the same mask and replay the same anchor randomness.
        masked,info=self.mask(inputs,lengths)
        with preserved_rng(before):masked_base,masked_features=anchor_view(anchor,masked)
        masked_residual=branch(masked_features)
        kd=distillation_loss(self.recipe['target'],base,masked_base,residual,masked_residual,branch.global_bias)
        stats=gradient_stats(gt,kd,branch.parameters())
        regularized=gradient_stats(supervised,kd,branch.parameters())
        beta=calibrated_beta(stats['gt_norm'],stats['kd_norm'])
        scale=min(1.,(epoch+1)/self.ramp_epochs)
        coefficient=self.recipe['weight']*scale*beta
        ratio=coefficient*stats['kd_norm']/stats['gt_norm'] if stats['gt_norm']>GRAD_EPS else 0.
        self.steps+=1;self.beta_sum+=beta;self.beta_min=min(self.beta_min,beta);self.beta_max=max(self.beta_max,beta)
        self.zero+=int(min(stats['gt_norm'],stats['kd_norm'])<=GRAD_EPS)
        self.capped+=int(beta==BETA_CAP);self.max_ratio=max(self.max_ratio,ratio)
        if coefficient*stats['kd_norm']>0 and self.first_positive is None:
            self.first_positive=dict(epoch=epoch+1,step=self.steps)
        report=dict(gt=float(gt.detach()),centered_mse=float(centered.detach()),residual_mean_penalty=float(mean_term.detach()),
            raw_mse=float(F.mse_loss(corrected.detach(),label.reshape(-1))),kd=float(kd.detach()),
            energy=float(energy.detach()),weighted_energy=self.recipe['energy']*float(energy.detach()),
            supervised=float(supervised.detach()),regularized_gt_norm=regularized['gt_norm'],regularized_cosine=regularized['cosine'],
            active=True,scale=scale,beta=beta,kd_ratio=ratio,coefficient=coefficient,weighted_kd=coefficient*float(kd.detach()),
            modality=info['modality'],anchor_disagreement=float(F.smooth_l1_loss(masked_base,base)),
            residual_disagreement=float(F.smooth_l1_loss(masked_residual.detach(),residual.detach())),
            residual_mean=float(residual.detach().mean()),residual_std=float(residual.detach().std(unbiased=False)),
            residual_saturation=float((residual.detach().abs()>=.95*branch.config['bound']).float().mean()),**stats)
        return supervised+coefficient*kd,report

    def calibration_summary(self):
        return dict(calibration_contract(),steps=self.steps,zero_gradient_batches=self.zero,capped_batches=self.capped,
            beta_min=self.beta_min if self.steps else 0.,beta_max=self.beta_max,beta_mean=self.beta_sum/max(1,self.steps),
            max_effective_ratio=self.max_ratio,first_positive=self.first_positive)
