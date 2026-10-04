"""Training-label pair repair; gated residual-pair consistency, no anchor gradient."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from models.score_qmd import MeanSegmentMask,rng_state
from tools.qmd_residual import anchor_view,preserved_rng,gradient_stats
from tools.qmd_complement import ComplementBranch,centered_supervision
from tools.qmd_pair_config import (RECIPES,V13_GT,RAMP_EPOCHS,TARGET_RATIO,RANK_RATIO,BETA_CAP,
    GRAD_EPS,LABEL_GAP,RESIDUAL_BOUND,MARGIN_SCALE,MARGIN_CAP,PROTECT_WEIGHT,GATE_EPS,calibration_contract)


class PairBranch(ComplementBranch):
    def __init__(self,tag,input_dim=768,seed=0):
        if tag not in RECIPES:raise ValueError('unknown v15 recipe')
        super().__init__(V13_GT,input_dim,seed)


def reliable_pairs(label):
    y=label.detach().reshape(-1)
    index=torch.triu_indices(y.numel(),y.numel(),offset=1,device=y.device)
    i,j=index[0],index[1];gap=y[i]-y[j];keep=gap.abs()>=LABEL_GAP
    return i[keep],j[keep],gap[keep]


def mean_or_zero(value,zero):
    return value.mean() if value.numel() else zero.sum()*0.


def pair_losses(base,full_residual,masked_residual,label):
    # The common global bias cancels in every pair difference.
    b=base.detach().reshape(-1);r=full_residual.reshape(-1);rm=masked_residual.reshape(-1)
    i,j,gap=reliable_pairs(label);sign=gap.sign()
    anchor_gap=b[i]-b[j];residual_gap=r[i]-r[j];corrected_gap=anchor_gap+residual_gap
    margin=(MARGIN_SCALE*gap.abs()).clamp(max=MARGIN_CAP)
    deficit=margin-sign*anchor_gap
    hard=((deficit>0)&(deficit<2*RESIDUAL_BOUND-GATE_EPS)).detach()
    protected=(deficit<=0).detach()
    unreachable=(deficit>=2*RESIDUAL_BOUND-GATE_EPS).detach()
    violation=torch.relu(margin-sign*corrected_gap).square()
    repair=mean_or_zero(violation[hard],r)
    protect=mean_or_zero(violation[protected],r)
    ranking=repair+PROTECT_WEIGHT*protect
    # Strict improvement prevents the zero-initialized teacher from opening the gate.
    teacher_gap=corrected_gap.detach()
    gate=((sign*teacher_gap>0)&
          ((teacher_gap-gap).abs()+GATE_EPS<(anchor_gap-gap).abs())).detach()
    target=residual_gap.detach()
    masked_gap=rm[i]-rm[j]
    discrepancy=F.smooth_l1_loss(masked_gap,target,reduction='none')
    kd=mean_or_zero(discrepancy[gate],rm)
    with torch.no_grad():
        anchor_wrong=sign*anchor_gap<=0;current_wrong=sign*teacher_gap<=0
        info=dict(reliable_pairs=int(gap.numel()),hard_pairs=int(hard.sum()),
            protected_pairs=int(protected.sum()),unreachable_pairs=int(unreachable.sum()),accepted_pairs=int(gate.sum()),
            anchor_wrong_pairs=int(anchor_wrong.sum()),corrected_wrong_pairs=int(current_wrong.sum()),
            repaired_pairs=int((anchor_wrong&~current_wrong).sum()),
            broken_pairs=int((~anchor_wrong&current_wrong).sum()),
            repair_loss=float(repair.detach()),protect_loss=float(protect.detach()))
    return ranking,kd,info


def pair_metrics(base,residual,label):
    # Reporting only; test labels never enter branch.forward or gate at inference.
    with torch.no_grad():
        _,_,info=pair_losses(base.double(),residual.double(),residual.double(),label.double())
    return info


def calibrated_beta(gt_norm,aux_norm,ratio):
    if not all(math.isfinite(v) and v>=0 for v in (gt_norm,aux_norm,ratio)):
        raise FloatingPointError('invalid calibration norm')
    if min(gt_norm,aux_norm)<=GRAD_EPS:return 0.
    return min(BETA_CAP,ratio*gt_norm/aux_norm)


class PairObjective:
    def __init__(self,seed,tag,smoke=False):
        self.recipe=RECIPES[tag];self.mask=MeanSegmentMask(seed,.15)
        self.ramp_epochs=1 if smoke else RAMP_EPOCHS
        self.steps=0;self.zero=0;self.capped=0;self.beta_sum=0.
        self.beta_min=float('inf');self.beta_max=0.;self.max_ratio=0.;self.first_positive=None
        self.rank_max_ratio=0.;self.rank_first_positive=None;self.rank_positive_steps=0

    def losses(self,anchor,branch,inputs,label,lengths,epoch):
        before=(rng_state(),np.random.get_state())
        base,features=anchor_view(anchor,inputs)
        residual=branch(features);corrected=base+branch.global_bias+residual
        gt,centered,mean_term=centered_supervision(corrected,label,residual)
        masked,mask_info=self.mask(inputs,lengths)
        with preserved_rng(before):masked_base,masked_features=anchor_view(anchor,masked)
        masked_residual=branch(masked_features)
        ranking,kd,pair_info=pair_losses(base,residual,masked_residual,label)
        stats=gradient_stats(gt,kd,branch.parameters())
        rank_stats=gradient_stats(gt,ranking,branch.parameters())
        scale=min(1.,(epoch+1)/self.ramp_epochs)
        rank_beta=calibrated_beta(rank_stats['gt_norm'],rank_stats['kd_norm'],RANK_RATIO)
        rank_coefficient=self.recipe['rank']*scale*rank_beta
        supervised=gt+rank_coefficient*ranking
        regularized=gradient_stats(supervised,kd,branch.parameters())
        beta=calibrated_beta(stats['gt_norm'],stats['kd_norm'],TARGET_RATIO)
        coefficient=self.recipe['weight']*scale*beta
        ratio=coefficient*stats['kd_norm']/stats['gt_norm'] if stats['gt_norm']>GRAD_EPS else 0.
        rank_ratio=rank_coefficient*rank_stats['kd_norm']/stats['gt_norm'] if stats['gt_norm']>GRAD_EPS else 0.
        self.steps+=1;self.beta_sum+=beta;self.beta_min=min(self.beta_min,beta);self.beta_max=max(self.beta_max,beta)
        self.zero+=int(min(stats['gt_norm'],stats['kd_norm'])<=GRAD_EPS)
        self.capped+=int(beta==BETA_CAP);self.max_ratio=max(self.max_ratio,ratio)
        self.rank_max_ratio=max(self.rank_max_ratio,rank_ratio)
        if coefficient*stats['kd_norm']>0 and self.first_positive is None:
            self.first_positive=dict(epoch=epoch+1,step=self.steps)
        if rank_coefficient*rank_stats['kd_norm']>0:
            self.rank_positive_steps+=1
            if self.rank_first_positive is None:self.rank_first_positive=dict(epoch=epoch+1,step=self.steps)
        report=dict(gt=float(gt.detach()),centered_mse=float(centered.detach()),residual_mean_penalty=float(mean_term.detach()),
            raw_mse=float(F.mse_loss(corrected.detach(),label.reshape(-1))),kd=float(kd.detach()),
            energy=0.,weighted_energy=0.,supervised=float(supervised.detach()),
            regularized_gt_norm=regularized['gt_norm'],regularized_cosine=regularized['cosine'],
            active=True,scale=scale,beta=beta,kd_ratio=ratio,coefficient=coefficient,weighted_kd=coefficient*float(kd.detach()),
            rank=float(ranking.detach()),rank_beta=rank_beta,rank_coefficient=rank_coefficient,rank_ratio=rank_ratio,
            rank_norm=rank_stats['kd_norm'],rank_cosine=rank_stats['cosine'],
            weighted_rank=rank_coefficient*float(ranking.detach()),
            modality=mask_info['modality'],anchor_disagreement=float(F.smooth_l1_loss(masked_base,base)),
            residual_disagreement=float(F.smooth_l1_loss(masked_residual.detach(),residual.detach())),
            residual_mean=float(residual.detach().mean()),residual_std=float(residual.detach().std(unbiased=False)),
            residual_saturation=float((residual.detach().abs()>=.95*branch.config['bound']).float().mean()),
            **pair_info,**stats)
        return supervised+coefficient*kd,report

    def calibration_summary(self):
        return dict(calibration_contract(),steps=self.steps,zero_gradient_batches=self.zero,capped_batches=self.capped,
            beta_min=self.beta_min if self.steps else 0.,beta_max=self.beta_max,beta_mean=self.beta_sum/max(1,self.steps),
            max_effective_ratio=self.max_ratio,first_positive=self.first_positive,
            max_rank_ratio=self.rank_max_ratio,rank_first_positive=self.rank_first_positive,
            rank_positive_steps=self.rank_positive_steps)
