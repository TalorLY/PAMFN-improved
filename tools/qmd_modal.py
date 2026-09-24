"""Read-only unimodal capture and low-capacity modality-specific residual heads."""
import numpy as np
import torch
from torch import nn
from tools.qmd_residual import assert_frozen,preserved_rng,gradient_stats
from tools.qmd_complement import ComplementBranch,centered_supervision
from tools.qmd_pair import PairObjective,pair_losses,calibrated_beta
from tools.qmd_modal_config import RECIPES,MODALITIES,branch_config,TARGET_RATIO
from models.score_qmd import rng_state,MeanSegmentMask
from torch.nn import functional as F


def modal_view(model,inputs):
    """Capture both paths in one unchanged forward; never invoke extra anchor passes."""
    assert_frozen(model);captured={k:[] for k in ('fused',)+MODALITIES};hooks=[]
    try:
        hooks.append(model.fc.register_forward_pre_hook(lambda m,a:captured['fused'].append(a[0].detach())))
        for key,name in zip(MODALITIES,('model_r','model_f','model_a')):
            module=getattr(model,name)
            def capture(m,args,output,key=key):
                captured[key].append(output[1]['feats'][-1].detach())
            hooks.append(module.register_forward_hook(capture))
        with torch.no_grad():score,_=model(inputs)
    finally:
        for hook in hooks:hook.remove()
    if any(len(v)!=1 or v[0].ndim!=3 or v[0].shape[2]!=1 for v in captured.values()):
        raise ValueError('expected one pooled [B,C,1] feature per fused/unimodal path')
    fused=captured['fused'][0].squeeze(2)
    modal=[captured[k][0].squeeze(2) for k in MODALITIES]
    if any(v.shape[0]!=fused.shape[0] or v.shape[1]*3!=fused.shape[1] for v in modal):
        raise ValueError('unimodal/fused channel contract changed')
    packed=torch.cat([fused]+modal,dim=1).detach()
    if not torch.isfinite(packed).all():raise FloatingPointError('nonfinite frozen features')
    return score.detach().reshape(-1),packed


def make_cache(anchor,loader,device,replay):
    import pipeline
    scores,features,labels=[],[],[]
    with preserved_rng(replay):
        for batch in loader:
            inputs,y=pipeline._prepare_batch(batch,device)
            score,feature=modal_view(anchor,inputs)
            scores.append(score);features.append(feature);labels.append(y.reshape(-1))
    return torch.cat(scores),torch.cat(features),torch.cat(labels)


class ModalBranch(ComplementBranch):
    def __init__(self,tag,input_dim=768,seed=0):
        nn.Module.__init__(self)
        self.config=branch_config(tag,input_dim);dim=input_dim
        def head(width):
            value=nn.Sequential(nn.Linear(width,16),nn.GELU(),nn.Linear(16,1,bias=False))
            nn.init.zeros_(value[-1].weight)
            return value
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed+20260915)
            # Joint head initialization matches v13 E exactly.
            self.net=(head(dim) if self.config['heads']=='joint' else
                      nn.ModuleDict({k:head(dim//3) for k in MODALITIES}))
        self.register_buffer('feature_mean',torch.zeros(dim))
        self.register_buffer('feature_scale',torch.ones(dim))
        self.register_buffer('global_bias',torch.zeros(()))
        self.register_buffer('calibrated',torch.tensor(False))

    def selected_features(self,packed):
        d=self.config['input_dim']
        if packed.ndim!=2 or packed.shape[1]!=2*d:raise ValueError('expected [B,2*input_dim] packed features')
        return packed[:,:d] if self.config['features']=='fused' else packed[:,d:]

    def fit_training_statistics(self,cache):
        base,packed,y=cache
        stats=super().fit_training_statistics((base,self.selected_features(packed),y))
        stats.update(feature_source=self.config['features'],packed_feature_dim=packed.shape[1])
        return stats

    def components(self,packed):
        if not bool(self.calibrated):raise RuntimeError('fit training statistics before correction')
        feat=self.selected_features(packed).detach()
        x=((feat-self.feature_mean)/self.feature_scale).clamp(-5.,5.)
        if self.config['heads']=='joint':return self.config['bound']*torch.tanh(self.net(x))
        return torch.cat([self.config['bound']*torch.tanh(self.net[k](v))
                          for k,v in zip(MODALITIES,x.chunk(3,dim=1))],dim=1)

    def forward(self,packed):return self.components(packed).mean(dim=1)


class ModalObjective(PairObjective):
    def __init__(self,seed,tag,smoke=False):
        # Reuse the proven per-batch calibration accounting, with rank always off.
        super().__init__(seed,'ordered_pair_gate',smoke)
        self.recipe=RECIPES[tag];self.mask=MeanSegmentMask(seed,.15)

    def losses(self,anchor,branch,inputs,label,lengths,epoch):
        before=(rng_state(),np.random.get_state())
        base,features=modal_view(anchor,inputs)
        residual=branch(features);corrected=base+branch.global_bias+residual
        gt,centered,mean_term=centered_supervision(corrected,label,residual)
        masked,mask_info=self.mask(inputs,lengths)
        with preserved_rng(before):masked_base,masked_features=modal_view(anchor,masked)
        masked_residual=branch(masked_features)
        _,kd,pair_info=pair_losses(base,residual,masked_residual,label)
        stats=gradient_stats(gt,kd,branch.parameters())
        scale=min(1.,(epoch+1)/self.ramp_epochs)
        beta=calibrated_beta(stats['gt_norm'],stats['kd_norm'],TARGET_RATIO)
        coefficient=self.recipe['weight']*scale*beta
        ratio=coefficient*stats['kd_norm']/stats['gt_norm'] if stats['gt_norm']>1e-12 else 0.
        self.steps+=1;self.beta_sum+=beta;self.beta_min=min(self.beta_min,beta);self.beta_max=max(self.beta_max,beta)
        self.zero+=int(min(stats['gt_norm'],stats['kd_norm'])<=1e-12)
        self.capped+=int(beta==10000.);self.max_ratio=max(self.max_ratio,ratio)
        if coefficient*stats['kd_norm']>0 and self.first_positive is None:self.first_positive=dict(epoch=epoch+1,step=self.steps)
        report=dict(gt=float(gt.detach()),centered_mse=float(centered.detach()),residual_mean_penalty=float(mean_term.detach()),
            raw_mse=float(F.mse_loss(corrected.detach(),label.reshape(-1))),kd=float(kd.detach()),
            energy=0.,weighted_energy=0.,supervised=float(gt.detach()),regularized_gt_norm=stats['gt_norm'],regularized_cosine=stats['cosine'],
            active=True,scale=scale,beta=beta,kd_ratio=ratio,coefficient=coefficient,weighted_kd=coefficient*float(kd.detach()),
            rank=0.,rank_beta=0.,rank_coefficient=0.,rank_ratio=0.,rank_norm=0.,rank_cosine=None,weighted_rank=0.,
            modality=mask_info['modality'],anchor_disagreement=float(F.smooth_l1_loss(masked_base,base)),
            residual_disagreement=float(F.smooth_l1_loss(masked_residual.detach(),residual.detach())),
            residual_mean=float(residual.detach().mean()),residual_std=float(residual.detach().std(unbiased=False)),
            residual_saturation=float((residual.detach().abs()>=.0285).float().mean()),**pair_info,**stats)
        return gt+coefficient*kd,report

    def calibration_summary(self):
        result=super().calibration_summary();result['rank_ratio']=0.
        return result
