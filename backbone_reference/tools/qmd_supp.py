"""Frozen E prediction plus trainable modal supplement; no KD or label-based routing."""
import torch
from torch import nn
from torch.nn import functional as F
from tools.qmd_modal import modal_view
from tools.qmd_residual import assert_frozen
from tools.qmd_supp_config import RECIPES,LR,GATE_LR,GATE_PENALTY,branch_config


class SupplementBranch(nn.Module):
    def __init__(self,tag,input_dim=768,seed=0):
        super().__init__();self.config=branch_config(tag,input_dim)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed+20260916)
            heads=[]
            for _ in range(3):
                head=nn.Sequential(nn.Linear(input_dim//3,16),nn.GELU(),nn.Linear(16,1,bias=False))
                nn.init.zeros_(head[-1].weight);heads.append(head)
            self.heads=nn.ModuleList(heads)
        self.register_buffer('feature_mean',torch.zeros(input_dim))
        self.register_buffer('feature_scale',torch.ones(input_dim))
        self.register_buffer('bound',torch.zeros(()))
        self.register_buffer('calibrated',torch.tensor(False))
        if RECIPES[tag]['shrink']:self.gate=nn.Parameter(torch.ones(()))
        else:self.register_buffer('gate',torch.ones(()))

    def fit_training_statistics(self,e_scores,packed):
        if bool(self.calibrated):raise RuntimeError('training statistics already fitted')
        dim=self.config['input_dim'];x=packed.detach()[:,dim:]
        if x.shape!=(e_scores.numel(),dim) or e_scores.numel()<2:raise ValueError('bad calibration shape')
        if not torch.isfinite(x).all() or not torch.isfinite(e_scores).all():raise ValueError('nonfinite training cache')
        self.feature_mean.copy_(x.double().mean(0));self.feature_scale.copy_(x.double().std(0,unbiased=False).clamp_min(.01))
        spread=float(e_scores.detach().double().std(unbiased=False))
        bound=min(self.config['abs_bound'],self.config['relative_ratio']*spread) if self.config['recipe']['relative'] else self.config['abs_bound']
        self.bound.fill_(bound);self.calibrated.fill_(True)
        return dict(source_split='train',samples=x.shape[0],passes=1,feature_dim=dim,packed_feature_dim=2*dim,
            e_prediction_std=spread,bound=float(self.bound),feature_scale_min=float(self.feature_scale.min()),
            feature_scale_max=float(self.feature_scale.max()),feature_source='modal',new_global_bias=False)

    def raw_components(self,packed):
        if not bool(self.calibrated):raise RuntimeError('fit training statistics first')
        dim=self.config['input_dim']
        if packed.ndim!=2 or packed.shape[1]!=2*dim:raise ValueError('wrong packed features')
        x=((packed.detach()[:,dim:]-self.feature_mean)/self.feature_scale).clamp(-5.,5.)
        return self.bound*torch.cat([torch.tanh(h(v)) for h,v in zip(self.heads,x.chunk(3,dim=1))],dim=1)

    def components(self,packed):return self.gate.clamp(0.,1.)*self.raw_components(packed)

    def forward(self,packed):return self.components(packed).mean(1)

    def project_gate(self):
        with torch.no_grad():self.gate.clamp_(0.,1.)

    def penalty(self):
        return GATE_PENALTY*self.bound.square()*self.gate if self.config['recipe']['shrink'] else self.bound*0.

    def optimizer(self):
        groups=[dict(params=list(self.heads.parameters()),lr=LR,weight_decay=.01)]
        if self.config['recipe']['shrink']:groups.append(dict(params=[self.gate],lr=GATE_LR,weight_decay=0.))
        return torch.optim.AdamW(groups)


def e_score(e,base,packed):
    assert_frozen(e)
    with torch.no_grad():return e.evaluation_score(base,packed)


def objective(anchor,e,branch,inputs,label):
    base,packed=modal_view(anchor,inputs)
    frozen=e_score(e,base,packed)
    residual=branch(packed).double();corrected=frozen+residual;y=label.reshape(-1).double()
    centered=F.mse_loss(corrected-corrected.mean(),y-y.mean())
    mean_penalty=.1*residual.mean().square();shrink=branch.penalty().double()
    loss=centered+mean_penalty+shrink
    return loss,dict(loss=float(loss.detach()),centered_mse=float(centered.detach()),
        mean_penalty=float(mean_penalty.detach()),shrink_penalty=float(shrink.detach()),
        residual_std=float(residual.detach().std(unbiased=False)),gate=float(branch.gate.detach()))
