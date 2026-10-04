"""Two controls only; the original E/U models remain in legacy/ unchanged."""
from contextlib import contextmanager
from unittest.mock import patch
import bootstrap
import torch
from torch.nn import functional as F
import worker as engine
from modules import EBranch, UBranch, _features, _vector

class PlainE(EBranch):
    def forward(self,features):
        if not bool(self.calibrated):raise RuntimeError('Fit train statistics first')
        features=_features(features,self.input_dim)
        x=((features.detach()-self.feature_mean)/self.feature_scale).clamp(-5.,5.)
        return .03*self.net(x).reshape(-1)

def ordinary_mse(e,base,features,labels):
    p=_vector(base,features.shape[0],'base').detach().float()
    y=_vector(labels,features.shape[0],'labels').detach().float()
    delta=e(features); loss=F.mse_loss(p+e.global_bias+delta,y)
    return loss,dict(loss=float(loss.detach()),objective='ordinary score MSE')

class FusedU(UBranch):
    def fit(self,e_scores,features):
        stats=super().fit(e_scores,features)
        stats['feature_source']='three copies of frozen fused256; not V/F/A'
        return stats

@contextmanager
def control_engine(method):
    """Process-local injection into the pinned branch loop; no source mutation."""
    original_view=engine.frozen_view
    original_mechanism=engine.mechanism
    def fused_view(*args,**kwargs):
        p,j,_=original_view(*args,**kwargs)
        return p,j,j.repeat(1,3)
    def diagnostic(*args,**kwargs):
        value=original_mechanism(*args,**kwargs)
        if method=='mc_fused_u':
            for index,name in enumerate(('V','F','A')):
                value['variants']['without_head_%d'%index]=value['variants'].pop('without_'+name)
            value['feature_source']='fused; head removals are NOT modality removals'
        return value
    updates=dict(epochs=lambda m,t,smoke=False:2 if smoke else 500,
        mechanism=diagnostic,benchmark=lambda *a,**k:dict(status='separate_v39_benchmark',reason='control training has no inline GPU timing'))
    if method=='mc_plain_e':updates.update(EBranch=PlainE,e_objective=ordinary_mse)
    elif method=='mc_fused_u':updates.update(UBranch=FusedU,frozen_view=fused_view)
    else:raise ValueError('Unknown control')
    with patch.multiple(engine,**updates):yield engine
