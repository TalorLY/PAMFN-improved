"""Same operations and dtypes; remove repeated Python/host synchronization work.

This is an explicitly scoped inference session, never a training substitute.
Native MCMoE forward (including native missing-modality checks) is unchanged.
"""
import bootstrap
import torch
from adapter import assert_frozen,historical_inference

def e_score(e,p,j):
    x=((j.detach()-e.feature_mean)/e.feature_scale).clamp(-5.,5.)
    delta=.03*torch.tanh(e.net(x).reshape(-1))
    return p.detach().reshape(-1).double()+e.global_bias.double()+delta.double()

def u_delta(u,m):
    x=((m.detach()-u.feature_mean)/u.feature_scale).clamp(-5.,5.)
    components=u.bound*torch.cat([head(v) for head,v in zip(u.heads,x.chunk(3,dim=1))],dim=1)
    return components.mean(1)

class InferenceSession:
    def __init__(self,model,e=None,u=None):
        self.model=model;self.e=e;self.u=u;self.active=False;self.hooks=[];self.enc=[];self.joint=[]
        if u is not None and e is None:raise ValueError('U requires E')
    def __enter__(self):
        if self.active:raise RuntimeError('Inference session is not reentrant')
        assert_frozen(self.model)
        for branch in (self.e,self.u):
            if branch is not None:
                assert_frozen(branch)
                if not bool(branch.calibrated):raise ValueError('Uncalibrated deployment branch')
                if any(not bool(torch.isfinite(v).all()) for v in branch.state_dict().values() if v.is_floating_point()):
                    raise ValueError('Nonfinite deployment checkpoint')
        self.scope=historical_inference(self.model);self.scope.__enter__()
        try:
            native=self.model.backbone
            if self.e is not None:
                self.hooks.append(native.regressor.register_forward_pre_hook(lambda m,a:self.joint.append(a[0].detach().mean(1))))
            if self.u is not None:
                self.hooks.append(native.transformer.encoder.register_forward_hook(lambda m,a,o:self.enc.append(o.detach().mean(1))))
            self.active=True;return self
        except BaseException:
            self.__exit__(None,None,None);raise
    def features(self,v,a,f):
        if not self.active or torch.is_grad_enabled():raise RuntimeError('Use active no-grad session')
        if any(x.ndim!=3 for x in (v,a,f)):raise ValueError('Expected batched temporal inputs')
        self.enc.clear();self.joint.clear()
        p=self.model(v,a,f,[1,1,1])['output'].reshape(-1)
        j=self.joint[0] if self.e is not None and len(self.joint)==1 else None
        if self.e is not None and j is None:raise RuntimeError('Fused hook path changed')
        m=None
        if self.u is not None:
            if len(self.enc)!=3:raise RuntimeError('Modal hook path changed')
            m=torch.cat([self.enc[0],self.enc[2],self.enc[1]],dim=1)
        return p,j,m
    def __call__(self,v,a,f):
        p,j,m=self.features(v,a,f)
        if self.e is None:return p
        score=e_score(self.e,p,j)
        return score if self.u is None else score+u_delta(self.u,m).double()
    def __exit__(self,typ,value,tb):
        for hook in self.hooks:hook.remove()
        self.hooks.clear();self.enc.clear();self.joint.clear();self.active=False
        if hasattr(self,'scope'):return self.scope.__exit__(typ,value,tb)
