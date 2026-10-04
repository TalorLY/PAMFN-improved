"""Read-only hooks into the pinned native MCMoE. No replacement forward path."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch
from torch import nn
from spec import HERE
from modules import OrderedHead

def seed_all(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)

def capture_rng():
    return (torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            np.random.get_state(),random.getstate())

def restore_rng(state):
    torch.set_rng_state(state[0])
    if torch.cuda.is_available() and state[1]: torch.cuda.set_rng_state_all(state[1])
    np.random.set_state(state[2]); random.setstate(state[3])

@contextmanager
def preserved_rng(state=None):
    before=capture_rng()
    try:
        if state is not None: restore_rng(state)
        yield
    finally: restore_rng(before)

def build_native(cpu=False):
    path=str(HERE/'upstream'/'mcmoe')
    if path not in sys.path: sys.path.insert(0,path)
    from models.model import MoE_AQA
    if cpu:
        from unittest.mock import patch
        with patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self):
            return MoE_AQA(768,256,2,3,3,4,.3,None)
    return MoE_AQA(768,256,2,3,3,4,.3,None)

class TransferModel(nn.Module):
    def __init__(self, ordered=False, thresholds=None, seed=0, cpu=False):
        super().__init__()
        self.backbone=build_native(cpu)
        self.quality=OrderedHead(input_dim=256,thresholds=thresholds,seed=seed) if ordered else None

    def forward(self,*args,**kwargs): return self.backbone(*args,**kwargs)

class TransferObjective(nn.Module):
    def __init__(self,model,weight=1e-4):
        super().__init__()
        from models.loss import LossFun
        self.native=LossFun(alpha=1.,margin=1.)
        # Avoid registering the model as an objective child or duplicating parameter ownership.
        object.__setattr__(self,'model',model)
        self.weight=weight; self.last_ordinal=None

    def forward(self,pred,label,feat,feat2,recon_loss,args):
        result=self.native(pred,label,feat,feat2,recon_loss,args)
        if not torch.isfinite(result):raise FloatingPointError('Nonfinite native MCMoE objective')
        if self.model.quality is None or self.weight==0:
            self.last_ordinal=0.; return result
        ordinal=self.model.quality.loss(feat.mean(1),label.reshape(-1))
        if not torch.isfinite(ordinal):raise FloatingPointError('Nonfinite Ordered objective')
        self.last_ordinal=float(ordinal.detach())
        return result+self.weight*ordinal

def freeze(model):
    model.eval()
    # Parameter requires_grad metadata can affect floating-point dispatch even
    # under no_grad (observed in native cross-attention on PyTorch 2.9 CPU).
    # Retain the checkpoint's original evaluation metadata before freezing;
    # repeated freeze() calls must not replace it with all-False flags.
    if not hasattr(model,'_v38_native_requires_grad'):
        object.__setattr__(model,'_v38_native_requires_grad',
                           {name:p.requires_grad for name,p in model.named_parameters()})
    for p in model.parameters(): p.requires_grad_(False); p.grad=None
    return model

def assert_frozen(model):
    if any(m.training for m in model.modules()) or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError('Frozen parent unexpectedly trainable or in train mode')

def state_digest(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

@contextmanager
def historical_inference(model):
    """No-gradient native inference with original parameter dispatch metadata.

    This does not train/unfreeze the parent: autograd is disabled for the entire
    scope, no optimizer runs, and all False flags are restored even on error.
    It preserves exact selected-checkpoint predictions instead of relaxing the
    parent replay comparison to a numerical tolerance.
    """
    assert_frozen(model)
    parameters=dict(model.named_parameters())
    flags=getattr(model,'_v38_native_requires_grad',None)
    if flags is None or set(flags)!=set(parameters):
        raise RuntimeError('Frozen parent lacks original inference metadata')
    with torch.no_grad():
        try:
            for name,p in parameters.items():p.requires_grad_(flags[name])
            yield
        finally:
            for p in parameters.values():p.requires_grad_(False)

def frozen_view(model, video,audio,flow):
    assert_frozen(model); native=model.backbone; enc=[]; joint=[]; hooks=[]
    try:
        hooks.append(native.transformer.encoder.register_forward_hook(lambda m,a,o:enc.append(o.detach().mean(1))))
        hooks.append(native.regressor.register_forward_pre_hook(lambda m,a:joint.append(a[0].detach().mean(1))))
        with historical_inference(model): output=model(video,audio,flow,[1,1,1])['output'].reshape(-1)
    finally:
        for hook in hooks: hook.remove()
    if len(enc)!=3 or len(joint)!=1: raise ValueError('MCMoE native hook path changed')
    fused=joint[0]; modal=torch.cat([enc[0],enc[2],enc[1]],dim=1) # native calls V,A,F; expose V,F,A
    if fused.shape!=(output.numel(),256) or modal.shape!=(output.numel(),768): raise ValueError('Feature dimensions changed')
    if not all(torch.isfinite(v).all() for v in (output,fused,modal)): raise FloatingPointError('Nonfinite frozen view')
    return output.detach(),fused,modal

def tensor_save(path,value):
    path=Path(path); temp=path.with_name(path.name+'.tmp')
    torch.save(value,str(temp)); os.replace(str(temp),str(path))

def load(path):
    try: return torch.load(str(path),map_location='cpu',weights_only=False)
    except TypeError: return torch.load(str(path),map_location='cpu')
