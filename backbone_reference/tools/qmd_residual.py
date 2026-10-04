"""v12: frozen Ordered anchor and bounded, sample-dependent score correction.

No change to PAMFN, its BN buffers, or its original training implementation.
Only the residual branch receives gradients. The anchor remains stochastic
under eval (legacy Gumbel); paired/replayed RNG is therefore essential.
"""
from contextlib import contextmanager
import hashlib
import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.score_qmd import MeanSegmentMask, ramp, rng_state, set_rng_state


@contextmanager
def preserved_rng(replay=None):
    saved = (rng_state(), np.random.get_state(), random.getstate())
    try:
        if replay is not None:
            set_rng_state(replay[0]); np.random.set_state(replay[1])
        yield
    finally:
        set_rng_state(saved[0]); np.random.set_state(saved[1]); random.setstate(saved[2])


def freeze_anchor(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False); p.grad = None
    return model


def assert_frozen(model):
    if any(m.training for m in model.modules()):
        raise RuntimeError('Ordered anchor entered train mode')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError('gradient reached the frozen Ordered anchor')


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def anchor_view(model, inputs):
    """Capture the existing fc input without editing the historical forward."""
    assert_frozen(model)
    captured = []
    hook = model.fc.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    try:
        with torch.no_grad():
            score, _ = model(inputs)
    finally:
        hook.remove()
    if len(captured) != 1 or captured[0].ndim != 3 or captured[0].shape[2] != 1:
        raise ValueError('expected exactly one pooled [batch,channels,1] score-head input')
    return score.reshape(-1).detach(), captured[0].squeeze(2).detach()


class ResidualBranch(nn.Module):
    def __init__(self, input_dim=768, hidden=64, bound=.1, seed=0):
        super().__init__()
        if input_dim < 1 or hidden < 1 or not math.isfinite(bound) or bound <= 0:
            raise ValueError('invalid branch dimensions/bound')
        self.config = dict(input_dim=input_dim, hidden=hidden, bound=bound)
        # CPU-only private initialization: creating a branch must not perturb
        # the anchor, dataloader, or the other method's random trajectory.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed + 20260914)
            self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden),
                                     nn.GELU(), nn.Linear(hidden, 1))
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.bound = bound

    def forward(self, feature):
        return self.bound * torch.tanh(self.net(feature.detach()).reshape(-1))

    def score(self, anchor_score, feature):
        # No sigmoid/clamp on the corrected score: avoid reintroducing a
        # saturated scoring head. The learned correction itself is bounded.
        return anchor_score.detach().reshape(-1) + self(feature)


def gradient_stats(main_loss, kd_loss, parameters):
    params = list(parameters)
    main = torch.autograd.grad(main_loss, params, retain_graph=True, allow_unused=True)
    kd = torch.autograd.grad(kd_loss, params, retain_graph=True, allow_unused=True)
    device = params[0].device
    zero = torch.zeros((), device=device, dtype=torch.float64)
    norm1, norm2, dot = zero, zero.clone(), zero.clone()
    for a, b in zip(main, kd):
        if a is not None: norm1 = norm1 + a.detach().double().square().sum()
        if b is not None: norm2 = norm2 + b.detach().double().square().sum()
        if a is not None and b is not None: dot = dot + (a.detach().double()*b.detach().double()).sum()
    n1, n2, d = float(norm1.sqrt()), float(norm2.sqrt()), float(dot)
    if not all(math.isfinite(v) for v in (n1,n2,d)):
        raise FloatingPointError('nonfinite residual-branch gradients')
    return dict(gt_norm=n1, kd_norm=n2, cosine=d/(n1*n2) if n1*n2 > 1e-20 else None)


class ResidualObjective:
    def __init__(self, seed, epochs, weight, smoke=False):
        if weight not in (0., .5, 1.) or epochs < 1:
            raise ValueError('v12 uses weights 0/.5/1 and positive epochs')
        self.weight, self.epochs = weight, epochs
        self.warmup = self.rampup = 0. if smoke else .1
        self.mask = MeanSegmentMask(seed, .15)
        self.beta = None
        self.calibration = None

    def losses(self, anchor, branch, inputs, label, lengths, epoch):
        before = (rng_state(), np.random.get_state())
        base, features = anchor_view(anchor, inputs)
        corrected = branch.score(base, features)
        gt = F.mse_loss(corrected, label.reshape(-1))
        scale = ramp(epoch, self.epochs, self.warmup, self.rampup)
        kd, stats = gt.new_zeros(()), dict(gt_norm=0., kd_norm=0., cosine=None)
        modality = None
        anchor_disagreement = 0.
        # The GT-only control also performs the SAME diagnostic masked forward.
        # It has zero KD coefficient but identical anchor/dropout/Gumbel RNG use.
        if scale > 0:
            masked, mask_info = self.mask(inputs, lengths)
            modality = mask_info['modality']
            with preserved_rng(before):
                masked_base, masked_features = anchor_view(anchor, masked)
            masked_score = branch.score(masked_base, masked_features)
            anchor_disagreement = float(F.smooth_l1_loss(masked_base, base))
            kd = F.smooth_l1_loss(masked_score, base.detach())
            stats = gradient_stats(gt, kd, branch.parameters())
            if self.beta is None:
                raw = .075 * stats['gt_norm']/stats['kd_norm'] if stats['kd_norm'] > 1e-12 else 0.
                self.beta = min(1., raw)
                self.calibration = dict(epoch=epoch+1, scope='all_residual_parameters_first_active_batch',
                    beta=self.beta, uncapped_beta=raw, target_ratio=.075, beta_cap=1.,
                    gt_norm=stats['gt_norm'], kd_norm=stats['kd_norm'],
                    achieved_ratio=self.weight*self.beta*stats['kd_norm']/stats['gt_norm'] if stats['gt_norm'] > 1e-12 else 0.)
        coefficient = self.weight * scale * (self.beta or 0.)
        objective = gt + coefficient * kd
        info = dict(gt=float(gt.detach()), kd=float(kd.detach()), scale=scale, coefficient=coefficient,
                    weighted_kd=coefficient*float(kd.detach()), active=scale>0,
                    modality=modality, anchor_disagreement=anchor_disagreement, **stats)
        return objective, info
