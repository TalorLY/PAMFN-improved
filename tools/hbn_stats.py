"""Train-only, head-local statistics; compatible with PyTorch 1.8/Python 3.8.

No optimizer, EMA, labels, backbone normalization updates or routing changes.
This module is also loaded beside the untouched July28 repository.
"""
from contextlib import contextmanager
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ALPHAS = (0.0, 0.1, 0.25)
CALIBRATION_SEED = 20260912


def recipe():
    return dict(alphas=list(ALPHAS), target='fc.2', source='train_only_legacy_crop',
                seed=CALIBRATION_SEED, batch_size=32, passes=1,
                variance='population_moments_distribution_mixture',
                upstream='unchanged_eval_mode_and_original_routing',
                training=False, checkpoint_selection='fixed_original_checkpoint')


@contextmanager
def isolated_rng(seed=None):
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if cuda is not None:
                torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def head_bn(model):
    bn = model.fc[2]
    if not isinstance(bn, nn.BatchNorm1d) or not bn.track_running_stats:
        raise ValueError('requires original final-head BatchNorm1d at fc.2')
    if not isinstance(model.fc[-1], nn.Sigmoid):
        raise ValueError('v7 preserves the original sigmoid scoring head')
    return bn


class ChannelMoments:
    """Sample-weighted Chan/Welford moments, not a mean of unequal batch variances."""
    def __init__(self):
        self.count, self.mean, self.m2 = 0, None, None

    def observe(self, value):
        x = value.detach().movedim(1, 0).reshape(value.shape[1], -1).double().cpu()
        if not torch.isfinite(x).all() or x.shape[1] == 0:
            raise ValueError('nonfinite or empty training activations')
        n, mean = x.shape[1], x.mean(1)
        m2 = ((x - mean[:, None]) ** 2).sum(1)
        if self.count == 0:
            self.count, self.mean, self.m2 = n, mean, m2
            return
        delta, total = mean - self.mean, self.count + n
        self.m2 += m2 + delta.square() * (self.count * n / total)
        self.mean += delta * (n / total)
        self.count = total

    def result(self):
        if self.count < 2:
            raise ValueError('need at least two training activation observations')
        return dict(count=self.count, mean=self.mean.clone(), var=self.m2 / self.count)


@torch.no_grad()
def collect_head_stats(model, dataset, prepare_batch, progress=None):
    if getattr(dataset, 'is_train', None) is not True:
        raise ValueError('only an explicitly marked training dataset may calibrate')
    if len(dataset) < 2:
        raise ValueError('need at least two training samples')
    bn, moments = head_bn(model), ChannelMoments()
    modes = [(m, m.training) for m in model.modules()]
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    versions = {n: p._version for n, p in model.named_parameters()}
    hook = bn.register_forward_pre_hook(lambda module, inputs: moments.observe(inputs[0]))
    seen = batches = 0
    try:
        model.eval()  # INCLUDING target BN: read inputs; never run BN in train mode.
        with isolated_rng(CALIBRATION_SEED):
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
            for batch in loader:
                inputs, _ = prepare_batch(batch, next(model.parameters()).device)
                model(inputs)  # No access to target scores for calibration.
                seen += next(iter(inputs.values())).shape[0]
                batches += 1
                if progress:
                    progress('calibrate', batches, len(loader))
    finally:
        hook.remove()
        for module, mode in modes:
            module.training = mode
    if seen != len(dataset):
        raise RuntimeError('training coverage mismatch')
    if any(not torch.equal(b, buffers[n]) for n, b in model.named_buffers()):
        raise RuntimeError('statistics collection changed model buffers')
    if any(p._version != versions[n] for n, p in model.named_parameters()):
        raise RuntimeError('statistics collection changed parameters')
    result = moments.result()
    result.update(samples=seen, batches=batches, layers=1, source='train')
    return result


def blended_stats(bn, stats, alpha):
    if alpha not in ALPHAS:
        raise ValueError('alpha outside the fixed v7 recipe')
    old_mean, old_var = bn.running_mean.detach().double().cpu(), bn.running_var.detach().double().cpu()
    mean, var = stats['mean'].double().cpu(), stats['var'].double().cpu()
    if (mean.shape != old_mean.shape or var.shape != old_var.shape
            or not all(torch.isfinite(x).all() for x in (mean, var, old_mean, old_var))
            or (var < 0).any() or (old_var < 0).any()):
        raise ValueError('invalid head moments')
    # Mixture of two distributions includes between-mean variance.
    mixed_mean = old_mean + alpha * (mean - old_mean)
    mixed_var = ((1-alpha)*old_var + alpha*var
                 + alpha*(1-alpha)*(old_mean-mean).square())
    return dict(mean=mixed_mean.to(bn.running_mean), var=mixed_var.to(bn.running_var))


@contextmanager
def use_head_stats(model, values):
    bn = head_bn(model)
    old_mean, old_var = bn.running_mean.clone(), bn.running_var.clone()
    try:
        with torch.no_grad():
            bn.running_mean.copy_(values['mean'])
            bn.running_var.copy_(values['var'])
        yield
    finally:
        with torch.no_grad():
            bn.running_mean.copy_(old_mean)
            bn.running_var.copy_(old_var)


@torch.no_grad()
def score_cached_head(model, cached_inputs):
    """Each cached batch retains its original shape/order; no new backbone pass."""
    predictions, logits = [], []
    device = next(model.parameters()).device
    for original in cached_inputs:
        value = original.to(device)
        for i in range(2, len(model.fc)-1):
            value = model.fc[i](value)
        logits.extend(value.detach().cpu().reshape(-1).tolist())
        predictions.extend(model.fc[-1](value).detach().cpu().reshape(-1).tolist())
    return predictions, logits
