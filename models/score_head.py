"""Two opt-in final-head ablations and passive, full-view diagnostics."""
import numpy as np
from torch import nn


VARIANTS = ('original', 'linear', 'groupnorm')


def configure_head(fc, variant):
    # Build the original head first: identical Conv initialization/RNG in all arms.
    if variant not in VARIANTS:
        raise ValueError('unknown score head: ' + str(variant))
    if variant == 'linear':
        fc[6] = nn.Identity()
    elif variant == 'groupnorm':
        fc[2] = nn.GroupNorm(1, fc[2].num_features, eps=fc[2].eps, affine=True)
    return fc


def forward_with_pre_output(fc, features):
    value = features
    for i in range(len(fc) - 1):
        value = fc[i](value)
    return fc[-1](value), value.detach()


def describe(values):
    x = np.asarray(values, dtype=np.float64)
    return dict(count=int(x.size), mean=float(x.mean()), std=float(x.std()),
                min=float(x.min()), max=float(x.max()))


def head_diagnostics(model, pre_output, predictions, labels):
    """No new forwards, randomness, gradients or updates to normalization buffers."""
    pre = np.asarray(pre_output, dtype=np.float64)
    pred, target = np.asarray(predictions, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    norm = model.fc[2]
    bounded = isinstance(model.fc[-1], nn.Sigmoid)
    result = dict(variant=model.score_head, normalization=type(norm).__name__,
                  output=type(model.fc[-1]).__name__,
                  trainable_parameters=sum(p.numel() for p in model.fc.parameters() if p.requires_grad),
                  pre_output=describe(pre), prediction=describe(pred), target=describe(target),
                  pre_abs_gt6_fraction=float(np.mean(np.abs(pre) > 6)),
                  mse=float(np.mean((pred - target)**2)),
                  outside_unit_interval_fraction=float(np.mean((pred < 0) | (pred > 1))),
                  saturation_fraction=float(np.mean((pred < .001) | (pred > .999))) if bounded else None)
    if isinstance(norm, nn.BatchNorm1d):
        result['running_mean'] = describe(norm.running_mean.detach().cpu().numpy())
        result['running_var'] = describe(norm.running_var.detach().cpu().numpy())
        result['num_batches_tracked'] = int(norm.num_batches_tracked.item())
    return result
