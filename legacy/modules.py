"""Retained Ordered/E/U rules, adapted only to MCMoE feature dimensions.

All statistics are fitted once using training data.  E and U detach their
parents; Ordered deliberately does not.  No legacy trainer import is needed.
"""
import torch
from torch import nn
from torch.nn import functional as F


def _features(value, width, name='features'):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != width:
        raise ValueError('%s must have shape [N, %d]' % (name, width))
    if not value.is_floating_point() or value.shape[0] == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError('%s must be nonempty and finite floating point' % name)
    return value


def _vector(value, count, name):
    if not isinstance(value, torch.Tensor) or value.shape not in ((count,), (count, 1)):
        raise ValueError('%s must have shape [N] or [N, 1]' % name)
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError('%s must be finite floating point' % name)
    return value.reshape(-1)


class OrderedHead(nn.Module):
    """Original four-bin ordered prototype objective; worker applies 1e-4."""
    def __init__(self, input_dim=256, thresholds=None, seed=0):
        super().__init__()
        if not isinstance(input_dim, int) or input_dim <= 0 or thresholds is None:
            raise ValueError('positive input_dim and three training thresholds are required')
        cuts = torch.as_tensor(thresholds, dtype=torch.float32).detach().cpu().clone()
        if cuts.shape != (3,) or not bool(torch.isfinite(cuts).all()) or bool((cuts[1:] < cuts[:-1]).any()):
            raise ValueError('three finite nondecreasing training thresholds are required')
        self.input_dim = input_dim
        self.ordinal_prototype_temperature = .1
        self.ordinal_prototype_order_margin = .1
        self.ordinal_prototype_order_scale = .1
        # Adding this auxiliary head must not change backbone/dropout/loader RNG.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(int(seed) + 20261001)
            self.quality_projector = nn.Linear(input_dim, 128)
            self.quality_prototypes = nn.Parameter(torch.empty(4, 128))
            self.quality_axis = nn.Parameter(torch.empty(128))
            nn.init.normal_(self.quality_prototypes, std=.02)
            nn.init.normal_(self.quality_axis, std=.02)
        self.register_buffer('ordinal_thresholds', cuts)

    def forward(self, features):
        _features(features, self.input_dim)
        return F.normalize(self.quality_projector(features), dim=1)

    def loss(self, features, labels):
        embedding = self(features)
        target = _vector(labels, features.shape[0], 'labels')
        prototypes = F.normalize(self.quality_prototypes, dim=1)
        logits = embedding @ prototypes.transpose(0, 1) / self.ordinal_prototype_temperature
        bins = torch.bucketize(target.detach(), self.ordinal_thresholds)
        class_ids = torch.arange(4, device=target.device).unsqueeze(0)
        distance = (class_ids - bins.unsqueeze(1)).abs()
        soft_target = torch.where(distance == 0, torch.ones_like(distance, dtype=embedding.dtype),
                                  torch.where(distance == 1,
                                              torch.full_like(distance, .1, dtype=embedding.dtype),
                                              torch.zeros_like(distance, dtype=embedding.dtype)))
        soft_target = soft_target / soft_target.sum(dim=1, keepdim=True)
        prototype_loss = -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        axis = F.normalize(self.quality_axis, dim=0)
        positions = prototypes @ axis
        gaps = positions[1:] - positions[:-1]
        order_loss = F.relu(self.ordinal_prototype_order_margin - gaps).mean()
        return prototype_loss + self.ordinal_prototype_order_scale * order_loss


class EBranch(nn.Module):
    """v13/v17 MLP_GT: train-only bias, standardized fused feature, bounded E."""
    def __init__(self, input_dim=256, seed=0):
        super().__init__()
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError('positive feature dimension required')
        self.input_dim = input_dim
        self.config = dict(input_dim=input_dim, hidden=16, bound=.03, feature_clip=5., std_floor=.01)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(int(seed) + 20260915)
            self.net = nn.Sequential(nn.Linear(input_dim, 16), nn.GELU(), nn.Linear(16, 1, bias=False))
            nn.init.zeros_(self.net[-1].weight)
        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('feature_scale', torch.ones(input_dim))
        self.register_buffer('global_bias', torch.zeros(()))
        self.register_buffer('calibrated', torch.tensor(False))

    def fit(self, base, features, labels):
        if bool(self.calibrated):
            raise RuntimeError('training statistics are immutable once fitted')
        features = _features(features, self.input_dim).detach()
        count = features.shape[0]
        base = _vector(base, count, 'base').detach()
        labels = _vector(labels, count, 'labels').detach()
        if count < 2:
            raise ValueError('at least two training samples required')
        with torch.no_grad():
            self.feature_mean.copy_(features.double().mean(0))
            self.feature_scale.copy_(features.double().std(0, unbiased=False).clamp_min(.01))
            self.global_bias.copy_((labels.double() - base.double()).mean())
            self.calibrated.fill_(True)
        return dict(source_split='train', samples=count, passes=1,
                    global_bias=float(self.global_bias), mean_label=float(labels.double().mean()),
                    mean_anchor=float(base.double().mean()), feature_dim=self.input_dim,
                    feature_scale_min=float(self.feature_scale.min()),
                    feature_scale_max=float(self.feature_scale.max()))

    def forward(self, features):
        if not bool(self.calibrated):
            raise RuntimeError('fit training statistics before correction')
        features = _features(features, self.input_dim)
        x = ((features.detach() - self.feature_mean) / self.feature_scale).clamp(-5., 5.)
        return .03 * torch.tanh(self.net(x).reshape(-1))

    def score(self, base, features):
        base = _vector(base, features.shape[0], 'base')
        return base.detach().reshape(-1) + self.global_bias + self(features)

    def evaluation_score(self, base, features):
        base = _vector(base, features.shape[0], 'base')
        residual = self(features).double()
        return base.detach().reshape(-1).double() + self.global_bias.double() + residual

    def optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=.0001, weight_decay=.01)


class UBranch(nn.Module):
    """Retained v30 unbounded_s: modal V/F/A heads, fixed mean, no tanh or KD.

    `bound` is the historical buffer name for a train-calibrated scale factor;
    it is not a hard bound on the unbounded heads' correction.
    """
    def __init__(self, input_dim=768, seed=0):
        super().__init__()
        if not isinstance(input_dim, int) or input_dim <= 0 or input_dim % 3:
            raise ValueError('positive modal dimension divisible by three required')
        self.input_dim = input_dim
        self.config = dict(input_dim=input_dim, hidden=16, modalities=['V', 'F', 'A'],
                           abs_bound=.003, relative_ratio=.1, std_floor=.01, feature_clip=5.,
                           paper_control='unbounded_s', combination='fixed mean of three heads')
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(int(seed) + 20260916)
            heads = []
            for _ in range(3):
                head = nn.Sequential(nn.Linear(input_dim // 3, 16), nn.GELU(), nn.Linear(16, 1, bias=False))
                nn.init.zeros_(head[-1].weight)
                heads.append(head)
            self.heads = nn.ModuleList(heads)
        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('feature_scale', torch.ones(input_dim))
        self.register_buffer('bound', torch.zeros(()))
        self.register_buffer('calibrated', torch.tensor(False))

    def fit(self, e_scores, modal_features):
        if bool(self.calibrated):
            raise RuntimeError('training statistics are immutable once fitted')
        features = _features(modal_features, self.input_dim, 'modal_features').detach()
        count = features.shape[0]
        e_scores = _vector(e_scores, count, 'e_scores').detach()
        if count < 2:
            raise ValueError('at least two training samples required')
        spread = float(e_scores.double().std(unbiased=False))
        with torch.no_grad():
            self.feature_mean.copy_(features.double().mean(0))
            self.feature_scale.copy_(features.double().std(0, unbiased=False).clamp_min(.01))
            self.bound.fill_(min(.003, .1 * spread))
            self.calibrated.fill_(True)
        return dict(source_split='train', samples=count, passes=1, feature_dim=self.input_dim,
                    e_prediction_std=spread, bound=float(self.bound),
                    feature_scale_min=float(self.feature_scale.min()),
                    feature_scale_max=float(self.feature_scale.max()),
                    feature_source='modal', new_global_bias=False, paper_control='unbounded_s')

    def components(self, modal_features):
        if not bool(self.calibrated):
            raise RuntimeError('fit training statistics before correction')
        features = _features(modal_features, self.input_dim, 'modal_features')
        x = ((features.detach() - self.feature_mean) / self.feature_scale).clamp(-5., 5.)
        return self.bound * torch.cat([head(value) for head, value in zip(self.heads, x.chunk(3, dim=1))], dim=1)

    def forward(self, modal_features):
        return self.components(modal_features).mean(1)

    def optimizer(self):
        return torch.optim.AdamW([dict(params=list(self.heads.parameters()), lr=.0001, weight_decay=.01)])


def centered_loss(corrected, label, residual):
    if not isinstance(corrected, torch.Tensor) or corrected.ndim != 1 or corrected.numel() == 0:
        raise ValueError('corrected scores must be a nonempty vector')
    count = corrected.numel()
    corrected = _vector(corrected, count, 'corrected')
    label = _vector(label, count, 'labels').detach()
    residual = _vector(residual, count, 'residual')
    centered = F.mse_loss(corrected - corrected.mean(), label.reshape(-1) - label.mean())
    zero_mean = residual.mean().square()
    return centered + .1 * zero_mean, centered, zero_mean


def _report(loss, centered, mean_square, residual):
    return dict(loss=float(loss.detach()), centered_mse=float(centered.detach()),
                mean_penalty=.1 * float(mean_square.detach()),
                residual_mean_penalty=float(mean_square.detach()),
                residual_mean=float(residual.detach().mean()),
                residual_std=float(residual.detach().std(unbiased=False)))


def e_objective(e, base, features, labels):
    """Float32 training exactly as retained E; only E receives gradients."""
    base = _vector(base, features.shape[0], 'base').detach().float()
    labels = _vector(labels, features.shape[0], 'labels').detach().float()
    residual = e(features)
    corrected = base + e.global_bias + residual
    loss, centered, mean_square = centered_loss(corrected, labels, residual)
    return loss, _report(loss, centered, mean_square, residual)


def u_objective(u, frozen_e, modal_features, labels):
    """Double score accumulation as retained U; frozen_e is a score tensor."""
    count = modal_features.shape[0]
    frozen_e = _vector(frozen_e, count, 'frozen_e').detach().double()
    labels = _vector(labels, count, 'labels').detach().double()
    residual = u(modal_features).double()
    corrected = frozen_e + residual
    loss, centered, mean_square = centered_loss(corrected, labels, residual)
    return loss, _report(loss, centered, mean_square, residual)
