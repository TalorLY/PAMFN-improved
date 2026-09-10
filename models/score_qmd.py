"""Training-only score distillation; no parameters or inference changes."""
import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def rng_state():
    return (torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def set_rng_state(state):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


@contextmanager
def private_bn_buffers(model):
    """Swap buffers, not copy_ them: both forward graphs must remain valid."""
    saved = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for name in ('running_mean', 'running_var', 'num_batches_tracked'):
                value = getattr(module, name)
                if value is not None:
                    saved.append((module, name, value))
                    setattr(module, name, value.clone())
    try:
        yield
    finally:
        for module, name, value in saved:
            setattr(module, name, value)


def paired_masked_forward(model, masked_input, before_full):
    after_full = rng_state()
    try:
        set_rng_state(before_full)
        with private_bn_buffers(model):
            return model(masked_input)
    finally:
        set_rng_state(after_full)


def ramp(epoch, epochs, warmup=0.1, rampup=0.1):
    warm = int(math.ceil(epochs * warmup))
    rise = int(math.ceil(epochs * rampup))
    if epoch < warm:
        return 0.0
    return min(1.0, (epoch - warm + 1) / max(1, rise))


class MeanSegmentMask:
    """One shuffled/balanced modality per batch; private CPU randomness."""
    def __init__(self, seed, ratio=0.15):
        if not 0 < ratio < 1:
            raise ValueError('mask ratio must be between zero and one')
        self.ratio = ratio
        self.generator = torch.Generator(device='cpu')
        self.generator.manual_seed(int(seed) + 20260909)
        self.pending = []

    def __call__(self, inputs, lengths):
        if not self.pending:
            self.pending = torch.randperm(3, generator=self.generator).tolist()
        modality = ('V', 'F', 'A')[self.pending.pop()]
        value = inputs[modality]
        valid = torch.as_tensor(lengths[modality], device='cpu').reshape(-1)
        if value.ndim != 3 or len(valid) != value.shape[0]:
            raise ValueError('expected [batch,time,features] and one valid length per sample')
        masked = value.clone()
        spans = []
        for index, count in enumerate(valid.tolist()):
            count = int(count)
            if count < 1 or count > value.shape[1]:
                raise ValueError('invalid valid length; padding must be excluded')
            width = min(count, max(1, int(math.ceil(count * self.ratio))))
            start = int(torch.randint(count - width + 1, (1,), generator=self.generator))
            masked[index, start:start + width] = value[index, :count].mean(dim=0)
            spans.append((start, width, count))
        result = dict(inputs)
        result[modality] = masked
        return result, {'modality': modality, 'spans': spans}


def score_kd(masked_score, full_score):
    return F.smooth_l1_loss(masked_score.reshape(-1), full_score.detach().reshape(-1))


def reliable_rank_kd(masked_score, full_score, label, gap=0.05, gamma=0.1):
    """Train-only, unique i<j pairs; retain absolute teacher confidence.

    Dividing by eligible pair count (not sum of weights) keeps a uniformly
    unreliable teacher weak. No teacher/label/confidence gradient is allowed.
    """
    student = masked_score.reshape(-1)
    teacher, target = full_score.detach().reshape(-1), label.detach().reshape(-1)
    if len(student) != len(teacher) or len(student) != len(target):
        raise ValueError('rank distillation requires matching sample counts')
    if not math.isfinite(gap) or gap < 0 or not math.isfinite(gamma) or gamma <= 0:
        raise ValueError('rank gap must be nonnegative and gamma positive')
    pairs = torch.triu_indices(len(student), len(student), offset=1, device=student.device)
    i, j = pairs[0], pairs[1]
    teacher_diff, target_diff = teacher[i] - teacher[j], target[i] - target[j]
    eligible = (target_diff.abs() > gap) & (teacher_diff * target_diff > 0)
    count = int(eligible.sum())
    if count == 0:
        return student.sum() * 0.0, dict(eligible_pairs=0, total_pairs=len(i), mean_reliability=0.0)
    i, j = i[eligible], j[eligible]
    reliability = torch.exp(-(teacher - target).abs() / gamma)
    weights = torch.min(reliability[i], reliability[j])
    losses = F.smooth_l1_loss(student[i] - student[j], teacher_diff[eligible], reduction='none')
    loss = (weights * losses).mean()
    return loss, dict(eligible_pairs=count, total_pairs=len(pairs[0]),
                      mean_reliability=float(weights.mean()))


def gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    squares = [g.detach().double().square().sum() for g in gradients if g is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


class ScoreQMD:
    def __init__(self, mode, seed, epochs, weight=1.0, mask_ratio=0.15,
                 mask_gt_weight=0.5, target_ratio=0.075, beta_cap=1.0,
                 warmup=0.1, rampup=0.1, rank_weight=0.5, rank_gap=0.05, rank_gamma=0.1):
        if mode not in ('off', 'mask_gt', 'score_qmd', 'score_rank_qmd'):
            raise ValueError('invalid QMD mode')
        for value in (weight, mask_gt_weight, target_ratio, beta_cap, warmup, rampup):
            if not math.isfinite(value) or value < 0:
                raise ValueError('QMD weights and schedule must be finite and nonnegative')
        if epochs < 1 or warmup + rampup > 1:
            raise ValueError('invalid QMD schedule')
        if (not math.isfinite(rank_weight) or rank_weight < 0 or
                not math.isfinite(rank_gap) or rank_gap < 0 or
                not math.isfinite(rank_gamma) or rank_gamma <= 0):
            raise ValueError('invalid rank distillation settings')
        self.mode, self.epochs, self.weight = mode, epochs, weight
        self.rank_weight, self.rank_gap, self.rank_gamma = rank_weight, rank_gap, rank_gamma
        self.rank_pairs = self.rank_batches = self.rank_gradient_batches = 0
        self.mask_gt_weight, self.target_ratio = mask_gt_weight, target_ratio
        self.beta_cap, self.warmup, self.rampup = beta_cap, warmup, rampup
        self.mask = MeanSegmentMask(seed, mask_ratio)
        self.beta = None
        self.calibration = None
        self.active_batches = 0
        self.modality_counts = {m: 0 for m in ('V', 'F', 'A')}
        self.epoch_rows = []

    def scale(self, epoch):
        if self.mode == 'off' or self.weight == 0:
            return 0.0
        return self.weight * ramp(epoch, self.epochs, self.warmup, self.rampup)

    def auxiliary(self, model, inputs, lengths, label, full_score, full_mse,
                  before_full, epoch):
        masked, mask_info = self.mask(inputs, lengths)
        prediction, _ = paired_masked_forward(model, masked, before_full)
        gt = F.mse_loss(prediction.reshape(-1), label.reshape(-1))
        kd = score_kd(prediction, full_score)
        if self.mode in ('score_qmd', 'score_rank_qmd') and self.beta is None:
            # First active, deterministically ordered training batch after warm-up.
            # Use score-head parameter gradients shared by both view graphs.
            parameters = [p for p in model.fc.parameters() if p.requires_grad]
            main_norm = gradient_norm(full_mse, parameters)
            kd_norm = gradient_norm(kd, parameters)
            if not math.isfinite(main_norm + kd_norm):
                raise FloatingPointError('non-finite QMD calibration gradients')
            raw = self.target_ratio * main_norm / kd_norm if kd_norm > 1e-12 else 0.0
            self.beta = min(self.beta_cap, raw)
            self.calibration = {
                'epoch': epoch + 1, 'main_gradient_norm': main_norm,
                'kd_gradient_norm': kd_norm, 'uncapped_beta': raw, 'beta': self.beta,
                'target_ratio': self.target_ratio, 'beta_cap': self.beta_cap,
                'achieved_ratio_at_full_ramp': (
                    self.weight * self.beta * kd_norm / main_norm if main_norm > 1e-12 else 0.0),
                'scope': 'score_head_parameters_first_active_training_batch',
                'degenerate': main_norm <= 1e-12 or kd_norm <= 1e-12,
                'capped': raw > self.beta_cap,
            }
        beta = self.beta if self.mode in ('score_qmd', 'score_rank_qmd') else 0.0
        self.active_batches += 1
        self.modality_counts[mask_info['modality']] += 1
        loss = self.scale(epoch) * (self.mask_gt_weight * gt + beta * kd)
        diagnostics = {'mask_gt': float(gt.detach()), 'score_kd': float(kd.detach()),
                       'beta': beta, 'scale': self.scale(epoch)}
        if self.mode == 'score_rank_qmd' and self.rank_weight > 0:
            rank, info = reliable_rank_kd(prediction, full_score, label,
                                           self.rank_gap, self.rank_gamma)
            loss = loss + self.scale(epoch) * self.rank_weight * rank
            self.rank_pairs += info['eligible_pairs']
            self.rank_batches += int(info['eligible_pairs'] > 0)
            # An eligible pair with identical differences has zero rank gradient.
            # Probe only until the first verified contribution; no extra backward update.
            if self.rank_gradient_batches == 0 and float(rank.detach()) > 0:
                gradient = torch.autograd.grad(rank, prediction, retain_graph=True)[0]
                self.rank_gradient_batches += int(bool(torch.isfinite(gradient).all())
                                                  and float(gradient.abs().sum()) > 0)
            diagnostics.update(rank_kd=float(rank.detach()), **info)
        return loss, diagnostics

    def summary(self):
        result = {'mode': self.mode, 'active_batches': self.active_batches,
                'modality_counts': self.modality_counts, 'calibration': self.calibration,
                'epochs': self.epoch_rows}
        if self.mode == 'score_rank_qmd':
            result['rank'] = dict(weight=self.rank_weight, gap=self.rank_gap, gamma=self.rank_gamma,
                                  normalization='eligible_pair_count', eligible_pairs=self.rank_pairs,
                                  eligible_batches=self.rank_batches,
                                  gradient_verified=self.rank_gradient_batches > 0)
        return result
