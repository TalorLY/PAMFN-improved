"""Continuous ordered quality manifold used as a train-only regularizer."""

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def preserve_torch_rng_state():
    """Restore CPU and already-initialized CUDA RNG states on exit."""
    cpu_state = torch.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_states = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class ContinuousOrderedQualityManifold(nn.Module):
    """Map a video representation to a learned one-dimensional arc."""

    def __init__(
            self, input_dim, embedding_dim=128, num_anchors=4,
            temperature=0.1, theta_max=math.pi / 3,
            anchor_scores=None):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least two")
        if num_anchors < 2:
            raise ValueError("num_anchors must be at least two")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if (
                not math.isfinite(theta_max)
                or theta_max <= 0
                or theta_max > math.pi / 2):
            raise ValueError(
                "theta_max must be in the interval (0, pi / 2]"
            )

        # Linear uses the run's current initialization stream, but this
        # auxiliary branch must not shift later shuffle/dropout randomness.
        with preserve_torch_rng_state():
            self.projector = nn.Linear(input_dim, embedding_dim)

        raw_u = torch.zeros(embedding_dim)
        raw_v = torch.zeros(embedding_dim)
        raw_u[0] = 1.0
        raw_v[1] = 1.0
        self.raw_u = nn.Parameter(raw_u)
        self.raw_v = nn.Parameter(raw_v)
        self.temperature = float(temperature)
        self.theta_max = float(theta_max)
        self.num_anchors = int(num_anchors)
        if anchor_scores is None:
            anchor_scores = torch.linspace(
                0.0, 1.0, self.num_anchors
            )
        else:
            anchor_scores = torch.as_tensor(
                anchor_scores, dtype=torch.float32
            ).reshape(-1)
            if anchor_scores.numel() != self.num_anchors:
                raise ValueError(
                    "anchor_scores must match num_anchors"
                )
            if not bool(torch.isfinite(anchor_scores).all()):
                raise ValueError("anchor_scores must be finite")
            if not bool(
                    torch.all(anchor_scores[1:] > anchor_scores[:-1])):
                raise ValueError(
                    "anchor_scores must be strictly increasing"
                )
            if float(anchor_scores[0]) < 0.0 or (
                    float(anchor_scores[-1]) > 1.0):
                raise ValueError(
                    "anchor_scores must lie in [0, 1]"
                )
        self.register_buffer(
            "anchor_scores",
            anchor_scores.clone().detach(),
        )

    def orthogonal_basis(self):
        u = F.normalize(self.raw_u, dim=0, eps=1e-6)
        v = self.raw_v - torch.sum(self.raw_v * u) * u
        v = F.normalize(v, dim=0, eps=1e-6)
        return u, v

    def manifold_point(self, score):
        score = score.reshape(-1)
        u, v = self.orthogonal_basis()
        theta = (2.0 * score - 1.0) * self.theta_max
        return (
            torch.cos(theta).unsqueeze(1) * u.unsqueeze(0)
            + torch.sin(theta).unsqueeze(1) * v.unsqueeze(0)
        )

    def continuous_targets(self, score, dtype=None):
        """Interpolate between the two adjacent fixed anchor scores."""
        score = score.detach().reshape(-1)
        centers = self.anchor_scores.to(
            device=score.device, dtype=score.dtype
        )
        right = torch.bucketize(score, centers)
        right = right.clamp(1, self.num_anchors - 1)
        left = right - 1
        lower = centers[left]
        upper = centers[right]
        fraction = ((score - lower) / (upper - lower)).clamp(0, 1)
        target_dtype = dtype or score.dtype
        left_target = F.one_hot(
            left, num_classes=self.num_anchors
        ).to(target_dtype)
        right_target = F.one_hot(
            right, num_classes=self.num_anchors
        ).to(target_dtype)
        return (
            left_target * (1.0 - fraction).to(target_dtype).unsqueeze(1)
            + right_target * fraction.to(target_dtype).unsqueeze(1)
        )

    def forward(self, stage3_feature):
        if stage3_feature.ndim != 2:
            raise ValueError("stage3_feature must have shape [batch, dim]")
        embedding = F.normalize(
            self.projector(stage3_feature), dim=1, eps=1e-6
        )
        prototypes = self.manifold_point(
            self.anchor_scores.to(dtype=embedding.dtype)
        )
        logits = (
            embedding @ prototypes.transpose(0, 1)
            / self.temperature
        )
        probabilities = torch.softmax(logits, dim=1)
        expected_quality = torch.sum(
            probabilities
            * self.anchor_scores.to(dtype=probabilities.dtype).unsqueeze(0),
            dim=1,
        )
        return {
            "embedding": embedding,
            "prototypes": prototypes,
            "logits": logits,
            "probabilities": probabilities,
            "expected_quality": expected_quality,
        }

    def loss_components(self, output, target, coord_scale=1.0):
        if not math.isfinite(coord_scale) or coord_scale < 0:
            raise ValueError("coord_scale must be finite and non-negative")
        target = target.reshape(-1)
        embedding = output["embedding"]
        curve_target = self.manifold_point(
            target.to(dtype=embedding.dtype)
        )
        curve = (
            1.0 - torch.sum(embedding * curve_target, dim=1)
        ).mean()
        soft_target = self.continuous_targets(
            target, dtype=embedding.dtype
        )
        anchor = -(
            soft_target * F.log_softmax(output["logits"], dim=1)
        ).sum(dim=1).mean()
        coord = F.smooth_l1_loss(
            output["expected_quality"],
            target.to(dtype=output["expected_quality"].dtype),
        )
        total = curve + anchor + float(coord_scale) * coord
        return {
            "total": total,
            "curve": curve,
            "anchor": anchor,
            "coord": coord,
        }
