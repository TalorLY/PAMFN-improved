import math
import unittest

import torch

from models.coqm import ContinuousOrderedQualityManifold


class COQMGeometryTest(unittest.TestCase):
    def test_basis_and_curve_are_unit_and_orthogonal(self):
        model = ContinuousOrderedQualityManifold(12, embedding_dim=8)
        u, v = model.orthogonal_basis()
        self.assertAlmostEqual(
            float(torch.linalg.norm(u).detach()), 1.0, places=6
        )
        self.assertAlmostEqual(
            float(torch.linalg.norm(v).detach()), 1.0, places=6
        )
        self.assertAlmostEqual(
            float(torch.sum(u * v).detach()), 0.0, places=6
        )

        scores = torch.linspace(0.0, 1.0, 17)
        points = model.manifold_point(scores)
        self.assertTrue(torch.allclose(
            torch.linalg.norm(points, dim=1), torch.ones(17)
        ))

    def test_anchor_spacing_matches_fixed_arc(self):
        model = ContinuousOrderedQualityManifold(
            12, embedding_dim=8, num_anchors=4,
            theta_max=math.pi / 3,
        )
        prototypes = model.manifold_point(model.anchor_scores)
        adjacent_cosine = torch.sum(
            prototypes[:-1] * prototypes[1:], dim=1
        )
        expected = torch.full(
            (3,), math.cos(2 * math.pi / 9)
        )
        self.assertTrue(torch.allclose(adjacent_cosine, expected))

    def test_forward_and_losses_have_finite_gradients(self):
        model = ContinuousOrderedQualityManifold(12, embedding_dim=8)
        features = torch.randn(5, 12, requires_grad=True)
        labels = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0])
        output = model(features)
        losses = model.loss_components(output, labels)
        losses["total"].backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
