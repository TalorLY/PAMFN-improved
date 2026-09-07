import unittest

import torch

from models.coqm import ContinuousOrderedQualityManifold


class COQMTargetsTest(unittest.TestCase):
    def setUp(self):
        self.model = ContinuousOrderedQualityManifold(
            6, embedding_dim=4, num_anchors=4
        )

    def test_anchor_boundaries_are_exact_one_hot_targets(self):
        target = self.model.continuous_targets(
            self.model.anchor_scores
        )
        self.assertTrue(torch.allclose(target, torch.eye(4)))

    def test_midpoint_interpolates_only_adjacent_anchors(self):
        target = self.model.continuous_targets(torch.tensor([0.5]))
        self.assertTrue(torch.allclose(
            target, torch.tensor([[0.0, 0.5, 0.5, 0.0]])
        ))

    def test_out_of_range_scores_are_clamped_to_end_anchors(self):
        target = self.model.continuous_targets(
            torch.tensor([-0.1, 1.1])
        )
        self.assertTrue(torch.allclose(
            target,
            torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
        ))

    def test_no_coord_ablation_removes_only_coordinate_term(self):
        labels = torch.tensor([0.1, 0.7])
        output = self.model(torch.randn(2, 6))
        losses = self.model.loss_components(
            output, labels, coord_scale=0.0
        )
        self.assertTrue(torch.allclose(
            losses["total"], losses["curve"] + losses["anchor"]
        ))


if __name__ == "__main__":
    unittest.main()
