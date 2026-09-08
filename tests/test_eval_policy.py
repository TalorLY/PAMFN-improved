import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from models.coqm import preserve_torch_rng_state
from models.pamfn import BaseModel, CMFusion, PAMFN
from pipeline import test_epoch


class EvaluationPolicyTest(unittest.TestCase):
    @staticmethod
    def _pretrained_model():
        with preserve_torch_rng_state():
            torch.manual_seed(991)
            return BaseModel(4, 4, 0.1)

    @classmethod
    def _build_pamfn(cls, eval_policy="argmax", **overrides):
        config = {
            "model_dim": 4,
            "fc_drop": 0.1,
            "fc_r": 2,
            "feat_drop": 0.1,
            "K": 2,
            "ms_heads": 1,
            "cm_heads": 1,
            "ckpt_dir": ".",
            "rgb_ckpt_name": "x",
            "flow_ckpt_name": "x",
            "audio_ckpt_name": "x",
            "dataset_name": "TES",
            "eval_policy": eval_policy,
        }
        config.update(overrides)
        pretrained = cls._pretrained_model()
        with patch(
                "models.pamfn._torch_load",
                side_effect=lambda _: copy.deepcopy(pretrained)):
            return PAMFN(**config)

    @staticmethod
    def _inputs():
        generator = torch.Generator().manual_seed(812)
        return {
            "V": torch.randn(2, 16, 4, generator=generator),
            "F": torch.randn(2, 16, 4, generator=generator),
            "A": torch.randn(2, 16, 4, generator=generator),
        }

    def test_argmax_eval_is_repeatable_and_does_not_advance_rng(self):
        model = self._build_pamfn("argmax")
        model.eval()
        inputs = self._inputs()
        before = torch.get_rng_state().clone()
        first_prediction, first_info = model(inputs)
        middle = torch.get_rng_state().clone()
        second_prediction, second_info = model(inputs)
        after = torch.get_rng_state().clone()

        self.assertTrue(torch.equal(before, middle))
        self.assertTrue(torch.equal(middle, after))
        self.assertTrue(torch.equal(first_prediction, second_prediction))
        for first_action, second_action in zip(
                first_info["action"], second_info["action"]):
            self.assertTrue(torch.equal(first_action, second_action))

    def test_training_and_legacy_eval_still_use_gumbel(self):
        features = [torch.randn(2, 4, 4) for _ in range(3)]
        fusion = CMFusion(4, 2, gap=True, eval_policy="argmax")
        original = torch.nn.functional.gumbel_softmax
        with patch(
                "models.pamfn.F.gumbel_softmax",
                wraps=original) as gumbel:
            fusion.train()
            fusion.gen_fusion_kind(features, torch.tensor(10.0))
            self.assertEqual(gumbel.call_count, 1)
            fusion.eval()
            fusion.gen_fusion_kind(features, torch.tensor(10.0))
            self.assertEqual(gumbel.call_count, 1)

        legacy = CMFusion(
            4, 2, gap=True, eval_policy="legacy_gumbel"
        )
        legacy.eval()
        with patch(
                "models.pamfn.F.gumbel_softmax",
                wraps=original) as gumbel:
            legacy.gen_fusion_kind(features, torch.tensor(10.0))
            self.assertEqual(gumbel.call_count, 1)

    def test_all_multimodal_variants_share_requested_policy(self):
        variants = (
            {},
            {
                "ordinal_prototype_weight": 1e-4,
                "ordinal_quality_thresholds": (0.25, 0.5, 0.75),
                "quality_aux_rng_isolated": True,
            },
            {
                "coqm_weight": 1e-4,
                "quality_aux_rng_isolated": True,
            },
        )
        for overrides in variants:
            model = self._build_pamfn("argmax", **overrides)
            policies = [
                stage.f_fusion.eval_policy
                for stage in (model.stage1, model.stage2, model.stage3)
            ]
            self.assertEqual(policies, ["argmax"] * 3)

    def test_mc_metrics_use_mean_prediction_per_sample(self):
        model = _AlternatingPredictionModel()
        batches = [
            _batch(0.2),
            _batch(0.8),
        ]
        before = torch.get_rng_state().clone()
        result = test_epoch(
            model,
            batches,
            eval_policy="mc_gumbel",
            eval_mc_samples=2,
            eval_seed=73,
        )
        after = torch.get_rng_state().clone()

        self.assertTrue(torch.equal(before, after))
        self.assertEqual(model.forward_calls, 4)
        self.assertAlmostEqual(result["coef"], 100.0)
        self.assertAlmostEqual(result["loss_x100"], 0.0, places=6)
        self.assertEqual(result["evaluation"]["passes"], 2)
        self.assertTrue(result["evaluation"]["rng_isolated"])


class _AlternatingPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.eval_policy = "mc_gumbel"
        self.forward_calls = 0
        self.loss = nn.MSELoss()

    def forward(self, model_input):
        offset = -0.1 if self.forward_calls % 2 == 0 else 0.1
        self.forward_calls += 1
        prediction = model_input["V"].reshape(-1, 1) + offset
        return prediction + self.anchor * 0.0, {}

    def call_loss(self, prediction, label, **kwargs):
        del kwargs
        return self.loss(prediction.reshape(-1), label.reshape(-1))


def _batch(value):
    feature = torch.tensor([[[value]]], dtype=torch.float32)
    label = torch.tensor([value], dtype=torch.float32)
    return feature, feature, feature, label, None


if __name__ == "__main__":
    unittest.main()
