import copy
import io
import unittest
from unittest.mock import patch

import torch

from models.coqm import (
    ContinuousOrderedQualityManifold,
    preserve_torch_rng_state,
)
from models.pamfn import BaseModel, PAMFN


class COQMRNGIsolationTest(unittest.TestCase):
    @staticmethod
    def _build_pamfn(seed, **overrides):
        with preserve_torch_rng_state():
            torch.manual_seed(991)
            pretrained = BaseModel(4, 4, 0.1)
        torch.manual_seed(seed)
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
        }
        config.update(overrides)
        with patch(
                "models.pamfn._torch_load",
                side_effect=lambda _: copy.deepcopy(pretrained)):
            model = PAMFN(**config)
        return model, torch.get_rng_state().clone()

    def test_standalone_head_does_not_advance_global_rng(self):
        torch.manual_seed(19)
        before = torch.get_rng_state().clone()
        ContinuousOrderedQualityManifold(12, embedding_dim=8)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_coqm_preserves_main_path_state_and_following_rng(self):
        base, base_rng = self._build_pamfn(23)
        coqm, coqm_rng = self._build_pamfn(
            23, coqm_weight=1e-4, quality_aux_rng_isolated=True
        )
        self.assertTrue(torch.equal(base_rng, coqm_rng))
        base_state = base.state_dict()
        coqm_state = coqm.state_dict()
        for name, value in base_state.items():
            self.assertTrue(torch.equal(value, coqm_state[name]))
        self.assertFalse(any(name.startswith("coqm.") for name in base_state))

    def test_rng_isolated_ordered_comparator_is_also_causal(self):
        base, base_rng = self._build_pamfn(29)
        ordered, ordered_rng = self._build_pamfn(
            29,
            ordinal_prototype_weight=1e-4,
            ordinal_quality_thresholds=(0.25, 0.5, 0.75),
            quality_aux_rng_isolated=True,
        )
        self.assertTrue(torch.equal(base_rng, ordered_rng))
        ordered_state = ordered.state_dict()
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, ordered_state[name]))

    def test_state_dict_round_trip_preserves_outputs(self):
        source = ContinuousOrderedQualityManifold(6, embedding_dim=4)
        buffer = io.BytesIO()
        torch.save(source.state_dict(), buffer)
        buffer.seek(0)
        restored = ContinuousOrderedQualityManifold(6, embedding_dim=4)
        restored.load_state_dict(torch.load(buffer, map_location="cpu"))
        features = torch.randn(3, 6)
        source_output = source(features)
        restored_output = restored(features)
        for name in source_output:
            self.assertTrue(torch.allclose(
                source_output[name], restored_output[name]
            ))

    def test_integrated_model_forward_and_training_loss(self):
        model, _ = self._build_pamfn(
            31, coqm_weight=1e-4, quality_aux_rng_isolated=True
        )
        captured = {}

        def capture_stage3(_, values):
            captured["stage3"] = values[0]

        hook = model.fc.register_forward_pre_hook(capture_stage3)
        inputs = {
            "V": torch.randn(2, 16, 4),
            "F": torch.randn(2, 16, 4),
            "A": torch.randn(2, 16, 4),
        }
        labels = torch.tensor([0.2, 0.8])
        prediction, info = model(inputs)
        hook.remove()
        stage3 = captured["stage3"]
        score_loss = model.mse(prediction.reshape(-1), labels)
        auxiliary_loss = model.coqm.loss_components(
            info["coqm_output"], labels
        )["total"]
        score_gradient = torch.autograd.grad(
            score_loss, stage3, retain_graph=True
        )[0]
        auxiliary_gradient = torch.autograd.grad(
            auxiliary_loss, stage3, retain_graph=True
        )[0]
        loss = model.call_training_loss(prediction, labels, **info)
        loss.backward()
        self.assertEqual(tuple(prediction.shape), (2, 1))
        self.assertIn("coqm_output", info)
        self.assertTrue(torch.isfinite(score_gradient).all())
        self.assertTrue(torch.isfinite(auxiliary_gradient).all())
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
