import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

import main as entry
from models.score_head import configure_head, forward_with_pre_output, head_diagnostics
from models.score_qmd import ScoreQMD
from pipeline import train_epoch, test_epoch
from tests import test_eval_policy as fixtures
from tools.verify_sr_qmd_runtime import frozen_module


class HeadModel(nn.Module):
    def __init__(self, variant='original', diagnostics=False):
        super().__init__()
        self.score_head, self.score_head_diagnostics = variant, diagnostics
        self.fc = nn.Sequential(nn.Dropout(.1), nn.Conv1d(4, 8, 1), nn.BatchNorm1d(8),
                                nn.ReLU(), nn.Dropout(.5), nn.Conv1d(8, 1, 1), nn.Sigmoid())
        configure_head(self.fc, variant)
        self.eval_policy = 'argmax'

    def forward(self, inputs):
        features = sum(inputs.values()).mean(1).unsqueeze(-1)
        if self.score_head_diagnostics:
            score, pre = forward_with_pre_output(self.fc, features)
            return score.squeeze(-1), dict(score_head_pre_output=pre)
        return self.fc(features).squeeze(-1), {}

    def call_loss(self, prediction, label, **kwargs):
        return (prediction.flatten() - label.flatten()).square().mean()


class HeadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_full_model_matches_frozen_v4(self):
        old = frozen_module('3192f15447961413e7af7073b595f3a1c07dbd63', 'models/pamfn.py', 'old_pamfn')
        config = dict(model_dim=4, fc_drop=.1, fc_r=2, feat_drop=.1, K=2, ms_heads=1, cm_heads=1,
                      ckpt_dir='.', rgb_ckpt_name='x', flow_ckpt_name='x', audio_ckpt_name='x',
                      dataset_name='TES', eval_policy='argmax')
        pretrained = fixtures.EvaluationPolicyTest._pretrained_model()
        old._torch_load = lambda _: copy.deepcopy(pretrained)
        torch.manual_seed(31)
        expected = old.PAMFN(**config)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(31)
        actual = fixtures.EvaluationPolicyTest._build_pamfn()
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertEqual(set(expected.state_dict()), set(actual.state_dict()))
        self.assertTrue(all(torch.equal(v, actual.state_dict()[k]) for k,v in expected.state_dict().items()))
        for model in (expected, actual):
            model.eval()
        with torch.no_grad():
            x = fixtures.EvaluationPolicyTest._inputs()
            self.assertTrue(torch.equal(expected(x)[0], actual(x)[0]))

    def test_both_variants_change_only_final_head_without_initialization_rng(self):
        states = []
        for variant in ('original', 'linear', 'groupnorm'):
            torch.manual_seed(21)
            model = fixtures.EvaluationPolicyTest._build_pamfn(score_head=variant)
            states.append((dict(model.named_parameters()), model.state_dict(), torch.get_rng_state()))
        for params, buffers, rng in states[1:]:
            self.assertEqual(set(params), set(states[0][0]))
            self.assertTrue(all(torch.equal(v, params[k]) for k,v in states[0][0].items()))
            self.assertTrue(torch.equal(rng, states[0][2]))
            self.assertTrue(all(torch.equal(v, buffers[k]) for k,v in states[0][1].items() if not k.startswith('fc.')))

    def test_diagnostics_do_not_change_training_rng_parameters_or_qmd(self):
        generator = torch.Generator().manual_seed(82)
        batch = tuple(torch.randn(4, 12, 4, generator=generator) for _ in range(3)) + (
            torch.tensor([.1,.4,.6,.9]), {m: torch.tensor([12]*4) for m in ('V','F','A')})
        for variant in ('original', 'linear', 'groupnorm'):
            outcomes = []
            for diagnostic in (False, True):
                torch.manual_seed(5)
                model = HeadModel(variant, diagnostic)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                qmd = ScoreQMD('score_qmd', 0, 2, warmup=0, rampup=0)
                metrics = train_epoch(model, optimizer, None, [batch, batch], 0, qmd=qmd)
                metrics.pop('time')
                if diagnostic:
                    d = metrics.pop('score_head')
                    self.assertEqual(d['pre_output']['count'], 8)
                outcomes.append((copy.deepcopy(model.state_dict()), torch.get_rng_state().clone(), metrics, qmd.summary()))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.fc.parameters()))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.fc.parameters()), 0)
            for k,v in outcomes[0][0].items():
                self.assertTrue(torch.equal(v, outcomes[1][0][k]))
            self.assertTrue(torch.equal(outcomes[0][1], outcomes[1][1]))
            self.assertEqual(outcomes[0][2:], outcomes[1][2:])

    def test_linear_predictions_are_not_clipped_or_labeled_saturated(self):
        model = HeadModel('linear', True).eval()
        with torch.no_grad():
            model.fc[5].weight.zero_()
            model.fc[5].bias.fill_(2.)
        inputs = {m: torch.randn(4, 8, 4) for m in ('V','F','A')}
        pred, info = model(inputs)
        self.assertTrue(torch.equal(pred, torch.full_like(pred, 2.)))
        d = head_diagnostics(model, info['score_head_pre_output'].flatten().tolist(),
                             pred.flatten().tolist(), [.1,.3,.5,.9])
        self.assertIsNone(d['saturation_fraction'])
        self.assertEqual(d['outside_unit_interval_fraction'], 1.)
        pred.sum().backward()
        self.assertEqual(float(model.fc[5].bias.grad), 4.)

    def test_groupnorm_output_independent_of_other_batch_members(self):
        model = HeadModel('groupnorm', True).train()
        for layer in model.fc:
            if isinstance(layer, nn.Dropout):
                layer.p = 0
        x = torch.randn(4,4,1)
        y = x.clone()
        y[1:] = 100*y[1:] + 30
        self.assertTrue(torch.allclose(model.fc(x)[0], model.fc(y)[0], atol=1e-7, rtol=0))
        self.assertEqual(list(model.fc[2].buffers()), [])

    def test_cli_keeps_old_qmd_and_records_new_head(self):
        argv = ['main.py','--gpu','-1','--feats','2','--action','Ball','--multi_modality','--qmd_mode','score_qmd']
        with patch('sys.argv', argv):
            old = entry.parser_args()
            self.assertNotIn('score_head', entry._run_metadata(old, 500))
        for variant in ('linear','groupnorm'):
            with patch('sys.argv', argv + ['--score_head',variant,'--score_head_diagnostics']):
                args = entry.parser_args()
                self.assertEqual(entry.qmd_config(args), entry.qmd_config(old))
                self.assertEqual(entry._run_metadata(args,500)['score_head'], dict(variant=variant,diagnostics=True))
                conf = entry.build_configs(args, entry.load_config('Ball',2))[3]['model_conf']
                self.assertEqual(conf['score_head'],variant)
        with self.assertRaises(ValueError):
            configure_head(HeadModel().fc, 'unknown')

    def test_evaluation_diagnostics_passive_and_linear_saturation_not_applicable(self):
        x = fixtures.EvaluationPolicyTest._inputs()
        batch = (x['V'], x['A'], x['F'], torch.tensor([.2,.8]), None)
        for variant in ('linear','groupnorm'):
            model = HeadModel(variant, False)
            before = torch.get_rng_state().clone()
            plain = test_epoch(model, [batch], eval_policy='argmax')
            model.score_head_diagnostics = True
            measured = test_epoch(model, [batch], eval_policy='argmax')
            self.assertEqual(plain['coef'], measured['coef'])
            self.assertEqual(plain['loss_x100'], measured['loss_x100'])
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(measured['score_head']['pre_output']['count'], 2)
            if variant == 'linear':
                self.assertIsNone(measured['saturation_fraction'])


if __name__ == '__main__':
    unittest.main()
