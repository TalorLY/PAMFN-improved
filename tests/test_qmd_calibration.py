import unittest
from unittest.mock import patch

import torch

from main import parser_args, qmd_config
from models.qmd_calibration import WindowCalibration
from models.score_qmd import ScoreQMD, rng_state
from tests.test_score_qmd import TinyModel
from tools.verify_sr_qmd_runtime import frozen_module


class CalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ratio_of_weighted_norms_and_not_mean_ratios(self):
        c = WindowCalibration('shared', 6, .075, 3., 1.)
        for i in range(6):
            c.observe(('V', 'F', 'A')[i % 3], 2. if i < 3 else 4.,
                      1. if i < 3 else 8., 32 if i < 3 else 8, i)
        self.assertAlmostEqual(c.report['beta']['shared'], .075)
        self.assertAlmostEqual(c.report['groups']['shared']['main_gradient_norm'], 2.4)
        self.assertAlmostEqual(c.report['groups']['shared']['achieved_ratio_at_full_ramp'], .075)
        with self.assertRaises(RuntimeError):
            c.observe('V', 1., 1., 4, 8)

    def test_modality_coefficients_cap_and_degenerate(self):
        c = WindowCalibration('modality', 9, .075, 3., 1.)
        for i in range(9):
            m = ('V', 'F', 'A')[i % 3]
            c.observe(m, 2., dict(V=1., F=.01, A=0.)[m], 32, 4)
        self.assertEqual(c.report['beta'], dict(V=.15, F=3., A=0.))
        self.assertTrue(c.report['groups']['F']['capped'])
        self.assertTrue(c.report['groups']['A']['degenerate'])
        self.assertEqual(c.report['kd_starts_active_batch'], 10)

    def test_reject_invalid_nonfinite_or_unbalanced_window(self):
        for n in (1, 4, 7, 9.0):
            with self.assertRaises(ValueError):
                WindowCalibration('shared', n, .075, 3., 1.)
        for value in (float('nan'), float('inf'), -1.):
            with self.assertRaises(FloatingPointError):
                WindowCalibration('shared', 9, .075, 3., 1.).observe('V', 1., value, 4, 0)
        c = WindowCalibration('shared', 6, .075, 3., 1.)
        with self.assertRaises(ValueError):
            for _ in range(6):
                c.observe('V', 1., 1., 4, 0)

    def train(self, cls=ScoreQMD, **kwargs):
        torch.manual_seed(29)
        model = TinyModel().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        q = cls('score_qmd', 0, 3, warmup=0, rampup=0, **kwargs)
        states, diagnostics = [], []
        for i in range(15):
            inputs = {m: torch.randn(4, 12, 4) for m in ('V', 'F', 'A')}
            target = torch.rand(4)
            before = rng_state()
            full, _ = model(inputs)
            main = model.call_loss(full, target)
            buffers = {k: v.clone() for k, v in model.named_buffers()}
            after = torch.get_rng_state().clone()
            aux, d = q.auxiliary(model, inputs, {m: [12]*4 for m in inputs}, target, full, main, before, i // 5)
            self.assertTrue(torch.equal(after, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(v, buffers[k]) for k, v in model.named_buffers()))
            optimizer.zero_grad()
            (main + aux).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            optimizer.step()
            states.append({k: v.clone() for k, v in model.state_dict().items()})
            diagnostics.append(d)
        return states, diagnostics, q.summary(), torch.get_rng_state()

    def test_window_is_shared_until_frozen_then_applied_without_rng_bn_changes(self):
        shared = self.train(calibration_mode='shared', beta_cap=3.)
        modal = self.train(calibration_mode='modality', beta_cap=3.)
        for i in range(9):
            self.assertEqual(shared[1][i]['beta'], 0.)
            self.assertEqual(modal[1][i]['beta'], 0.)
            for key in shared[0][i]:
                self.assertTrue(torch.equal(shared[0][i][key], modal[0][i][key]))
        for result in (shared, modal):
            report = result[2]['calibration']
            self.assertEqual(report['batches'], 9)
            self.assertEqual(report['modality_counts'], dict(V=3, F=3, A=3))
            self.assertTrue(all(d['beta'] > 0 for d in result[1][9:]))
            self.assertEqual(sum(result[2]['window_progress']['kd_applied_batches'].values()), 6)
        self.assertEqual(shared[2]['calibration']['observations'], modal[2]['calibration']['observations'])
        self.assertTrue(torch.equal(shared[3], modal[3]))

    def test_default_exact_frozen_v3_all_training_states_and_reports(self):
        old = frozen_module('947db12de4dbf60a3c3f807c4edd2a768619e3c2', 'models/score_qmd.py', 'v3_score')
        ref, now = self.train(old.ScoreQMD), self.train()
        self.assertEqual(ref[1:3], now[1:3])
        self.assertTrue(torch.equal(ref[3], now[3]))
        for expected, actual in zip(ref[0], now[0]):
            self.assertTrue(all(torch.equal(actual[k], expected[k]) for k in expected))

    def test_cli_new_settings_recorded_only_when_enabled(self):
        argv = ['main.py', '--gpu', '-1', '--feats', '2', '--action', 'Ball', '--multi_modality', '--qmd_mode', 'score_qmd']
        with patch('sys.argv', argv):
            self.assertNotIn('calibration_mode', qmd_config(parser_args()))
        with patch('sys.argv', argv + ['--qmd_calibration_mode', 'modality', '--qmd_beta_cap', '3']):
            result = qmd_config(parser_args())
            self.assertEqual(result['calibration_mode'], 'modality')
            self.assertEqual(result['calibration_batches'], 9)
        with self.assertRaises(ValueError):
            ScoreQMD('score_rank_qmd', 0, 100, calibration_mode='shared')


if __name__ == '__main__':
    unittest.main()
