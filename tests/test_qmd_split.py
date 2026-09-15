import io
from contextlib import redirect_stderr
import subprocess
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

import main as entry
import pipeline
from models.score_qmd import ScoreQMD, rng_state
from tests.test_july28_qmd import TinyModel, TinyDataset

BASE = '4d95c157f0efd745d534d604d9502895bb586c76'


def frozen_qmd():
    module = types.ModuleType('frozen_score_qmd_v8')
    code = subprocess.check_output(['git', 'show', BASE + ':models/score_qmd.py']).decode()
    exec(compile(code, BASE + '/models/score_qmd.py', 'exec'), module.__dict__)
    return module.ScoreQMD


def assert_tree(test, left, right):
    if torch.is_tensor(left):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, np.ndarray):
        test.assertTrue(np.array_equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_tree(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree(test, a, b)
    else:
        test.assertEqual(left, right)


class SplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def trajectory(self, cls, **config):
        pipeline.setup_seed(17)
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        loader = DataLoader(TinyDataset(), batch_size=4, shuffle=True)
        qmd = cls(config.pop('mode', 'score_qmd'), 17, 2, warmup=0, rampup=0, **config)
        metrics = []
        for epoch in range(2):
            row = pipeline.train_epoch(model, optimizer, None, loader, epoch, qmd=qmd)
            row.pop('time'); metrics.append(row)
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(),
                    numpy=np.random.get_state(), metrics=metrics), qmd.summary()

    def test_default_full_and_guard_are_bit_identical_to_v8(self):
        for config in ({}, {'grad_mode': 'kd'}):
            with self.subTest(config=config):
                old, old_report = self.trajectory(frozen_qmd(), **config)
                new, new_report = self.trajectory(ScoreQMD, **config)
                assert_tree(self, old, new)
                assert_tree(self, old_report, new_report)

    def test_gt_split_matches_old_mask_gt_parameter_updates(self):
        old, _ = self.trajectory(frozen_qmd(), mode='mask_gt')
        new, report = self.trajectory(ScoreQMD, terms='gt')
        assert_tree(self, old, new)
        self.assertEqual(report['components']['weighted_kd_sum'], 0)
        self.assertGreater(report['components']['weighted_gt_sum'], 0)
        self.assertTrue(report['components']['gradient_verified'])
        # Retained calibration is measurement only, not a hidden KD update.
        self.assertIsNotNone(report['calibration'])

    def test_kd_split_matches_zero_gt_old_updates_and_calibration(self):
        old, old_report = self.trajectory(frozen_qmd(), mask_gt_weight=0)
        new, report = self.trajectory(ScoreQMD, terms='kd')
        assert_tree(self, old, new)
        assert_tree(self, old_report['calibration'], report['calibration'])
        self.assertEqual(report['components']['weighted_gt_sum'], 0)
        self.assertGreater(report['components']['weighted_kd_sum'], 0)
        self.assertTrue(report['components']['gradient_verified'])

    def test_warmup_off_path_does_not_change_rng_or_optimizer(self):
        old, _ = self.trajectory(frozen_qmd(), weight=0)
        for terms in ('gt', 'kd'):
            # Explicit warmup fixture with no active batches.
            def warm(*a, **kw):
                kw['warmup'] = 1.; kw['rampup'] = 0.
                return ScoreQMD(*a, terms=terms, **kw)
            actual, report = self.trajectory(warm)
            assert_tree(self, old, actual)
            self.assertEqual(report['active_batches'], 0)

    def test_loss_selection_teacher_detach_and_private_bn_rng(self):
        import models.score_qmd as module
        for terms in ('gt', 'kd'):
            pipeline.setup_seed(41)
            model = TinyModel()
            x = {m: torch.rand(4, 8, 4) for m in ('V', 'F', 'A')}
            y = torch.linspace(.1, .9, 4)
            before = rng_state()
            full, _ = model(x); full.retain_grad()
            mse = (full.reshape(-1) - y).square().mean()
            after = rng_state()
            original = module.paired_masked_forward
            captured = []
            def forward(*args):
                out = original(*args); captured.append(out[0]); return out
            qmd = ScoreQMD('score_qmd', 41, 1, terms=terms, warmup=0, rampup=0)
            with patch.object(module, 'paired_masked_forward', side_effect=forward):
                loss, _ = qmd.auxiliary(model, x, {m: [8] * 4 for m in x}, y, full, mse, before, 0)
            expected = (.5 * (captured[0].reshape(-1) - y).square().mean() if terms == 'gt'
                        else qmd.beta * module.score_kd(captured[0], full))
            self.assertTrue(torch.equal(loss, expected))
            assert_tree(self, rng_state(), after)
            self.assertEqual(int(model.fc[1].num_batches_tracked), 1)
            # Calibration differentiates full_mse and can populate retain_grad;
            # verify the subsequent auxiliary update does not reach the teacher.
            full.grad = None
            loss.backward()
            self.assertIsNone(full.grad)

    def test_bad_signal_and_guard_combinations_are_rejected(self):
        for kw in ({'terms': 'bad'}, {'terms': 'gt', 'grad_mode': 'kd'}, {'terms': 'kd', 'weight': 0},
                   {'terms': 'gt', 'mask_gt_weight': 0}, {'terms': 'kd', 'target_ratio': 0},
                   {'terms': 'kd', 'beta_cap': 0}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                ScoreQMD('score_qmd', 0, 2, **kw)

    def test_cli_metadata_default_unchanged_and_split_explicit(self):
        argv = ['main.py', '--gpu', '-1', '--feats', '2', '--action', 'Ball', '--multi_modality',
                '--qmd_mode', 'score_qmd', '--ordinal_prototype_weight', '.0001']
        with patch('sys.argv', argv):
            default = entry.qmd_config(entry.parser_args())
        self.assertNotIn('terms', default)
        for term in ('gt', 'kd'):
            with patch('sys.argv', argv + ['--qmd_terms', term]):
                config = entry.qmd_config(entry.parser_args())
            self.assertEqual(config, dict(default, terms=term))
        with patch('sys.argv', argv + ['--qmd_terms', 'kd', '--qmd_grad_mode', 'kd']), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                entry.parser_args()


if __name__ == '__main__':
    unittest.main()
