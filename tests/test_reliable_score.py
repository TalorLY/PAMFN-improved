import unittest
from unittest.mock import patch

import torch

from main import parser_args, qmd_config
from models.score_qmd import ScoreQMD, reliable_score_kd, rng_state
from tests.test_score_qmd import TinyModel


class ReliableScoreTest(unittest.TestCase):
    def test_only_better_teacher_and_no_teacher_or_label_gradient(self):
        student = torch.tensor([.6, .2, .4], requires_grad=True)
        teacher = torch.tensor([.4, .5, .4], requires_grad=True)
        target = torch.tensor([.3, .2, .3], requires_grad=True)
        before = torch.get_rng_state().clone()
        loss, info = reliable_score_kd(student, teacher, target)
        self.assertAlmostEqual(float(loss.detach()), .5 * .2**2 / 3, places=7)
        self.assertEqual(info['accepted_samples'], 1)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNone(target.grad)
        self.assertEqual(student.grad[1:].tolist(), [0., 0.])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_no_accepted_sample_still_backward_safe(self):
        student = torch.tensor([.2, .7], requires_grad=True)
        loss, info = reliable_score_kd(student, torch.tensor([.6, .1]), student.detach())
        loss.backward()
        self.assertEqual(info['accepted_samples'], 0)
        self.assertEqual(student.grad.tolist(), [0., 0.])

    def test_reject_shape_mismatch(self):
        with self.assertRaises(ValueError):
            reliable_score_kd(torch.ones(2), torch.ones(1), torch.ones(2))

    def test_calibration_is_ungated_and_mask_gt_stays_active(self):
        torch.set_num_threads(1)
        outputs = []
        for mode in ('score_qmd', 'reliable_score_qmd'):
            torch.manual_seed(21)
            model = TinyModel().train()
            inputs = {m: torch.randn(4, 8, 4) for m in ('V', 'F', 'A')}
            y = torch.tensor([.1, .4, .6, .9])
            qmd = ScoreQMD(mode, 0, 1, warmup=0, rampup=0)
            before = rng_state()
            full, _ = model(inputs)
            mse = model.call_loss(full, y)
            after = torch.get_rng_state().clone()
            buffers = {k: v.clone() for k, v in model.named_buffers()}
            aux, diagnostic = qmd.auxiliary(model, inputs, {m: [8]*4 for m in inputs}, y, full, mse, before, 0)
            self.assertTrue(torch.equal(after, torch.get_rng_state()))
            for k, v in model.named_buffers():
                self.assertTrue(torch.equal(buffers[k], v), k)
            (mse + aux).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
            outputs.append((qmd.calibration, diagnostic))
        self.assertEqual(outputs[0][0], outputs[1][0])
        self.assertEqual(outputs[0][1]['mask_gt'], outputs[1][1]['mask_gt'])
        self.assertLessEqual(outputs[1][1]['gated_score_kd'], outputs[1][1]['score_kd'])

    def test_cli_no_rank_in_reliable_combination(self):
        argv = ['main.py', '--gpu', '-1', '--feats', '2', '--action', 'Ball', '--multi_modality',
                '--qmd_mode', 'reliable_score_qmd', '--ordinal_prototype_weight', '.0001',
                '--ordinal_prototype_target_mode', 'continuous', '--quality_aux_rng_isolated']
        with patch('sys.argv', argv):
            config = qmd_config(parser_args())
        self.assertEqual(config['mode'], 'reliable_score_qmd')
        self.assertNotIn('rank_weight', config)


if __name__ == '__main__':
    unittest.main()
