import io
import json
import tempfile
from pathlib import Path
from contextlib import redirect_stderr
import unittest
from unittest.mock import patch

from tools.summarize_score_qmd import ACTIONS, SEEDS, aggregate, validate_cell
from tools.run_score_qmd_experiments import run_cell
from main import parser_args, build_configs, load_config


class QMDToolsTest(unittest.TestCase):
    def test_completed_resume_skips_training_even_if_gpu_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'Ball/seed0/score_qmd'
            path.mkdir(parents=True)
            self.fixture(path)
            (path/'model.pth').write_bytes(b'test-checkpoint')
            (path/'complete.json').write_text('{}')
            with patch('tools.run_score_qmd_experiments.command') as command:
                self.assertEqual(run_cell(root,'Ball',0,'score_qmd','1',True,1),'SKIP')
                self.assertEqual(run_cell(root,'Ball',0,'score_qmd','2',True,1),'SKIP')
                command.assert_not_called()

    def test_resume_after_training_only_runs_reload_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'Ball/seed0/score_qmd'
            path.mkdir(parents=True)
            self.fixture(path)
            (path/'model.pth').write_bytes(b'test-checkpoint')
            with patch('tools.run_score_qmd_experiments.command') as command:
                self.assertEqual(run_cell(root,'Ball',0,'score_qmd','1',True,1),'FINISH')
                self.assertEqual(command.call_count,1)
                self.assertIn('--test',command.call_args[0][0])
            self.assertTrue((path/'complete.json').is_file())

    def fixture(self, path):
        run = dict(action='Ball', seed=0, tag='score_qmd', feats=2, epochs=1,
                   eval_policy='argmax', eval_seed=20260908, coqm_weight=0,
                   ordinal_prototype_weight=0,
                   qmd=dict(mode='score_qmd', weight=1., mask_ratio=.15,
                            mask_gt_weight=.5, target_ratio=.075, beta_cap=1.,
                            warmup=0., rampup=0.))
        values = {'train_result.json':dict(run=run, saved_model_srcc=.8),
                  'test_result.json':dict(run=run, coef=80., loss_x100=1., mae=.05,
                    bias=0., prediction_std_ratio=.8,
                    evaluation=dict(policy='argmax', seed=20260908, passes=1, rng_isolated=True)),
                  'qmd_training.json':dict(active_batches=4,
                    calibration=dict(degenerate=False,beta=.5))}
        for name, value in values.items():
            (path/name).write_text(json.dumps(value), encoding='utf-8')
        return values

    def test_reload_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            values = self.fixture(path)
            self.assertEqual(validate_cell(path,'Ball',0,'score_qmd',1)['srcc'],80.)
            values['test_result.json']['coef'] = 79.
            (path/'test_result.json').write_text(json.dumps(values['test_result.json']))
            with self.assertRaisesRegex(ValueError, 'reload mismatch'):
                validate_cell(path,'Ball',0,'score_qmd',1)

    def test_inactive_auxiliary_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            values = self.fixture(path)
            values['qmd_training.json']['active_batches'] = 0
            (path/'qmd_training.json').write_text(json.dumps(values['qmd_training.json']))
            with self.assertRaisesRegex(ValueError, 'never activated'):
                validate_cell(path,'Ball',0,'score_qmd',1)

    def test_smoke_budget_cannot_be_used_as_full_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.fixture(path)
            with self.assertRaisesRegex(ValueError, 'epoch budget'):
                validate_cell(path,'Ball',0,'score_qmd',500)

    def rows(self):
        return [dict(action=a, seed=s, method=m, srcc=value)
                for a in ACTIONS for s in SEEDS
                for m,value in [('base',80.0),('mask_gt',80.5),('score_qmd',80.8)]]

    def test_expected_positive_screen(self):
        result = aggregate(self.rows())
        self.assertEqual(result['decision'], 'SCORE_QMD_SIGNAL')
        self.assertEqual(result['checks']['score_qmd']['paired_wins'], 18)
        self.assertAlmostEqual(result['score_qmd_minus_mask_gt_pp'], 0.3)

    def test_missing_or_duplicate_cells_are_rejected(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            aggregate(rows[:-1])
        rows[-1] = rows[0]
        with self.assertRaises(ValueError):
            aggregate(rows)

    def test_ball_drop_blocks_promotion(self):
        rows = self.rows()
        for r in rows:
            if r['action']=='Ball' and r['method']!='base':
                r['srcc'] = 78.0
        self.assertEqual(aggregate(rows)['decision'], 'QMD_NO_CLEAR_GAIN')

    def test_mask_gt_can_win_without_claiming_kd_success(self):
        rows = self.rows()
        for r in rows:
            if r['method']=='mask_gt':
                r['srcc'] += 0.5
        self.assertEqual(aggregate(rows)['decision'], 'MASK_GT_SIGNAL')

    def test_cli_zero_weight_preserves_base_configs(self):
        def configs(extra):
            with patch('sys.argv', ['main.py','--gpu','-1','--feats','2','--action','TES',
                                    '--multi_modality']+extra):
                args = parser_args()
            return build_configs(args, load_config('TES',2))
        self.assertEqual(configs([]), configs(['--qmd_mode','score_qmd','--qmd_weight','0']))
        active = configs(['--qmd_mode','score_qmd'])
        self.assertTrue(active[2]['return_valid_lengths'])
        self.assertEqual(configs([])[3], active[3])

    def test_rejects_invalid_qmd_settings(self):
        for extra in (['--qmd_mask_ratio','1'], ['--qmd_weight','nan'],
                      ['--qmd_warmup','0.7','--qmd_rampup','0.5'],
                      ['--qmd_mode','score_qmd','--coqm_weight','0.006']):
            with patch('sys.argv', ['main.py','--gpu','-1','--feats','2','--action','TES',
                                    '--multi_modality']+extra):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser_args()


if __name__ == '__main__':
    unittest.main()
