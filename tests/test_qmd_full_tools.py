from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from tools import run_qmd_full as run
from tools import run_qmd_dual as core


def row(c):
    p, a, s, m = c
    return dict(protocol=p, action=a, seed=s, method=m, srcc_x100=(70. if s == 1 else 82.) +
                {run.BASE:0, run.ORDERED:1, run.KD:2, run.COMBO:.5}[m], warnings=[])


class FullRunnerTest(unittest.TestCase):
    def test_matrix30_reuse6_and_all54_primary_cells(self):
        jobs, reused = run.matrix(), run.reused_cells()
        self.assertEqual(len(jobs), 30)
        self.assertEqual(len(set(jobs)), 30)
        self.assertEqual(len(reused), 6)
        self.assertFalse(set(jobs).intersection(reused))
        for method in (run.BASE, run.KD):
            self.assertEqual({(c[1], c[2]) for c in jobs + reused if c[3] == method},
                             {(a, s) for a in core.ACTIONS for s in run.SEEDS})
        self.assertEqual(len(run.grad.reference_cells(jobs)), 36)
        self.assertEqual(len(run.smoke_cells(jobs)), 4)
        self.assertEqual(run.smoke_cells([]), [])

    def test_fixed_base_and_exact_v9_kd_arguments(self):
        for c in run.matrix() + run.reused_cells():
            for smoke in (False, True):
                argv = run.cell_args(c, '3', smoke)
                if c[3] == run.KD:
                    self.assertEqual(argv, run.split.cell_args(c, '3', smoke))
                else:
                    self.assertEqual(core.mode_for(c[3]), 'off')
                    self.assertEqual(argv[argv.index('--qmd_mode') + 1], 'off')
                    self.assertEqual(argv[argv.index('--ordinal_prototype_weight') + 1], '0')
                    self.assertIn('--save_test_predictions', argv)
                    for option in ('--qmd_terms', '--qmd_grad_mode', '--shared_rank_weight', '--shared_list_weight'):
                        self.assertNotIn(option, argv)
                if smoke:
                    self.assertEqual(argv[argv.index('--epochs') + 1], '2')
                else:
                    self.assertNotIn('--epochs', argv)

    def test_base_predictions_and_disabled_auxiliary_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); c = ('july28', 'Ball', 0, run.BASE)
            qmd = dict(mode='off', weight=1., mask_ratio=.15, mask_gt_weight=.5, target_ratio=.075, beta_cap=1., warmup=.1, rampup=.1)
            y = np.linspace(.1, .9, 50).tolist()
            data = dict(run=dict(qmd=qmd), coef=100., loss_x100=0., predictions=dict(label=y, prediction=y))
            core.save(root/'train_result.json', data); core.save(root/'test_result.json', data)
            with patch.object(core, 'validate_cell', return_value=row(c)):
                run.validate_cell(root, c)
                data['predictions']['prediction'] = y[::-1]
                core.save(root/'test_result.json', data)
                with self.assertRaisesRegex(ValueError, 'disagree'):
                    run.validate_cell(root, c)
                data['run']['qmd']['mode'] = 'score_qmd'
                core.save(root/'test_result.json', data)
                with self.assertRaisesRegex(ValueError, 'unexpectedly enabled'):
                    run.validate_cell(root, c)

    def test_changed_argv_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); c = run.matrix()[0]
            core.save(root/'run_config.json', dict(argv=run.cell_args(c, '4')[5:], protocol='july28'))
            run.check_args(root, c)
            core.save(root/'run_config.json', {})
            with self.assertRaisesRegex(ValueError, 'arguments changed'):
                run.check_args(root, c)

    def test_summary_has_matched_base_ordered_and_keeps_low_seed(self):
        rows = [row(('july28', a, s, m)) for a in core.ACTIONS for s in run.SEEDS
                for m in (run.BASE, run.ORDERED, run.KD, run.COMBO)]
        result = run.aggregate(rows)
        self.assertEqual(len(result['raw']), 72)
        self.assertEqual(len(result['methods']), 4)
        kd = result['methods'][2]
        self.assertEqual(kd['cells'], 18)
        self.assertEqual(kd['srcc_x100'], 80)
        self.assertEqual(kd['delta_vs_base'], 2)
        self.assertEqual(kd['delta_vs_ordered'], 1)
        self.assertEqual(kd['per_seed']['1']['srcc_x100'], 72)
        self.assertEqual(set(kd['per_action']), set(core.ACTIONS))
        self.assertAlmostEqual(kd['seed_mean_std'], 12 / (3 ** .5))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            run.aggregate(rows[:-1])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            run.aggregate(rows[:-1] + [rows[0]])

    def test_missing_or_wrong_v9_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = dict(commits=run.V9_COMMITS, inputs={}, environment={})
            core.save(root/'source.json', source)
            core.save(root/'completion.json', dict(source=source, new_cells=14, experiments_complete=True))
            with self.assertRaisesRegex(ValueError, 'completed frozen v9'):
                run.checked_v9(root, source, root/'new')
            core.save(root/'completion.json', dict(source=source, new_cells=15, experiments_complete=True))
            with self.assertRaisesRegex(ValueError, 'environment mismatch'):
                run.checked_v9(root, dict(source, environment={'torch':'different'}), root/'new')

    def test_training_source_change_rejected(self):
        with patch.object(run.subprocess, 'check_output', return_value='changed model'):
            with self.assertRaisesRegex(ValueError, 'preserve'):
                run.verify_training_unchanged(Path('.'))

    def test_new_root_cannot_overlap_old_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run.check_result_roots(root/'new', [root/'old'])
            for new, old in [(root,root), (root/'old/new',root/'old'), (root,root/'old')]:
                with self.assertRaisesRegex(ValueError, 'separate'):
                    run.check_result_roots(new, [old])

    def test_only_new_grid_dispatched_and_smoke_uses_one_gpu(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            args = types.SimpleNamespace(stage='all', gpus='0 1 3 4', july28_repo=temp, reference_root=temp, v9_root=temp)
            with patch.object(run.cal, 'check_resources'), patch.object(run, 'verify_training_unchanged'), \
                 patch.object(core, 'provenance', return_value={}), patch.object(run.grad, 'checked_reference'), \
                 patch.object(run, 'checked_v9'), patch.object(core, 'command'), patch.object(run, 'run_wave') as wave, \
                 patch.object(run, 'validate_cell', side_effect=lambda out,c,smoke=False:row(c)), patch.object(run, 'write_report'):
                run.execute(args, root)
                self.assertEqual(wave.call_args_list[0].args[3], ['0'])
                self.assertEqual(len(wave.call_args_list[0].args[2]), 4)
                self.assertEqual(wave.call_args_list[1].args[2], run.matrix())
                self.assertEqual(wave.call_args_list[1].args[3], ['0', '1', '3', '4'])
                self.assertEqual(core.read(root/'completion.json')['new_cells'], 30)

    def test_failed_smoke_blocks_training(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            args = types.SimpleNamespace(stage='all', gpus='4', july28_repo=temp, reference_root=temp, v9_root=temp)
            with patch.object(run.cal, 'check_resources'), patch.object(run, 'verify_training_unchanged'), \
                 patch.object(core, 'provenance', return_value={}), patch.object(run.grad, 'checked_reference'), \
                 patch.object(run, 'checked_v9'), patch.object(core, 'command', side_effect=RuntimeError('unit failed')), \
                 patch.object(run, 'run_wave') as wave:
                with self.assertRaisesRegex(RuntimeError, 'unit failed'):
                    run.execute(args, root)
                wave.assert_not_called()
                self.assertFalse((root/'completion.json').exists())

    def test_existing_gate_does_not_repeat_smoke(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp); core.save(root/'smoke/gate.json', dict(decision='QMD_FULL_SMOKE_PASS', cells=4, source={}))
            args = types.SimpleNamespace(stage='screen', gpus='4', july28_repo=temp, reference_root=temp, v9_root=temp)
            with patch.object(run.cal, 'check_resources'), patch.object(run, 'verify_training_unchanged'), \
                 patch.object(core, 'provenance', return_value={}), patch.object(run.grad, 'checked_reference'), \
                 patch.object(run, 'checked_v9'), patch.object(core, 'command') as cmd, \
                 patch.object(run, 'run_wave') as wave, patch.object(run, 'write_report'):
                run.execute(args, root)
                cmd.assert_not_called(); self.assertEqual(wave.call_count, 1)
                self.assertEqual(len(wave.call_args.args[2]), 30)

    def test_completed_resume_keeps_base_and_kd_checkpoints(self):
        # Exercise the shared resume path including off-mode Base (no qmd file).
        for method in (run.BASE, run.KD):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp); c = ('july28','Ball',0,method); out = core.cell_path(root,c)
                out.mkdir(parents=True)
                (out/'model.pth').write_bytes(b'fixture')
                (out/'model.pth.eval_rng.pt').write_bytes(b'rng')
                core.save(out/'complete.json', dict(files={}, checkpoint_sha256=core.sha(out/'model.pth'),
                    eval_rng_sha256=core.sha(out/'model.pth.eval_rng.pt')))
                with patch.object(run, 'validate_cell', return_value=row(c)), patch.object(run.split, 'check_warmup'), patch.object(core, 'command') as cmd:
                    state, _ = run.run_cell(root, {}, c, '4', False)
                    self.assertEqual(state, 'SKIP'); cmd.assert_not_called()

    def test_package_validates_completed_grid_and_excludes_weights(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root=Path(temp); core.save(root/'completion.json', dict(experiments_complete=True))
            with patch.object(run, 'write_report', side_effect=ValueError('incomplete grid')):
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    run.execute(types.SimpleNamespace(stage='package'), root)
            (root/'model.pth').write_bytes(b'fixture'); core.save(root/'summary.json', {'fixture':True})
            output=core.package(root, '_partial')
            with zipfile.ZipFile(output) as z:
                self.assertIn('summary.json', z.namelist())
                self.assertNotIn('model.pth', z.namelist())

    def test_status_reports_separate_new_and_frozen_counts(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()) as output:
            run.status(Path(temp))
            self.assertIn('Base new: 0/18', output.getvalue())
            self.assertIn('KD new: 0/12', output.getvalue())
            self.assertIn('KD full coverage: 0/18', output.getvalue())
            self.assertIn('Frozen Ordered: 0/18', output.getvalue())


if __name__ == '__main__':
    unittest.main()
