from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

from models.qmd_calibration import WindowCalibration
from tools import run_qmd_dual as core
from tools import run_qmd_cal as cal
from tools import qmd_process


def fixture(out, cell, smoke=True):
    out.mkdir(parents=True, exist_ok=True)
    p, a, s, m = cell
    epochs = cal.SMOKE_EPOCHS if smoke else core.BUDGET[a]
    config = cal.settings(cell, smoke)
    run = dict(action=a, seed=s, tag=m, epochs=epochs, feats=2, eval_policy='argmax',
               eval_seed=20260908, ordinal_prototype_weight=0., qmd=config)
    (out/'model.pth').write_bytes(b'fixture')
    core.save(out/'train_result.json', dict(run=run, saved_model_srcc=.8))
    core.save(out/'test_result.json', dict(run=run, coef=80., loss_x100=1., prediction_std=.1))
    core.save(out/'model.pth.selection.json', dict(epoch=1, coef=80.))
    (out/'model.pth.epochs.jsonl').write_text(''.join(json.dumps(dict(epoch=e, coef=80.))+'\n' for e in range(1, epochs+1)))
    c = WindowCalibration(config['calibration_mode'], 9, .075, 3., 1.)
    for i in range(9):
        c.observe(('V', 'F', 'A')[i % 3], 1., .025, 32, 0)
    core.save(out/'qmd_training.json', dict(mode='score_qmd', active_batches=7*epochs,
        calibration=c.report, epochs=[dict(epoch=e, active_batches=7) for e in range(1, epochs+1)],
        window_progress=dict(mode=config['calibration_mode'], frozen=True, requested_batches=9,
                             collected_batches=9, kd_applied_batches=dict(V=2, F=2, A=1)),
        view_diagnostics=dict(rows=[dict(epoch=1, modality=m, samples=32, accepted_samples=16,
            teacher_abs_error_sum=.3, student_abs_error_sum=.4) for m in ('V', 'F', 'A')])))


class CalToolsTest(unittest.TestCase):
    def test_matrix_recipe_and_full_budgets(self):
        jobs = cal.matrix()
        self.assertEqual(len(jobs), 24)
        self.assertEqual(len(set(jobs)), 24)
        self.assertEqual(len(cal.smoke_cells(jobs)), 4)
        self.assertEqual(len(cal.matrix('confirm', ('score_qmd_cal_modality',))), 6)
        self.assertEqual(core.BUDGET['Ball'], 500)
        for cell in jobs:
            argv = cal.cell_args(cell, '3')
            self.assertNotIn('--epochs', argv)
            self.assertNotIn('--ordinal_prototype_weight', argv)
            self.assertNotIn('--qmd_rank_weight', argv)
            self.assertEqual(argv[argv.index('--qmd_beta_cap')+1], '3')
            self.assertEqual(argv[5:], cal.cell_args(cell, '4')[5:])
        with self.assertRaises(ValueError):
            cal.matrix('confirm')

    def test_validator_recomputes_beta_and_rejects_tampering(self):
        cell = ('modern', 'Ball', 0, 'score_qmd_cal_modality')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            fixture(out, cell)
            for value in cal.validate_cell(out, cell, True)['beta'].values():
                self.assertAlmostEqual(value, 3.)
            q = core.read(out/'qmd_training.json')
            q['calibration']['beta']['V'] = 2.
            core.save(out/'qmd_training.json', q)
            with self.assertRaisesRegex(ValueError, 'raw training'):
                cal.validate_cell(out, cell, True)
            fixture(out, cell)
            t = core.read(out/'test_result.json')
            t['run']['qmd']['mask_ratio'] = .05
            core.save(out/'test_result.json', t)
            with self.assertRaises(ValueError):
                cal.validate_cell(out, cell, True)

    def test_restart_preserves_partial_and_skips_completed_on_different_gpu(self):
        cell = ('modern', 'Ball', 0, 'score_qmd_cal_shared')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = core.cell_path(root, cell)
            out.mkdir(parents=True)
            (out/'train.log').write_text('interrupted')
            calls = []
            def command(argv, cwd, log):
                calls.append(argv)
                fixture(out, cell)
            with patch.object(core, 'command', side_effect=command):
                self.assertEqual(cal.run_cell(root, dict(modern=root), cell, '3', True)[0], 'FINISH')
                self.assertEqual(cal.run_cell(root, dict(modern=root), cell, '4', True)[0], 'SKIP')
                self.assertEqual(len(calls), 2)
                self.assertEqual(len(list((out/'attempts').glob('*/train.log'))), 1)
                (out/'model.pth').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                    cal.run_cell(root, dict(modern=root), cell, '4', True)

    def test_command_tracks_success_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            core.command([sys.executable, '-c', 'print(123)'], out, out/'test.log')
            self.assertEqual(core.read(out/'test.process.json')['state'], 'finished')
            with self.assertRaises(subprocess.CalledProcessError):
                core.command([sys.executable, '-c', 'raise SystemExit(5)'], out, out/'train.log')
            self.assertEqual(core.read(out/'train.process.json')['returncode'], 5)

    def test_status_marks_dead_child_instead_of_claiming_running(self):
        cell = cal.matrix()[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core.save(root/'manifest.json', dict(study='screen', jobs=[cell]))
            core.save(root/'smoke/gate.json', {})
            out = core.cell_path(root/'screen', cell)
            core.save(out/'train.process.json', dict(pid=123, identity={}, state='running'))
            buf = io.StringIO()
            with patch.object(cal, 'alive', return_value=False), redirect_stdout(buf):
                cal.status(root)
            self.assertIn('New training completed: 0/1', buf.getvalue())
            self.assertIn('FAILED/INTERRUPTED', buf.getvalue())

    def test_resource_checks_do_not_use_busy_gpu(self):
        with patch.object(cal.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=20*1024**3)), \
             patch.object(cal.subprocess, 'check_output', return_value='0, 5\n3, 30000\n'):
            cal.check_resources(Path('.'), ['3'])
            with self.assertRaises(RuntimeError):
                cal.check_resources(Path('.'), ['0'])

    def test_pid_reuse_boot_change_and_zombie_are_not_live(self):
        record = dict(host=qmd_process.socket.gethostname(), pid=123, start_ticks='10', boot_id='boot', state='R')
        with patch.object(qmd_process, 'identity', return_value=dict(record)):
            self.assertTrue(qmd_process.alive(record))
        for changes in (dict(start_ticks='11'), dict(boot_id='new'), dict(state='Z')):
            with patch.object(qmd_process, 'identity', return_value=dict(record, **changes)):
                self.assertFalse(qmd_process.alive(record))

    def test_orphan_child_blocks_second_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core.save(root/'screen/train.process.json', dict(identity=dict(pid=123)))
            lock = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
            with patch.dict('sys.modules', {'fcntl': lock}), \
                 patch('sys.argv', ['runner', '--root', str(root)]), \
                 patch.object(cal, 'alive', return_value=True), patch.object(cal, 'execute') as launch:
                with self.assertRaisesRegex(RuntimeError, 'live child'):
                    cal.main()
                launch.assert_not_called()

    def test_all_stage_completes_24_cells_and_auto_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'study'
            refs = [dict(protocol='modern', action=a, seed=s, srcc_x100=81.)
                    for a in core.ACTIONS for s in (0, 1)]
            def wave(output, repos, cells, gpus, smoke=False):
                for c in cells:
                    out = core.cell_path(output, c)
                    fixture(out, c, smoke)
                    core.save(out/'complete.json', dict(fixture=True))
            lock = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
            with patch.dict('sys.modules', {'fcntl': lock}), \
                 patch('sys.argv', ['runner', '--root', str(root)]), \
                 patch.object(cal, 'check_resources'), \
                 patch.object(core, 'provenance', return_value=dict(inputs={}, environment={}, commits={})), \
                 patch.object(core, 'command'), patch.object(cal.light, 'checked_reference', return_value=refs), \
                 patch.object(cal, 'run_wave', side_effect=wave):
                cal.main()
            self.assertEqual(core.read(root/'completion.json')['new_cells'], 24)
            self.assertEqual(core.read(root/'completion.json')['reference_cells'], 12)
            self.assertEqual(core.read(root/'smoke/gate.json')['decision'], 'QMD_CAL_SMOKE_PASS')
            self.assertEqual(len(core.read(root/'pilot_summary.json')['raw']), 12)
            report = core.read(root/'summary.json')
            self.assertEqual(len(report['raw']), 24)
            self.assertTrue(all(r['delta_vs_control'] == -1. for r in report['methods']))
            with zipfile.ZipFile(root.parent/'study_return.zip') as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(json.loads(archive.read('runner.json'))['state'], 'finished')
                self.assertFalse(any(n.endswith('.pth') for n in archive.namelist()))


if __name__ == '__main__':
    unittest.main()
