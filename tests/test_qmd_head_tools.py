from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

from tools import run_qmd_head as head
from tools import run_qmd_dual as core


def fixture(out, cell, smoke=True):
    out.mkdir(parents=True, exist_ok=True)
    p, a, s, m = cell
    variant = head.RECIPES[m]
    epochs = head.SMOKE_EPOCHS if smoke else core.BUDGET[a]
    run = dict(action=a, seed=s, tag=m, epochs=epochs, feats=2, eval_policy='argmax',
               eval_seed=20260908, ordinal_prototype_weight=0., qmd=head.settings(cell, smoke),
               score_head=dict(variant=variant, diagnostics=True))
    # Different seeds deliberately have opposite gains: summarizer must keep both.
    score = 83. if s == 0 else 77.
    stats = dict(count=4, mean=.4, std=.1, min=.2, max=.6)
    d = dict(variant=variant, normalization='BatchNorm1d' if variant=='linear' else 'GroupNorm',
             output='Identity' if variant=='linear' else 'Sigmoid',
             trainable_parameters=49, pre_output=stats, prediction=stats, target=stats,
             saturation_fraction=None if variant=='linear' else 0.)
    (out/'model.pth').write_bytes(b'fixture')
    core.save(out/'train_result.json', dict(run=run, saved_model_srcc=score/100))
    core.save(out/'test_result.json', dict(run=run, coef=score, loss_x100=1., prediction_std=.1,
        prediction_std_ratio=1., score_head=d))
    core.save(out/'model.pth.selection.json', dict(epoch=1, coef=score))
    (out/'model.pth.epochs.jsonl').write_text(''.join(json.dumps(dict(epoch=e, coef=score))+'\n' for e in range(1,epochs+1)))
    (out/'model.pth.head.jsonl').write_text(''.join(json.dumps(dict(epoch=e, train=d, test=d))+'\n' for e in range(1,epochs+1)))
    core.save(out/'qmd_training.json', dict(mode='score_qmd', active_batches=7*epochs,
        calibration=dict(beta=.5, degenerate=False, scope='score_head_parameters_first_active_training_batch'),
        epochs=[dict(epoch=e, active_batches=7) for e in range(1,epochs+1)],
        view_diagnostics=dict(rows=[dict(epoch=1, modality=m, samples=32, accepted_samples=16,
            teacher_abs_error_sum=.3, student_abs_error_sum=.4) for m in ('V','F','A')])))


class HeadToolsTest(unittest.TestCase):
    def test_fixed_matrix_old_qmd_and_full_budgets(self):
        jobs = head.matrix()
        self.assertEqual(len(jobs), 24)
        self.assertEqual(len(set(jobs)), 24)
        self.assertEqual(len(head.smoke_cells(jobs)), 4)
        self.assertEqual(len(head.matrix('confirm', ('score_qmd_head_linear',))), 6)
        self.assertEqual(set(c[2] for c in head.matrix('confirm', ('score_qmd_head_gn',))), {42})
        for c in jobs:
            argv = head.cell_args(c,'0')
            self.assertNotIn('--epochs',argv)
            self.assertNotIn('--qmd_calibration_mode',argv)
            self.assertNotIn('--ordinal_prototype_weight',argv)
            self.assertNotIn('--qmd_rank_weight',argv)
            self.assertEqual(argv[argv.index('--qmd_beta_cap')+1],'1')
            self.assertEqual(argv[argv.index('--qmd_mask_ratio')+1],'.15')
            self.assertEqual(argv[5:],head.cell_args(c,'4')[5:])
        with self.assertRaises(ValueError):
            head.matrix('screen', ('score_qmd_head_linear',))
        with self.assertRaises(ValueError):
            head.matrix('confirm')

    def test_wrong_head_calibration_and_missing_diagnostics_rejected(self):
        cell = head.matrix()[0]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            fixture(out,cell)
            self.assertEqual(head.validate_cell(out,cell,True)['score_head'],'linear')
            data = core.read(out/'test_result.json')
            data['run']['qmd']['beta_cap'] = 3.
            core.save(out/'test_result.json',data)
            with self.assertRaisesRegex(ValueError,'recipe'):
                head.validate_cell(out,cell,True)
            fixture(out,cell)
            data = core.read(out/'test_result.json')
            data['score_head']['output'] = 'Sigmoid'
            core.save(out/'test_result.json',data)
            with self.assertRaisesRegex(ValueError,'actual score head'):
                head.validate_cell(out,cell,True)
            fixture(out,cell)
            (out/'model.pth.head.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError,'history is incomplete'):
                head.validate_cell(out,cell,True)

    def test_restart_keeps_partial_and_skips_finished_on_changed_gpu(self):
        cell = head.matrix()[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = core.cell_path(root,cell)
            out.mkdir(parents=True)
            (out/'train.log').write_text('interrupted')
            with patch.object(core,'command',side_effect=lambda *args: fixture(out,cell)) as command:
                self.assertEqual(head.run_cell(root,dict(modern=root),cell,'0',True)[0],'FINISH')
                self.assertEqual(head.run_cell(root,dict(modern=root),cell,'4',True)[0],'SKIP')
                self.assertEqual(command.call_count,2)
                self.assertEqual(len(list((out/'attempts').glob('*/train.log'))),1)
                (out/'model.pth.head.jsonl').write_text('')
                with self.assertRaises(ValueError):
                    head.run_cell(root,dict(modern=root),cell,'4',True)

    def test_status_has_four_smoke_cells_counts_and_dead_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = head.matrix()
            core.save(root/'manifest.json',dict(study='screen',jobs=jobs))
            out = core.cell_path(root/'smoke',head.smoke_cells(jobs)[0])
            core.save(out/'train.process.json',dict(pid=123,identity={},state='running'))
            buf = io.StringIO()
            with patch.object(head.cal,'alive',return_value=False), redirect_stdout(buf):
                head.status(root)
            self.assertIn('New training completed: 0/24',buf.getvalue())
            self.assertIn('Smoke: 0/4',buf.getvalue())
            self.assertIn('FAILED/INTERRUPTED',buf.getvalue())

    def test_orphan_blocks_second_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core.save(root/'screen/train.process.json',dict(identity=dict(pid=123)))
            lock = types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *args: None)
            with patch.dict('sys.modules',{'fcntl':lock}), patch('sys.argv',['runner','--root',str(root)]), \
                 patch.object(head,'alive',return_value=True), patch.object(head,'execute') as launch:
                with self.assertRaisesRegex(RuntimeError,'live child'):
                    head.main()
                launch.assert_not_called()

    def test_whole_flow_retains_bad_seed_and_auto_packages_all_24(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'study'
            refs = [dict(protocol='modern',action=a,seed=s,srcc_x100=81.) for a in core.ACTIONS for s in (0,1)]
            def wave(output,repos,cells,gpus,smoke=False):
                for c in cells:
                    out = core.cell_path(output,c)
                    fixture(out,c,smoke)
                    core.save(out/'complete.json',dict(fixture=True))
            lock = types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *args: None)
            with patch.dict('sys.modules',{'fcntl':lock}), patch('sys.argv',['runner','--root',str(root)]), \
                 patch.object(head.cal,'check_resources'), \
                 patch.object(core,'provenance',return_value=dict(inputs={},environment={},commits={})), \
                 patch.object(core,'command'), patch.object(head.light,'checked_reference',return_value=refs), \
                 patch.object(head,'run_wave',side_effect=wave):
                head.main()
            self.assertEqual(core.read(root/'completion.json')['new_cells'],24)
            self.assertEqual(core.read(root/'completion.json')['reference_cells'],12)
            self.assertEqual(core.read(root/'smoke/gate.json')['decision'],'QMD_HEAD_SMOKE_PASS')
            report = core.read(root/'summary.json')
            self.assertEqual(len(report['raw']),24)
            for g in report['methods']:
                self.assertEqual(g['mean_srcc_x100'],80.)
                self.assertEqual(g['delta_vs_control'],-1.)
                self.assertEqual(g['per_seed_score'],{'0':83.,'1':77.})
            self.assertEqual(len(core.read(root/'pilot_summary.json')['raw']),12)
            with zipfile.ZipFile(root.parent/'study_return.zip') as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(json.loads(archive.read('runner.json'))['state'],'finished')
                self.assertFalse(any(n.endswith('.pth') for n in archive.namelist()))
                self.assertEqual(len([n for n in archive.namelist() if n.startswith('screen/') and n.endswith('.head.jsonl')]),24)


if __name__ == '__main__':
    unittest.main()
