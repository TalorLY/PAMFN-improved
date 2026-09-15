from contextlib import redirect_stdout
from contextlib import nullcontext
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from tools import run_qmd_gradient as run
from tools import run_qmd_dual as core
from tools import reuse_qmd_gradient_v8 as reuse


def fake_row(cell):
    p,a,s,m=cell
    delta={'observe':0.,'kd':-.2,'all':.4}[run.mode_for(cell)]
    return dict(protocol=p,action=a,seed=s,method=m,srcc_x100=80.+delta,guard_mode=run.mode_for(cell),warnings=[])


class RunnerTest(unittest.TestCase):
    def test_fixed_matrices_and_native_protocol_arguments(self):
        self.assertEqual(len(run.matrix()),36)
        self.assertEqual(len(set(run.matrix())),36)
        self.assertEqual(len(run.matrix('full')),108)
        self.assertEqual(len(run.smoke_cells(run.matrix())),12)
        self.assertEqual(len(run.reference_cells(run.matrix())),18)
        self.assertEqual(run.smoke_cells([]),[])
        for c in run.matrix():
            argv=run.cell_args(c,'3',True)
            self.assertEqual(argv[argv.index('--run_tag')+1],c[3])
            self.assertEqual(argv[argv.index('--qmd_grad_mode')+1],run.mode_for(c))
            self.assertEqual(argv[argv.index('--epochs')+1],'2')
            self.assertEqual('--eval_policy' in argv,c[0]=='modern')
            self.assertNotIn('--quality_aux_rng_isolated',argv)
            self.assertEqual('--ordinal_prototype_weight' in argv,c[0]=='july28')
            self.assertNotIn('--score_head',argv); self.assertNotIn('--ema_screen',argv)
        with self.assertRaises(ValueError): run.matrix('bad')
        with self.assertRaises(ValueError): run.mode_for(('modern','Ball',0,'bad'))

    def test_predictions_and_guard_counts_are_validated(self):
        c=run.matrix()[0]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            y=np.linspace(.1,.9,50).tolist()
            for name in ('train_result.json','test_result.json'):
                core.save(out/name,dict(run=dict(qmd=run.settings(c)),coef=100.,loss_x100=0.,predictions=dict(label=y,prediction=y)))
            guard=dict(recipe=run.recipe('observe'),batches=2,projections=0,rows=[dict(epoch=51,modality='V',batches=2,projections=0,
                       gt_conflicts=1,kd_conflicts=0,total_conflicts=1)])
            qmd=dict(active_batches=2,gradient_guard=guard,epochs=[dict(epoch=51,active_batches=2)])
            core.save(out/'qmd_training.json',qmd)
            with patch.object(core,'validate_cell',return_value=fake_row(c)):
                self.assertEqual(run.validate_cell(out,c)['gt_conflict_fraction'],.5)
                guard['projections']=1; core.save(out/'qmd_training.json',qmd)
                with self.assertRaisesRegex(ValueError,'counts'): run.validate_cell(out,c)
                guard['projections']=0; core.save(out/'qmd_training.json',qmd)
                record=core.read(out/'test_result.json'); record['predictions']['prediction'].reverse()
                core.save(out/'test_result.json',record)
                with self.assertRaisesRegex(ValueError,'prediction'): run.validate_cell(out,c)

    def test_observe_checks_all_epochs_and_matching_nan(self):
        c=run.matrix()[0]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/'new'; out.mkdir()
            ref=core.cell_path(root/'reference',(c[0],c[1],c[2],run.BASE[c[0]])); ref.mkdir(parents=True)
            values=[dict(epoch=i,coef=80. if i==1 else float('nan'),loss_x100=1.,prediction_std=.1) for i in (1,2)]
            for directory in (out,ref):
                (directory/'model.pth.epochs.jsonl').write_text('\n'.join(json.dumps(x) for x in values))
            run.check_observe_reference(out,c,root/'reference')
            values[0]['coef']=79.
            (out/'model.pth.epochs.jsonl').write_text('\n'.join(json.dumps(x) for x in values))
            with self.assertRaisesRegex(ValueError,'drift'): run.check_observe_reference(out,c,root/'reference')

    def test_complete_execute_summarizes_all_seeds_and_packages_no_weights(self):
        jobs=run.matrix(); source=dict(commits={},inputs={},environment={})
        refs=[dict(protocol=p,action=a,seed=s,method=m,srcc_x100=80.1 if m=='ordered' else 80.)
              for p,a,s,m in run.reference_cells(jobs)]
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp)/'run'; root.mkdir()
            args=types.SimpleNamespace(stage='all',study='screen',gpus='0 1',july28_repo=str(run.REPO),reference_root=str(root/'old'))
            def wave(path,repos,cells,gpus,smoke=False):
                for c in cells:
                    out=core.cell_path(path,c); out.mkdir(parents=True,exist_ok=True)
                    core.save(out/'complete.json',{'fixture':True})
                    (out/'model.pth').write_bytes(b'fixture not a model')
            with patch.object(run.cal,'check_resources'), patch.object(core,'provenance',return_value=source), \
                 patch.object(run,'checked_reference',return_value=refs), patch.object(core,'command'), \
                 patch.object(run,'run_wave',side_effect=wave), patch.object(run,'validate_cell',side_effect=lambda out,c,smoke=False:fake_row(c)), \
                 patch.object(run,'check_observe_reference'):
                run.execute(args,root)
                report=core.read(root/'summary.json')
                self.assertEqual(len(report['raw']),36)
                self.assertEqual(len(report['methods']),6)
                self.assertEqual(core.read(root/'completion.json')['new_cells'],36)
                for group in report['methods']:
                    if group['method'].endswith('_all') and group['protocol']=='july28':
                        self.assertAlmostEqual(group['delta_vs_control'],.3)
                archive=core.package(root,'')
                with zipfile.ZipFile(archive) as z:
                    self.assertIn('summary.json',z.namelist())
                    self.assertFalse(any(n.endswith('.pth') for n in z.namelist()))

    def test_smoke_failure_blocks_training(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp); args=types.SimpleNamespace(stage='all',study='screen',gpus='0',july28_repo=str(run.REPO),reference_root=tmp)
            with patch.object(run.cal,'check_resources'), patch.object(core,'provenance',return_value={}), \
                 patch.object(run,'checked_reference',return_value=[]), patch.object(core,'command',side_effect=RuntimeError('unit failed')), \
                 patch.object(run,'run_wave') as wave:
                with self.assertRaisesRegex(RuntimeError,'unit failed'): run.execute(args,root)
                wave.assert_not_called()
                self.assertFalse((root/'completion.json').exists())

    def test_reference_source_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            core.save(root/'source.json',dict(commits={'modern':'wrong'}))
            core.save(root/'completion.json',{})
            with self.assertRaisesRegex(ValueError,'frozen'): run.checked_reference(root,{},run.matrix(),root/'out')

    def test_status_labels_interrupted_and_live_processes(self):
        c=run.matrix()[0]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); core.save(root/'manifest.json',dict(study='screen',jobs=[c]))
            core.save(root/'smoke/gate.json',dict(decision='QMD_GRAD_SMOKE_PASS'))
            out=core.cell_path(root/'screen',c)
            core.save(out/'train.process.json',dict(pid=12,identity={},log='train.log'))
            for live,label in [(False,'FAILED/INTERRUPTED'),(True,'RUNNING')]:
                output=io.StringIO()
                with patch.object(run.cal,'alive',return_value=live),redirect_stdout(output):
                    run.cal.status(root,smoke_selector=run.smoke_cells,smoke_epochs=2)
                self.assertIn(label,output.getvalue()); self.assertIn('0/1',output.getvalue())


class ReuseTest(unittest.TestCase):
    def fixture(self, tmp):
        origin=Path(tmp)/'v8'; root=Path(tmp)/'fix1'
        cell=('modern','Ball',0,run.method('modern','kd'))
        jobs=[cell,('july28','Ball',0,run.method('july28','kd'))]
        old=dict(commits=reuse.V8_COMMITS,inputs={'fixture':'same'},environment={'fixture':'same'})
        new=dict(old,commits=dict(modern='new',july28='new'))
        for directory,source in ((origin,old),(root,new)):
            core.save(directory/'source.json',source)
            core.save(directory/'manifest.json',dict(jobs=jobs))
        out=core.cell_path(origin/'screen',cell); out.mkdir(parents=True)
        core.save(out/'run_config.json',dict(argv=run.cell_args(cell,'4')[5:],protocol=cell[0]))
        (out/'model.pth').write_bytes(b'checkpoint fixture')
        (out/'train.log').write_text('finished training')
        core.save(out/'complete.json',dict(metrics=fake_row(cell),files={'run_config.json':core.sha(out/'run_config.json')},
                  checkpoint_sha256=core.sha(out/'model.pth')))
        failed=core.cell_path(origin/'screen',jobs[1]); failed.mkdir(parents=True)
        (failed/'train.log').write_text('original projection failure')
        core.save(failed/'failure.json',dict(error='projection failed'))
        return origin,root,out,new,jobs

    def test_verified_import_preserves_origin_skips_incomplete_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            origin,root,out,source,jobs=self.fixture(tmp)
            before={str(p.relative_to(origin)):core.sha(p) for p in origin.rglob('*') if p.is_file()}
            with patch.object(reuse,'source_lock',return_value=nullcontext()),patch.object(reuse,'verify_source'), \
                 patch.object(run,'validate_cell',side_effect=lambda out,c:fake_row(c)):
                reuse.reuse_completed(origin,root,{},source,jobs)
                manifest=core.read(root/'reuse_manifest.json')
                self.assertEqual([r['cell'] for r in manifest['cells']],[list(jobs[0])])
                dest=core.cell_path(root/'screen',jobs[0])
                self.assertEqual(core.sha(dest/'model.pth'),core.sha(out/'model.pth'))
                self.assertFalse((core.cell_path(root/'screen',jobs[1])/'complete.json').exists())
                self.assertTrue((core.cell_path(root/'original_v8/screen',jobs[1])/'train.log').exists())
                # Simulate interrupt after all copies but before the import manifest.
                (root/'reuse_manifest.json').unlink()
                reuse.reuse_completed(origin,root,{},source,jobs)
                self.assertEqual(manifest,core.read(root/'reuse_manifest.json'))
                with patch.object(reuse,'source_lock',side_effect=AssertionError('must not recopy')):
                    reuse.reuse_completed(origin,root,{},source,jobs)
            self.assertEqual(before,{str(p.relative_to(origin)):core.sha(p) for p in origin.rglob('*') if p.is_file()})

    def test_corrupt_checkpoint_and_changed_arguments_are_rejected(self):
        for corruption in ('checkpoint','arguments'):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as tmp:
                origin,root,out,source,jobs=self.fixture(tmp)
                if corruption=='checkpoint': (out/'model.pth').write_bytes(b'corrupt')
                else: core.save(out/'run_config.json',dict(argv=['different'],protocol='modern'))
                with patch.object(reuse,'source_lock',return_value=nullcontext()),patch.object(reuse,'verify_source'), \
                     patch.object(run,'validate_cell',side_effect=lambda out,c:fake_row(c)):
                    with self.assertRaises(ValueError): reuse.reuse_completed(origin,root,{},source,jobs)
                self.assertFalse((root/'reuse_manifest.json').exists())
                self.assertFalse((root/'screen').exists())

    def test_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            origin,root,out,source,jobs=self.fixture(tmp)
            dest=core.cell_path(root/'screen',jobs[0]); dest.mkdir(parents=True)
            (dest/'model.pth').write_bytes(b'user data')
            with patch.object(reuse,'source_lock',return_value=nullcontext()),patch.object(reuse,'verify_source'), \
                 patch.object(run,'validate_cell',side_effect=lambda out,c:fake_row(c)):
                with self.assertRaisesRegex(ValueError,'destination already differs'):
                    reuse.reuse_completed(origin,root,{},source,jobs)
            self.assertEqual((dest/'model.pth').read_bytes(),b'user data')

    def test_source_compatibility_rejects_other_training_changes(self):
        old=dict(commits=reuse.V8_COMMITS,inputs={},environment={})
        new=dict(old,commits={'modern':'new','july28':'new'})
        with self.assertRaisesRegex(ValueError,'exact original'):
            reuse.verify_source(dict(old,commits={}),new,{})
        with self.assertRaisesRegex(ValueError,'environment'):
            reuse.verify_source(old,dict(new,environment={'changed':True}),{})
        with patch.object(reuse.subprocess,'check_output',return_value='pipeline.py\n'):
            with self.assertRaisesRegex(ValueError,'outside fix1'):
                reuse.verify_source(old,new,{'modern':run.REPO})
        with patch.object(reuse.subprocess,'check_output',return_value='tools/qmd_gradient.py\n'):
            reuse.verify_source(old,new,{'modern':run.REPO})
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp); (repo/'tools').mkdir(); (repo/'tools/qmd_gradient.py').write_text('wrong patch')
            with patch.object(reuse.subprocess,'check_output',return_value='tools/qmd_gradient.py\n'):
                with self.assertRaisesRegex(ValueError,'validation-only'):
                    reuse.verify_source(old,new,{'modern':repo})

    def test_old_live_runner_or_child_blocks_import(self):
        fake=types.SimpleNamespace(LOCK_SH=1,LOCK_NB=4,flock=lambda *a:None)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'.runner.lock').touch()
            core.save(root/'train.process.json',dict(identity={'fixture':True}))
            with patch.dict('sys.modules',{'fcntl':fake}),patch.object(reuse,'alive',return_value=True):
                with self.assertRaisesRegex(RuntimeError,'still alive'):
                    with reuse.source_lock(root): self.fail('must block')

    def test_nested_roots_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'non-nested'):
                reuse.reuse_completed(tmp,Path(tmp)/'nested',{}, {}, [])

    def test_status_separates_imported_and_fix1_completed_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            origin,root,out,source,jobs=self.fixture(tmp)
            core.save(root/'reuse_manifest.json',dict(cells=[dict(cell=jobs[0])]))
            output=io.StringIO()
            with patch('sys.argv',['run','--stage','status','--root',str(root)]), \
                 patch.object(run.cal,'status'),redirect_stdout(output): run.main()
            self.assertIn('Reused original v8: 1 ; fix1 training completed: 0/1',output.getvalue())


if __name__=='__main__': unittest.main()
