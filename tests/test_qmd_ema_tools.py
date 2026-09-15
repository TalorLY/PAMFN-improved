from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

from models.ema_shadow import recipe
from tools import run_qmd_ema as runner
from tools import run_qmd_dual as core


def fixture(out, cell, smoke=False):
    out.mkdir(parents=True,exist_ok=True)
    _,a,s,m=cell
    epochs=runner.SMOKE_EPOCHS if smoke else core.BUDGET[a]
    run=dict(action=a,seed=s,tag=m,epochs=epochs,feats=2,eval_policy='argmax',eval_seed=20260908,
             ordinal_prototype_weight=0.,qmd=runner.settings(smoke),ema=dict(recipe=recipe(),view='online'))
    (out/'model.pth').write_bytes(b'fixture')
    core.save(out/'train_result.json',dict(run=run,saved_model_srcc=.81))
    core.save(out/'test_result.json',dict(run=run,coef=81.,loss_x100=1.,prediction_std=.1))
    core.save(out/'model.pth.selection.json',dict(epoch=1,coef=81.))
    (out/'model.pth.epochs.jsonl').write_text(''.join(json.dumps(dict(epoch=e,coef=81.))+'\n' for e in range(1,epochs+1)))
    core.save(out/'qmd_training.json',dict(mode='score_qmd',active_batches=7*epochs,
        calibration=dict(beta=.5,degenerate=False,scope='score_head_parameters_first_active_training_batch'),
        epochs=[dict(epoch=e,active_batches=7) for e in range(1,epochs+1)]))
    best={}
    for view,score in dict(online_bn=81.2,ema_short=83. if s==0 else 78.,ema_long=82. if s==0 else 79.).items():
        (out/(view+'.pth')).write_bytes(('fixture'+view).encode())
        metrics=dict(coef=score,loss_x100=1.,prediction_std=.1,prediction_std_ratio=1.,mae=.1,bias=.01)
        history=[dict(epoch=e,updates=7*e,view=view,bn_calibration=dict(source='train',samples=200,batches=7,layers=1),**metrics)
                 for e in range(1,epochs+1)]
        best[view]=history[0]
        core.save(out/(view+'.selection.json'),history[0])
        core.save(out/(view+'.test_result.json'),dict(run=dict(run,ema=dict(recipe=recipe(),view=view)),**metrics))
        (out/(view+'.epochs.jsonl')).write_text(''.join(json.dumps(h)+'\n' for h in history))
    core.save(out/'ema_training.json',dict(recipe=recipe(),updates=7*epochs,epochs=epochs,best=best))


class EMAToolsTest(unittest.TestCase):
    def test_fixed_six_training_trajectories_and_full_budgets(self):
        jobs=runner.matrix()
        self.assertEqual(len(jobs),6)
        self.assertEqual(len(set(jobs)),6)
        self.assertEqual({c[1] for c in jobs},{'Ball','Hoop','Ribbon'})
        self.assertEqual({c[2] for c in jobs},{0,1})
        self.assertEqual(len(runner.smoke_cells(jobs)),2)
        self.assertEqual(len(runner.matrix('all-actions')),12)
        self.assertEqual({c[2] for c in runner.matrix('confirm')},{42})
        for c in jobs:
            args=runner.cell_args(c,'0')
            self.assertIn('--ema_screen',args)
            self.assertNotIn('--epochs',args)
            self.assertNotIn('--score_head',args)
            self.assertNotIn('--ordinal_prototype_weight',args)
            self.assertEqual(args[5:],runner.cell_args(c,'4')[5:])

    def test_reload_recipe_and_histories_checked(self):
        c=runner.matrix()[0]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            fixture(out,c,True)
            self.assertEqual(len(runner.validate_cell(out,c,True)['ema_views']),3)
            t=core.read(out/'ema_short.test_result.json')
            t['coef']+=.1
            core.save(out/'ema_short.test_result.json',t)
            with self.assertRaisesRegex(ValueError,'mismatch'):
                runner.validate_cell(out,c,True)
            fixture(out,c,True)
            (out/'ema_long.epochs.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError,'history'):
                runner.validate_cell(out,c,True)

    def test_resume_validates_all_checkpoints_and_archives_incomplete_attempts(self):
        c=runner.matrix()[0]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            out=core.cell_path(root,c)
            out.mkdir(parents=True)
            (out/'train.log').write_text('interrupted')
            with patch.object(core,'command',side_effect=lambda *args:fixture(out,c,True)) as command:
                self.assertEqual(runner.run_cell(root,dict(modern=root),c,'0',True)[0],'FINISH')
                self.assertEqual(command.call_count,5)
                self.assertEqual(runner.run_cell(root,dict(modern=root),c,'4',True)[0],'SKIP')
                self.assertEqual(command.call_count,5)
            self.assertEqual(len(list((out/'attempts').glob('*/train.log'))),1)
            (out/'ema_short.pth').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'extra checkpoint changed'):
                runner.run_cell(root,dict(modern=root),c,'4',True)

    def test_whole_flow_counts_training_not_views_and_retains_negative_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'study'
            def reference(src,source,jobs,dest):
                rows=[]
                for p,a,s,m in jobs:
                    out=core.cell_path(dest/'reference',(p,a,s,'score_qmd'))
                    out.mkdir(parents=True,exist_ok=True)
                    (out/'model.pth.epochs.jsonl').write_text(''.join(json.dumps(dict(epoch=e,coef=81.))+'\n' for e in range(1,core.BUDGET[a]+1)))
                    rows.append(dict(action=a,seed=s,srcc_x100=81.))
                core.save(dest/'reference_manifest.json',dict(records=rows))
                return rows
            def wave(output,repos,jobs,gpus,smoke=False):
                for c in jobs:
                    out=core.cell_path(output,c)
                    fixture(out,c,smoke)
                    core.save(out/'complete.json',dict(fixture=True))
            lock=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *args:None)
            with patch.dict('sys.modules',{'fcntl':lock}),patch('sys.argv',['runner','--root',str(root)]), \
                 patch.object(runner.cal,'check_resources'),patch.object(core,'provenance',return_value={'fixture':True}), \
                 patch.object(runner.light,'checked_reference',side_effect=reference),patch.object(core,'command'), \
                 patch.object(runner,'run_wave',side_effect=wave),redirect_stdout(io.StringIO()):
                runner.main()
            complete=core.read(root/'completion.json')
            self.assertEqual(complete['new_cells'],6)
            self.assertEqual(complete['scoring_views'],24)
            result=core.read(root/'summary.json')
            self.assertEqual(len(result['raw']),24)
            short=next(g for g in result['groups'] if g['view']=='ema_short')
            self.assertEqual(short['delta_vs_online'],-.5)
            self.assertEqual(short['per_seed'],{'0':83.,'1':78.})
            with zipfile.ZipFile(root.parent/'study_return.zip') as z:
                self.assertIn('completion.json',z.namelist())
                self.assertTrue(any(n.endswith('ema_short.epochs.jsonl') for n in z.namelist()))
                self.assertFalse(any(n.endswith('.pth') for n in z.namelist()))
            buf=io.StringIO()
            with redirect_stdout(buf):
                runner.status(root)
            self.assertIn('New training completed: 6/6',buf.getvalue())
            self.assertIn('Shadow/BN reload results: 18/18',buf.getvalue())
            # Historical discrepancies are visible, never substituted for same-trajectory controls.
            ref=core.cell_path(root/'reference',('modern','Ball',0,'score_qmd'))/'model.pth.epochs.jsonl'
            ref.write_text(ref.read_text().replace('81.0','82.0'))
            checked=runner.summarize(root,runner.matrix(),core.read(root/'reference_manifest.json')['records'])
            self.assertFalse(checked['student_reference_trajectory_matches'])
            self.assertTrue(checked['warnings'])
            self.assertEqual(checked['groups'],result['groups'])

    def test_partial_archive_and_orphan_child_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            core.save(root/'manifest.json',dict(study='screen',jobs=runner.matrix()))
            runner.execute(types.SimpleNamespace(stage='package'),root)
            self.assertTrue((root.parent/(root.name+'_partial_return.zip')).exists())
            self.addCleanup((root.parent/(root.name+'_partial_return.zip')).unlink)
            core.save(root/'screen/train.process.json',dict(identity=dict(pid=123)))
            lock=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *args:None)
            with patch.dict('sys.modules',{'fcntl':lock}),patch('sys.argv',['runner','--root',str(root)]), \
                 patch.object(runner,'alive',return_value=True),patch.object(runner,'execute') as execute:
                with self.assertRaisesRegex(RuntimeError,'live child'):
                    runner.main()
                execute.assert_not_called()


if __name__=='__main__':
    unittest.main()
