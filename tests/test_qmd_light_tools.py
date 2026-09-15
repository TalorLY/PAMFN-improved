import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from tools import run_qmd_dual as core
from tools import run_qmd_light as light


def fixture(out,cell,smoke=True):
    p,a,s,m=cell
    epochs=1 if smoke else core.BUDGET[a]
    settings=light.qmd_settings(cell,smoke)
    run=dict(action=a,seed=s,tag=m,epochs=epochs,feats=2,
             eval_policy='argmax' if p=='modern' else 'legacy_gumbel',qmd=settings,
             ordinal_prototype_weight=.0001 if p=='july28' else 0.,
             ordinal_prototype_target_mode='discrete' if p=='july28' else 'continuous',
             quality_aux_rng_isolated=False if p=='july28' else True)
    out.mkdir(parents=True,exist_ok=True)
    (out/'model.pth').write_bytes(b'fixture')
    if p=='july28':
        (out/'model.pth.eval_rng.pt').write_bytes(b'rng')
    core.save(out/'train_result.json',dict(run=run,saved_model_srcc=.8))
    core.save(out/'test_result.json',dict(run=run,coef=80.,loss_x100=1.,prediction_std=.1,
                                       evaluation=dict(replayed_rng=p=='july28')))
    core.save(out/'model.pth.selection.json',dict(epoch=1,coef=80.))
    (out/'model.pth.epochs.jsonl').write_text('\n'.join(json.dumps(dict(epoch=e,coef=80.)) for e in range(1,epochs+1))+'\n')
    view=dict(epoch=1,modality='V',samples=2,accepted_samples=1,teacher_abs_error_sum=.2,student_abs_error_sum=.3)
    qmd=dict(mode=settings['mode'],epochs=[{}]*epochs,active_batches=1,
             calibration=dict(beta=.5,degenerate=False),view_diagnostics=dict(rows=[view]))
    if p=='modern' and 'reliable' in m:
        qmd['score_gate']=dict(floor=.5,gradient_verified=True,accepted_samples=1,total_samples=2)
    core.save(out/'qmd_training.json',qmd)


class LightToolsTest(unittest.TestCase):
    def test_fixed_counts_and_full_ball_budget(self):
        jobs=light.matrix()
        self.assertEqual(len(jobs),36)
        self.assertEqual(len(set(jobs)),36)
        self.assertEqual(len(light.smoke_cells(jobs)),6)
        self.assertEqual(core.BUDGET['Ball'],500)
        self.assertEqual({c[2] for c in jobs},{0,1})
        self.assertEqual({c[1] for c in jobs},set(core.ACTIONS))
        self.assertEqual([c[0] for c in light.dispatch_order(jobs)[:4]],['modern','july28','july28','modern'])

    def test_probe_is_separate_and_confirmation_uses_correct_third_seed(self):
        probe=light.matrix('ball-mask')
        self.assertEqual(len(probe),3)
        self.assertEqual({c[1] for c in probe},{'Ball'})
        self.assertEqual({c[2] for c in probe},{0,1,42})
        confirm=light.matrix('confirm',light.SCREEN)
        self.assertEqual(len(confirm),18)
        self.assertTrue(all(c[2]==(42 if c[0]=='modern' else 2) for c in confirm))
        with self.assertRaises(ValueError):
            light.matrix('confirm')

    def test_args_have_exact_weights_and_never_rank(self):
        for cell in light.matrix()+light.matrix('ball-mask'):
            argv=light.cell_args(cell,'4')
            config=light.qmd_settings(cell)
            self.assertEqual(float(argv[argv.index('--qmd_weight')+1]),config['weight'])
            self.assertEqual(float(argv[argv.index('--qmd_mask_ratio')+1]),config['mask_ratio'])
            self.assertNotIn('--qmd_rank_weight',argv)
            self.assertNotIn('--epochs',argv)
            if cell[0]=='july28':
                self.assertNotIn('--eval_policy',argv)
                self.assertNotIn('--quality_aux_rng_isolated',argv)
            self.assertEqual(argv[5:],light.cell_args(cell,'0')[5:])

    def test_validate_rejects_wrong_weight_floor_or_missing_views(self):
        cell=('modern','Ball',0,'reliable_score_qmd_floor05')
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            fixture(out,cell)
            self.assertEqual(light.validate_cell(out,cell,True)['srcc_x100'],80.)
            for filename,mutate in [('test_result.json',lambda x:x['run']['qmd'].update(weight=.25)),
                                     ('qmd_training.json',lambda x:x['score_gate'].update(floor=0)),
                                     ('qmd_training.json',lambda x:x.update(view_diagnostics={}))]:
                fixture(out,cell)
                value=core.read(out/filename); mutate(value); core.save(out/filename,value)
                with self.assertRaises(ValueError):
                    light.validate_cell(out,cell,True)

    def test_resume_skips_complete_cells_and_rejects_changed_checkpoint(self):
        cell=('july28','Ball',0,'ordered_score_qmd_w010')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            out=core.cell_path(root,cell)
            calls=[]
            def command(argv,cwd,log):
                calls.append(argv)
                fixture(out,cell)
            with patch.object(core,'command',side_effect=command):
                self.assertEqual(light.run_cell(root,dict(july28=root),cell,'0',True)[0],'FINISH')
                self.assertEqual(light.run_cell(root,dict(july28=root),cell,'4',True)[0],'SKIP')
                self.assertEqual(len(calls),2)
                (out/'model.pth').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'checkpoint changed'):
                    light.run_cell(root,dict(july28=root),cell,'0',True)

    def test_interrupted_cell_moves_partial_files_and_restarts_only_that_cell(self):
        cell=('modern','Ball',0,'reliable_score_qmd_floor05')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); out=core.cell_path(root,cell); out.mkdir(parents=True)
            (out/'train.log').write_text('partial training')
            def command(argv,cwd,log):
                fixture(out,cell)
            with patch.object(core,'command',side_effect=command):
                light.run_cell(root,dict(modern=root),cell,'0',True)
            preserved=list((out/'attempts').glob('*/train.log'))
            self.assertEqual(len(preserved),1)
            self.assertEqual(preserved[0].read_text(),'partial training')

    def test_reference_checks_environment_hash_and_recipe(self):
        cell=('modern','Ball',0,'reliable_score_qmd_floor05')
        source=dict(commits=dict(light.REFERENCE_COMMITS),environment=dict(torch='fixture'),inputs=dict(a='hash'))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); reference=root/'old'; output=root/'new'
            core.save(reference/'source.json',source)
            core.save(reference/'completion.json',dict(source=source,cells=126,experiments_complete=True))
            origin=core.cell_path(reference/'screen',('modern','Ball',0,'score_qmd'))
            expected=dict(mode='score_qmd',weight=1.,mask_ratio=.15,mask_gt_weight=.5,target_ratio=.075,beta_cap=1.,warmup=.1,rampup=.1)
            core.save(origin/'test_result.json',dict(run=dict(qmd=expected)))
            core.save(origin/'complete.json',dict(files={'test_result.json':core.sha(origin/'test_result.json')},metrics=dict(srcc_x100=82.)))
            record=dict(protocol='modern',action='Ball',seed=0,method='score_qmd',srcc_x100=82.)
            with patch.object(core,'validate_cell',return_value=record):
                rows=light.checked_reference(reference,source,[cell],output)
                self.assertEqual(rows[0]['origin'],'reused_v2_control')
                with self.assertRaisesRegex(ValueError,'environment'):
                    light.checked_reference(reference,dict(source,environment={}),[cell],output)
                (origin/'test_result.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'reference file changed'):
                    light.checked_reference(reference,source,[cell],output)

    def test_summary_uses_paired_controls_and_keeps_ball_probe_separate(self):
        jobs=light.matrix('ball-mask')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            refs=[]
            for c in jobs:
                fixture(core.cell_path(root/'screen',c),c,False)
                refs.append(dict(protocol=c[0],action=c[1],seed=c[2],srcc_x100=81.))
            result=light.summarize(root,jobs,refs)
            self.assertEqual(result['methods'][0]['delta_vs_control'],-1.)
            self.assertEqual(result['methods'][0]['actions'],['Ball'])
            self.assertEqual(len(result['raw']),3)

    def test_all_stage_produces_complete_36_cell_summary_and_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'screen'
            jobs=light.matrix()
            refs=[dict(protocol=p,action=a,seed=s,srcc_x100=81.)
                  for p,a,s in sorted(set((c[0],c[1],c[2]) for c in jobs))]
            def wave(output,repos,cells,gpus,smoke=False):
                for c in cells:
                    out=core.cell_path(output,c)
                    fixture(out,c,smoke)
                    core.save(out/'complete.json',dict(fixture=True))
            lock=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *args:None)
            with patch.dict('sys.modules',{'fcntl':lock}), \
                 patch('sys.argv',['runner','--stage','all','--root',str(root),'--gpus','0 1']), \
                 patch.object(core,'provenance',return_value=dict(environment={},inputs={},commits={})), \
                 patch.object(core,'command'),patch.object(light,'checked_reference',return_value=refs), \
                 patch.object(light,'run_wave',side_effect=wave):
                light.main()
            self.assertEqual(core.read(root/'completion.json')['new_cells'],36)
            self.assertEqual(core.read(root/'completion.json')['reference_cells'],24)
            self.assertEqual(core.read(root/'smoke/gate.json')['decision'],'QMD_LIGHT_SMOKE_PASS')
            self.assertEqual(len(core.read(root/'pilot_summary.json')['raw']),18)
            report=core.read(root/'summary.json')
            self.assertEqual(len(report['raw']),36)
            self.assertEqual(len(report['methods']),3)
            self.assertTrue(all(r['delta_vs_control']==-1. for r in report['methods']))
            self.assertTrue((root.parent/'screen_return.zip').exists())


if __name__=='__main__':
    unittest.main()
