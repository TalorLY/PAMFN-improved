from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

from tools import run_qmd_hbn as run
from tools import run_qmd_dual as core


def result_for(cell, smoke=False, delta=.2):
    return dict(cell=list(cell),recipe=run.recipe(),calibrated_layers=1,checkpoint_sha256='fixture',
        baseline=dict(coef=80.),checks=dict(original_replay=True,identity_exact=True,buffers_restored=True,
        parameters_unchanged=True,tiny_state_reload_exact=True,full_vs_cache_exact=smoke,
        test_labels_used_for_calibration=False),results=[dict(alpha=a,coef=80.+(delta if a else 0.),
        delta=(80.+delta)-80. if a else 0.) for a in run.ALPHAS])


class RunnerTest(unittest.TestCase):
    def test_fixed_matrix_and_protocol_arguments(self):
        cells=run.matrix()
        self.assertEqual(len(cells),36); self.assertEqual(len(set(cells)),36)
        self.assertEqual(len(run.smoke_cells()),6)
        self.assertEqual(set(c[2] for c in cells),{0,1})
        self.assertTrue(all(c[1]=='Ball' for c in cells[:6]))
        for p,a,s,m in cells:
            argv=core.cell_args(p,a,s,m,'-1')[3:]
            self.assertEqual(argv[argv.index('--run_tag')+1],m)
            self.assertNotIn('--epochs',argv)
            self.assertEqual('--eval_policy' in argv,p=='modern')
            self.assertEqual('--quality_aux_rng_isolated' in argv,False)

    def test_preflight_authenticates_rng_separate_field_and_writes_readonly_specs(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            tmp=Path(tmp); ref=tmp/'ref'; out=tmp/'out'
            source=dict(commits=dict(run.REFERENCE_COMMITS),inputs={'fixture':'same'},environment={'torch':'same'})
            core.save(ref/'source.json',source)
            core.save(ref/'completion.json',dict(source=source,cells=126,experiments_complete=True))
            for p,a,s,m in run.matrix():
                cell=(p,a,s,m); origin=core.cell_path(ref/'screen',cell); origin.mkdir(parents=True)
                (origin/'model.pth').write_bytes(b'fixture checkpoint')
                qmd=dict(mode='off' if m=='ordered' else 'score_qmd',weight=1.,mask_ratio=.15,
                         mask_gt_weight=.5,target_ratio=.075,beta_cap=1.,warmup=.1,rampup=.1)
                core.save(origin/'test_result.json',dict(run=dict(qmd=qmd),coef=80.,loss_x100=1.,prediction_std=.1))
                record=dict(files={'test_result.json':core.sha(origin/'test_result.json')},
                            checkpoint_sha256=core.sha(origin/'model.pth'),metrics=dict(srcc_x100=80.))
                if p=='july28':
                    (origin/'model.pth.eval_rng.pt').write_bytes(b'fixture RNG')
                    record['eval_rng_sha256']=core.sha(origin/'model.pth.eval_rng.pt')
                core.save(origin/'complete.json',record)
            repos=dict(modern=tmp,july28=tmp)
            with patch.object(core,'provenance',return_value=source), patch.object(core,'validate_cell',return_value=dict(srcc_x100=80.)):
                run.preflight(out,repos,ref)
                target=core.cell_path(out/'reference',('july28','Ball',0,'ordered'))
                spec=core.read(target/'spec.json')
                self.assertTrue(spec['eval_rng_sha256'])
                self.assertFalse((target/'model.pth').exists())
                self.assertFalse((target/'model.pth.eval_rng.pt').exists())
                origin=core.cell_path(ref/'screen',('july28','Ball',0,'ordered'))
                (origin/'model.pth.eval_rng.pt').write_bytes(b'changed RNG')
                with self.assertRaisesRegex(ValueError,'RNG is not authenticated'):
                    run.preflight(out,repos,ref)

    def test_validation_rejects_wrong_identity_partial_matrix_and_missing_smoke(self):
        cell=run.matrix()[0]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); result=result_for(cell)
            core.save(out/'result.json',result)
            run.validate_result(out,cell)
            with self.assertRaisesRegex(ValueError,'smoke'):
                run.validate_result(out,cell,True)
            result['results'].pop(); core.save(out/'result.json',result)
            with self.assertRaisesRegex(ValueError,'incomplete'):
                run.validate_result(out,cell)

    def test_completion_hash_tamper_rejected(self):
        cell=run.matrix()[0]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); core.save(out/'result.json',result_for(cell))
            core.save(out/'complete.json',dict(files={'result.json':core.sha(out/'result.json')},checkpoint_sha256='fixture'))
            run.validate_completed(out,cell)
            core.save(out/'result.json',result_for(cell,delta=-4.))
            with self.assertRaisesRegex(ValueError,'changed'):
                run.validate_completed(out,cell)

    def test_summary_all_108_rows_and_no_cross_protocol_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(run,'validate_completed',side_effect=lambda out,cell:result_for(cell)):
                report=run.summarize(root)
            self.assertEqual(len(report['raw']),108)
            self.assertEqual(len(report['groups']),9)
            self.assertEqual(len(report['july28_combination_pairs']),36)
            self.assertAlmostEqual(report['groups'][1]['delta'],.2)
            self.assertTrue(all(r['combined_minus_ordered']==0 for r in report['july28_combination_pairs']))
            self.assertIsNone(run.mean_complete([80.,None,90.]))

    def test_safe_resume_skips_completed_and_preserves_partial_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'screen'; cell=run.matrix()[0]; out=core.cell_path(root,cell)
            spec=core.cell_path(root.parent/'reference',cell)/'spec.json'
            core.save(spec,dict(checkpoint_sha256='fixture'))
            core.save(out/'result.json',result_for(cell))
            core.save(out/'complete.json',dict(files={'result.json':core.sha(out/'result.json')},checkpoint_sha256='fixture'))
            with patch.object(core,'command') as command:
                self.assertEqual(run.run_cell(root,{'modern':Path(tmp)},cell,'0')[0],'SKIP')
                command.assert_not_called()
            # A second incomplete cell must retain the failed log on retry.
            cell=run.matrix()[1]; out=core.cell_path(root,cell); out.mkdir(parents=True)
            (out/'eval.log').write_text('interrupted fixture')
            def fake_command(argv,cwd,log):
                core.save(out/'result.json',result_for(cell))
            with patch.object(core,'command',side_effect=fake_command):
                self.assertEqual(run.run_cell(root,{'modern':Path(tmp)},cell,'0')[0],'FINISH')
            self.assertEqual(len(list((out/'attempts').rglob('eval.log'))),1)

    def test_status_shows_fixed_completion_counts_without_live_process_claim(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as output:
            run.status(Path(tmp))
            self.assertIn('0/36',output.getvalue()); self.assertIn('0/108',output.getvalue())
            self.assertIn('NEW TRAINING: 0',output.getvalue())

    def test_package_omits_weights_but_includes_statistics_json_and_predictions(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp)/'results'; root.mkdir()
            for name in ('model.pth','alpha_0.1.bn.pt','calibration.json','predictions.csv','summary.json'):
                (root/name).write_text('fixture')
            path=core.package(root,'_partial')
            with zipfile.ZipFile(path) as z:
                self.assertEqual(set(z.namelist()),{'calibration.json','predictions.csv','summary.json'})

    def test_full_launcher_smoke_screen_auto_package_and_second_run_skips(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp)/'study'; source=dict(fixture='source')
            def preflight(out,repos,ref):
                core.save(out/'source.json',source)
                for cell in run.matrix():
                    core.save(core.cell_path(out/'reference',cell)/'spec.json',dict(cell=list(cell),checkpoint_sha256='fixture'))
                return source
            def command(argv,cwd,log):
                if '--spec' not in argv:  # Startup unit checks are covered by the real suite itself.
                    return
                cell=core.read(argv[argv.index('--spec')+1])['cell']
                out=Path(argv[argv.index('--output')+1])
                core.save(out/'result.json',result_for(cell,'--smoke' in argv))
            fake_lock=types.SimpleNamespace(flock=lambda *a:None,LOCK_EX=1,LOCK_NB=2)
            argv=['runner','--root',str(root),'--gpus','0 1']
            with patch.dict('sys.modules',{'fcntl':fake_lock}), patch('sys.argv',argv), \
                 patch.object(run,'check_resources'), patch.object(run,'preflight',side_effect=preflight), \
                 patch.object(core,'command',side_effect=command) as calls:
                run.main()
                self.assertEqual(calls.call_count,43)  # Unit + 6 smoke + 36 formal checkpoint cells.
                self.assertEqual(core.read(root/'runner.json')['state'],'finished')
                self.assertEqual(core.read(root/'completion.json')['scoring_views'],108)
                self.assertTrue((root.parent/'study_return.zip').exists())
                calls.reset_mock(); run.main(); calls.assert_not_called()
                gate=core.read(root/'smoke/gate.json'); gate['decision']='QMD_HBN_SMOKE_FAIL'
                core.save(root/'smoke/gate.json',gate)
                with self.assertRaisesRegex(RuntimeError,'smoke PASS'):
                    run.main()

    def test_orphan_process_blocks_duplicate_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); core.save(root/'cell/eval.process.json',dict(identity={'pid':1}))
            fake_lock=types.SimpleNamespace(flock=lambda *a:None,LOCK_EX=1,LOCK_NB=2)
            with patch.dict('sys.modules',{'fcntl':fake_lock}), patch('sys.argv',['runner','--root',str(root)]), \
                 patch.object(run,'alive',return_value=True), self.assertRaisesRegex(RuntimeError,'live child'):
                run.main()


if __name__=='__main__':
    unittest.main()
