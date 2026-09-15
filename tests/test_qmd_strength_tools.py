from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from tools import run_qmd_strength as run
from tools import run_qmd_dual as core


def row(c):
    p, a, s, m = c
    offsets = dict(zip(run.METHODS, (0., 1., .8, .7, 1.2, 1.1)))
    return dict(protocol=p, action=a, seed=s, method=m, srcc_x100=(70. if s == 2 else 82.) + offsets[m], warnings=[])


def fixture(out, c):
    y = np.linspace(.1, .9, 50).tolist()
    result = dict(run=dict(qmd=run.settings(c)), coef=100., loss_x100=0., predictions=dict(label=y, prediction=y))
    for name in ('train_result.json', 'test_result.json'):
        core.save(out/name, result)
    core.save(out/'run_config.json', dict(argv=run.cell_args(c, '4')[5:], protocol='july28'))
    qmd = dict(active_batches=2, epochs=[dict(active_batches=2)], components=dict(terms='kd', batches=2,
        weighted_gt_sum=0., weighted_kd_sum=.1, gradient_verified=True, first_positive_head_gradient_norm=.01))
    core.save(out/'qmd_training.json', qmd)
    return result, qmd


class StrengthRunnerTest(unittest.TestCase):
    def test_all_actions_three_seeds_36_new_no_pilot(self):
        jobs = run.matrix()
        self.assertEqual(len(jobs), 36)
        self.assertEqual(len(set(jobs)), 36)
        for m in run.WEIGHTS:
            self.assertEqual({(c[1], c[2]) for c in jobs if c[3] == m},
                             {(a,s) for a in core.ACTIONS for s in (0,1,2)})
        self.assertEqual(len(run.reference_cells()), 72)
        self.assertFalse(set(jobs).intersection(run.reference_cells()))
        smoke = run.smoke_cells(jobs)
        self.assertEqual(len(smoke), 12)
        self.assertEqual({c[1] for c in smoke}, set(core.ACTIONS))
        self.assertEqual(run.smoke_cells([]), [])

    def test_only_weight_and_tag_change_from_frozen_kd(self):
        for c in run.matrix():
            for smoke in (False, True):
                old = run.full.cell_args(c[:3]+(run.KD,), '3', smoke)
                new = run.cell_args(c, '3', smoke)
                new[new.index('--run_tag')+1] = run.KD
                new[new.index('--qmd_weight')+1] = '1'
                self.assertEqual(old, new)
                self.assertEqual(new[new.index('--qmd_terms')+1], 'kd')
                if smoke:
                    self.assertEqual(new[new.index('--epochs')+1], '2')
                else:
                    self.assertNotIn('--epochs', new)
                for flag in ('--qmd_grad_mode','--score_head','--qmd_rank_weight','--quality_aux_rng_isolated'):
                    self.assertNotIn(flag,new)

    def test_wrong_cell_rejected(self):
        for c in [('modern','Ball',0,next(iter(run.WEIGHTS))), ('july28','Ball',3,next(iter(run.WEIGHTS))),
                  ('july28','Ball',0,run.KD), ('july28','unknown',0,next(iter(run.WEIGHTS)))]:
            with self.assertRaisesRegex(ValueError, 'outside fixed'):
                run.settings(c)

    def test_recorded_strength_and_disabled_gt_validated(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp); c=('july28','Ball',0,next(iter(run.WEIGHTS)))
            data,qmd=fixture(out,c)
            with patch.object(core,'validate_cell',side_effect=lambda *a:row(c)):
                actual=run.validate_cell(out,c)
                self.assertEqual(actual['qmd_weight'], .25)
                self.assertEqual(actual['weighted_gt_sum'], 0)
                data['run']['qmd']['weight']=1.
                core.save(out/'test_result.json',data)
                with self.assertRaisesRegex(ValueError,'configuration mismatch'):
                    run.validate_cell(out,c)
                data['run']['qmd']['weight']=.25
                core.save(out/'test_result.json',data)
                qmd['components']['weighted_gt_sum']=.01
                core.save(out/'qmd_training.json',qmd)
                with self.assertRaisesRegex(ValueError,'disabled component'):
                    run.validate_cell(out,c)

    def test_missing_gradient_is_retained_as_warning(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);c=('july28','Ball',0,next(iter(run.WEIGHTS)))
            _,qmd=fixture(out,c)
            qmd['components'].update(gradient_verified=False,weighted_kd_sum=0)
            core.save(out/'qmd_training.json',qmd)
            with patch.object(core,'validate_cell',side_effect=lambda *a:row(c)):
                self.assertIn('no_verified_split_gradient',run.validate_cell(out,c)['warnings'])

    def test_changed_argv_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);c=('july28','Ball',0,next(iter(run.WEIGHTS)))
            fixture(out,c)
            run.check_args(out,c)
            core.save(out/'run_config.json',{})
            with self.assertRaisesRegex(ValueError,'arguments changed'):
                run.check_args(out,c)

    def test_calibration_beta_unchanged_and_achieved_ratio_scales(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(run.split,'check_warmup') as warm:
            root=Path(temp); c=('july28','Ball',0,next(iter(run.WEIGHTS)))
            ref=core.cell_path(root/'reference',c[:3]+(run.KD,))
            calibration=dict(beta=1.,kd_gradient_norm=.1,main_gradient_norm=1.,epoch=51,
                             achieved_ratio_at_full_ramp=.1,capped=True,scope='fixture')
            core.save(ref/'qmd_training.json',dict(calibration=calibration))
            core.save(root/'new/qmd_training.json',dict(calibration=dict(calibration,achieved_ratio_at_full_ramp=.025)))
            run.check_pre_aux(root/'new',c,root/'reference');warm.assert_called_once()
            core.save(root/'new/qmd_training.json',dict(calibration=dict(calibration,beta=.5)))
            with self.assertRaisesRegex(ValueError,'calibration changed'):
                run.check_pre_aux(root/'new',c,root/'reference')

    def test_full_108_row_summary_all_controls_and_low_seed(self):
        rows=[row(c) for c in run.reference_cells()+run.matrix()]
        report=run.aggregate(rows)
        self.assertEqual(len(report['raw']),108)
        self.assertEqual(len(report['methods']),6)
        weak=next(g for g in report['methods'] if g['method']==next(iter(run.WEIGHTS)))
        self.assertEqual(weak['cells'],18)
        self.assertAlmostEqual(weak['srcc_x100'],79.2)
        self.assertAlmostEqual(weak['delta_vs_base'],1.2)
        self.assertAlmostEqual(weak['delta_vs_ordered'],.2)
        self.assertAlmostEqual(weak['delta_vs_kd1'],.4)
        self.assertAlmostEqual(weak['per_seed']['2']['srcc_x100'],71.2)
        self.assertEqual(set(weak['per_action']),set(core.ACTIONS))
        self.assertAlmostEqual(weak['seed_mean_std'],12/(3**.5))
        for bad in (rows[:-1], rows[:-1]+[rows[0]]):
            with self.assertRaisesRegex(ValueError,'incomplete or duplicate'):
                run.aggregate(bad)

    def test_nonfinite_score_or_mixed_protocol_rejected(self):
        for change in (dict(srcc_x100=float('nan')),dict(protocol='modern')):
            rows=[row(c) for c in run.reference_cells()+run.matrix()]
            rows[0].update(change)
            with self.assertRaisesRegex(ValueError,'nonfinite score or wrong protocol'):
                run.aggregate(rows)

    def test_v10_incomplete_or_different_source_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old=dict(commits=run.V10_COMMITS,inputs={},environment={})
            core.save(root/'source.json',old)
            completion=dict(source=old,experiments_complete=True,new_cells=29,reference_cells=36,reused_kd_cells=6,primary_cells=54)
            core.save(root/'completion.json',completion)
            with self.assertRaisesRegex(ValueError,'completed frozen v10'):
                run.checked_v10(root,old,root/'new')
            completion['new_cells']=30;core.save(root/'completion.json',completion)
            with self.assertRaisesRegex(ValueError,'environment mismatch'):
                run.checked_v10(root,dict(old,environment={'torch':'different'}),root/'new')

    def test_exact_trainer_required(self):
        with patch.object(run.subprocess,'check_output',return_value=run.V10_COMMITS['july28']+'\n'):
            run.verify_training_unchanged(Path('.'))
        with patch.object(run.subprocess,'check_output',return_value='wrong commit'):
            with self.assertRaisesRegex(ValueError,'exact frozen'):
                run.verify_training_unchanged(Path('.'))

    def test_frozen_reference_without_optional_text_logs_can_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);reference=root/'old';c=('july28','Ball',0,run.BASE)
            old=dict(commits=run.V10_COMMITS,inputs={},environment={})
            core.save(reference/'source.json',old)
            core.save(reference/'completion.json',dict(source=old,experiments_complete=True,new_cells=30,
                reference_cells=36,reused_kd_cells=6,primary_cells=54))
            core.save(reference/'manifest.json',dict(fixture=True))
            origin=core.cell_path(reference/'screen',c)
            core.save(origin/'fixture.json',dict(value=1))
            core.save(origin/'complete.json',dict(files={'fixture.json':core.sha(origin/'fixture.json')}))
            with patch.object(run.full,'summarize'),patch.object(run,'reference_cells',return_value=[c]), \
                 patch.object(run,'validate_reference',return_value=row(c)):
                self.assertEqual(run.checked_v10(reference,old,root/'new'),[row(c)])
                self.assertEqual(run.checked_v10(reference,old,root/'new'),[row(c)])
            dest=core.cell_path(root/'new/reference',c)
            self.assertTrue((dest/'fixture.json').exists())
            self.assertFalse((dest/'train.log').exists())
            (dest/'fixture.json').write_text('changed')
            with patch.object(run.full,'summarize'),patch.object(run,'reference_cells',return_value=[c]):
                with self.assertRaisesRegex(ValueError,'already differs'):
                    run.checked_v10(reference,old,root/'new')

    def stage(self, root, stage='all'):
        args=types.SimpleNamespace(stage=stage,gpus='0 1 3 4',july28_repo=str(root),reference_root=str(root/'old'))
        return args

    def mocked_execution(self, stack, root, stage='all', warnings=False):
        for obj,name in [(run.cal,'check_resources'),(run,'verify_training_unchanged'),(run,'checked_v10'),(run,'write_report')]:
            stack.enter_context(patch.object(obj,name))
        stack.enter_context(patch.object(core,'provenance',return_value={}))
        stack.enter_context(patch.object(run,'validate_cell',side_effect=lambda out,c,smoke=False:dict(row(c),warnings=['fixture'] if warnings else [])))
        cmd=stack.enter_context(patch.object(core,'command'))
        wave=stack.enter_context(patch.object(run,'run_wave'))
        return self.stage(root,stage),cmd,wave

    def test_single_gpu_functional_smoke_then_all36_no_pilot(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root=Path(temp);args,cmd,wave=self.mocked_execution(stack,root)
            run.execute(args,root)
            self.assertEqual(wave.call_count,2)
            self.assertEqual(wave.call_args_list[0].args[3],['0'])
            self.assertEqual(len(wave.call_args_list[0].args[2]),12)
            self.assertEqual(wave.call_args_list[1].args[2],run.matrix())
            self.assertEqual(wave.call_args_list[1].args[3],['0','1','3','4'])
            self.assertIn('--kd_weights',cmd.call_args_list[2].args[0])
            self.assertEqual(core.read(root/'completion.json')['new_cells'],36)
            self.assertEqual(core.read(root/'completion.json')['total_rows'],108)

    def test_failed_unit_or_smoke_blocks_full_training(self):
        for failed_unit in (True,False):
            with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
                root=Path(temp);args,cmd,wave=self.mocked_execution(stack,root,warnings=not failed_unit)
                if failed_unit:cmd.side_effect=RuntimeError('unit failed')
                with self.assertRaises(RuntimeError):run.execute(args,root)
                self.assertEqual(wave.call_count,0 if failed_unit else 1)
                self.assertFalse((root/'completion.json').exists())

    def test_screen_requires_gate_and_resume_skips_smoke(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root=Path(temp);args,cmd,wave=self.mocked_execution(stack,root,'screen')
            with self.assertRaisesRegex(RuntimeError,'matching smoke PASS'):run.execute(args,root)
            wave.assert_not_called()
            core.save(root/'smoke/gate.json',dict(decision='QMD_STRENGTH_SMOKE_PASS',cells=12,source={}))
            run.execute(args,root)
            cmd.assert_not_called();self.assertEqual(wave.call_count,1)
            self.assertEqual(wave.call_args.args[2],run.matrix())

    def test_smoke_only_does_not_start_full_training(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root=Path(temp);args,_,wave=self.mocked_execution(stack,root,'smoke')
            package=stack.enter_context(patch.object(core,'package'))
            run.execute(args,root)
            self.assertEqual(wave.call_count,1)
            self.assertFalse((root/'completion.json').exists())
            package.assert_called_once_with(root,'_smoke')

    def test_completed_cell_resumes_without_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c=run.matrix()[0];out=core.cell_path(root,c);out.mkdir(parents=True)
            (out/'model.pth').write_bytes(b'fixture');(out/'model.pth.eval_rng.pt').write_bytes(b'rng')
            core.save(out/'complete.json',dict(files={},checkpoint_sha256=core.sha(out/'model.pth'),eval_rng_sha256=core.sha(out/'model.pth.eval_rng.pt')))
            with patch.object(run,'validate_cell',return_value=row(c)),patch.object(run,'check_pre_aux'),patch.object(core,'command') as cmd:
                state,_=run.run_cell(root,{},c,'4',False)
                self.assertEqual(state,'SKIP');cmd.assert_not_called()

    def test_report_exports_each_seed_and_action(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);report=run.aggregate([row(c) for c in run.reference_cells()+run.matrix()])
            with patch.object(run,'summarize',return_value=report):run.write_report(root)
            self.assertEqual(len((root/'scores.csv').read_text().splitlines()),109)
            for s in run.SEEDS:self.assertEqual(len((root/('scores_seed%d.csv'%s)).read_text().splitlines()),37)
            self.assertEqual(len((root/'per_action.csv').read_text().splitlines()),37)
            self.assertEqual(len((root/'per_seed.csv').read_text().splitlines()),19)

    def test_package_revalidates_full_grid_and_excludes_weights(self):
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            root=Path(temp);core.save(root/'completion.json',dict(experiments_complete=True))
            with patch.object(run,'write_report',side_effect=ValueError('incomplete grid')):
                with self.assertRaisesRegex(ValueError,'incomplete'):run.execute(types.SimpleNamespace(stage='package'),root)
            (root/'model.pth').write_bytes(b'fixture');core.save(root/'summary.json',dict(fixture=True))
            output=core.package(root,'_partial')
            with zipfile.ZipFile(output) as z:
                self.assertIn('summary.json',z.namelist());self.assertNotIn('model.pth',z.namelist())

    def test_status_reports_each_weight_and_frozen_counts(self):
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()) as output:
            root=Path(temp);core.save(root/'manifest.json',dict(jobs=run.matrix(),study='full'))
            run.status(root)
            for value in ('0/36','KD weight 0.25: 0/18','KD weight 0.50: 0/18','Frozen base: 0/18','No pilot filtering'):
                self.assertIn(value,output.getvalue())


if __name__ == '__main__':
    unittest.main()
