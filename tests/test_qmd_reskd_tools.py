from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from tools import run_qmd_reskd as run
from tools import run_qmd_dual as core
from tools.qmd_reskd_config import metadata,parameter_count
from tests.test_qmd_residual_tools import teacher


def fixture(root,cell,smoke=False):
    out=core.cell_path(root/('smoke' if smoke else 'screen'),cell)
    t=teacher(root,cell[1],cell[2]);source={k:t[k] for k in ('checkpoint_sha256','eval_rng_sha256','expected_srcc_x100','action','seed')}
    n=100 if cell[1] in ('TES','PCS') else 50;y=np.linspace(.1,.9,n)
    epochs=2 if smoke else core.BUDGET[cell[1]];weight=run.RECIPES[cell[3]]['weight']
    meta=metadata(cell[1],cell[2],cell[3],epochs,smoke)
    def data(p):return dict(coef=100.,loss_x100=float(np.mean((p-y)**2)*100),prediction_std=float(np.std(p)),
        predictions=dict(label=y.tolist(),prediction=p.tolist(),order='native_test_loader'))
    bias=.2;initial='initial-'+str(meta['branch']['hidden'])
    stats=dict(source_split='train',samples=200,passes=1,feature_dim=768,global_bias=bias,
        mean_label=.5,mean_anchor=.3,feature_scale_min=.01,feature_scale_max=1.,statistics_sha256='stats')
    check=dict(source=source,baseline=data(y-bias),bias_only=data(y),zero_branch_exact=True,bias_only_rank_preserved=True,
        branch_parameters=parameter_count(meta['branch']),branch_config=meta['branch'],
        initial_branch_sha256=initial,initial_anchor_sha256='anchor',training_statistics_sha256='stats')
    test=dict(data(y),run=meta,source=source,evaluation=dict(policy=meta['eval_policy'],replayed_rng=True),
        baseline_srcc_x100=100.,training_statistics_sha256='stats',global_bias=bias,
        correction_mean=0.,correction_std=0.,correction_max_abs=0.,correction_saturation=0.)
    train=dict(run=meta,source=source,saved_model_srcc=1.)
    compact={k:v for k,v in test.items() if k not in ('predictions','run','source','evaluation','baseline_srcc_x100','training_statistics_sha256')}
    qmd=dict(mode='residual_space_kd',recipe=run.RECIPES[cell[3]],anchor_unchanged=True,anchor_gradients_none=True,
        anchor_state_before='anchor',anchor_state_after='anchor',branch_updated=True,
        initial_branch_sha256=initial,final_branch_sha256='updated',epoch_zero_eligible=False,
        statistics_before='stats',statistics_after='stats',statistics_unchanged=True,
        lr=.0001,weight_decay=.01,optimizer='AdamW',kd_first_epoch=1,
        active_batches=epochs,weighted_kd_sum=epochs*weight,positive_kd_gradient_batches=epochs if weight else 0,
        conflicts=0,cosine_batches=epochs,calibration=dict(run.calibration_contract(),steps=epochs,zero_gradient_batches=0,capped_batches=0,
            beta_min=.3,beta_max=.3,beta_mean=.3,max_effective_ratio=weight*.075*.5,
            first_positive=dict(epoch=1,step=1) if weight else None),extra_stage_epochs=epochs,
        epochs=[dict(epoch=e,batches=1,active_batches=1,weighted_kd_sum=weight,
            positive_kd_gradient_batches=1 if weight else 0,zero_gradient_batches=0,capped_batches=0,
            beta_min=.3,beta_mean=.3,beta_max=.3,kd_ratio_mean=weight*.075*.5*min(1.,e/(1 if smoke else 5)),
            kd_ratio_max=weight*.075*.5*min(1.,e/(1 if smoke else 5)),gt_mean=.1,energy_mean=.01,
            weighted_energy_mean=run.RECIPES[cell[3]]['energy']*.01,
            supervised_mean=.1+run.RECIPES[cell[3]]['energy']*.01) for e in range(1,epochs+1)])
    qmd['calibration']['max_effective_ratio']=max(r['kd_ratio_max'] for r in qmd['epochs'])
    for name,value in [('teacher_check.json',check),('test_result.json',test),('train_result.json',train),
        ('training_statistics.json',stats),('qmd_training.json',qmd),('model.pth.selection.json',dict(epoch=1,**compact)),
        ('last_epoch_result.json',dict(epoch=epochs,**compact)),
        ('run_config.json',dict(argv=run.cell_args(root,cell,'3',smoke)[5:],protocol='july28'))]:core.save(out/name,value)
    (out/'model.pth.epochs.jsonl').write_text(''.join(json.dumps(dict(epoch=e,**compact))+'\n' for e in range(1,epochs+1)))
    return out


class ResidualKDRunnerTest(unittest.TestCase):
    def test_fixed_factorial_grid_and_all_seeds(self):
        self.assertEqual(len(run.matrix()),108);self.assertEqual(len(set(run.matrix())),108)
        self.assertEqual(len(run.smoke_cells(run.matrix())),36);self.assertEqual(run.smoke_cells([]),[])
        self.assertEqual(len(run.reference_cells()),72)
        for m in run.RECIPES:
            self.assertEqual({(c[1],c[2]) for c in run.matrix() if c[3]==m},
                {(a,s) for a in core.ACTIONS for s in (0,1,2)})
        self.assertEqual({r['energy'] for r in run.RECIPES.values()},{0.,1.})

    def test_matched_energy_gt_controls_and_kd_strength(self):
        for r in run.RECIPES.values():
            self.assertTrue(any(c['energy']==r['energy'] and c['weight']==0 for c in run.RECIPES.values()))
        self.assertEqual({r['weight'] for r in run.RECIPES.values() if r['weight']},{.5})

    def test_retained_references_are_base_ordered_e_f(self):
        self.assertEqual(run.REFERENCES,('base','ordered','ordered_bc_mlp16_gt','ordered_bc_mlp16_kd_w050'))

    def test_reference_import_cannot_write_inside_old_results(self):
        with tempfile.TemporaryDirectory() as temp:
            old=Path(temp)/'old'
            for new in (old,old/'new',old.parent):
                with self.assertRaisesRegex(ValueError,'separate from old'):
                    run.checked_v13(old,{},new)

    def test_native_budgets_and_separate_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);teacher(root)
            for m in run.RECIPES:
                c=('july28','Ball',0,m);args=run.cell_args(root,c,'4')
                self.assertEqual(args[2],'tools/train_qmd_reskd.py');self.assertNotIn('--epochs',args)
                self.assertEqual(run.cell_args(root,c,'4',True)[-3:],['--smoke','--epochs','2'])
            with self.assertRaises(ValueError):run.cell_args(root,('july28','Ball',3,run.GT),'4')

    def test_all_six_valid_smoke_and_formal_records(self):
        for smoke in (False,True):
            for m in run.RECIPES:
                with tempfile.TemporaryDirectory() as temp:
                    cell=('july28','Ball',0,m);out=fixture(Path(temp),cell,smoke)
                    row=run.validate_cell(out,cell,smoke)
                    self.assertEqual(row['srcc_x100'],100.);self.assertFalse(row['warnings'])

    def tamper(self,file,key,value,message,method=None):
        with tempfile.TemporaryDirectory() as temp:
            c=('july28','Ball',0,method or run.GT);out=fixture(Path(temp),c)
            d=core.read(out/file);d[key]=value;core.save(out/file,d)
            with self.assertRaisesRegex(ValueError,message):run.validate_cell(out,c)

    def test_test_statistics_rejected(self):self.tamper('training_statistics.json','source_split','test','only from training')
    def test_statistics_refit_rejected(self):self.tamper('qmd_training.json','statistics_after','bad','statistics changed')
    def test_anchor_change_rejected(self):self.tamper('qmd_training.json','anchor_state_after','bad','Ordered state')
    def test_epoch_zero_rejected(self):self.tamper('model.pth.selection.json','epoch',0,'epoch zero')
    def test_changed_optimizer_rejected(self):self.tamper('qmd_training.json','lr',.001,'optimizer changed')
    def test_nonfinite_selected_rejected(self):self.tamper('train_result.json','saved_model_srcc',float('nan'),'selection/reload')
    def test_changed_bias_rejected(self):self.tamper('test_result.json','global_bias',.3,'bias or sample')
    def test_gt_cannot_contain_kd(self):self.tamper('qmd_training.json','positive_kd_gradient_batches',1,'GT-only')

    def mutate_calibration(self,mutation,message):
        with tempfile.TemporaryDirectory() as temp:
            c=('july28','Ball',0,run.ENERGY_RESIDUAL);out=fixture(Path(temp),c)
            q=core.read(out/'qmd_training.json');mutation(q);core.save(out/'qmd_training.json',q)
            with self.assertRaisesRegex(ValueError,message):run.validate_cell(out,c)

    def test_gradient_ratio_bound_rejected(self):
        self.mutate_calibration(lambda q:q['epochs'][0].update(kd_ratio_max=1.),'bounded ratio')

    def test_energy_accounting_rejected(self):
        self.mutate_calibration(lambda q:q['epochs'][0].update(supervised_mean=1.),'energy objective')

    def test_wrong_adaptive_policy_rejected(self):
        self.mutate_calibration(lambda q:q['calibration'].update(policy='fixed_first_batch'),'contract changed')

    def test_wrong_first_effective_step_rejected(self):
        self.mutate_calibration(lambda q:q['calibration'].update(first_positive=dict(epoch=1,step=500)),'outside history')

    def test_ineffective_kd_reported_not_silently_passed(self):
        with tempfile.TemporaryDirectory() as temp:
            c=('july28','Ball',0,run.RESIDUAL);out=fixture(Path(temp),c)
            q=core.read(out/'qmd_training.json');q['positive_kd_gradient_batches']=0;q['weighted_kd_sum']=0.
            q['calibration'].update(first_positive=None,max_effective_ratio=0.)
            for e in q['epochs']:e.update(positive_kd_gradient_batches=0,weighted_kd_sum=0.,kd_ratio_mean=0.,kd_ratio_max=0.)
            core.save(out/'qmd_training.json',q)
            self.assertIn('no_effective_branch_kd',run.validate_cell(out,c)['warnings'])

    def test_pairing_and_initialization(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(core,'ACTIONS',('Ball',)):
            root=Path(temp)
            for m in run.RECIPES:fixture(root,('july28','Ball',0,m))
            run.check_pairing(root/'screen',seeds=(0,))
            path=core.cell_path(root/'screen',('july28','Ball',0,run.GT))/'teacher_check.json'
            d=core.read(path);d['initial_branch_sha256']='bad';core.save(path,d)
            with self.assertRaisesRegex(ValueError,'initialization differs'):run.check_pairing(root/'screen',seeds=(0,))

    def test_live_status_has_completed_and_total(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);core.save(root/'manifest.json',dict(jobs=run.matrix(),study='full'))
            buf=io.StringIO()
            with redirect_stdout(buf):run.status(root)
            self.assertIn('0/108',buf.getvalue());self.assertIn('0/36',buf.getvalue())
            self.assertEqual(buf.getvalue().count(': 0/18'),6)

    def test_aggregate_keeps_lower_seed1_and_rejects_omission(self):
        rows=[dict(protocol='july28',action=a,seed=s,method=m,srcc_x100=(60. if s==1 else 80.)+i)
            for a in core.ACTIONS for s in (0,1,2) for i,m in enumerate(run.METHODS)]
        report=run.full.aggregate(rows,methods=run.METHODS,controls=((run.ORDERED,'ordered'),(run.ENERGY_GT,'energy_gt')))
        self.assertEqual(len(report['raw']),180)
        self.assertEqual(report['methods'][-1]['per_seed']['1']['srcc_x100'],69.)
        self.assertAlmostEqual(report['methods'][-1]['srcc_x100'],(89+69+89)/3)
        with self.assertRaises(ValueError):run.full.aggregate([r for r in rows if r['seed']!=1],methods=run.METHODS)

    def test_resume_requires_all_training_artifacts(self):
        with patch.object(core,'run_cell') as runner,patch.object(run,'cell_args',return_value=[]):
            run.run_cell(Path('/tmp/root/screen'),{},('july28','Ball',0,run.GT),'0',False)
            self.assertEqual(runner.call_args.kwargs['extra_required'],
                ('teacher_check.json','training_statistics.json','last_epoch_result.json'))

    def test_full_summary_csv_and_return_archive(self):
        import csv
        from zipfile import ZipFile
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'v14';teachers={}
            for c in run.matrix():
                out=fixture(root,c)
                teachers[c[1:3]]=core.read(root/'teacher_manifest.json')['records'][0]
                if c[3]==run.GT:
                    core.save(core.cell_path(root/'reference',c[:3]+(run.V13_GT,))/'test_result.json',
                        core.read(out/'test_result.json'))
            core.save(root/'teacher_manifest.json',dict(records=list(teachers.values())))
            # Reference validation has separate real-archive checks; here isolate report assembly.
            reference=lambda out,c:dict(protocol=c[0],action=c[1],seed=c[2],method=c[3],srcc_x100=100.)
            with patch.object(run,'validate_reference',side_effect=reference), \
                 patch.object(run.split,'verify_record',return_value=dict(metrics=dict(srcc_x100=100.))):
                run.write_report(root)
            summary=core.read(root/'summary.json')
            self.assertEqual(len(summary['raw']),180);self.assertEqual(len(summary['v13_gt_replay']),18)
            self.assertTrue(all(r['max_prediction_delta']==0 for r in summary['v13_gt_replay']))
            with (root/'scores_seed1.csv').open(newline='') as handle:
                self.assertEqual(len(list(csv.DictReader(handle))),60)
            core.package(root,'_partial')
            with ZipFile(root.with_name(root.name+'_partial_return.zip')) as archive:
                self.assertIn('scores_seed1.csv',archive.namelist());self.assertIsNone(archive.testzip())
                self.assertFalse(any(n.endswith('.pth') for n in archive.namelist()))


if __name__=='__main__':unittest.main()
