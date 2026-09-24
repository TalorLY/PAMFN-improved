from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from tools import run_qmd_pair as run
from tools import run_qmd_dual as core
from tools.qmd_pair_config import metadata,parameter_count
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
    test=dict(data(y),repaired_pairs=0,broken_pairs=0,run=meta,source=source,evaluation=dict(policy=meta['eval_policy'],replayed_rng=True),
        baseline_srcc_x100=100.,training_statistics_sha256='stats',global_bias=bias,
        correction_mean=0.,correction_std=0.,correction_max_abs=0.,correction_saturation=0.)
    train=dict(run=meta,source=source,saved_model_srcc=1.)
    compact={k:v for k,v in test.items() if k not in ('predictions','run','source','evaluation','baseline_srcc_x100','training_statistics_sha256')}
    qmd=dict(mode='selective_pair_correction',recipe=run.RECIPES[cell[3]],anchor_unchanged=True,anchor_gradients_none=True,
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
            kd_ratio_max=weight*.075*.5*min(1.,e/(1 if smoke else 5)),gt_mean=.1,energy_mean=0.,
            weighted_energy_mean=0.,
            supervised_mean=.1+run.RECIPES[cell[3]]['rank']*.02) for e in range(1,epochs+1)])
    rank=run.RECIPES[cell[3]]['rank']
    for r in qmd['epochs']:
        r.update(rank_mean=.02,weighted_rank_mean=rank*.02,weighted_rank_sum=rank*.02,
            rank_ratio_mean=rank*.05*min(1.,r['epoch']/(1 if smoke else 5)),
            rank_ratio_max=rank*.05*min(1.,r['epoch']/(1 if smoke else 5)),
            reliable_pairs=6,hard_pairs=2,protected_pairs=3,unreachable_pairs=1,accepted_pairs=3,
            gate_open_batches=1,positive_rank_batches=1 if rank else 0)
    qmd.update(weighted_rank_sum=epochs*rank*.02,positive_rank_batches=epochs if rank else 0,
        accepted_pairs=epochs*3,reliable_pairs=epochs*6)
    qmd['calibration'].update(rank_positive_steps=epochs if rank else 0,
        rank_first_positive=dict(epoch=1,step=1) if rank else None,
        max_rank_ratio=max(r['rank_ratio_max'] for r in qmd['epochs']))
    qmd['calibration']['max_effective_ratio']=max(r['kd_ratio_max'] for r in qmd['epochs'])
    for name,value in [('teacher_check.json',check),('test_result.json',test),('train_result.json',train),
        ('training_statistics.json',stats),('qmd_training.json',qmd),('model.pth.selection.json',dict(epoch=1,**compact)),
        ('last_epoch_result.json',dict(epoch=epochs,**compact)),
        ('run_config.json',dict(argv=run.cell_args(root,cell,'3',smoke)[5:],protocol='july28'))]:core.save(out/name,value)
    (out/'model.pth.epochs.jsonl').write_text(''.join(json.dumps(dict(epoch=e,**compact))+'\n' for e in range(1,epochs+1)))
    return out


class PairRunnerTest(unittest.TestCase):
    def test_full_factorial_matrix_all_actions_and_seeds(self):
        self.assertEqual(len(run.matrix()),54);self.assertEqual(len(set(run.matrix())),54)
        self.assertEqual(len(run.smoke_cells(run.matrix())),18);self.assertEqual(len(run.reference_cells()),72)
        self.assertEqual(len(run.METHODS),7)
        for m in run.RECIPES:
            self.assertEqual({(c[1],c[2]) for c in run.matrix() if c[3]==m},
                {(a,s) for a in core.ACTIONS for s in (0,1,2)})
        self.assertEqual({(r['rank'],r['weight']) for r in run.RECIPES.values()},{(1.,0.),(0.,.5),(1.,.5)})

    def test_same_gt_and_f_reference(self):
        self.assertEqual(run.REFERENCES,('base','ordered','ordered_bc_mlp16_gt','ordered_bc_mlp16_kd_w050'))

    def test_default_trainer_and_native_budgets(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);teacher(root)
            for m in run.RECIPES:
                c=('july28','Ball',0,m);args=run.cell_args(root,c,'4')
                self.assertEqual(args[2],'tools/train_qmd_pair.py');self.assertNotIn('--epochs',args)
                self.assertEqual(run.cell_args(root,c,'4',True)[-3:],['--smoke','--epochs','2'])
            with self.assertRaises(ValueError):run.cell_args(root,('july28','Ball',42,run.RANK),'4')
        script=(run.REPO/'tools/run_qmd_pair.sh').read_text()
        self.assertIn('PAMFN-july28-qmd-pair-v15',script)

    def test_all_three_smoke_and_formal_records_validate(self):
        for smoke in (False,True):
            for m in run.RECIPES:
                with tempfile.TemporaryDirectory() as temp:
                    c=('july28','Ball',0,m);out=fixture(Path(temp),c,smoke)
                    self.assertEqual(run.validate_cell(out,c,smoke)['srcc_x100'],100.)

    def mutation(self,file,change,message,method=None):
        with tempfile.TemporaryDirectory() as temp:
            c=('july28','Ball',0,method or run.COMBO);out=fixture(Path(temp),c)
            d=core.read(out/file);change(d);core.save(out/file,d)
            with self.assertRaisesRegex(ValueError,message):run.validate_cell(out,c)

    def test_changed_anchor_rejected(self):
        self.mutation('qmd_training.json',lambda d:d.update(anchor_state_after='bad'),'Ordered state')

    def test_test_statistics_rejected(self):
        self.mutation('training_statistics.json',lambda d:d.update(source_split='test'),'only from training')

    def test_epoch_zero_rejected(self):
        self.mutation('model.pth.selection.json',lambda d:d.update(epoch=0),'epoch zero')

    def test_rank_ratio_bound_rejected(self):
        self.mutation('qmd_training.json',lambda d:d['epochs'][0].update(rank_ratio_max=1.),'rank gradient')

    def test_kd_ratio_bound_rejected(self):
        self.mutation('qmd_training.json',lambda d:d['epochs'][0].update(kd_ratio_max=1.),'KD gradient')

    def test_gate_cannot_accept_more_than_candidates(self):
        self.mutation('qmd_training.json',lambda d:d['epochs'][0].update(accepted_pairs=7),'pair selection')

    def test_gate_closed_cannot_update_kd(self):
        self.mutation('qmd_training.json',lambda d:d['epochs'][0].update(gate_open_batches=0),'outside useful-teacher gate')

    def test_rank_loss_accounting(self):
        self.mutation('qmd_training.json',lambda d:d['epochs'][0].update(supervised_mean=1.),'rank objective')

    def test_no_rank_in_gate_only_arm(self):
        self.mutation('qmd_training.json',lambda d:d.update(positive_rank_batches=1),'rank positive',run.GATE)

    def test_rank_first_step_within_history(self):
        self.mutation('qmd_training.json',lambda d:d['calibration'].update(rank_first_positive=dict(epoch=1,step=999)), 'outside history')

    def test_closed_gate_is_recorded_not_forced_open(self):
        with tempfile.TemporaryDirectory() as temp:
            c=('july28','Ball',0,run.GATE);out=fixture(Path(temp),c)
            q=core.read(out/'qmd_training.json')
            q.update(weighted_kd_sum=0.,positive_kd_gradient_batches=0,accepted_pairs=0)
            q['calibration'].update(first_positive=None,max_effective_ratio=0.)
            for r in q['epochs']:r.update(weighted_kd_sum=0.,positive_kd_gradient_batches=0,accepted_pairs=0,
                gate_open_batches=0,kd_ratio_mean=0.,kd_ratio_max=0.)
            core.save(out/'qmd_training.json',q)
            self.assertIn('no_effective_branch_kd_gate_or_gradient_zero',run.validate_cell(out,c)['warnings'])

    def test_progress_counts_and_each_arm(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);core.save(root/'manifest.json',dict(jobs=run.matrix(),study='full'))
            buf=io.StringIO()
            with redirect_stdout(buf):run.status(root)
            self.assertIn('0/54',buf.getvalue());self.assertIn('0/18',buf.getvalue())
            for m in run.RECIPES:self.assertIn(m+': 0/18',buf.getvalue())

    def test_pairing_rejects_initialization_drift(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(core,'ACTIONS',('Ball',)):
            root=Path(temp)
            for m in run.RECIPES:fixture(root,('july28','Ball',0,m))
            run.check_pairing(root/'screen',seeds=(0,))
            p=core.cell_path(root/'screen',('july28','Ball',0,run.RANK))/'teacher_check.json'
            d=core.read(p);d['initial_branch_sha256']='different';core.save(p,d)
            with self.assertRaisesRegex(ValueError,'initialization differs'):run.check_pairing(root/'screen',seeds=(0,))

    def test_report_keeps_lower_seed_and_rejects_missing_seed(self):
        rows=[dict(protocol='july28',action=a,seed=s,method=m,srcc_x100=(60. if s==1 else 80.)+i)
            for a in core.ACTIONS for s in (0,1,2) for i,m in enumerate(run.METHODS)]
        report=run.full.aggregate(rows,methods=run.METHODS,controls=((run.V13_GT,'gt'),(run.ORDERED,'ordered')))
        self.assertEqual(len(report['raw']),126)
        self.assertEqual(report['methods'][-1]['per_seed']['1']['srcc_x100'],66.)
        with self.assertRaises(ValueError):run.full.aggregate([r for r in rows if r['seed']!=1],methods=run.METHODS)

    def test_old_results_are_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            old=Path(temp)/'old'
            for new in (old,old/'new',old.parent):
                with self.assertRaisesRegex(ValueError,'separate from old'):run.checked_v13(old,{},new)

    def test_resume_requires_every_training_artifact(self):
        with patch.object(core,'run_cell') as runner,patch.object(run,'cell_args',return_value=[]):
            run.run_cell(Path('/tmp/root/screen'),{},('july28','Ball',0,run.RANK),'0',False)
            self.assertEqual(runner.call_args.kwargs['extra_required'],
                ('teacher_check.json','training_statistics.json','last_epoch_result.json'))

    def test_summary_interaction_seed_csv_and_archive(self):
        import csv
        from zipfile import ZipFile
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'v15';teachers={}
            for c in run.matrix():
                fixture(root,c);teachers[c[1:3]]=core.read(root/'teacher_manifest.json')['records'][0]
            core.save(root/'teacher_manifest.json',dict(records=list(teachers.values())))
            reference=lambda out,c:dict(protocol=c[0],action=c[1],seed=c[2],method=c[3],srcc_x100=100.)
            with patch.object(run,'validate_reference',side_effect=reference), \
                 patch.object(run.split,'verify_record',return_value=dict(metrics=dict(srcc_x100=100.))):
                run.write_report(root)
            report=core.read(root/'summary.json')
            self.assertEqual(len(report['raw']),126);self.assertEqual(len(report['factorial']),18)
            self.assertTrue(all(r['interaction']==0 for r in report['factorial']))
            with (root/'scores_seed1.csv').open(newline='') as f:self.assertEqual(len(list(csv.DictReader(f))),42)
            core.package(root,'_partial')
            with ZipFile(root.with_name(root.name+'_partial_return.zip')) as z:
                self.assertIn('scores_seed1.csv',z.namelist());self.assertIsNone(z.testzip())
                self.assertFalse(any(n.endswith('.pth') for n in z.namelist()))


if __name__=='__main__':unittest.main()
