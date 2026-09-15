from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from tools import run_qmd_complement as run
from tools import run_qmd_dual as core
from tools.qmd_complement_config import metadata,parameter_count
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
    qmd=dict(mode='bias_separated_kd',recipe=run.RECIPES[cell[3]],anchor_unchanged=True,anchor_gradients_none=True,
        anchor_state_before='anchor',anchor_state_after='anchor',branch_updated=True,
        initial_branch_sha256=initial,final_branch_sha256='updated',epoch_zero_eligible=False,
        statistics_before='stats',statistics_after='stats',statistics_unchanged=True,
        lr=.0001,weight_decay=.01,optimizer='AdamW',kd_first_epoch=1,
        active_batches=epochs,weighted_kd_sum=epochs*weight,positive_kd_gradient_batches=epochs if weight else 0,
        conflicts=1,cosine_batches=epochs,calibration=dict(beta=.3,epoch=1),extra_stage_epochs=epochs,
        epochs=[dict(epoch=e,active_batches=1,weighted_kd_sum=weight) for e in range(1,epochs+1)])
    for name,value in [('teacher_check.json',check),('test_result.json',test),('train_result.json',train),
        ('training_statistics.json',stats),('qmd_training.json',qmd),('model.pth.selection.json',dict(epoch=1,**compact)),
        ('last_epoch_result.json',dict(epoch=epochs,**compact)),
        ('run_config.json',dict(argv=run.cell_args(root,cell,'3',smoke)[5:],protocol='july28'))]:core.save(out/name,value)
    (out/'model.pth.epochs.jsonl').write_text(''.join(json.dumps(dict(epoch=e,**compact))+'\n' for e in range(1,epochs+1)))
    return out


class ComplementRunnerTest(unittest.TestCase):
    def test_fixed_108_grid_36_smoke_and_72_references(self):
        self.assertEqual(len(run.matrix()),108);self.assertEqual(len(set(run.matrix())),108)
        self.assertEqual(len(run.smoke_cells(run.matrix())),36);self.assertEqual(run.smoke_cells([]),[])
        self.assertEqual(len(run.reference_cells()),72)
        for m in run.RECIPES:
            self.assertEqual({(c[1],c[2]) for c in run.matrix() if c[3]==m},{(a,s) for a in core.ACTIONS for s in (0,1,2)})

    def test_each_kd_has_matched_capacity_gt_control(self):
        for recipe in run.RECIPES.values():
            self.assertTrue(any(v['hidden']==recipe['hidden'] and v['weight']==0 for v in run.RECIPES.values()))
        self.assertEqual(run.RECIPES['ordered_bc_linear_raw_kd']['weight'],run.RECIPES['ordered_bc_linear_kd_w050']['weight'])

    def test_single_entry_native_budgets_and_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);teacher(root)
            for m in run.RECIPES:
                c=('july28','Ball',0,m);argv=run.cell_args(root,c,'4')
                self.assertEqual(argv[2],'tools/train_qmd_complement.py');self.assertNotIn('--epochs',argv)
                self.assertEqual(run.cell_args(root,c,'4',True)[-3:],['--smoke','--epochs','2'])
            with self.assertRaises(ValueError):run.cell_args(root,('july28','Ball',3,run.LINEAR_GT),'4')

    def test_all_six_valid_cells(self):
        for smoke in (False,True):
            for m in run.RECIPES:
                with tempfile.TemporaryDirectory() as temp:
                    c=('july28','Ball',0,m);out=fixture(Path(temp),c,smoke)
                    row=run.validate_cell(out,c,smoke)
                    self.assertEqual(row['srcc_x100'],100.);self.assertFalse(row['warnings'])
                    self.assertFalse(row['selected_before_aux'])

    def check_tamper(self,file,field,value,message):
        with tempfile.TemporaryDirectory() as temp:
            c=('july28','Ball',0,run.LINEAR_GT);out=fixture(Path(temp),c)
            data=core.read(out/file);data[field]=value;core.save(out/file,data)
            with self.assertRaisesRegex(ValueError,message):run.validate_cell(out,c)

    def test_reject_test_calibration(self):self.check_tamper('training_statistics.json','source_split','test','only from training')
    def test_reject_statistics_refit(self):self.check_tamper('qmd_training.json','statistics_after','bad','statistics changed')
    def test_reject_anchor_change(self):self.check_tamper('qmd_training.json','anchor_state_after','bad','Ordered state')
    def test_reject_late_kd(self):self.check_tamper('qmd_training.json','kd_first_epoch',51,'epoch one')
    def test_reject_epoch_zero(self):self.check_tamper('model.pth.selection.json','epoch',0,'epoch zero')
    def test_reject_bad_optimizer(self):self.check_tamper('qmd_training.json','lr',.0008,'optimizer changed')
    def test_reject_nonfinite_selected(self):self.check_tamper('train_result.json','saved_model_srcc',float('nan'),'selection/reload')
    def test_reject_bias_changed_at_test(self):self.check_tamper('test_result.json','global_bias',.3,'bias or sample')
    def test_reject_correction_diagnostic(self):self.check_tamper('test_result.json','correction_std',.01,'diagnostic mismatch')

    def test_reject_gt_kd_gradient(self):self.check_tamper('qmd_training.json','positive_kd_gradient_batches',1,'GT-only')

    def test_ineffective_kd_flagged_not_hidden(self):
        with tempfile.TemporaryDirectory() as temp:
            c=('july28','Ball',0,'ordered_bc_linear_kd_w050');out=fixture(Path(temp),c)
            q=core.read(out/'qmd_training.json');q['positive_kd_gradient_batches']=0;core.save(out/'qmd_training.json',q)
            self.assertIn('no_effective_branch_kd',run.validate_cell(out,c)['warnings'])

    def test_pairing_checks_all_six_but_not_prewarm(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(core,'ACTIONS',('Ball',)):
            root=Path(temp)
            for m in run.RECIPES:fixture(root,('july28','Ball',0,m))
            run.check_pairing(root/'screen',seeds=(0,))
            path=core.cell_path(root/'screen',('july28','Ball',0,run.MLP_GT))/'training_statistics.json'
            data=core.read(path);data['global_bias']=.1;core.save(path,data)
            with self.assertRaisesRegex(ValueError,'paired training statistics'):run.check_pairing(root/'screen',seeds=(0,))

    def test_status_realtime_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);core.save(root/'manifest.json',dict(jobs=run.matrix(),study='full'))
            capture=io.StringIO()
            with redirect_stdout(capture):run.status(root)
            self.assertIn('0/108',capture.getvalue());self.assertIn('0/36',capture.getvalue())
            self.assertEqual(capture.getvalue().count(': 0/18'),6)

    def test_aggregate_preserves_all_seeds_and_controls(self):
        rows=[dict(protocol='july28',action=a,seed=s,method=m,srcc_x100=(60. if s==2 else 80.)+i)
            for a in core.ACTIONS for s in (0,1,2) for i,m in enumerate(run.METHODS)]
        report=run.full.aggregate(rows,methods=run.METHODS,controls=((run.ORDERED,'ordered'),(run.LINEAR_GT,'linear_gt'),(run.MLP_GT,'mlp_gt')))
        self.assertEqual(len(report['raw']),180)
        self.assertEqual(report['methods'][-1]['per_seed']['2']['srcc_x100'],69.)
        self.assertEqual(report['methods'][-1]['delta_vs_mlp_gt'],1.)
        with self.assertRaises(ValueError):run.full.aggregate(rows[:-1],methods=run.METHODS)

    def test_extra_files_required_for_safe_resume(self):
        with patch.object(core,'run_cell') as runner,patch.object(run,'cell_args',return_value=[]):
            run.run_cell(Path('/tmp/root/screen'),{},('july28','Ball',0,run.LINEAR_GT),'0',False)
            self.assertEqual(runner.call_args.kwargs['extra_required'],
                ('teacher_check.json','training_statistics.json','last_epoch_result.json'))


if __name__=='__main__':unittest.main()
