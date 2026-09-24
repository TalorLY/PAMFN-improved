from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from tools import run_qmd_modal as run
from tools import run_qmd_dual as core
from tools.qmd_modal_config import metadata,parameter_count
from tests.test_qmd_residual_tools import teacher


def fixture(root,cell,smoke=False):
    out=core.cell_path(root/('smoke' if smoke else 'screen'),cell)
    t=teacher(root,cell[1],cell[2]);source={k:t[k] for k in ('checkpoint_sha256','eval_rng_sha256','expected_srcc_x100','action','seed')}
    n=100 if cell[1] in ('TES','PCS') else 50;y=np.linspace(.1,.9,n)
    epochs=2 if smoke else core.BUDGET[cell[1]];weight=run.RECIPES[cell[3]]['weight']
    meta=metadata(cell[1],cell[2],cell[3],epochs,smoke)
    def data(p):return dict(coef=100.,loss_x100=float(np.mean((p-y)**2)*100),prediction_std=float(np.std(p)),
        predictions=dict(label=y.tolist(),prediction=p.tolist(),order='native_test_loader'))
    bias=.2;initial='initial-'+meta['branch']['heads']
    stats=dict(source_split='train',samples=200,passes=1,feature_dim=768,global_bias=bias,
        mean_label=.5,mean_anchor=.3,feature_scale_min=.01,feature_scale_max=1.,statistics_sha256='stats',feature_source=meta['branch']['features'],packed_feature_dim=1536)
    check=dict(source=source,baseline=data(y-bias),bias_only=data(y),zero_branch_exact=True,bias_only_rank_preserved=True,
        branch_parameters=parameter_count(meta['branch']),branch_config=meta['branch'],
        initial_branch_sha256=initial,initial_anchor_sha256='anchor',training_statistics_sha256='stats')
    test=dict(data(y),repaired_pairs=0,broken_pairs=0,run=meta,source=source,evaluation=dict(policy=meta['eval_policy'],replayed_rng=True),
        baseline_srcc_x100=100.,training_statistics_sha256='stats',global_bias=bias,
        correction_mean=0.,correction_std=0.,correction_max_abs=0.,correction_saturation=0.)
    test['expert_metrics']={};test['expert_predictions']={}
    if meta['branch']['heads']=='separate':
        for k in ('V','F','A'):
            test['expert_predictions'][k]=data(y)['predictions']
            test['expert_metrics'][k]=dict(coef=100.,loss_x100=0.,correction_std=0.)
    train=dict(run=meta,source=source,saved_model_srcc=1.)
    compact={k:v for k,v in test.items() if k not in ('predictions','expert_predictions','run','source','evaluation','baseline_srcc_x100','training_statistics_sha256')}
    qmd=dict(mode='independent_modality_correction',recipe=run.RECIPES[cell[3]],anchor_unchanged=True,anchor_gradients_none=True,
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
        ('last_epoch_result.json',dict(epoch=epochs,predictions=test['predictions'],expert_predictions=test['expert_predictions'],**compact)),
        ('run_config.json',dict(argv=run.cell_args(root,cell,'3',smoke)[5:],protocol='july28'))]:core.save(out/name,value)
    (out/'model.pth.epochs.jsonl').write_text(''.join(json.dumps(dict(epoch=e,**compact))+'\n' for e in range(1,epochs+1)))
    return out


class ModalRunnerTest(unittest.TestCase):
    def test_full_grid_all_seeds_actions(self):
        self.assertEqual(len(run.matrix()),72);self.assertEqual(len(set(run.matrix())),72)
        self.assertEqual(len(run.smoke_cells(run.matrix())),24);self.assertEqual(len(run.reference_cells()),90)
        self.assertEqual(len(run.METHODS),9)
        for m in run.RECIPES:
            self.assertEqual({c[1:3] for c in run.matrix() if c[3]==m},{(a,s) for a in core.ACTIONS for s in (0,1,2)})
    def test_four_records_and_budgets(self):
        for smoke in (False,True):
            for m in run.RECIPES:
                with tempfile.TemporaryDirectory() as temp:
                    c=('july28','Ball',0,m);out=fixture(Path(temp),c,smoke)
                    self.assertEqual(run.validate_cell(out,c,smoke)['srcc_x100'],100.)
                    args=run.cell_args(Path(temp),c,'4',smoke)
                    self.assertEqual(args[2],'tools/train_qmd_modal.py')
                    self.assertEqual('--epochs' in args,smoke)
    def mutation(self,file,change,message,method=None):
        with tempfile.TemporaryDirectory() as temp:
            c=('july28','Ball',0,method or run.KD);out=fixture(Path(temp),c)
            d=core.read(out/file);change(d);core.save(out/file,d)
            with self.assertRaisesRegex(ValueError,message):run.validate_cell(out,c)
    def test_anchor_changed_rejected(self):self.mutation('qmd_training.json',lambda d:d.update(anchor_state_after='bad'),'Ordered state')
    def test_test_statistics_rejected(self):self.mutation('training_statistics.json',lambda d:d.update(source_split='test'),'only from training')
    def test_wrong_feature_source_rejected(self):self.mutation('training_statistics.json',lambda d:d.update(feature_source='fused'),'feature source')
    def test_wrong_parameter_count_rejected(self):self.mutation('teacher_check.json',lambda d:d.update(branch_parameters=12320),'architecture')
    def test_no_epoch_zero(self):self.mutation('model.pth.selection.json',lambda d:d.update(epoch=0),'epoch zero')
    def test_no_rank_update(self):self.mutation('qmd_training.json',lambda d:d['epochs'][0].update(rank_ratio_max=.01),'rank gradient')
    def test_kd_bound(self):self.mutation('qmd_training.json',lambda d:d['epochs'][0].update(kd_ratio_max=1.),'KD gradient')
    def test_closed_gate_cannot_update(self):self.mutation('qmd_training.json',lambda d:d['epochs'][0].update(gate_open_batches=0),'outside useful-teacher')
    def test_expert_missing_rejected(self):self.mutation('test_result.json',lambda d:d['expert_predictions'].pop('F'),'missing expert')
    def test_expert_label_order_rejected(self):self.mutation('test_result.json',lambda d:d['expert_predictions']['V']['label'].reverse(),'order/shape')
    def test_final_prediction_missing_rejected(self):
        self.mutation('last_epoch_result.json',lambda d:d['predictions'].update(prediction=[]),'final-epoch predictions')
    def test_final_metrics_recomputed(self):
        self.mutation('last_epoch_result.json',lambda d:d['predictions']['prediction'].__setitem__(0,.8),'final-epoch prediction metrics')
    def test_fixed_expert_average_is_checked(self):
        from scipy.stats import spearmanr
        def change(d):
            p=d['expert_predictions']['V'];x=np.asarray(p['prediction'])+.001;y=np.asarray(p['label'])
            p['prediction']=x.tolist();d['expert_metrics']['V'].update(coef=spearmanr(x,y)[0]*100,loss_x100=np.mean((x-y)**2)*100)
        self.mutation('test_result.json',change,'expert average')
    def test_modal_pairing_rejects_drift(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(core,'ACTIONS',('Ball',)):
            root=Path(temp)
            for m in run.RECIPES:fixture(root,('july28','Ball',0,m),True)
            run.check_pairing(root/'smoke',seeds=(0,))
            p=core.cell_path(root/'smoke',('july28','Ball',0,run.KD))/'teacher_check.json'
            d=core.read(p);d['initial_branch_sha256']='wrong';core.save(p,d)
            with self.assertRaisesRegex(ValueError,'initialization differs'):run.check_pairing(root/'smoke',seeds=(0,))
    def test_status_realtime_completion_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);core.save(root/'manifest.json',dict(jobs=run.matrix(),study='full'))
            buf=io.StringIO()
            with redirect_stdout(buf):run.status(root)
            self.assertIn('0/72',buf.getvalue());self.assertIn('0/24',buf.getvalue())
            for m in run.RECIPES:self.assertIn(m+': 0/18',buf.getvalue())
    def test_seed1_cannot_be_dropped(self):
        rows=[dict(protocol='july28',action=a,seed=s,method=m,srcc_x100=(60 if s==1 else 80)+i)
            for a in core.ACTIONS for s in (0,1,2) for i,m in enumerate(run.METHODS)]
        report=run.full.aggregate(rows,methods=run.METHODS,controls=((run.BASE,'base'),(run.ORDERED,'ordered')))
        self.assertEqual(len(report['raw']),162)
        with self.assertRaises(ValueError):run.full.aggregate([r for r in rows if r['seed']!=1],methods=run.METHODS,controls=((run.BASE,'base'),(run.ORDERED,'ordered')))
    def test_old_roots_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            old=Path(temp)/'old'
            for new in (old,old/'new',old.parent):
                with self.assertRaisesRegex(ValueError,'separate from old'):run.checked_v15(old,{},new)
    def test_resume_keeps_last_prediction_artifact(self):
        with patch.object(core,'run_cell') as call,patch.object(run,'cell_args',return_value=[]):
            run.run_cell(Path('/tmp/root/screen'),{},('july28','Ball',0,run.KD),'0',False)
            self.assertIn('last_epoch_result.json',call.call_args.kwargs['extra_required'])
    def test_summary_decomposition_csv_and_archive(self):
        import csv
        from zipfile import ZipFile
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'v16';teachers={}
            for c in run.matrix():
                out=fixture(root,c);teachers[c[1:3]]=core.read(root/'teacher_manifest.json')['records'][0]
                if c[3]==run.FUSED:
                    ref=core.cell_path(root/'reference',c[:3]+(run.V13_GT,))
                    for name in ('test_result.json','model.pth.epochs.jsonl'):
                        ref.mkdir(parents=True,exist_ok=True);(ref/name).write_bytes((out/name).read_bytes())
            core.save(root/'teacher_manifest.json',dict(records=list(teachers.values())))
            reference=lambda out,c:dict(protocol=c[0],action=c[1],seed=c[2],method=c[3],srcc_x100=100.)
            with patch.object(run,'validate_reference',side_effect=reference),patch.object(run.split,'verify_record',return_value=dict(metrics=dict(srcc_x100=100.))):
                run.write_report(root)
            report=core.read(root/'summary.json');self.assertEqual(len(report['raw']),162)
            self.assertTrue(all(r['kd_gain']==0 and r['feature_gain']==0 and r['factorization_gain']==0 for r in report['factorial']))
            with (root/'scores_seed1.csv').open(newline='') as f:self.assertEqual(len(list(csv.DictReader(f))),54)
            core.package(root,'_partial')
            with ZipFile(root.with_name(root.name+'_partial_return.zip')) as z:
                self.assertIn('scores_seed1.csv',z.namelist());self.assertIsNone(z.testzip())
                self.assertFalse(any(n.endswith('.pth') for n in z.namelist()))


if __name__=='__main__':unittest.main()
