from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import run_qmd_residual as run
from tools import run_qmd_dual as core


def teacher(root,a='Ball',s=0):
    record=dict(action=a,seed=s,model=str(root/'old/model.pth'),rng=str(root/'old/rng.pt'),
        checkpoint_sha256='teacher-sha',eval_rng_sha256='rng-sha',expected_srcc_x100=100.)
    core.save(root/'teacher_manifest.json',dict(records=[record]))
    return record


def fixture(root,cell,smoke=False):
    out=core.cell_path(root/('smoke' if smoke else 'screen'),cell)
    t=teacher(root,cell[1],cell[2]);source={k:t[k] for k in ('checkpoint_sha256','eval_rng_sha256','expected_srcc_x100','action','seed')}
    n=100 if cell[1] in ('TES','PCS') else 50;y=np.linspace(.1,.9,n).tolist()
    epochs=2 if smoke else core.BUDGET[cell[1]];weight=run.WEIGHTS[cell[3]]
    meta=dict(action=cell[1],seed=cell[2],tag=cell[3],epochs=epochs,feats=2,protocol='july28_frozen_residual',
        weight=weight,branch=dict(input_dim=768,hidden=64,bound=.1),anchor='July28 Ordered frozen checkpoint',
        eval_policy='fixed_ordered_eval_rng',mask_ratio=.15,warmup=0. if smoke else .1,rampup=0. if smoke else .1,
        target_ratio=.075,beta_cap=1.,gt='full corrected score',kd_target='frozen full-input Ordered score')
    data=dict(coef=100.,loss_x100=0.,prediction_std=float(np.std(y)),predictions=dict(label=y,prediction=y,order='native_test_loader'))
    check=dict(source=source,baseline=data,zero_branch_exact=True,branch_parameters=50817,branch_config=meta['branch'],
        initial_branch_sha256='initial',initial_anchor_sha256='anchor')
    test=dict(data,run=meta,source=source,evaluation=dict(policy='fixed_ordered_eval_rng',replayed_rng=True),baseline_srcc_x100=100.)
    train=dict(run=meta,source=source,saved_model_srcc=1.)
    compact={k:v for k,v in data.items() if k!='predictions'}
    qmd=dict(mode='residual_kd',weight=weight,anchor_unchanged=True,anchor_gradients_none=True,
        anchor_state_before='anchor',anchor_state_after='anchor',branch_updated=True,
        initial_branch_sha256='initial',final_branch_sha256='updated',epoch_zero_eligible=False,
        active_batches=epochs,weighted_kd_sum=epochs*weight,positive_kd_gradient_batches=epochs if weight else 0,
        conflicts=1,cosine_batches=epochs,calibration=dict(beta=.3),extra_stage_epochs=epochs,
        epochs=[dict(epoch=e,active_batches=1,weighted_kd_sum=weight) for e in range(1,epochs+1)])
    for name,value in [('teacher_check.json',check),('test_result.json',test),('train_result.json',train),
                       ('qmd_training.json',qmd),('model.pth.selection.json',dict(epoch=1,**compact)),
                       ('run_config.json',dict(argv=run.cell_args(root,cell,'3',smoke)[5:],protocol='july28'))]:
        core.save(out/name,value)
    (out/'model.pth.epochs.jsonl').write_text(''.join(json.dumps(dict(epoch=e,**compact))+'\n' for e in range(1,epochs+1)))
    return out


class ResidualRunnerTest(unittest.TestCase):
    def test_fixed_54_grid_and_18_smoke(self):
        self.assertEqual(len(run.matrix()),54);self.assertEqual(len(set(run.matrix())),54)
        for m in run.WEIGHTS:
            self.assertEqual({(c[1],c[2]) for c in run.matrix() if c[3]==m},{(a,s) for a in core.ACTIONS for s in (0,1,2)})
        self.assertEqual(len(run.smoke_cells(run.matrix())),18);self.assertEqual(run.smoke_cells([]),[])
        self.assertEqual(len(run.reference_cells()),72)

    def test_args_native_budgets_and_single_trainer(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);teacher(root)
            for m in run.WEIGHTS:
                c=('july28','Ball',0,m)
                argv=run.cell_args(root,c,'4')
                self.assertEqual(argv[2],'tools/train_qmd_residual.py');self.assertNotIn('--epochs',argv)
                self.assertNotIn('--qmd_grad_mode',argv)
                self.assertEqual(run.cell_args(root,c,'4',True)[-3:],['--smoke','--epochs','2'])
            with self.assertRaises(ValueError):run.cell_args(root,('july28','Ball',3,run.GT),'4')
        with patch.object(core,'command') as command:
            run.unit_check(Path('.'),Path('unit.log'),['tests.test_qmd_residual'],'4')
            self.assertEqual(command.call_args.args[0][:2],['env','CUDA_VISIBLE_DEVICES=4'])

    def test_valid_each_arm_with_saved_predictions(self):
        for m in run.WEIGHTS:
            with tempfile.TemporaryDirectory() as temp:
                c=('july28','Ball',0,m);out=fixture(Path(temp),c)
                row=run.validate_cell(out,c)
                self.assertEqual(row['srcc_x100'],100.);self.assertEqual(row['warnings'],[])

    def test_validation_rejects_metadata_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            c=('july28','Ball',0,run.GT);out=fixture(Path(temp),c)
            data=core.read(out/'test_result.json');data['run']['weight']=1.;core.save(out/'test_result.json',data)
            with self.assertRaisesRegex(ValueError,'recipe'):run.validate_cell(out,c)

    def test_epoch_zero_not_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            c=('july28','Ball',0,run.GT);out=fixture(Path(temp),c)
            data=core.read(out/'model.pth.selection.json');data['epoch']=0;core.save(out/'model.pth.selection.json',data)
            with self.assertRaisesRegex(ValueError,'epoch zero'):run.validate_cell(out,c)

    def test_nonfinite_saved_coefficient_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            c=('july28','Ball',0,run.GT);out=fixture(Path(temp),c)
            data=core.read(out/'train_result.json');data['saved_model_srcc']=float('nan');core.save(out/'train_result.json',data)
            with self.assertRaisesRegex(ValueError,'selection'):run.validate_cell(out,c)

    def test_anchor_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            c=('july28','Ball',0,run.GT);out=fixture(Path(temp),c)
            data=core.read(out/'qmd_training.json');data['anchor_state_after']='changed';core.save(out/'qmd_training.json',data)
            with self.assertRaisesRegex(ValueError,'Ordered state'):run.validate_cell(out,c)

    def test_gt_control_must_have_zero_kd_update(self):
        with tempfile.TemporaryDirectory() as temp:
            c=('july28','Ball',0,run.GT);out=fixture(Path(temp),c)
            data=core.read(out/'qmd_training.json');data['weighted_kd_sum']=1.;data['epochs'][0]['weighted_kd_sum']=1.
            core.save(out/'qmd_training.json',data)
            with self.assertRaisesRegex(ValueError,'GT-only'):run.validate_cell(out,c)

    def test_ineffective_kd_flagged_not_dropped(self):
        with tempfile.TemporaryDirectory() as temp:
            c=('july28','Ball',0,'ordered_residual_kd_w050');out=fixture(Path(temp),c)
            data=core.read(out/'qmd_training.json');data['positive_kd_gradient_batches']=0
            core.save(out/'qmd_training.json',data)
            self.assertIn('no_effective_branch_kd',run.validate_cell(out,c)['warnings'])

    def test_missing_epoch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            c=('july28','Ball',0,run.GT);out=fixture(Path(temp),c)
            p=out/'model.pth.epochs.jsonl';p.write_text(p.read_text().splitlines()[0]+'\n')
            with self.assertRaisesRegex(ValueError,'epochs'):run.validate_cell(out,c)

    def test_actual_prediction_metrics_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            c=('july28','Ball',0,run.GT);out=fixture(Path(temp),c)
            data=core.read(out/'test_result.json');data['predictions']['prediction'].reverse();core.save(out/'test_result.json',data)
            with self.assertRaisesRegex(ValueError,'metrics'):run.validate_cell(out,c)

    def test_same_root_resume_delegates_to_tested_core(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(core,'run_cell',return_value=('SKIP',{})) as call:
            root=Path(temp);teacher(root)
            result=run.run_cell(root/'screen',{},('july28','Ball',0,run.GT),'4',False)
            self.assertEqual(result[0],'SKIP');self.assertEqual(call.call_args.kwargs['extra_required'],('teacher_check.json',))

    def test_missing_teachers_fail_before_training(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(run.split,'verify_record',return_value=dict(
                checkpoint_sha256='a',eval_rng_sha256='b',metrics=dict(srcc_x100=80.))):
            with self.assertRaisesRegex(FileNotFoundError,'No training has started'):
                run.resolve_teachers(Path(temp),Path(temp)/'missing')

    def test_teacher_resolution_exact_hash_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);origin=core.cell_path(root/'old/screen',('july28','Ball',0,run.ORDERED))
            origin.mkdir(parents=True);(origin/'model.pth').write_bytes(b'model');(origin/'model.pth.eval_rng.pt').write_bytes(b'rng')
            record=dict(checkpoint_sha256=core.sha(origin/'model.pth'),eval_rng_sha256=core.sha(origin/'model.pth.eval_rng.pt'),metrics=dict(srcc_x100=80.))
            with patch.object(core,'ACTIONS',('Ball',)),patch.object(run,'SEEDS',(0,)),patch.object(run.split,'verify_record',return_value=record):
                first=run.resolve_teachers(root,root/'old');self.assertEqual(first,run.resolve_teachers(root,root/'old'))
                (origin/'model.pth').write_bytes(b'wrong')
                with self.assertRaises(FileNotFoundError):run.resolve_teachers(root,root/'old')

    def test_all_seeds_retained_in_aggregate_and_branch_control(self):
        rows=[dict(protocol='july28',action=a,seed=s,method=m,srcc_x100=(60. if s==2 else 80.)+i)
              for a in core.ACTIONS for s in (0,1,2) for i,m in enumerate(run.METHODS)]
        result=run.full.aggregate(rows,methods=run.METHODS,controls=((run.ORDERED,'ordered'),(run.GT,'branch_gt')))
        self.assertEqual(len(result['raw']),126)
        self.assertEqual(result['methods'][-1]['per_seed']['2']['srcc_x100'],66.)
        self.assertEqual(result['methods'][-1]['delta_vs_branch_gt'],2.)
        with self.assertRaises(ValueError):run.full.aggregate(rows[:-1],methods=run.METHODS)

    def test_status_has_completion_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);core.save(root/'manifest.json',dict(jobs=run.matrix(),study='full'))
            capture=io.StringIO()
            with redirect_stdout(capture):run.status(root)
            self.assertIn('0/54',capture.getvalue());self.assertIn('0/18',capture.getvalue())

    def test_result_root_cannot_overlap_reference(self):
        with self.assertRaises(ValueError):run.full.check_result_roots(Path('/a/b'),(Path('/a'),))

    def test_prewarm_check_accepts_paired_arms_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(core,'ACTIONS',('Ball',)),patch.object(run,'SEEDS',(0,)):
            root=Path(temp)
            for m in run.WEIGHTS:fixture(root,('july28','Ball',0,m))
            run.check_prewarm(root/'screen')
            out=core.cell_path(root/'screen',('july28','Ball',0,'ordered_residual_kd_w050'))
            p=out/'model.pth.epochs.jsonl';lines=[json.loads(v) for v in p.read_text().splitlines()]
            lines[0]['loss_x100']=1.;p.write_text(''.join(json.dumps(v)+'\n' for v in lines))
            with self.assertRaisesRegex(ValueError,'prewarm'):run.check_prewarm(root/'screen')

    def test_v11_provenance_must_be_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);core.save(root/'old/source.json',dict(commits={},inputs={},environment={}))
            core.save(root/'old/completion.json',dict(experiments_complete=True))
            with self.assertRaisesRegex(ValueError,'completed frozen v11'):
                run.checked_v11(root/'old',{},root/'new')


if __name__=='__main__':unittest.main()
