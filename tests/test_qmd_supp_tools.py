from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tools import run_qmd_supp as run
from tools import run_qmd_dual as core
from tools.qmd_supp_config import RECIPES,FIXED,SCALED,SHRINK,metadata,parameter_count
from tests.test_qmd_residual_tools import teacher


def fixture(root,cell,smoke=False):
    out=core.cell_path(root/('smoke' if smoke else 'screen'),cell);a,s,m=cell[1:]
    t=teacher(root,a,s);e=dict(t,model=str(root/'old/e.pth'),record=str(root/'e_reference/complete.json'),checkpoint_sha256='e-sha')
    core.save(root/'e_manifest.json',dict(records=[e]));fields=('action','seed','checkpoint_sha256','eval_rng_sha256','expected_srcc_x100')
    source=dict(ordered={k:t[k] for k in fields},e={k:e[k] for k in fields});epochs=2 if smoke else core.BUDGET[a]
    y=np.linspace(.1,.9,100 if a in ('TES','PCS') else 50)
    baseline=dict(coef=100.,loss_x100=0.,prediction_std=float(y.std()),predictions=dict(label=y.tolist(),prediction=y.tolist(),order='native_test_loader'))
    core.save(root/'e_reference/test_result.json',baseline)
    meta=metadata(a,s,m,epochs,smoke);bound=.001 if RECIPES[m]['relative'] else .003
    stats=dict(source_split='train',samples=400 if a in ('TES','PCS') else 200,passes=1,new_global_bias=False,
        e_prediction_std=.01,bound=bound,feature_scale_min=.01,feature_scale_max=1.,statistics_sha256='stats')
    check=dict(source=source,baseline=baseline,ordered=baseline,zero_branch_exact=True,branch_config=meta['branch'],
        trainable_parameters=parameter_count(m),initial_frozen_sha256=dict(ordered='ord',e='e'),initial_branch_sha256='start',
        initial_heads_sha256='heads',training_statistics_sha256='stats')
    test=dict(baseline,run=meta,source=source,evaluation=dict(policy=meta['eval_policy'],replayed_rng=True),training_statistics_sha256='stats',
        gate=1.,bound=bound,e_srcc_x100=100.,correction_std=0.,correction_mean=0.,correction_max_abs=0.,repaired_pairs=0,broken_pairs=0,
        expert_metrics={k:{f:v for f,v in baseline.items() if f!='predictions'} for k in ('V','F','A')},
        expert_predictions={k:copy.deepcopy(baseline['predictions']) for k in ('V','F','A')})
    compact={k:v for k,v in test.items() if k not in ('predictions','expert_predictions','run','source','evaluation','training_statistics_sha256')}
    q=dict(mode='frozen_E_supplement',recipe=RECIPES[m],frozen_unchanged=True,frozen_gradients_none=True,
        frozen_before=dict(ordered='ord',e='e'),frozen_after=dict(ordered='ord',e='e'),initial_branch_sha256='start',final_branch_sha256='end',branch_updated=True,
        statistics_before='stats',statistics_after='stats',statistics_unchanged=True,epoch_zero_eligible=False,extra_stage_epochs=epochs,
        kd_weight=0.,weighted_kd_sum=0.,positive_kd_gradient_batches=0,rank_weight=0.,positive_gradient_batches=epochs,
        epochs=[dict(epoch=i,batches=1,positive_gradient_batches=1,gate_mean=1.,loss_mean=.1,centered_mse_mean=.1,
            mean_penalty_mean=0.,shrink_penalty_mean=0.) for i in range(1,epochs+1)])
    for f,v in [('train_result.json',dict(run=meta,source=source,saved_model_srcc=1.)),('test_result.json',test),('teacher_check.json',check),
        ('training_statistics.json',stats),('qmd_training.json',q),('model.pth.selection.json',dict(epoch=1,**compact)),
        ('last_epoch_result.json',dict(test,epoch=epochs)),('run_config.json',dict(argv=run.cell_args(root,cell,'3',smoke)[5:],protocol='july28'))]:core.save(out/f,v)
    (out/'model.pth.epochs.jsonl').write_text(''.join(json.dumps(dict(epoch=i,**compact))+'\n' for i in range(1,epochs+1)))
    return out


class SupplementRunnerTest(unittest.TestCase):
    def test_matrix_and_smoke(self):
        self.assertEqual(len(set(run.matrix())),54);self.assertEqual(len(run.smoke_cells(run.matrix())),18)
        self.assertEqual(len(run.prior.reference_cells()),90);self.assertEqual(len(run.METHODS),8)
        self.assertEqual(run.smoke_cells([]),[])
    def test_each_recipe_and_native_budget(self):
        for m in RECIPES:
            for smoke in (False,True):
                with tempfile.TemporaryDirectory() as temp:
                    root=Path(temp);c=('july28','Ball',0,m);out=fixture(root,c,smoke)
                    self.assertEqual(run.validate_cell(out,c,smoke)['srcc_x100'],100.)
                    argv=run.cell_args(root,c,'4',smoke);self.assertEqual('--epochs' in argv,smoke)
                    self.assertIn('--e_model',argv);self.assertNotIn('--qmd_weight',argv)
    def mutation(self,file,edit,message,method=SHRINK):
        with tempfile.TemporaryDirectory() as temp:
            c=('july28','Ball',0,method);out=fixture(Path(temp),c);d=core.read(out/file);edit(d);core.save(out/file,d)
            with self.assertRaisesRegex(ValueError,message):run.validate_cell(out,c)
    def test_kd_forbidden(self):self.mutation('qmd_training.json',lambda d:d.update(kd_weight=.1),'disabled')
    def test_e_frozen(self):self.mutation('qmd_training.json',lambda d:d['frozen_after'].update(e='changed'),'updated')
    def test_test_stats_forbidden(self):self.mutation('training_statistics.json',lambda d:d.update(source_split='test'),'training-only')
    def test_nan_spread_rejected(self):self.mutation('training_statistics.json',lambda d:d.update(e_prediction_std=float('nan')),'nonfinite')
    def test_bound_tampering(self):self.mutation('training_statistics.json',lambda d:d.update(bound=.1),'bound')
    def test_gate_range(self):self.mutation('test_result.json',lambda d:d.update(gate=1.1),'gate/bound')
    def test_fixed_gate_unchanged(self):self.mutation('test_result.json',lambda d:d.update(gate=.5),'fixed gate',FIXED)
    def test_no_epoch_zero(self):self.mutation('model.pth.selection.json',lambda d:d.update(epoch=0),'history')
    def test_missing_expert(self):self.mutation('test_result.json',lambda d:d['expert_metrics'].pop('F'),'expert keys')
    def test_wrong_pairs(self):self.mutation('test_result.json',lambda d:d.update(broken_pairs=1),'pair counts')
    def test_reload_score(self):self.mutation('train_result.json',lambda d:d.update(saved_model_srcc=.5),'selection/reload')
    def test_resume_skips_completed_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c=('july28','Ball',0,SHRINK);out=fixture(root,c,True)
            (out/'model.pth').write_bytes(b'model');(out/'model.pth.eval_rng.pt').write_bytes(b'rng')
            core.save(out/'complete.json',dict(files={p.name:core.sha(p) for p in out.iterdir() if p.suffix in ('.json','.jsonl')},
                checkpoint_sha256=core.sha(out/'model.pth'),eval_rng_sha256=core.sha(out/'model.pth.eval_rng.pt')))
            with patch.object(core,'command') as command:
                state,_=run.run_cell(root/'smoke',{},c,'4',True)
                self.assertEqual(state,'SKIP');command.assert_not_called()
    def test_missing_e_stops_before_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root,('july28','Ball',0,FIXED))
            with self.assertRaisesRegex(FileNotFoundError,'v16 FusedGT'):run.resolve_e(root)
    def test_status_and_archive(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'result';root.mkdir();core.save(root/'manifest.json',dict(jobs=run.matrix(),study='full'))
            text=io.StringIO()
            with redirect_stdout(text):run.status(root)
            self.assertIn('0/54',text.getvalue());self.assertIn('e_modal_shrink: 0/18',text.getvalue())
            (root/'model.pth').write_bytes(b'not for archive');core.save(root/'report.json',dict(ok=True))
            archive=core.package(root,'_partial')
            with zipfile.ZipFile(archive) as z:self.assertNotIn('model.pth',z.namelist());self.assertIn('report.json',z.namelist())


if __name__=='__main__':unittest.main()
