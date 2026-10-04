"""CPU regression tests. Synthetic scores never enter paper tables."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import bootstrap
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from adapter import (TransferModel,seed_all,freeze,state_digest,capture_rng,tensor_save,
                     frozen_view,historical_inference,assert_frozen)
from modules import EBranch,UBranch
from controls import PlainE,FusedU,ordinary_mse,control_engine
from deployment import InferenceSession,e_score,u_delta
from contract import (POLICY,METHODS,cells,budget,ref_file,reference,save,sha,read,completed)
import worker as legacy_worker

class Synthetic(Dataset):
    def __init__(self):self.rows=[('synthetic_%d'%i,.2+.075*i) for i in range(8)]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        rng=torch.Generator().manual_seed(i+91)
        return tuple(torch.randn(12,d,generator=rng) for d in (768,768,1024))+(torch.tensor(self.rows[i][1],dtype=torch.float32),)

def fixture(root):
    seed_all(0);ds=Synthetic();train=DataLoader(ds,batch_size=4,shuffle=True);test=DataLoader(ds,batch_size=1)
    model=TransferModel(cpu=True).eval();native=root/'weights'/'mcmoe';native.mkdir(parents=True)
    tensor_save(native/'model.pth',model.backbone.state_dict());rng=capture_rng();tensor_save(native/'model.pth.eval_rng.pt',rng)
    freeze(model);cache=legacy_worker.make_cache(model,test,torch.device('cpu'),rng)
    e=EBranch();e.fit(cache[0],cache[1],cache[3]);e.eval()
    with torch.no_grad():e.net[-1].weight.normal_(0,.03)
    u=UBranch();u.fit(e.evaluation_score(cache[0],cache[1]).detach(),cache[2]);u.eval()
    with torch.no_grad():
        for head in u.heads:head[-1].weight.normal_(0,.03)
    results={'mcmoe':legacy_worker.metrics(cache[0].tolist(),cache[3].tolist()),
             'mc_e':legacy_worker.branch_evaluate(cache,e),'mc_u':legacy_worker.branch_evaluate(cache,e,u)}
    for name,branch in [('mcmoe',None),('mc_e',e),('mc_u',u)]:
        folder=root/'weights'/name;folder.mkdir(parents=True,exist_ok=True)
        if branch is not None:tensor_save(folder/'model.pth',dict(branch=branch.state_dict()))
        result=results[name];result.update(task='TES',seed=0,method=name,phase='smoke',ids=[v[0] for v in ds.rows])
        save(folder/'test_result.json',result)
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
        dest=ref_file(root,'smoke','TES',0,name)
        save(dest,dict(directory=str(folder),files=files));save(dest.parent/'test_result.json',result)
    return model,freeze(e),freeze(u),train,test,cache

class ControlsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_matrix_and_fixed_protocol(self):
        self.assertEqual(len(cells()),42);self.assertEqual(len(cells(True)),4)
        self.assertEqual(budget(),500);self.assertEqual(budget(True),2)
        self.assertEqual(POLICY['inherited']['selection'],POLICY['selection'])
        self.assertFalse(POLICY['inherited']['new_hyperparameter_search'])
    def test_capacity_and_initialization_match(self):
        for cls,base,width in [(PlainE,EBranch,256),(FusedU,UBranch,768)]:
            old=base(width,2);new=cls(width,2)
            self.assertEqual(sum(p.numel() for p in old.parameters()),sum(p.numel() for p in new.parameters()))
            for k,v in old.state_dict().items():self.assertTrue(torch.equal(v,new.state_dict()[k]),k)
    def test_control_constructor_does_not_consume_rng(self):
        torch.manual_seed(53);before=torch.get_rng_state().clone();PlainE();FusedU()
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
    def test_ordinary_loss_definition_and_detach(self):
        e=PlainE();x=torch.randn(8,256,requires_grad=True);p=torch.randn(8,requires_grad=True);y=torch.randn(8)
        e.fit(p,x,y);loss,_=ordinary_mse(e,p,x,y)
        self.assertTrue(torch.equal(loss,torch.nn.functional.mse_loss(p.detach()+e.global_bias+e(x),y)))
        loss.backward();self.assertIsNone(x.grad);self.assertIsNone(p.grad)
        self.assertGreater(float(e.net[-1].weight.grad.abs().sum()),0)
    def test_plain_is_unbounded(self):
        e=PlainE();x=torch.ones(8,256);e.fit(torch.zeros(8),x,torch.zeros(8))
        with torch.no_grad():e.net[0].bias.fill_(10);e.net[-1].weight.fill_(10)
        self.assertGreater(float(e(x).detach().abs().max()),.03)
    def test_fused_control_only_changes_feature_source(self):
        p=torch.randn(2);j=torch.randn(2,256);m=torch.randn(2,768)
        with patch.object(legacy_worker,'frozen_view',return_value=(p,j,m)):
            with control_engine('mc_fused_u') as engine:
                q,k,v=engine.frozen_view(None)
                self.assertIs(q,p);self.assertIs(k,j);self.assertTrue(torch.equal(v,j.repeat(1,3)))
        self.assertIs(legacy_worker.EBranch,EBranch)
        self.assertIs(legacy_worker.UBranch,UBranch)
    def test_unknown_control_rejected(self):
        with self.assertRaises(ValueError):
            with control_engine('unknown'):pass

class DeploymentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.model,cls.e,cls.u,cls.train,cls.test,cls.cache=fixture(cls.root)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_direct_branch_formulas_batch1_and_batch8(self):
        for n in (1,8):
            p,j,m=self.cache[:3];p,j,m=p[:n],j[:n],m[:n]
            with torch.no_grad():
                self.assertTrue(torch.equal(e_score(self.e,p,j),self.e.evaluation_score(p,j)))
                self.assertTrue(torch.equal(u_delta(self.u,m),self.u(m)))
    def test_native_session_parity_and_state(self):
        xs=next(iter(self.test))[:3];before=state_digest(self.model)
        p,j,m=frozen_view(self.model,*xs)
        with torch.no_grad():expected=self.e.evaluation_score(p,j)+self.u(m).double()
        with InferenceSession(self.model,self.e,self.u) as session:
            self.assertTrue(torch.equal(session(*xs),expected))
            self.assertTrue(torch.equal(session(*xs),expected))
        assert_frozen(self.model);self.assertEqual(state_digest(self.model),before)
        self.assertEqual(len(self.model.backbone.regressor._forward_pre_hooks),0)
        self.assertEqual(len(self.model.backbone.transformer.encoder._forward_hooks),0)
    def test_exception_restores_flags_and_hooks(self):
        with self.assertRaisesRegex(RuntimeError,'test error'):
            with InferenceSession(self.model,self.e,self.u):raise RuntimeError('test error')
        assert_frozen(self.model);self.assertEqual(len(self.model.backbone.regressor._forward_pre_hooks),0)
    def test_requires_context(self):
        s=InferenceSession(self.model,self.e,self.u)
        with self.assertRaises(RuntimeError):s(*next(iter(self.test))[:3])
    def test_requires_calibration_and_eval(self):
        e=freeze(EBranch())
        with self.assertRaises(ValueError):
            with InferenceSession(self.model,e):pass
    def test_full_dataset_replay(self):
        from benchmark_worker import replay
        paths={k:reference(self.root,'smoke','TES',0,m) for k,m in zip(('anchor','e','u'),('mcmoe','mc_e','mc_u'))}
        checked,_=replay(self.model,self.e,self.u,self.test,torch.device('cpu'),paths)
        self.assertEqual(checked['status'],'PASS');self.assertEqual(checked['samples'],8)
    def test_reference_tamper_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'old';out.mkdir();save(out/'model.json',{'x':1})
            record=dict(directory=str(out),files={'model.json':sha(out/'model.json')})
            save(ref_file(root,'smoke','TES',0,'mcmoe'),record)
            save(out/'model.json',{'x':2})
            with self.assertRaises(ValueError):reference(root,'smoke','TES',0,'mcmoe')

class RunnerTest(unittest.TestCase):
    def test_storage_counts_and_reserve(self):
        import run
        with tempfile.TemporaryDirectory() as tmp:
            from collections import namedtuple
            usage=namedtuple('usage','total used free')(100*2**30,50*2**30,50*2**30)
            with patch('storage.shutil.disk_usage',return_value=usage):r=run.disk_check(Path(tmp),2)
            self.assertEqual(sum(r['remaining_counts'].values()),46)
            self.assertGreaterEqual(r['required_bytes'],int(9.5*1024**3))
    def test_pack_does_not_include_weights_or_read_links(self):
        import run,zipfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'results';root.mkdir();save(root/'value.json',dict(ok=True))
            (root/'model.pth').write_bytes(b'not for upload')
            archive=run.pack(root,True)
            with zipfile.ZipFile(archive) as z:
                self.assertFalse(any(n.endswith('.pth') for n in z.namelist()))
                self.assertIn('RETURN_CONTENTS.json',z.namelist())
    def test_benchmark_matrix(self):
        import run
        self.assertEqual(len(run.bench_cells()),42);self.assertEqual(len(run.bench_cells(True)),4)
    def test_completed_hash_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=Path(tmp);out=c/'attempt_1';out.mkdir();save(out/'result.json',{'a':1})
            save(c/'complete.json',dict(attempt='attempt_1',files={'result.json':sha(out/'result.json')}))
            self.assertEqual(completed(c,True),out)
            save(out/'result.json',{'a':2})
            with self.assertRaises(ValueError):completed(c,True)

class ExportTest(unittest.TestCase):
    def records(self):
        from paper_export import ALL
        result={}
        for t in ('TES','PCS','SS','TR','PE','CO','IN'):
            for s in (0,1,2):
                for m in ALL:
                    p=list(range(12));y=list(range(12))
                    if m=='mc_plain_e':p[0],p[7]=p[7],p[0]
                    if m=='mc_u':p[0],p[3]=p[3],p[0]
                    from scipy.stats import spearmanr
                    result[t,s,m]=dict(task=t,seed=s,method=m,coef=float(spearmanr(p,y)[0]*100),
                        ids=['video%d'%i for i in range(12)],predictions=dict(prediction=p,label=y))
        return result
    def test_export_signed_design_evidence(self):
        from paper_export import summarize
        rows,seeds,tasks,pairs=summarize(self.records())
        self.assertEqual((len(rows),len(seeds),len(tasks)),(8,24,56))
        self.assertGreater(pairs[0]['mean_gain'],0)
        self.assertLess(pairs[1]['mean_gain'],0)
        self.assertEqual(pairs[1]['interpretation'],'does not support mean advantage')
    def test_prediction_audit_rejects_wrong_score_and_duplicates(self):
        from paper_export import check_prediction
        r=self.records()['TES',0,'mcmoe'];check_prediction(r,12)
        r['coef']=90
        with self.assertRaises(ValueError):check_prediction(r,12)
        r['coef']=100;r['ids'][1]=r['ids'][0]
        with self.assertRaises(ValueError):check_prediction(r,12)
    def test_case_queue_keeps_all_seeds_and_failures(self):
        from paper_export import cases
        rows,queue=cases(self.records())
        self.assertEqual(len(rows),7*3*12)
        self.assertEqual(len(queue['failure']),2)
        self.assertTrue(all(len(r['seed_gains'])==3 for r in queue['failure']))
        self.assertLess(queue['failure'][0]['mean_rank_error_reduction'],0)
    def test_cost_raw_repetitions_required(self):
        from paper_export import audit_cost
        modes=('legacy_anchor','legacy_e','legacy_u','deploy_anchor','deploy_e','deploy_u')
        r=dict(task='TES',seed=0,parent='mcmoe',session_setup_wall_ms=1.,
            parameters=dict(anchor=9122183,e=4128,u=12384),parity=dict(batch1_vs_reference_max_abs=0.),
            cost=dict(status='measured',observed_exclusive=False,device='synthetic',
                variants={m:dict(wall_ms=[1.]*60,cuda_ms=[.9]*60,wall_ms_median=1.,cuda_ms_median=.9) for m in modes}))
        self.assertEqual(audit_cost(r)['deploy_u_overhead_pct'],0.)
        r['cost']['variants']['deploy_u']['wall_ms'].pop()
        with self.assertRaises(ValueError):audit_cost(r)
    def test_missing_controls_cannot_finalize(self):
        from paper_export import build
        from contract import POLICY
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'manifest.json',dict(policy=POLICY))
            with self.assertRaises((ValueError,FileNotFoundError)):build(root)
    def test_full_table_export_fixture(self):
        # Presentation/integration fixture only: validators are mocked explicitly;
        # this test cannot create a production completion flag or scientific score.
        import paper_export as pe
        from run import bench_root
        with tempfile.TemporaryDirectory(prefix='synthetic_v39_export_') as tmp:
            root=Path(tmp);save(root/'manifest.json',dict(policy=POLICY))
            n=247;pred=list(range(n));ids=['synthetic_%d'%i for i in range(n)]
            folders={}
            for t in pe.TASKS:
                for s in pe.SEEDS:
                    for m in pe.ALL:
                        c=root/'formal'/t/('seed%d'%s)/m;out=c/'attempt_1'
                        r=dict(task=t,seed=s,method=m,coef=100.,ids=ids,predictions=dict(prediction=pred,label=pred))
                        save(out/'test_result.json',r)
                        save(c/'complete.json',dict(attempt='attempt_1',files={'test_result.json':sha(out/'test_result.json')}))
                        folders[t,s,m]=out
                    for parent in ('mcmoe','mc_ordered'):
                        c=bench_root(root,'formal')/t/('seed%d'%s)/parent;out=c/'attempt_1'
                        modes=('legacy_anchor','legacy_e','legacy_u','deploy_anchor','deploy_e','deploy_u')
                        r=dict(task=t,seed=s,parent=parent,session_setup_wall_ms=1.,parameters=dict(anchor=9122183,e=4128,u=12384),
                            parity=dict(batch1_vs_reference_max_abs=0.),cost=dict(status='measured',observed_exclusive=False,
                            device='SYNTHETIC TEST ONLY',variants={m:dict(wall_ms=[1.]*60,cuda_ms=[1.]*60,wall_ms_median=1.,cuda_ms_median=1.) for m in modes}))
                        save(out/'benchmark.json',r);save(c/'complete.json',dict(attempt='attempt_1',files={'benchmark.json':sha(out/'benchmark.json')}))
            with patch.object(pe,'reference',side_effect=lambda r,p,t,s,m:folders[t,s,m]), \
                 patch('run.validate_control',side_effect=lambda out,*a:read(out/'test_result.json')), \
                 patch('run.deps',return_value={}), \
                 patch('run.validate_benchmark',side_effect=lambda out,*a:read(out/'benchmark.json')):
                result=pe.build(root)
            self.assertEqual(result['scores'],168);self.assertEqual(result['cost_records'],42)
            self.assertTrue((root/'paper/v39_table.tex').is_file())
            self.assertFalse((root/'completion.json').exists())
            self.assertIn('shared-GPU',' '.join(read(root/'paper/gaps.json')['open_items']))
            self.assertEqual(len(read(root/'paper/paired_predictions.json')),168)

def integration():
    """Actual native model, two controls, two epochs; no production completion flag."""
    from design_worker import execute
    from benchmark_worker import execute as benchmark
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='v39_integration_') as tmp:
        root=Path(tmp);model,e,u,train,test,_=fixture(root)
        before={m:read(ref_file(root,'smoke','TES',0,m))['files'] for m in ('mcmoe','mc_e','mc_u')}
        for method in METHODS:
            args=argparse.Namespace(root=str(root),phase='smoke',task='TES',seed=0,method=method,
                out=str(root/method),data='',cpu=True,workers=0)
            result=execute(args,train,test)
            assert result['reloaded_exact'] and result['epochs']==2
            assert read(root/method/'qmd_training.json')['branch_updated']
            assert result['synthetic_cpu_test_only']
        args=argparse.Namespace(root=str(root),phase='smoke',task='TES',seed=0,parent='mcmoe',
            out=str(root/'timing'),data='',cpu=True,physical_gpu='0',require_exclusive=False)
        result=benchmark(args,test);assert result['parity']['status']=='PASS' and result['cost']['status']=='not_measured'
        for m in before:
            folder=reference(root,'smoke','TES',0,m)
            assert {k:sha(folder/k) for k in before[m]}==before[m]
        print('V39_ACTUAL_NATIVE_CPU_INTEGRATION_PASS; synthetic only')

if __name__=='__main__':
    import sys
    if '--integration' in sys.argv:integration()
    else:unittest.main()
