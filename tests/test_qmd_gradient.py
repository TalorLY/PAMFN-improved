import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import subprocess
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.score_qmd import ScoreQMD
from tools.qmd_gradient import GradientGuard, check_projection, correction_for, geometry, recipe


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(4, 4)
        self.fc = nn.Sequential(nn.Dropout(.1), nn.Conv1d(4, 8, 1), nn.BatchNorm1d(8),
                                nn.ReLU(), nn.Dropout(.3), nn.Conv1d(8, 1, 1), nn.Sigmoid())
        self.ordered = nn.Parameter(torch.randn(4))
        self.unused = nn.Parameter(torch.ones(2))

    def forward(self, inputs):
        feature = self.trunk(sum(inputs.values()).mean(1))
        return self.fc(feature.unsqueeze(-1)).squeeze(-1), dict(feature=feature)

    def call_loss(self, prediction, label, **info):
        return ((prediction.flatten()-label.flatten())**2).mean()

    def call_training_loss(self, prediction, label, **info):
        # Only main uses this Ordered-like head, so projection must not touch it.
        return self.call_loss(prediction,label) + .0001*(info['feature']-self.ordered).square().mean()


class TinyData(Dataset):
    def __init__(self):
        self.x = torch.randn(8,12,4,generator=torch.Generator().manual_seed(99))
    def __len__(self): return len(self.x)
    def __getitem__(self,i):
        x = self.x[i] + .01*random.random() + .01*np.random.random()
        return x, x*.4, x*.7, torch.tensor((i+1)/10), {m:12 for m in ('V','F','A')}


def assert_equal_tree(test, left, right):
    test.assertEqual(type(left),type(right))
    if torch.is_tensor(left): test.assertTrue(torch.equal(left,right))
    elif isinstance(left,np.ndarray): test.assertTrue(np.array_equal(left,right))
    elif isinstance(left,dict):
        test.assertEqual(left.keys(),right.keys())
        for k in left: assert_equal_tree(test,left[k],right[k])
    elif isinstance(left,(tuple,list)):
        test.assertEqual(len(left),len(right))
        for a,b in zip(left,right): assert_equal_tree(test,a,b)
    else: test.assertEqual(left,right)


class GradientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_projection_removes_only_negative_parallel_component(self):
        main=(torch.tensor([1.,0.]),torch.tensor([7.]),None)
        aux=(torch.tensor([-2.,3.]),None,torch.tensor([6.]))
        delta,before,after,coefficient=correction_for(main,aux)
        self.assertTrue(torch.equal(delta[0],torch.tensor([2.,0.])))
        self.assertIsNone(delta[1]); self.assertIsNone(delta[2])
        self.assertEqual(coefficient,2.)
        self.assertEqual(before['dot'],-2.); self.assertEqual(after['dot'],0.)
        self.assertTrue(torch.equal(aux[0],torch.tensor([-2.,3.])))

    def test_aligned_zero_and_disjoint_gradients_do_not_change(self):
        for main,aux in [((torch.ones(2),),(torch.ones(2),)),((torch.zeros(2),),(torch.ones(2),)),
                         ((None,torch.ones(1)),(torch.ones(2),None))]:
            delta,_,_,coef=correction_for(main,aux)
            self.assertEqual(coef,0.); self.assertTrue(all(x is None for x in delta))

    def test_nonfinite_rejected(self):
        with self.assertRaises(FloatingPointError):
            correction_for((torch.ones(2),),(torch.tensor([float('nan'),1.]),))

    def test_fix1_near_orthogonal_float32_roundoff_regression(self):
        main=(torch.tensor([-1.0809345245361328, 1.40631902217865]),)
        aux=(torch.tensor([-0.9482316374778748, -0.7288362979888916]),)
        before=geometry(main,aux)
        coefficient=-before['dot']/before['main_sq']
        expected=main[0]*coefficient
        after=geometry(main,(aux[0]+expected,))
        self.assertLess(after['dot'],-1e-5*abs(before['dot']))  # Old code raises.
        with self.assertWarnsRegex(RuntimeWarning,'roundoff'):
            delta,b,a,c=correction_for(main,aux)
        self.assertTrue(torch.equal(delta[0],expected))  # No gradient rewrite.
        self.assertEqual((b,a,c),(before,after,coefficient))
        self.assertLess(abs(a['cosine']),torch.finfo(torch.float32).eps)

    def test_fix1_material_projection_failure_is_not_silenced(self):
        main=(torch.tensor([1.,0.]),); aux=(torch.tensor([-2.,3.]),)
        before=geometry(main,aux)
        with self.assertRaisesRegex(FloatingPointError,'after_dot=.*tolerance='):
            check_projection(before,dict(before),main,aux,2.)
        for value in (float('nan'),float('inf')):
            with self.assertRaises(FloatingPointError):
                correction_for(main,(torch.tensor([value,1.]),))

    def test_fix1_projection_across_dtypes_scales_and_tensor_partitions(self):
        import warnings
        generator=torch.Generator().manual_seed(231)
        for dtype in (torch.float32,torch.float64):
            for scale in (1e-8,1.,1e8):
                for trial in range(20):
                    m=torch.randn(24,generator=generator,dtype=dtype)*scale
                    a=torch.randn(24,generator=generator,dtype=dtype)*scale
                    if trial%2:
                        a=(a.double()-m.double()*(m.double().dot(a.double())/
                           m.double().dot(m.double()))).to(dtype)
                    main=(m[:9],m[9:],None)
                    aux=(a[:9],a[9:],torch.ones(1,dtype=dtype))
                    before=geometry(main,aux)
                    coefficient=-before['dot']/before['main_sq'] if before['dot']<0 else 0.
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',RuntimeWarning)
                        delta,_,after,c=correction_for(main,aux)
                    self.assertEqual(c,coefficient)
                    self.assertIsNone(delta[2])
                    for i in (0,1):
                        if coefficient: self.assertTrue(torch.equal(delta[i],main[i]*coefficient))
                        else: self.assertIsNone(delta[i])
                    self.assertGreaterEqual(after['cosine'],-32*torch.finfo(dtype).eps)

    def test_fix1_successful_v8_kd_all_optimizer_and_rng_paths_are_identical(self):
        import pipeline
        def old_check(before,after,main,target,coefficient):
            if coefficient and after['dot'] < -1e-5*max(abs(before['dot']),1e-24):
                raise FloatingPointError('old projection check')
        for mode in ('kd','all'):
            outcomes=[]
            for checker in (old_check,check_projection):
                pipeline.setup_seed(10); random.seed(10)
                model=TinyModel(); opt=torch.optim.AdamW(model.parameters(),lr=.001)
                qmd=ScoreQMD('score_qmd',0,2,grad_mode=mode,warmup=0,rampup=0)
                loader=DataLoader(TinyData(),batch_size=4,shuffle=True,num_workers=0)
                metrics=[]
                with patch('tools.qmd_gradient.check_projection',side_effect=checker),redirect_stdout(io.StringIO()):
                    for epoch in range(2):
                        result=pipeline.train_epoch(model,opt,None,loader,epoch,qmd=qmd)
                        result.pop('time'); metrics.append(result)
                outcomes.append((copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),
                                 torch.get_rng_state(),np.random.get_state(),random.getstate(),metrics,qmd.summary()))
            assert_equal_tree(self,outcomes[0],outcomes[1])

    def test_kd_and_all_are_distinct_and_main_only_parameter_untouched(self):
        observed={}
        for mode in ('observe','kd','all'):
            shared=nn.Parameter(torch.tensor([1.,1.]))
            main_only=nn.Parameter(torch.tensor([2.]))
            unused=nn.Parameter(torch.ones(1))
            pars=[shared,main_only,unused]
            main=shared[0]+main_only.sum()
            gt=-3*shared[0]+shared[1]
            kd=shared[0]
            guard=GradientGuard(mode)
            rng=torch.get_rng_state().clone()
            correction=guard.prepare(main,(gt,kd,'V'),pars,0)
            self.assertTrue(all(p.grad is None for p in pars))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            (main+gt+kd).backward()
            guard.apply(pars,correction)
            observed[mode]=shared.grad.clone()
            self.assertTrue(torch.equal(main_only.grad,torch.ones(1)))
            self.assertIsNone(unused.grad)
            self.assertEqual(guard.summary()['batches'],1)
        self.assertTrue(torch.equal(observed['observe'],observed['kd']))
        self.assertTrue(torch.equal(observed['all'],torch.tensor([1.,1.])))

    def test_two_epochs_off_observe_match_frozen_weights_optimizer_bn_rng_metrics(self):
        import pipeline
        # Full original pipeline + original QMD are loaded as source, not imported
        # from another worktree. The complete bundle carries both ancestors.
        # Reliable protocol detection from native CLI capabilities.
        import main as entry
        with patch('sys.argv',['main.py','--gpu','-1','--feats','2','--action','Ball']):
            july = not hasattr(entry.parser_args(),'eval_policy')
        commit='ad13082a15b36041268f0eb9685978a0c87e346b' if july else '09a38c6009c17f5c1df3e25f5ad1d8a5aac8750f'
        old=[]
        for path in ('pipeline.py','models/score_qmd.py'):
            module=types.ModuleType('frozen_'+path.replace('/','_'))
            code=subprocess.check_output(['git','show',commit+':'+path]).decode()
            exec(compile(code,commit+'/'+path,'exec'),module.__dict__)
            old.append(module)
        outcomes=[]
        for mode in ('frozen','off','observe'):
            pipeline.setup_seed(21); random.seed(21)
            loader=DataLoader(TinyData(),batch_size=4,shuffle=True,num_workers=0)
            model=TinyModel(); opt=torch.optim.AdamW(model.parameters(),lr=.001)
            cls=old[1].ScoreQMD if mode=='frozen' else ScoreQMD
            extra={} if mode=='frozen' else dict(grad_mode=mode)
            qmd=cls('score_qmd',0,2,warmup=0,rampup=0,**extra)
            metrics=[]
            with redirect_stdout(io.StringIO()):
                for epoch in range(2):
                    result=(old[0] if mode=='frozen' else pipeline).train_epoch(model,opt,None,loader,epoch,qmd=qmd)
                    result.pop('time'); metrics.append(result)
            report=qmd.summary()
            guard=report.pop('gradient_guard',None)
            if mode=='observe':
                self.assertEqual(guard['batches'],4); self.assertEqual(guard['projections'],0)
            outcomes.append((copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),
                             torch.get_rng_state().clone(),np.random.get_state(),random.getstate(),metrics,report))
        assert_equal_tree(self,outcomes[0],outcomes[1])
        assert_equal_tree(self,outcomes[0],outcomes[2])

    def test_active_modes_train_and_report_both_auxiliary_gradients(self):
        import pipeline
        for mode in ('kd','all'):
            pipeline.setup_seed(10); random.seed(10)
            model=TinyModel(); opt=torch.optim.SGD(model.parameters(),lr=.01)
            qmd=ScoreQMD('score_qmd',0,2,grad_mode=mode,warmup=0,rampup=0)
            with redirect_stdout(io.StringIO()):
                pipeline.train_epoch(model,opt,None,DataLoader(TinyData(),batch_size=4),0,qmd=qmd)
            report=qmd.summary()['gradient_guard']
            self.assertEqual(report['batches'],2)
            self.assertTrue(all('gt_conflicts' in r and 'kd_conflicts' in r for r in report['rows']))
            self.assertIsNone(qmd._grad_terms)
            self.assertTrue(all(torch.isfinite(v).all() for v in model.state_dict().values()))

    def test_default_metadata_and_invalid_mode(self):
        q=ScoreQMD('score_qmd',0,2)
        self.assertIsNone(q.guard); self.assertNotIn('gradient_guard',q.summary())
        for mode in ('bad','off'):
            with self.assertRaises(ValueError): GradientGuard(mode)
        with self.assertRaises(ValueError): ScoreQMD('score_rank_qmd',0,2,grad_mode='all')
        self.assertEqual(recipe('all')['inference'],'unchanged')

    def test_prediction_capture_preserves_scores_and_rng(self):
        import pipeline
        outcomes=[]
        for enabled in (False,True):
            pipeline.setup_seed(30); random.seed(30)
            model=TinyModel()
            result=pipeline.test_epoch(model,DataLoader(TinyData(),batch_size=4),return_predictions=enabled)
            result.pop('time')
            if enabled:
                values=result.pop('predictions')
                self.assertEqual(len(values['label']),8)
                from scipy.stats import spearmanr
                self.assertAlmostEqual(result['coef'],spearmanr(values['label'],values['prediction'])[0]*100)
            outcomes.append((result,torch.get_rng_state(),np.random.get_state(),random.getstate()))
        assert_equal_tree(self,outcomes[0],outcomes[1])

    def test_native_pamfn_graph_with_ordered_and_all_guard_modes(self):
        import pipeline
        from models.pamfn import PAMFN, BaseModel
        outcomes={}
        for mode in ('off','observe','kd','all'):
            pipeline.setup_seed(42)
            pretrained=BaseModel(4,4,.1)
            config=dict(model_dim=4,fc_drop=.1,fc_r=2,feat_drop=.1,K=2,ms_heads=1,cm_heads=1,
                        ckpt_dir='.',rgb_ckpt_name='x',flow_ckpt_name='x',audio_ckpt_name='x',dataset_name='TES',
                        ordinal_prototype_weight=.0001,ordinal_quality_thresholds=(.25,.5,.75))
            with patch('models.pamfn._torch_load',side_effect=lambda _:copy.deepcopy(pretrained)):
                model=PAMFN(**config)
            x=torch.randn(2,16,4)
            batch=(x,x*.7,x*.4,torch.tensor([.3,.8]),{m:torch.tensor([16,16]) for m in ('V','F','A')})
            qmd=ScoreQMD('score_qmd',0,2,grad_mode=mode,warmup=0,rampup=0)
            opt=torch.optim.AdamW(model.parameters(),lr=.001)
            with redirect_stdout(io.StringIO()): pipeline.train_epoch(model,opt,None,[batch],0,qmd=qmd)
            outcomes[mode]=(copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),torch.get_rng_state())
            if mode!='off': self.assertEqual(qmd.guard.summary()['batches'],1)
        assert_equal_tree(self,outcomes['off'],outcomes['observe'])


if __name__=='__main__': unittest.main()
