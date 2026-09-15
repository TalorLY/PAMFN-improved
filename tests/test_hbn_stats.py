import copy
import csv
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import pipeline
from tools import hbn_stats as hbn
from tools import eval_qmd_hbn as worker
from tools.verify_sr_qmd_runtime import frozen_module
from tests.test_qmd_head import HeadModel
from tests import test_eval_policy as fixtures


class Data(Dataset):
    def __init__(self, n=33, train=True):
        self.is_train = train
        self.x = torch.randn(n,16,4,generator=torch.Generator().manual_seed(104))
        self.labels = [('sample%d'%i,(i+1)/(n+1)) for i in range(n)]
    def __len__(self):
        return len(self.x)
    def __getitem__(self,i):
        return self.x[i], self.x[i]*.6, self.x[i]*.3, self.labels[i][1], 0


class StochasticHead(HeadModel):
    def forward(self, inputs):
        # Mimic July28 routing draws without changing any normalization buffers.
        if self.eval_policy=='legacy_gumbel':
            inputs={k:v+torch.rand(())*.02 for k,v in inputs.items()}
        return super().forward(inputs)


class HBNTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_moments_unequal_batches_and_between_batch_mean(self):
        x=torch.arange(66,dtype=torch.float64).reshape(33,2,1)
        stats=hbn.ChannelMoments()
        stats.observe(x[:32]); stats.observe(x[32:])
        result=stats.result()
        self.assertEqual(result['count'],33)
        self.assertTrue(torch.allclose(result['mean'],x.mean((0,2))))
        self.assertTrue(torch.allclose(result['var'],x.var((0,2),unbiased=False)))

    def test_moments_reject_nonfinite_empty_and_single_observation(self):
        for x in (torch.full((2,2,1),float('nan')),torch.zeros(0,2,1)):
            with self.assertRaises(ValueError):
                hbn.ChannelMoments().observe(x)
        one=hbn.ChannelMoments(); one.observe(torch.ones(1,2,1))
        with self.assertRaises(ValueError):
            one.result()

    def test_mixture_variance_includes_mean_shift_and_identity_is_exact(self):
        bn=nn.BatchNorm1d(2)
        stats=dict(mean=torch.tensor([2.,4.]),var=torch.tensor([3.,5.]))
        result=hbn.blended_stats(bn,stats,.25)
        self.assertTrue(torch.equal(result['mean'],torch.tensor([.5,1.])))
        self.assertTrue(torch.equal(result['var'],torch.tensor([2.25,5.])))
        zero=hbn.blended_stats(bn,stats,0.)
        self.assertTrue(torch.equal(zero['mean'],bn.running_mean))
        self.assertTrue(torch.equal(zero['var'],bn.running_var))
        for alpha in (-1,.5,float('nan')):
            with self.assertRaises(ValueError):
                hbn.blended_stats(bn,stats,alpha)

    def test_collection_preserves_full_state_modes_rng_and_singleton_tail(self):
        model=HeadModel().train()
        old=copy.deepcopy(model.state_dict())
        state=torch.get_rng_state(); py=random.getstate(); np_state=np.random.get_state()
        stats=hbn.collect_head_stats(model,Data(),pipeline._prepare_batch)
        self.assertEqual((stats['samples'],stats['batches'],stats['layers']),(33,2,1))
        self.assertTrue(all(torch.equal(v,model.state_dict()[n]) for n,v in old.items()))
        self.assertTrue(all(m.training for m in model.modules()))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(py,random.getstate())
        self.assertTrue(np.array_equal(np_state[1],np.random.get_state()[1]))

    def test_no_label_dependence_and_no_test_dataset(self):
        model=HeadModel().eval(); data=Data()
        first=hbn.collect_head_stats(model,data,pipeline._prepare_batch)
        data.labels=[(n,1000-v) for n,v in data.labels]
        second=hbn.collect_head_stats(model,data,pipeline._prepare_batch)
        self.assertTrue(torch.equal(first['mean'],second['mean']))
        self.assertTrue(torch.equal(first['var'],second['var']))
        data.is_train=False
        with self.assertRaisesRegex(ValueError,'training dataset'):
            hbn.collect_head_stats(model,data,pipeline._prepare_batch)

    def test_context_and_failed_collection_restore_state_and_remove_hook(self):
        model=HeadModel().train(); before=copy.deepcopy(model.state_dict())
        with self.assertRaises(RuntimeError):
            with hbn.use_head_stats(model,dict(mean=torch.ones(8),var=torch.ones(8)*2)):
                raise RuntimeError('fixture')
        self.assertTrue(all(torch.equal(v,model.state_dict()[n]) for n,v in before.items()))
        def fail(*args):
            torch.rand(3); np.random.rand(); random.random(); raise RuntimeError('fixture')
        state=torch.get_rng_state(); np_state=np.random.get_state(); py=random.getstate()
        with self.assertRaises(RuntimeError):
            hbn.collect_head_stats(model,Data(),fail)
        self.assertEqual(len(model.fc[2]._forward_pre_hooks),0)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(py,random.getstate())
        self.assertTrue(np.array_equal(np_state[1],np.random.get_state()[1]))

    def test_real_pamfn_changes_only_two_head_buffers_and_cache_matches_full(self):
        for policy in ('argmax','legacy_gumbel'):
            with self.subTest(policy=policy), hbn.isolated_rng(5):
                model=fixtures.EvaluationPolicyTest._build_pamfn(policy).eval()
                old=copy.deepcopy(model.state_dict())
                # K=2 small architecture has 55 BN layers; production K differs.
                self.assertEqual(sum(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in model.modules()),55)
                stats=hbn.collect_head_stats(model,Data(3),pipeline._prepare_batch)
                self.assertTrue(all(torch.equal(v,model.state_dict()[n]) for n,v in old.items()))
                inputs=fixtures.EvaluationPolicyTest._inputs(); cache=[]
                hook=model.fc[2].register_forward_pre_hook(lambda m,x:cache.append(x[0].detach().clone()))
                with hbn.isolated_rng(19), torch.no_grad():
                    baseline=model(inputs)[0].flatten().tolist()
                hook.remove()
                original_cache=cache[0].clone()
                pred,_=hbn.score_cached_head(model,cache)
                self.assertEqual(pred,baseline)
                with hbn.use_head_stats(model,hbn.blended_stats(model.fc[2],stats,.1)):
                    changes={n for n,v in old.items() if not torch.equal(v,model.state_dict()[n])}
                    self.assertTrue(changes.issubset({'fc.2.running_mean','fc.2.running_var'}))
                    candidate,_=hbn.score_cached_head(model,cache)
                    with hbn.isolated_rng(19), torch.no_grad():
                        full=model(inputs)[0].flatten().tolist()
                    self.assertEqual(candidate,full)
                self.assertTrue(torch.equal(cache[0],original_cache))
                self.assertTrue(all(torch.equal(v,model.state_dict()[n]) for n,v in old.items()))

    def test_reject_alternative_heads(self):
        for variant in ('linear','groupnorm'):
            with self.assertRaises(ValueError):
                hbn.head_bn(HeadModel(variant))

    def test_constant_scores_retained_as_null_not_zero_or_dropped(self):
        result=worker.metrics([1.,1.,1.],[0.,.5,1.],[-1.,0.,1.])
        self.assertIsNone(result['coef']); self.assertEqual(result['pre_sigmoid_coef'],100.)

    def test_complete_worker_with_native_pipelines_modern_and_july28_rng(self):
        # Small deterministic data/model; run complete worker logic, native metric code,
        # checkpoints, buffer files, RNG replay, cache/full smoke and CSV handoff.
        repo=Path(pipeline.__file__).resolve().parent
        for protocol in ('modern','july28'):
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as tmp, hbn.isolated_rng(21):
                tmp=Path(tmp); train=Data(33); test=Data(8,False)
                loaders=(DataLoader(train,batch_size=32),DataLoader(test,batch_size=1))
                model=StochasticHead().eval()
                model.eval_policy='argmax' if protocol=='modern' else 'legacy_gumbel'
                if protocol=='modern':
                    native=pipeline
                else:
                    # Historical pipeline imports an unused training scheduler eagerly.
                    # No scheduler/optimizer is exercised by this checkpoint-only test.
                    stub=types.ModuleType('utils.lr_scheduler')
                    stub.build_scheduler=lambda *a,**kw: self.fail('evaluation must not build a scheduler')
                    with patch.dict(sys.modules,{'utils.lr_scheduler':stub}):
                        native=frozen_module('ad13082a15b36041268f0eb9685978a0c87e346b','pipeline.py','july28_fixture')
                cp=tmp/'model.pth'; torch.save(model.state_dict(),str(cp))
                rng=(__import__('models.score_qmd',fromlist=['rng_state']).rng_state(),np.random.get_state())
                rp=tmp/'model.pth.eval_rng.pt'; torch.save(rng,str(rp))
                with hbn.isolated_rng():
                    expected=native.test_epoch(model,loaders[1],eval_policy='argmax') if protocol=='modern' else native.test_epoch(model,loaders[1])
                spec=dict(cell=[protocol,'Ball',0,'score_qmd' if protocol=='modern' else 'ordered'],
                    repo=str(repo),recipe=hbn.recipe(),argv=[],checkpoint=str(cp),checkpoint_sha256=worker.digest(cp),
                    expected_test={k:expected[k] for k in ('coef','loss_x100','prediction_std')},
                    eval_rng=str(rp),eval_rng_sha256=worker.digest(rp))
                worker.save(tmp/'spec.json',spec)
                entry=types.SimpleNamespace(__file__=str(repo/'main.py'),
                    parser_args=lambda:object(),load_config=lambda *a:None,
                    build_configs=lambda *a:({},500,{}, {},{},{}))
                native.__file__=str(repo/'pipeline.py')
                before_cwd=Path.cwd(); before_path=list(sys.path)
                try:
                    with patch.dict(sys.modules,{'main':entry,'pipeline':native,'hbn_stats':hbn}), \
                         patch.object(native,'build_dataset',return_value=loaders), \
                         patch.object(native,'build_model',side_effect=lambda **kw:copy.deepcopy(model)), \
                         patch.dict(os.environ,{},clear=False), \
                         patch('sys.argv',['worker','--spec',str(tmp/'spec.json'),'--output',str(tmp/'out'),'--gpu','-1','--smoke']), \
                         redirect_stdout(io.StringIO()):
                        worker.main()
                finally:
                    os.chdir(str(before_cwd)); sys.path[:]=before_path
                result=json.loads((tmp/'out/result.json').read_text())
                self.assertEqual(len(result['results']),3)
                self.assertTrue(result['checks']['full_vs_cache_exact'])
                self.assertEqual(result['results'][0]['delta'],0)
                self.assertEqual(worker.digest(cp),spec['checkpoint_sha256'])
                self.assertEqual(len((tmp/'out/predictions.csv').read_text().splitlines()),25)
                with (tmp/'out/predictions.csv').open() as handle:
                    prediction_rows=list(csv.DictReader(handle))
                self.assertEqual(float(prediction_rows[0]['label']),float(torch.tensor(test.labels[0][1],dtype=torch.float32)))


if __name__=='__main__':
    unittest.main()
