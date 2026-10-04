import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.score_qmd import rng_state
from tools.qmd_residual import (ResidualBranch, ResidualObjective, anchor_view, assert_frozen,
    freeze_anchor, gradient_stats, preserved_rng, state_digest)
from tools.train_qmd_residual import (evaluate, load, make_cache, metrics, parser_args, save,
    sha, tensor_save, train_branch, verify_teacher)


class TinyAnchor(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat=nn.Sequential(nn.Conv1d(4,4,3,padding=1),nn.BatchNorm1d(4),nn.ReLU())
        self.fc=nn.Sequential(nn.Conv1d(4,1,1),nn.Sigmoid())

    def forward(self,x):
        feature=self.feat((x['V']+x['F']*.5+x['A']*.25).transpose(1,2))
        feature=feature[:,:,:1]+feature.mean(2,keepdim=True)+torch.rand_like(feature[:,:,:1])*.03
        return self.fc(feature).squeeze(2),{}

    def call_loss(self,pred,y,**kwargs): return (pred.flatten()-y.flatten()).square().mean()


class TinyData(Dataset):
    def __len__(self):return 12
    def __getitem__(self,i):
        x=torch.arange(32,dtype=torch.float32).view(8,4)/32+i*.11
        return x,x*.6,x*.8,.15+i*.055,dict(V=8,F=8,A=8)


class ResidualTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(27);np.random.seed(27)
        self.anchor=freeze_anchor(TinyAnchor())
        self.x={m:torch.rand(4,8,4) for m in ('V','F','A')}
        self.y=torch.tensor([.2,.45,.6,.8])
        self.lengths={m:[8]*4 for m in self.x}

    def test_private_initialization_and_exact_zero(self):
        state=torch.get_rng_state().clone()
        branch=ResidualBranch(4,seed=3)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        base,feature=anchor_view(self.anchor,self.x)
        self.assertTrue(torch.equal(base,branch.score(base,feature)))
        self.assertFalse(feature.requires_grad)

    def test_bound_and_no_batchnorm_or_dropout(self):
        branch=ResidualBranch(4,bound=.1)
        with torch.no_grad():branch.net[-1].weight.fill_(100.)
        self.assertLessEqual(float(branch(torch.rand(10,4)).detach().abs().max()),.100001)
        self.assertFalse(any(isinstance(m,(nn.modules.batchnorm._BatchNorm,nn.Dropout)) for m in branch.modules()))

    def test_freeze_guard(self):
        assert_frozen(self.anchor)
        self.anchor.train()
        with self.assertRaises(RuntimeError):anchor_view(self.anchor,self.x)

    def test_hook_removed_on_failure(self):
        with patch.object(self.anchor,'forward',side_effect=RuntimeError('fixture')):
            with self.assertRaises(RuntimeError):anchor_view(self.anchor,self.x)
        self.assertFalse(self.anchor.fc._forward_pre_hooks)

    def test_gt_and_kd_only_reach_branch(self):
        original=state_digest(self.anchor)
        for weight in (0.,.5,1.):
            branch=ResidualBranch(4); obj=ResidualObjective(0,2,weight,True)
            optimizer=torch.optim.AdamW(branch.parameters(),lr=.001)
            before=state_digest(branch)
            loss,info=obj.losses(self.anchor,branch,self.x,self.y,self.lengths,0)
            optimizer.zero_grad();loss.backward();optimizer.step()
            self.assertNotEqual(before,state_digest(branch));assert_frozen(self.anchor)
            self.assertEqual(original,state_digest(self.anchor))
            self.assertEqual(info['weighted_kd']>0,weight>0)
            self.assertGreater(info['kd_norm'],0)

    def test_same_batch_scaling_and_rng_for_all_arms(self):
        state=rng_state(); observed=[]
        for weight in (0.,.5,1.):
            branch=ResidualBranch(4);obj=ResidualObjective(0,2,weight,True)
            with preserved_rng((state,np.random.get_state())):
                loss,info=obj.losses(self.anchor,branch,self.x,self.y,self.lengths,0)
                observed.append((float(loss.detach()),info,obj.calibration,torch.get_rng_state().clone()))
        for value in observed[1:]:
            self.assertTrue(torch.equal(value[3],observed[0][3]))
            self.assertEqual(value[2]['beta'],observed[0][2]['beta'])
            self.assertEqual(value[1]['kd_norm'],observed[0][1]['kd_norm'])
        self.assertAlmostEqual(observed[1][1]['weighted_kd']*2,observed[2][1]['weighted_kd'])

    def test_replayed_cache_matches_original_and_restores_rng(self):
        import pipeline
        loader=DataLoader(TinyData(),batch_size=1)
        replay=(rng_state(),np.random.get_state())
        with preserved_rng(replay):old=pipeline.test_epoch(self.anchor,loader,True)
        state=torch.get_rng_state().clone()
        cache=make_cache(self.anchor,loader,torch.device('cpu'),replay)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(metrics(cache[0],cache[2])['predictions'],old['predictions'])
        self.assertEqual(evaluate(ResidualBranch(4),cache)['coef'],old['coef'])

    def test_real_pamfn_fc_feature_capture_and_frozen_buffers(self):
        from models.pamfn import PAMFN,BaseModel
        with patch('models.pamfn._torch_load',side_effect=[BaseModel(4,8,0.,m) for m in ('V','F','A')]):
            anchor=PAMFN(8,0.,1,0.,2,1,1,'.','x','x','x','Ball',
                ordinal_prototype_weight=.0001,ordinal_quality_thresholds=(.25,.5,.75))
        freeze_anchor(anchor); before=state_digest(anchor)
        x={m:torch.rand(4,16,4) for m in ('V','F','A')}
        replay=(rng_state(),np.random.get_state())
        with preserved_rng(replay):old,_=anchor(x)
        with preserved_rng(replay):score,feat=anchor_view(anchor,x)
        self.assertEqual(feat.shape,(4,24));self.assertTrue(torch.equal(old.flatten(),score))
        branch=ResidualBranch(24);obj=ResidualObjective(0,2,1.,True)
        loss,info=obj.losses(anchor,branch,x,self.y,{m:[16]*4 for m in x},0)
        loss.backward();assert_frozen(anchor)
        self.assertEqual(before,state_digest(anchor));self.assertGreater(info['kd_norm'],0.)

    def test_train_save_reload_and_epoch_zero_excluded(self):
        replay=(rng_state(),np.random.get_state())
        cache=make_cache(self.anchor,DataLoader(TinyData(),batch_size=1),torch.device('cpu'),replay)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);tensor_save(root/'teacher_rng.pt',replay)
            source=dict(expected_srcc_x100=metrics(cache[0],cache[2])['coef'])
            branch=ResidualBranch(4)
            result=train_branch(self.anchor,branch,DataLoader(TinyData(),batch_size=4),cache,
                2,.5,0,.001,True,root,source,root/'teacher_rng.pt')
            loaded=ResidualBranch(4);loaded.load_state_dict(load(root/'model.pth')['branch'])
            self.assertAlmostEqual(evaluate(loaded,cache)['coef'],result['saved_model_srcc']*100)
            report=json.loads((root/'qmd_training.json').read_text())
            self.assertTrue(report['anchor_unchanged']);self.assertFalse(report['epoch_zero_eligible'])
            self.assertIn(result['selected_epoch'],(1,2));self.assertTrue(report['branch_updated'])
            self.assertGreater(report['positive_kd_gradient_batches'],0)

    def test_teacher_hash_and_identity_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);tensor_save(root/'model.pth',self.anchor.state_dict());tensor_save(root/'rng.pt',rng_state())
            args=argparse.Namespace(teacher_model=str(root/'model.pth'),teacher_rng=str(root/'rng.pt'),
                teacher_record=str(root/'complete.json'),action='Ball',seed=0)
            record=dict(checkpoint_sha256=sha(args.teacher_model),eval_rng_sha256=sha(args.teacher_rng),
                metrics=dict(protocol='july28',action='Ball',seed=0,method='ordered',srcc_x100=80.))
            save(args.teacher_record,record);verify_teacher(args)
            args.seed=1
            with self.assertRaises(ValueError):verify_teacher(args)
            args.seed=0;record['checkpoint_sha256']='bad';save(args.teacher_record,record)
            with self.assertRaises(ValueError):verify_teacher(args)

    def test_gradient_conflict_measurement(self):
        p=nn.Parameter(torch.tensor([1.]))
        stats=gradient_stats(p.square().sum(),-p.square().sum(),[p])
        self.assertAlmostEqual(stats['cosine'],-1.)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available locally')
    def test_branch_initialization_does_not_reseed_cuda(self):
        before=torch.cuda.get_rng_state_all()
        ResidualBranch(4,seed=2)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,torch.cuda.get_rng_state_all())))

    def test_invalid_recipe_rejected(self):
        for weight in (-1,.25,2.):
            with self.assertRaises(ValueError):ResidualObjective(0,10,weight)
        with self.assertRaises(ValueError):ResidualBranch(4,bound=float('nan'))


if __name__=='__main__':unittest.main()
