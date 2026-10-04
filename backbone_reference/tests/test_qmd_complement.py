import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tests.test_qmd_residual import TinyAnchor,TinyData
from tools.qmd_residual import freeze_anchor,anchor_view,preserved_rng,state_digest,assert_frozen
from models.score_qmd import rng_state
from tools.qmd_complement_config import RECIPES,LINEAR_GT,MLP_GT,parameter_count,metadata
from tools.qmd_complement import ComplementBranch,ComplementObjective,centered_supervision,teacher_target
from tools.train_qmd_complement import (evaluate,make_cache,metrics,train_branch,load,tensor_save,
    statistics_digest,parser_args)


class ComplementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(27);np.random.seed(27)
        self.anchor=freeze_anchor(TinyAnchor())
        self.inputs={m:torch.rand(4,8,4) for m in ('V','F','A')}
        self.y=torch.tensor([.2,.45,.6,.8]);self.lengths={m:[8]*4 for m in self.inputs}
        self.cache=make_cache(self.anchor,DataLoader(TinyData(),batch_size=1),torch.device('cpu'),
                              (rng_state(),np.random.get_state()))

    def branch(self,tag=LINEAR_GT):
        branch=ComplementBranch(tag,4)
        branch.fit_training_statistics(self.cache)
        return branch

    def test_train_only_statistics_and_immutable_refit(self):
        b=ComplementBranch(LINEAR_GT,4)
        stats=b.fit_training_statistics(self.cache)
        self.assertEqual(stats['source_split'],'train');self.assertEqual(stats['samples'],12)
        self.assertAlmostEqual(float(b.global_bias),float((self.cache[2]-self.cache[0]).mean()),places=6)
        with self.assertRaises(RuntimeError):b.fit_training_statistics(self.cache)
        with self.assertRaises(RuntimeError):ComplementBranch(LINEAR_GT,4)(self.cache[1])

    def test_bias_does_not_saturate_sample_branch(self):
        b=ComplementBranch(LINEAR_GT,4)
        b.fit_training_statistics((torch.zeros(4),torch.rand(4,4),torch.full((4,),.7)))
        self.assertAlmostEqual(float(b.global_bias),.7,places=6)
        self.assertEqual(float(b(torch.rand(4,4)).detach().abs().max()),0.)

    def test_zero_initialization_preserves_bias_only_ranking(self):
        b=self.branch();base=torch.tensor([1.,1.+2**-23,1.+2**-22])
        b.global_bias.fill_(10.)
        self.assertLess(len(torch.unique(base+b.global_bias)),3)
        p=b.evaluation_score(base,torch.rand(3,4))
        self.assertEqual(len(torch.unique(p)),3);self.assertEqual(p.dtype,torch.float64)
        self.assertTrue(torch.equal(p,base.double()+b.global_bias.double()))

    def test_private_initialization_and_architecture_counts(self):
        state=torch.get_rng_state().clone()
        for tag in RECIPES:
            b=ComplementBranch(tag)
            self.assertEqual(sum(p.numel() for p in b.parameters()),parameter_count(b.config))
            self.assertEqual(parameter_count(b.config),768 if not RECIPES[tag]['hidden'] else 12320)
            self.assertFalse(any(isinstance(m,(nn.modules.batchnorm._BatchNorm,nn.Dropout)) for m in b.modules()))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_zero_final_layer_and_normalization_floor(self):
        b=ComplementBranch(MLP_GT,4)
        b.fit_training_statistics((torch.zeros(4),torch.ones(4,4),torch.arange(4).float()))
        self.assertTrue(torch.isfinite(b(torch.full((4,4),1e10))).all())
        self.assertTrue(torch.allclose(b.feature_scale,torch.full((4,),.01)))
        with torch.no_grad():b.net[-1].weight.fill_(100.)
        self.assertLessEqual(float(b(torch.rand(4,4)).detach().abs().max()),.030001)

    def test_centered_gt_is_invariant_to_global_offsets(self):
        pred=torch.tensor([.1,.3,.7,.8],dtype=torch.float64,requires_grad=True)
        label=torch.tensor([.2,.35,.6,.9],dtype=torch.float64)
        r=torch.tensor([.01,-.01,.02,-.02],dtype=torch.float64)
        loss=centered_supervision(pred,label,r)[0]
        shifted=centered_supervision(pred+3,label-7,r)[0]
        self.assertAlmostEqual(float(loss.detach()),float(shifted.detach()),places=12)
        self.assertTrue(torch.allclose(torch.autograd.grad(loss,pred,retain_graph=True)[0],
                                      torch.autograd.grad(shifted,pred)[0],atol=1e-12))

    def test_targets_stop_all_teacher_gradients(self):
        raw=torch.randn(4,requires_grad=True);corrected=torch.randn(4,requires_grad=True)
        for kind,expected in [('raw',raw),('corrected',corrected)]:
            target=teacher_target(kind,raw,corrected)
            self.assertFalse(target.requires_grad);self.assertTrue(torch.equal(target,expected))
        with self.assertRaises(ValueError):teacher_target('bad',raw,corrected)

    def test_corrected_target_does_not_penalize_global_bias_for_identity_mask(self):
        b=self.branch();b.global_bias.fill_(.3)
        values={}
        state=(rng_state(),np.random.get_state())
        for tag in ['ordered_bc_linear_raw_kd','ordered_bc_linear_kd_w050']:
            obj=ComplementObjective(0,tag)
            obj.mask=lambda x,l:(x,dict(modality='V'))
            with preserved_rng(state):_,info=obj.losses(self.anchor,b,self.inputs,self.y,self.lengths,0)
            values[obj.recipe['target']]=info['kd']
        self.assertGreater(values['raw'],.04)
        self.assertEqual(values['corrected'],0.)

    def test_every_arm_updates_branch_not_anchor_or_statistics(self):
        frozen=state_digest(self.anchor)
        for tag in RECIPES:
            b=self.branch(tag);before=state_digest(b);stats=statistics_digest(b)
            obj=ComplementObjective(0,tag);optimizer=torch.optim.AdamW(b.parameters(),lr=.0001)
            loss,info=obj.losses(self.anchor,b,self.inputs,self.y,self.lengths,0)
            optimizer.zero_grad();loss.backward();optimizer.step()
            self.assertNotEqual(state_digest(b),before)
            self.assertEqual(statistics_digest(b),stats);assert_frozen(self.anchor)
            self.assertEqual(state_digest(self.anchor),frozen)
            self.assertEqual(info['coefficient']>0,RECIPES[tag]['weight']>0)
            self.assertAlmostEqual(info['scale'],.2);self.assertEqual(obj.calibration['epoch'],1)

    def test_same_rng_and_first_batch_calibration_for_matched_weights(self):
        state=(rng_state(),np.random.get_state());results=[]
        for tag in [LINEAR_GT,'ordered_bc_linear_kd_w050','ordered_bc_linear_kd_w100']:
            b=self.branch(tag);obj=ComplementObjective(0,tag)
            with preserved_rng(state):
                _,info=obj.losses(self.anchor,b,self.inputs,self.y,self.lengths,0)
                results.append((info,obj.calibration,torch.get_rng_state().clone()))
        self.assertEqual(results[0][1]['beta'],results[1][1]['beta'])
        self.assertEqual(results[1][1]['beta'],results[2][1]['beta'])
        self.assertAlmostEqual(results[1][0]['weighted_kd']*2,results[2][0]['weighted_kd'])
        self.assertTrue(all(torch.equal(results[0][2],v[2]) for v in results))

    def test_training_statistics_pass_preserves_rng(self):
        before=torch.get_rng_state().clone()
        make_cache(self.anchor,DataLoader(TinyData(),batch_size=4,shuffle=True),torch.device('cpu'),
                   (rng_state(),np.random.get_state()))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_calibration_rejects_nonfinite_data(self):
        b=ComplementBranch(LINEAR_GT,4)
        with self.assertRaises(ValueError):
            b.fit_training_statistics((torch.zeros(4),torch.full((4,4),float('nan')),torch.zeros(4)))

    def test_all_six_train_save_and_reload(self):
        for tag in RECIPES:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);b=self.branch(tag)
                replay=(rng_state(),np.random.get_state());tensor_save(root/'rng.pt',replay)
                source=dict(expected_srcc_x100=metrics(self.cache[0],self.cache[2])['coef'])
                stats=dict(source_split='train',samples=12)
                result=train_branch(self.anchor,b,DataLoader(TinyData(),batch_size=4),self.cache,
                    2,tag,0,True,root,source,root/'rng.pt',stats)
                saved=load(root/'model.pth');restored=ComplementBranch(tag,4);restored.load_state_dict(saved['branch'])
                self.assertAlmostEqual(result['saved_model_srcc']*100,evaluate(restored,self.cache)['coef'])
                self.assertEqual(saved['statistics_sha256'],statistics_digest(restored))
                self.assertIn(result['selected_epoch'],(1,2))
                q=json.loads((root/'qmd_training.json').read_text())
                self.assertEqual(q['kd_first_epoch'],1);self.assertTrue(q['statistics_unchanged'])
                self.assertFalse(q['epoch_zero_eligible'])

    def test_evaluation_labels_do_not_affect_statistics_or_predictions(self):
        b=self.branch();before=statistics_digest(b)
        a=evaluate(b,self.cache)
        alternate=(self.cache[0],self.cache[1],self.cache[2].flip(0))
        other=evaluate(b,alternate)
        self.assertEqual(a['predictions']['prediction'],other['predictions']['prediction'])
        self.assertEqual(before,statistics_digest(b))

    def test_smoke_and_native_budget_contract(self):
        argv=['--gpu','-1','--action','Ball','--seed','0','--run_tag',LINEAR_GT,
              '--teacher_model','m','--teacher_rng','r','--teacher_record','c',
              '--result_json','o','--save_final_model','model.pth']
        self.assertEqual(parser_args(argv).epochs,None)
        self.assertEqual(parser_args(argv+['--smoke','--epochs','2']).epochs,2)
        with self.assertRaises(SystemExit):parser_args(argv+['--epochs','3'])
        self.assertEqual(metadata('Ball',0,LINEAR_GT,500)['warmup_epochs'],0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable locally')
    def test_private_initialization_preserves_cuda_rng(self):
        state=torch.cuda.get_rng_state_all()
        for tag in RECIPES:ComplementBranch(tag)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(state,torch.cuda.get_rng_state_all())))


if __name__=='__main__':unittest.main()
