import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from main import parser_args, qmd_config
from models.score_qmd import ScoreQMD, reliable_rank_kd, rng_state
from pipeline import train_epoch
from tests.test_score_qmd import TinyModel, TinyDataset
from tests.test_coqm_rng_isolation import COQMRNGIsolationTest
from tools.verify_sr_qmd_runtime import SCORE_COMMIT, frozen_module


class LengthDataset(TinyDataset):
    def __getitem__(self,index):
        a,b,c,y,_ = super().__getitem__(index)
        return a,b,c,y,dict(V=8,F=8,A=8)


class SRQMDTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pairs_are_unique_correct_and_confidence_detached(self):
        teacher = torch.tensor([.1,.8,.4],requires_grad=True)
        target = torch.tensor([.1,.5,.9],requires_grad=True)
        student = torch.tensor([.3,.6,.5],requires_grad=True)
        loss, info = reliable_rank_kd(student,teacher,target)
        self.assertEqual(info['total_pairs'],3)
        self.assertEqual(info['eligible_pairs'],2)  # teacher reverses samples 1/2
        expected = (np.exp(-3)*.5*(.3-.7)**2 + np.exp(-5)*.5*(.2-.3)**2)/2
        self.assertAlmostEqual(float(loss.detach()),expected,places=8)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNone(target.grad)
        self.assertGreater(float(student.grad.abs().sum()),0)

    def test_no_pair_and_single_sample_have_finite_zero_gradients(self):
        for teacher,target in [([.5],[.5]),([.8,.2],[.2,.8]),([.2,.2],[.2,.8]),([.2,.4],[.5,.51])]:
            student = torch.ones(len(teacher),requires_grad=True)
            loss,info = reliable_rank_kd(student,torch.tensor(teacher),torch.tensor(target))
            self.assertEqual(info['eligible_pairs'],0)
            self.assertEqual(float(loss.detach()),0.)
            loss.backward()
            self.assertTrue(torch.equal(student.grad,torch.zeros_like(student)))

    def test_uniformly_bad_teacher_is_not_renormalized_to_full_strength(self):
        student = torch.tensor([.3,.4],requires_grad=True)
        target = torch.tensor([.2,.8])
        good,_ = reliable_rank_kd(student,target,target)
        bad,_ = reliable_rank_kd(student,target+1.,target)
        self.assertAlmostEqual(float((bad/good).detach()),np.exp(-10),places=7)

    def test_rank_loss_is_shift_invariant_and_global_rng_free(self):
        x,y,t = torch.tensor([.1,.7,.5]),torch.tensor([.2,.6,.9]),torch.tensor([.3,.5,.8])
        before = torch.get_rng_state().clone()
        left,_ = reliable_rank_kd(x,t,y)
        right,_ = reliable_rank_kd(x+2,t,y)
        self.assertAlmostEqual(float(left),float(right),places=7)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_score_v1_and_zero_rank_remain_exact_over_two_epochs(self):
        old = frozen_module(SCORE_COMMIT,'models/score_qmd.py','sr_test_old_qmd')
        old_pipe = frozen_module(SCORE_COMMIT,'pipeline.py','sr_test_old_pipeline')
        def run(cls, function, mode, **extra):
            torch.manual_seed(33)
            np.random.seed(33)
            model = TinyModel()
            optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
            qmd = cls(mode,33,2,warmup=0,rampup=0,**extra)
            loader = DataLoader(LengthDataset(),batch_size=4,shuffle=True,num_workers=0)
            metrics=[]
            for epoch in range(2):
                result = function(model,optimizer,None,loader,epoch,qmd=qmd)
                result.pop('time')
                metrics.append(result)
            return model.state_dict(),torch.get_rng_state(),metrics,qmd.calibration
        expected = run(old.ScoreQMD,old_pipe.train_epoch,'score_qmd')
        for mode,extra in [('score_qmd',{}),('score_rank_qmd',dict(rank_weight=0))]:
            actual = run(ScoreQMD,train_epoch,mode,**extra)
            self.assertEqual(expected[2:],actual[2:])
            self.assertTrue(torch.equal(expected[1],actual[1]))
            for name,value in expected[0].items():
                self.assertTrue(torch.equal(value,actual[0][name]),name)

    def test_combined_pamfn_backward_bn_rng_and_checkpoint(self):
        ordered = dict(ordinal_prototype_weight=.0001,ordinal_quality_thresholds=(.25,.5,.75),
                       ordinal_prototype_target_mode='continuous',ordinal_grade_centers=(.1,.35,.65,.9),
                       quality_aux_rng_isolated=True,eval_policy='argmax')
        model, state = COQMRNGIsolationTest._build_pamfn(37,**ordered)
        base, base_state = COQMRNGIsolationTest._build_pamfn(37,eval_policy='argmax')
        self.assertTrue(torch.equal(state,base_state))
        for name,value in base.state_dict().items():
            self.assertTrue(torch.equal(value,model.state_dict()[name]),name)
        model.train()
        inputs = {m:torch.randn(4,16,4) for m in ('V','F','A')}
        before = rng_state()
        full, info = model(inputs)
        # Synthetic training labels follow this teacher, guaranteeing eligible pairs.
        labels = full.detach().reshape(-1) + .02
        rng_after = torch.get_rng_state().clone()
        buffers = {n:b.clone() for n,b in model.named_buffers()}
        qmd = ScoreQMD('score_rank_qmd',37,1,warmup=0,rampup=0,rank_gap=0)
        # The tiny pretrained fixture uses V for all three branches; perturb V.
        qmd.mask.pending = [0]
        aux, diagnostic = qmd.auxiliary(model,inputs,{m:[16]*4 for m in inputs},labels,
                                       full,model.call_loss(full,labels),before,0)
        self.assertTrue(torch.equal(rng_after,torch.get_rng_state()))
        for n,value in buffers.items():
            self.assertTrue(torch.equal(value,dict(model.named_buffers())[n]),n)
        objective = model.call_training_loss(full,labels,**info)+aux
        objective.backward()
        self.assertGreater(diagnostic['eligible_pairs'],0)
        self.assertGreater(diagnostic['rank_kd'],0)
        self.assertTrue(qmd.summary()['rank']['gradient_verified'])
        self.assertGreater(float(model.quality_projector.weight.grad.abs().sum()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        restored,_ = COQMRNGIsolationTest._build_pamfn(39,**ordered)
        restored.load_state_dict(model.state_dict())
        model.eval()
        restored.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(inputs)[0],restored(inputs)[0]))

    def test_cli_combination_and_invalid_settings(self):
        common=['main.py','--gpu','-1','--feats','2','--action','Ball','--multi_modality']
        good=['--qmd_mode','score_rank_qmd','--ordinal_prototype_weight','.0001',
              '--ordinal_prototype_target_mode','continuous','--quality_aux_rng_isolated']
        with patch('sys.argv',common+good):
            args=parser_args()
            self.assertEqual(qmd_config(args)['rank_weight'],.5)
        for extra in [good[:-1],['--qmd_rank_gamma','0'],['--qmd_rank_weight','nan'],
                      ['--qmd_rank_gap','-1'],['--qmd_mode','score_rank_qmd','--coqm_weight','.01']]:
            with patch('sys.argv',common+extra),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                parser_args()


if __name__ == '__main__':
    unittest.main()
