import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader

from tests.test_qmd_residual import TinyAnchor,TinyData
from tools.qmd_residual import freeze_anchor,preserved_rng,state_digest,assert_frozen
from models.score_qmd import rng_state
from tools.qmd_complement import ComplementBranch,ComplementObjective
from tools.qmd_reskd import ResidualKDBranch,ResidualKDObjective,distillation_loss,calibrated_beta
from tools.qmd_reskd_config import (RECIPES,GT,SCORE,RESIDUAL,ENERGY_GT,ENERGY_RESIDUAL,V13_GT,
    BETA_CAP,TARGET_RATIO,metadata,parameter_count)
from tools.train_qmd_reskd import (evaluate,make_cache,metrics,train_branch,load,tensor_save,statistics_digest,parser_args)


class ResidualKDTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(27);np.random.seed(27)
        self.anchor=freeze_anchor(TinyAnchor())
        self.inputs={m:torch.rand(4,8,4) for m in ('V','F','A')}
        self.y=torch.tensor([.2,.45,.6,.8]);self.lengths={m:[8]*4 for m in self.inputs}
        self.cache=make_cache(self.anchor,DataLoader(TinyData(),batch_size=1),torch.device('cpu'),
            (rng_state(),np.random.get_state()))

    def branch(self,tag=GT):
        b=ResidualKDBranch(tag,4);b.fit_training_statistics(self.cache);return b

    def assert_state_equal(self,a,b):
        if torch.is_tensor(a):self.assertTrue(torch.equal(a,b))
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for k in a:self.assert_state_equal(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assert_state_equal(x,y)
        else:self.assertEqual(a,b)

    def test_six_recipes_same_capacity_private_init(self):
        state=torch.get_rng_state().clone();digests=[]
        for tag in RECIPES:
            b=ResidualKDBranch(tag)
            self.assertEqual(sum(p.numel() for p in b.parameters()),12320)
            self.assertEqual(parameter_count(b.config),12320);digests.append(state_digest(b))
        self.assertEqual(len(set(digests)),1);self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_v13_architecture_and_initialization_unchanged(self):
        self.assert_state_equal(ComplementBranch(V13_GT,4).state_dict(),ResidualKDBranch(GT,4).state_dict())

    def test_residual_loss_ignores_frozen_anchor_mask_drift(self):
        a=torch.tensor([.2,.8]);am=a+.3;r=torch.tensor([.01,-.01]);rm=r+.005
        x=distillation_loss('residual',a,am,r,rm,.1)
        self.assertTrue(torch.equal(x,distillation_loss('residual',a-2,am+3,r,rm,7.)))
        self.assertGreater(float(distillation_loss('score',a,am,r,rm,.1)),float(x))

    def test_targets_stop_full_branch_and_anchor_gradients(self):
        for kind in ('score','residual'):
            a=torch.randn(4,requires_grad=True);am=torch.randn(4,requires_grad=True)
            r=torch.randn(4,requires_grad=True);rm=torch.randn(4,requires_grad=True)
            distillation_loss(kind,a,am,r,rm,torch.tensor(.1)).backward()
            self.assertIsNone(a.grad);self.assertIsNone(am.grad);self.assertIsNone(r.grad)
            self.assertGreater(float(rm.grad.abs().sum()),0.)

    def test_reject_unknown_target(self):
        with self.assertRaises(ValueError):distillation_loss('bad',None,None,None,None,None)

    def test_beta_zero_then_recovers(self):
        self.assertEqual(calibrated_beta(1.,0.),0.)
        self.assertAlmostEqual(calibrated_beta(1.,1.),TARGET_RATIO)
        self.assertEqual(calibrated_beta(0.,1.),0.)
        self.assertEqual(calibrated_beta(1.,1e-10),BETA_CAP)
        self.assertAlmostEqual(calibrated_beta(1.,1.),TARGET_RATIO)

    def test_nonfinite_calibration_rejected(self):
        for x in (float('nan'),float('inf'),-1.):
            with self.assertRaises(FloatingPointError):calibrated_beta(x,1.)
            with self.assertRaises(FloatingPointError):calibrated_beta(1.,x)

    def test_bounded_ratio_over_many_gradient_scales(self):
        for g in (0.,1e-14,1e-8,1e-3,1.,1e3):
            for k in (0.,1e-14,1e-8,1e-3,1.,1e3):
                beta=calibrated_beta(g,k)
                self.assertLessEqual(beta,BETA_CAP)
                if g>1e-12:self.assertLessEqual(beta*k/g,TARGET_RATIO+1e-12)

    def test_zero_residual_first_batch_does_not_disable_later_kd(self):
        b=self.branch(RESIDUAL);obj=ResidualKDObjective(0,RESIDUAL);opt=torch.optim.AdamW(b.parameters(),lr=.0001)
        reports=[]
        for step in range(8):
            opt.zero_grad();loss,info=obj.losses(self.anchor,b,self.inputs,self.y,self.lengths,0)
            loss.backward();opt.step();reports.append(info)
        self.assertEqual(reports[0]['kd'],0.);self.assertEqual(reports[0]['coefficient'],0.)
        self.assertTrue(any(r['coefficient']*r['kd_norm']>0 for r in reports[1:]))
        self.assertGreater(max(r['kd_ratio'] for r in reports[1:]),.001)
        self.assertEqual(obj.calibration_summary()['first_positive']['epoch'],1)

    def test_energy_loss_decreases_residual_magnitude(self):
        r=torch.tensor([-.02,.01],requires_grad=True);energy=r.square().mean()
        grad=torch.autograd.grad(energy,r)[0]
        self.assertTrue(torch.equal(grad,r.detach()))
        self.assertLess(float((r-.1*grad).square().mean().detach()),float(energy.detach()))

    def test_energy_and_kd_loss_accounting(self):
        for tag in RECIPES:
            b=self.branch(tag)
            with torch.no_grad():b.net[-1].weight.fill_(.02)
            loss,r=ResidualKDObjective(0,tag).losses(self.anchor,b,self.inputs,self.y,self.lengths,0)
            self.assertAlmostEqual(float(loss.detach()),r['gt']+r['weighted_energy']+r['weighted_kd'],places=6)
            self.assertAlmostEqual(r['weighted_energy'],RECIPES[tag]['energy']*r['energy'])
            self.assertLessEqual(r['kd_ratio'],.5*.2*TARGET_RATIO+1e-8)

    def test_paired_rng_all_six(self):
        state=(rng_state(),np.random.get_state());out=[]
        for tag in RECIPES:
            b=self.branch(tag)
            with preserved_rng(state):
                _,r=ResidualKDObjective(0,tag).losses(self.anchor,b,self.inputs,self.y,self.lengths,0)
                out.append((torch.get_rng_state().clone(),r['modality'],r['gt']))
        for result in out[1:]:
            self.assertTrue(torch.equal(out[0][0],result[0]));self.assertEqual(out[0][1:],result[1:])

    def test_v13_gt_training_optimizer_and_rng_parity(self):
        state=(rng_state(),np.random.get_state());out=[]
        for old in (True,False):
            with preserved_rng(state):
                b=ComplementBranch(V13_GT,4) if old else ResidualKDBranch(GT,4)
                b.fit_training_statistics(self.cache)
                obj=ComplementObjective(0,V13_GT) if old else ResidualKDObjective(0,GT)
                opt=torch.optim.AdamW(b.parameters(),lr=.0001,weight_decay=.01)
                for epoch in range(3):
                    for step in range(3):
                        opt.zero_grad();loss,_=obj.losses(self.anchor,b,self.inputs,self.y,self.lengths,epoch)
                        loss.backward();opt.step()
                out.append((copy.deepcopy(b.state_dict()),copy.deepcopy(opt.state_dict()),torch.get_rng_state().clone()))
        self.assert_state_equal(out[0],out[1])

    def test_no_anchor_or_statistics_updates(self):
        anchor=state_digest(self.anchor)
        for tag in RECIPES:
            b=self.branch(tag);before=state_digest(b);stats=statistics_digest(b)
            loss,_=ResidualKDObjective(0,tag).losses(self.anchor,b,self.inputs,self.y,self.lengths,0)
            opt=torch.optim.AdamW(b.parameters(),lr=.0001);loss.backward();opt.step()
            self.assertNotEqual(before,state_digest(b));self.assertEqual(stats,statistics_digest(b))
            assert_frozen(self.anchor);self.assertEqual(anchor,state_digest(self.anchor))

    def test_gt_arms_do_not_apply_kd(self):
        for tag in (GT,ENERGY_GT):
            b=self.branch(tag);obj=ResidualKDObjective(0,tag)
            for _ in range(3):
                _,r=obj.losses(self.anchor,b,self.inputs,self.y,self.lengths,0)
                self.assertEqual(r['weighted_kd'],0.);self.assertEqual(r['kd_ratio'],0.)
            self.assertIsNone(obj.calibration_summary()['first_positive'])

    def test_bounded_branch_and_training_only_statistics(self):
        b=self.branch();before=statistics_digest(b)
        with self.assertRaises(RuntimeError):b.fit_training_statistics(self.cache)
        with torch.no_grad():b.net[-1].weight.fill_(100.)
        self.assertLessEqual(float(b(self.cache[1]).detach().abs().max()),.030001)
        self.assertEqual(before,statistics_digest(b))

    def test_test_labels_do_not_change_predictions(self):
        b=self.branch();a=evaluate(b,self.cache)
        other=evaluate(b,(self.cache[0],self.cache[1],self.cache[2].flip(0)))
        self.assertEqual(a['predictions']['prediction'],other['predictions']['prediction'])

    def test_all_six_train_save_reload(self):
        for tag in RECIPES:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);b=self.branch(tag);tensor_save(root/'rng.pt',(rng_state(),np.random.get_state()))
                source=dict(expected_srcc_x100=metrics(self.cache[0],self.cache[2])['coef'])
                result=train_branch(self.anchor,b,DataLoader(TinyData(),batch_size=4),self.cache,2,tag,0,True,
                    root,source,root/'rng.pt',dict(source_split='train',samples=12))
                saved=load(root/'model.pth');restored=ResidualKDBranch(tag,4);restored.load_state_dict(saved['branch'])
                self.assertAlmostEqual(result['saved_model_srcc']*100,evaluate(restored,self.cache)['coef'])
                q=json.loads((root/'qmd_training.json').read_text());self.assertTrue(q['statistics_unchanged'])
                self.assertEqual(q['calibration']['steps'],6)
                self.assertEqual(q['positive_kd_gradient_batches']>0,RECIPES[tag]['weight']>0)

    def test_cli_rejects_seed_deletion_or_short_formal_budget(self):
        args=['--gpu','-1','--action','Ball','--seed','0','--run_tag',GT,'--teacher_model','m',
            '--teacher_rng','r','--teacher_record','c','--result_json','o','--save_final_model','model.pth']
        self.assertIsNone(parser_args(args).epochs)
        with self.assertRaises(SystemExit):parser_args(args+['--epochs','3'])
        with self.assertRaises(SystemExit):parser_args(args+['--seed','42'])
        self.assertEqual(metadata('Ball',1,GT,500)['seed'],1)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable locally')
    def test_private_init_preserves_cuda_rng(self):
        state=torch.cuda.get_rng_state_all()
        for tag in RECIPES:ResidualKDBranch(tag)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(state,torch.cuda.get_rng_state_all())))


if __name__=='__main__':unittest.main()
