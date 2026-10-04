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
from tools.qmd_pair import PairBranch,PairObjective,pair_losses,reliable_pairs,calibrated_beta
from tools.qmd_pair_config import RECIPES,GT,RANK,GATE,COMBO,V13_GT,metadata,parameter_count
from tools.train_qmd_pair import evaluate,make_cache,metrics,train_branch,load,tensor_save,statistics_digest,parser_args


class PairTest(unittest.TestCase):
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
        b=PairBranch(tag,4);b.fit_training_statistics(self.cache);return b

    def assert_state_equal(self,a,b):
        if torch.is_tensor(a):self.assertTrue(torch.equal(a,b))
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for k in a:self.assert_state_equal(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assert_state_equal(x,y)
        else:self.assertEqual(a,b)

    def test_same_private_initialization_and_capacity(self):
        before=torch.get_rng_state().clone();digests=[]
        for tag in RECIPES:
            b=PairBranch(tag);digests.append(state_digest(b))
            self.assertEqual(sum(p.numel() for p in b.parameters()),12320)
            self.assertEqual(parameter_count(b.config),12320)
        self.assertEqual(len(set(digests)),1);self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_v13_branch_state_unchanged(self):
        self.assert_state_equal(ComplementBranch(V13_GT,4).state_dict(),PairBranch(GT,4).state_dict())

    def test_supervised_pair_gradient_repairs_wrong_order(self):
        base=torch.tensor([.51,.49]);y=torch.tensor([.2,.8]);r=torch.zeros(2,requires_grad=True)
        loss,_,info=pair_losses(base,r,r,y);loss.backward()
        self.assertGreater(float(r.grad[0]),0.);self.assertLess(float(r.grad[1]),0.)
        self.assertEqual(info['hard_pairs'],1);self.assertEqual(info['anchor_wrong_pairs'],1)

    def test_confident_anchor_pair_protected_from_new_reversal(self):
        base=torch.tensor([.48,.52]);y=torch.tensor([.2,.8]);r=torch.tensor([.03,-.03],requires_grad=True)
        loss,_,info=pair_losses(base,r,r,y);loss.backward()
        self.assertEqual(info['broken_pairs'],1);self.assertEqual(info['protected_pairs'],1)
        self.assertGreater(info['protect_loss'],0.);self.assertGreater(float(r.grad[0]),0.)

    def test_correct_large_margin_needs_no_rank_change(self):
        r=torch.zeros(2,requires_grad=True)
        loss,_,info=pair_losses(torch.tensor([.2,.8]),r,r,torch.tensor([.2,.8]))
        self.assertEqual(float(loss),0.);self.assertEqual(info['hard_pairs'],0)

    def test_unreachable_pair_does_not_force_branch_saturation(self):
        r=torch.zeros(2,requires_grad=True)
        loss,_,info=pair_losses(torch.tensor([.6,.4]),r,r,torch.tensor([.2,.8]))
        self.assertEqual(info['unreachable_pairs'],1);self.assertEqual(info['hard_pairs'],0)
        loss.backward();self.assertEqual(float(r.grad.abs().sum()),0.)

    def test_ties_and_small_label_gaps_are_not_rank_targets(self):
        r=torch.zeros(3,requires_grad=True);rm=torch.zeros(3,requires_grad=True)
        loss,kd,info=pair_losses(torch.tensor([.8,.3,.2]),r,rm,torch.tensor([.5,.5,.51]))
        self.assertEqual(info['reliable_pairs'],0);(loss+kd).backward()
        self.assertEqual(float(r.grad.abs().sum()+rm.grad.abs().sum()),0.)

    def test_one_sample_and_empty_pairs_are_safe(self):
        r=torch.zeros(1,requires_grad=True)
        loss,kd,info=pair_losses(torch.tensor([.3]),r,r,torch.tensor([.5]))
        self.assertEqual(info['reliable_pairs'],0);(loss+kd).backward()
        self.assertEqual(float(r.grad),0.)

    def test_gate_closed_at_zero_residual(self):
        r=torch.zeros(2,requires_grad=True)
        _,kd,info=pair_losses(torch.tensor([.4,.6]),r,r,torch.tensor([.2,.8]))
        self.assertEqual(info['accepted_pairs'],0);self.assertEqual(float(kd),0.)

    def test_gate_rejects_correct_but_worse_teacher(self):
        r=torch.tensor([.01,-.01],requires_grad=True)
        _,kd,info=pair_losses(torch.tensor([.4,.6]),r,torch.zeros_like(r),torch.tensor([.2,.8]))
        self.assertEqual(info['accepted_pairs'],0);self.assertEqual(float(kd),0.)

    def test_gate_rejects_closer_but_still_wrong_teacher(self):
        r=torch.tensor([-.01,.01],requires_grad=True)
        _,_,info=pair_losses(torch.tensor([.6,.4]),r,torch.zeros_like(r),torch.tensor([.2,.8]))
        self.assertEqual(info['accepted_pairs'],0)

    def test_gate_accepts_correct_and_more_accurate_gap(self):
        r=torch.tensor([-.01,.01],requires_grad=True);rm=torch.zeros(2,requires_grad=True)
        _,kd,info=pair_losses(torch.tensor([.4,.6]),r,rm,torch.tensor([.2,.8]))
        self.assertEqual(info['accepted_pairs'],1);self.assertGreater(float(kd),0.)
        kd.backward();self.assertGreater(float(rm.grad.abs().sum()),0.);self.assertIsNone(r.grad)

    def test_teacher_anchor_and_labels_detached(self):
        base=torch.tensor([.4,.6],requires_grad=True);y=torch.tensor([.2,.8],requires_grad=True)
        r=torch.tensor([-.01,.01],requires_grad=True);rm=torch.zeros(2,requires_grad=True)
        _,kd,_=pair_losses(base,r,rm,y);kd.backward()
        self.assertIsNone(base.grad);self.assertIsNone(y.grad);self.assertIsNone(r.grad)

    def test_pair_permutation_and_global_offset_invariant(self):
        base=torch.tensor([.4,.6,.55]);y=torch.tensor([.2,.8,.5])
        r=torch.tensor([-.01,.01,0.],requires_grad=True);rm=torch.zeros(3,requires_grad=True);p=torch.tensor([2,0,1])
        a=pair_losses(base,r,rm,y);b=pair_losses(base[p],r[p],rm[p],y[p]);c=pair_losses(base+2,r,rm,y)
        for index in (0,1):
            self.assertAlmostEqual(float(a[index]),float(b[index]),places=7)
            self.assertAlmostEqual(float(a[index]),float(c[index]),places=7)
        self.assertEqual(a[2]['accepted_pairs'],b[2]['accepted_pairs'])

    def test_unique_unordered_pairs(self):
        i,j,gap=reliable_pairs(torch.tensor([.1,.3,.5,.7]))
        self.assertEqual(len(i),6);self.assertTrue(bool((i<j).all()))

    def test_calibration_zero_recovery_and_caps(self):
        for ratio in (.075,.1):
            self.assertEqual(calibrated_beta(1.,0.,ratio),0.)
            self.assertAlmostEqual(calibrated_beta(1.,1.,ratio),ratio)
            for g in (0.,1e-14,1e-8,1.,1e3):
                for k in (0.,1e-14,1e-8,1.,1e3):
                    b=calibrated_beta(g,k,ratio);self.assertLessEqual(b,10000.)
                    if g>1e-12:self.assertLessEqual(b*k/g,ratio+1e-12)

    def test_bad_norms_rejected(self):
        for value in (float('nan'),float('inf'),-1.):
            with self.assertRaises(FloatingPointError):calibrated_beta(value,1.,.1)

    def test_objective_accounting_and_auxiliary_ratio_bounds(self):
        for tag in RECIPES:
            b=self.branch(tag);obj=PairObjective(0,tag)
            with torch.no_grad():b.net[-1].weight.fill_(.02)
            loss,r=obj.losses(self.anchor,b,self.inputs,self.y,self.lengths,0)
            self.assertAlmostEqual(float(loss.detach()),r['gt']+r['weighted_rank']+r['weighted_kd'],places=6)
            self.assertLessEqual(r['rank_ratio'],RECIPES[tag]['rank']*.1*.2+1e-8)
            self.assertLessEqual(r['kd_ratio'],RECIPES[tag]['weight']*.075*.2+1e-8)
            self.assertLessEqual(r['rank_ratio']+r['kd_ratio'],.1375*.2+1e-8)

    def test_same_mask_and_rng_for_every_arm(self):
        state=(rng_state(),np.random.get_state());values=[]
        for tag in RECIPES:
            with preserved_rng(state):
                _,r=PairObjective(0,tag).losses(self.anchor,self.branch(tag),self.inputs,self.y,self.lengths,0)
                values.append((torch.get_rng_state().clone(),r['modality'],r['gt']))
        for value in values[1:]:self.assert_state_equal(values[0],value)

    def test_gt_optimizer_and_rng_exact_v13_parity(self):
        state=(rng_state(),np.random.get_state());values=[]
        for old in (True,False):
            with preserved_rng(state):
                b=ComplementBranch(V13_GT,4) if old else self.branch(GT)
                if old:b.fit_training_statistics(self.cache)
                obj=ComplementObjective(0,V13_GT) if old else PairObjective(0,GT)
                opt=torch.optim.AdamW(b.parameters(),lr=.0001,weight_decay=.01)
                for epoch in range(3):
                    for step in range(3):
                        opt.zero_grad();loss,_=obj.losses(self.anchor,b,self.inputs,self.y,self.lengths,epoch)
                        loss.backward();opt.step()
                values.append((copy.deepcopy(b.state_dict()),copy.deepcopy(opt.state_dict()),torch.get_rng_state().clone()))
        self.assert_state_equal(values[0],values[1])

    def test_anchor_and_statistics_frozen_all_arms(self):
        anchor=state_digest(self.anchor)
        for tag in RECIPES:
            b=self.branch(tag);before=state_digest(b);stats=statistics_digest(b)
            loss,_=PairObjective(0,tag).losses(self.anchor,b,self.inputs,self.y,self.lengths,0)
            loss.backward();torch.optim.AdamW(b.parameters(),lr=.0001).step()
            self.assertNotEqual(before,state_digest(b));self.assertEqual(stats,statistics_digest(b))
            assert_frozen(self.anchor);self.assertEqual(anchor,state_digest(self.anchor))

    def test_test_labels_do_not_change_predictions(self):
        b=self.branch(COMBO)
        with torch.no_grad():b.net[-1].weight.fill_(.02)
        a=evaluate(b,self.cache);other=evaluate(b,(self.cache[0],self.cache[1],self.cache[2].flip(0)))
        self.assertEqual(a['predictions']['prediction'],other['predictions']['prediction'])

    def test_all_arms_train_save_and_independent_reload(self):
        for tag in RECIPES:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);b=self.branch(tag);tensor_save(root/'rng.pt',(rng_state(),np.random.get_state()))
                source=dict(expected_srcc_x100=metrics(self.cache[0],self.cache[2])['coef'])
                result=train_branch(self.anchor,b,DataLoader(TinyData(),batch_size=4),self.cache,2,tag,0,True,
                    root,source,root/'rng.pt',dict(source_split='train',samples=12))
                saved=load(root/'model.pth');restored=self.branch(tag);restored.load_state_dict(saved['branch'])
                self.assertAlmostEqual(result['saved_model_srcc']*100,evaluate(restored,self.cache)['coef'])
                q=json.loads((root/'qmd_training.json').read_text());self.assertTrue(q['statistics_unchanged'])
                self.assertEqual(q['calibration']['steps'],6)

    def test_parser_rejects_changed_budget_or_unknown_seed(self):
        args=['--gpu','-1','--action','Ball','--seed','0','--run_tag',COMBO,'--teacher_model','m',
            '--teacher_rng','r','--teacher_record','c','--result_json','o','--save_final_model','model.pth']
        self.assertIsNone(parser_args(args).epochs)
        with self.assertRaises(SystemExit):parser_args(args+['--epochs','3'])
        with self.assertRaises(SystemExit):parser_args(args+['--seed','42'])
        self.assertEqual(metadata('Ball',1,COMBO,500)['gate']['inference'],
            'no labels or gate; always use learned bounded residual')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable locally')
    def test_cuda_pair_losses_and_detached_teacher(self):
        base=torch.tensor([.51,.49],device='cuda',requires_grad=True)
        label=torch.tensor([.2,.8],device='cuda',requires_grad=True)
        r=torch.tensor([-.015,.015],device='cuda',requires_grad=True)
        rm=torch.zeros(2,device='cuda',requires_grad=True)
        rank,kd,info=pair_losses(base,r,rm,label)
        self.assertEqual(info['accepted_pairs'],1)
        (rank+kd).backward()
        self.assertGreater(float(r.grad.abs().sum()),0.);self.assertGreater(float(rm.grad.abs().sum()),0.)
        self.assertIsNone(base.grad);self.assertIsNone(label.grad)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable locally')
    def test_private_initialization_preserves_cuda_rng(self):
        state=torch.cuda.get_rng_state_all()
        for tag in RECIPES:PairBranch(tag)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(state,torch.cuda.get_rng_state_all())))


if __name__=='__main__':unittest.main()
