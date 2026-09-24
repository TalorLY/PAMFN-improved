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
from tests.test_qmd_residual import TinyData
from tools.qmd_residual import freeze_anchor,preserved_rng,state_digest,assert_frozen,anchor_view
from tools.qmd_complement import ComplementBranch,ComplementObjective
from tools.qmd_modal import ModalBranch,ModalObjective,modal_view,make_cache
from tools.qmd_modal_config import RECIPES,FUSED,CONCAT,EXPERT,KD,V13_GT,parameter_count
from tools.train_qmd_modal import evaluate,train_branch,metrics,load,tensor_save,statistics_digest,parser_args
from models.score_qmd import rng_state


class TinyModality(nn.Module):
    def __init__(self,key,dim=2):
        super().__init__();self.key=key
        self.embedding=nn.Sequential(nn.Conv1d(4,dim,1),nn.BatchNorm1d(dim),nn.ReLU())
    def forward(self,inputs):
        x=self.embedding(inputs[self.key].transpose(1,2)).mean(2,keepdim=True)
        return x.mean(1),{'feats':[x,x,x,x]}


class TinyModalAnchor(nn.Module):
    def __init__(self,dim=2):
        super().__init__();self.model_r=TinyModality('V',dim);self.model_f=TinyModality('F',dim);self.model_a=TinyModality('A',dim)
        self.fc=nn.Sequential(nn.Conv1d(dim*3,1,1),nn.Sigmoid())
    def forward(self,inputs):
        _,v=self.model_r(inputs);_,a=self.model_a(inputs);_,f=self.model_f(inputs)
        z=torch.cat([v['feats'][-1],f['feats'][-1],a['feats'][-1]],1)
        z=z+torch.rand_like(z)*.005
        return self.fc(z).squeeze(2),{}


class ModalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        torch.manual_seed(27);np.random.seed(27);self.anchor=freeze_anchor(TinyModalAnchor())
        self.inputs={m:torch.rand(4,8,4) for m in ('V','F','A')}
        self.y=torch.tensor([.2,.45,.6,.8]);self.lengths={m:[8]*4 for m in self.inputs}
        self.cache=make_cache(self.anchor,DataLoader(TinyData(),batch_size=1),torch.device('cpu'),(rng_state(),np.random.get_state()))
    def branch(self,tag=EXPERT):
        b=ModalBranch(tag,6);b.fit_training_statistics(self.cache);return b
    def equal(self,a,b):
        if torch.is_tensor(a):self.assertTrue(torch.equal(a,b))
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for k in a:self.equal(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.equal(x,y)
        else:self.assertEqual(a,b)
    def test_private_initialization_and_parameter_counts(self):
        state=torch.get_rng_state().clone();digests={}
        for tag in RECIPES:
            b=ModalBranch(tag);digests[tag]=state_digest(b)
            self.assertEqual(sum(p.numel() for p in b.parameters()),parameter_count(b.config))
        self.equal(state,torch.get_rng_state());self.assertEqual(digests[FUSED],digests[CONCAT]);self.assertEqual(digests[EXPERT],digests[KD])
        self.assertEqual(parameter_count(ModalBranch(EXPERT).config),12384)
        self.assertEqual(parameter_count(ModalBranch(FUSED).config),12320)
    def test_joint_state_exact_old_mlp(self):self.equal(ModalBranch(FUSED,6).state_dict(),ComplementBranch(V13_GT,6).state_dict())
    def test_capture_preserves_anchor_output_rng_and_modal_order(self):
        state=(rng_state(),np.random.get_state())
        with preserved_rng(state):a,f=anchor_view(self.anchor,self.inputs);rng=torch.get_rng_state().clone()
        with preserved_rng(state):b,packed=modal_view(self.anchor,self.inputs);self.equal(rng,torch.get_rng_state())
        self.equal(a,b);self.equal(f,packed[:,:6])
        with torch.no_grad():
            expected=torch.cat([m(self.inputs)[1]['feats'][-1].squeeze(2) for m in (self.anchor.model_r,self.anchor.model_f,self.anchor.model_a)],1)
        self.equal(expected,packed[:,6:]);self.assertFalse(packed.requires_grad)
    def test_real_pamfn_forward_hook_contract_and_rng(self):
        from models.pamfn import PAMFN,BaseModel
        def pretrained(path):
            key='V' if '_rgb_' in path else ('F' if '_flow_' in path else 'A')
            return BaseModel(4,4,0.,key)
        with patch('models.pamfn._torch_load',side_effect=pretrained):
            anchor=freeze_anchor(PAMFN(model_dim=4,fc_drop=0.,fc_r=2,feat_drop=0.,K=2,
                ms_heads=1,cm_heads=1,ckpt_dir='unused',rgb_ckpt_name='r',flow_ckpt_name='f',audio_ckpt_name='a',dataset_name='Ball'))
        inputs={k:torch.rand(2,32,4) for k in ('V','F','A')};state=(rng_state(),np.random.get_state());before=state_digest(anchor)
        with preserved_rng(state):a,f=anchor_view(anchor,inputs);rng=torch.get_rng_state().clone()
        with preserved_rng(state):b,packed=modal_view(anchor,inputs);self.equal(rng,torch.get_rng_state())
        self.equal(a,b);self.equal(f,packed[:,:12]);self.assertEqual(packed.shape,(2,24))
        self.assertEqual(before,state_digest(anchor))
    def test_hooks_removed_after_failure(self):
        bad=copy.deepcopy(self.anchor);del bad.model_f
        with self.assertRaises(AttributeError):modal_view(bad,self.inputs)
        self.assertFalse(bad.fc._forward_pre_hooks);self.assertFalse(bad.model_r._forward_hooks)
    def test_hooks_removed_after_success(self):
        modal_view(self.anchor,self.inputs)
        for m in self.anchor.modules():self.assertFalse(m._forward_hooks);self.assertFalse(m._forward_pre_hooks)
    def test_modal_heads_only_read_own_channels(self):
        b=self.branch()
        with torch.no_grad():
            for head in b.net.values():head[-1].weight.fill_(.1)
        x=self.cache[1].clone();a=b.components(x);x[:,6:8]+=2.;changed=b.components(x)
        self.assertFalse(torch.equal(a[:,0],changed[:,0]));self.equal(a[:,1:],changed[:,1:])
        x=self.cache[1].clone();x[:,:6]+=2.;self.equal(a,b.components(x))
    def test_fused_ignores_modal_features(self):
        b=self.branch(FUSED)
        with torch.no_grad():b.net[-1].weight.fill_(.1)
        x=self.cache[1].clone();a=b(x);x[:,6:]+=2.;self.equal(a,b(x))
    def test_fixed_average_bounded_and_no_labels(self):
        b=self.branch()
        with torch.no_grad():
            for h in b.net.values():h[-1].weight.fill_(100.)
        c=b.components(self.cache[1]);self.equal(b(self.cache[1]),c.mean(1))
        self.assertLessEqual(float(c.detach().abs().max()),.03000001)
        a=evaluate(b,self.cache);other=evaluate(b,(self.cache[0],self.cache[1],self.cache[2].flip(0)))
        self.assertEqual(a['predictions']['prediction'],other['predictions']['prediction'])
    def test_joint_gt_replays_v13_optimizer_and_rng(self):
        state=(rng_state(),np.random.get_state());values=[]
        for old in (True,False):
            with preserved_rng(state):
                b=ComplementBranch(V13_GT,6) if old else self.branch(FUSED)
                if old:b.fit_training_statistics((self.cache[0],self.cache[1][:,:6],self.cache[2]))
                obj=ComplementObjective(0,V13_GT) if old else ModalObjective(0,FUSED)
                opt=torch.optim.AdamW(b.parameters(),lr=.0001,weight_decay=.01)
                for epoch in range(3):
                    for step in range(3):
                        opt.zero_grad();loss,_=obj.losses(self.anchor,b,self.inputs,self.y,self.lengths,epoch)
                        loss.backward();opt.step()
                values.append((copy.deepcopy(b.state_dict()),copy.deepcopy(opt.state_dict()),torch.get_rng_state().clone()))
        self.equal(values[0],values[1])
    def test_statistics_modal_match_and_immutable(self):
        stats=[]
        for tag in (CONCAT,EXPERT,KD):stats.append(statistics_digest(self.branch(tag)))
        self.assertEqual(len(set(stats)),1)
        with self.assertRaises(RuntimeError):self.branch().fit_training_statistics(self.cache)
    def test_zero_branch_preserves_bias_only_order(self):
        for tag in RECIPES:
            b=self.branch(tag);result=evaluate(b,self.cache)
            self.assertEqual(result['coef'],metrics(self.cache[0],self.cache[2])['coef'])
            self.assertEqual(result['correction_max_abs'],0.)
    def test_bad_shape_and_uncalibrated_rejected(self):
        with self.assertRaises(RuntimeError):ModalBranch(EXPERT,6)(self.cache[1])
        with self.assertRaises(ValueError):self.branch()(self.cache[1][:,:6])
        with self.assertRaises(ValueError):ModalBranch(EXPERT,7)
    def test_anchor_and_normalization_frozen_after_updates(self):
        before=state_digest(self.anchor)
        for tag in RECIPES:
            b=self.branch(tag);stats=statistics_digest(b);initial=state_digest(b)
            loss,info=ModalObjective(0,tag).losses(self.anchor,b,self.inputs,self.y,self.lengths,0)
            loss.backward();torch.optim.AdamW(b.parameters(),lr=.0001).step()
            self.assertNotEqual(initial,state_digest(b));self.assertEqual(stats,statistics_digest(b));assert_frozen(self.anchor)
            self.assertEqual(info['rank_ratio'],0.);self.assertEqual(info['weighted_rank'],0.)
        self.assertEqual(before,state_digest(self.anchor))
    def test_kd_activation_and_ratio(self):
        b=self.branch(KD);obj=ModalObjective(0,KD);opt=torch.optim.AdamW(b.parameters(),lr=.001)
        active=0
        for step in range(12):
            opt.zero_grad();loss,r=obj.losses(self.anchor,b,self.inputs,self.y,self.lengths,4)
            loss.backward();opt.step();active+=r['coefficient']*r['kd_norm']>0
            self.assertLessEqual(r['kd_ratio'],.03750001)
            self.assertAlmostEqual(float(loss),r['gt']+r['weighted_kd'],places=6)
        self.assertGreater(active,0);self.assertEqual(obj.calibration_summary()['rank_ratio'],0.)
    def test_mask_rng_same_all_arms(self):
        state=(rng_state(),np.random.get_state());values=[]
        for tag in RECIPES:
            with preserved_rng(state):
                _,r=ModalObjective(0,tag).losses(self.anchor,self.branch(tag),self.inputs,self.y,self.lengths,0)
                values.append((torch.get_rng_state().clone(),r['modality']))
        for v in values[1:]:self.equal(values[0],v)
    def test_all_arms_train_reload_and_final_predictions(self):
        for tag in RECIPES:
            with tempfile.TemporaryDirectory() as temp:
                out=Path(temp);b=self.branch(tag);tensor_save(out/'rng.pt',(rng_state(),np.random.get_state()))
                source=dict(expected_srcc_x100=metrics(self.cache[0],self.cache[2])['coef'])
                result=train_branch(self.anchor,b,DataLoader(TinyData(),batch_size=4),self.cache,2,tag,0,True,out,source,out/'rng.pt',{})
                restored=self.branch(tag);restored.load_state_dict(load(out/'model.pth')['branch'])
                self.assertAlmostEqual(result['saved_model_srcc']*100,evaluate(restored,self.cache)['coef'])
                last=json.loads((out/'last_epoch_result.json').read_text())
                self.assertEqual(last['predictions'],evaluate(b,self.cache)['predictions'])
                self.assertEqual(set(last['expert_predictions']),{'V','F','A'} if tag in (EXPERT,KD) else set())
    def test_formal_epochs_seed_locked(self):
        args=['--gpu','-1','--action','Ball','--seed','0','--run_tag',EXPERT,'--teacher_model','m',
            '--teacher_rng','r','--teacher_record','c','--result_json','o','--save_final_model','model.pth']
        self.assertIsNone(parser_args(args).epochs)
        with self.assertRaises(SystemExit):parser_args(args+['--epochs','3'])
        with self.assertRaises(SystemExit):parser_args(args+['--seed','42'])
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable locally')
    def test_cuda_capture_update_and_private_rng(self):
        state=torch.cuda.get_rng_state_all();b=ModalBranch(KD,6)
        self.assertTrue(all(torch.equal(a,c) for a,c in zip(state,torch.cuda.get_rng_state_all())))
        anchor=freeze_anchor(copy.deepcopy(self.anchor)).cuda();cache=tuple(x.cuda() for x in self.cache)
        b.cuda().fit_training_statistics(cache)
        loss,_=ModalObjective(0,KD).losses(anchor,b,{k:v.cuda() for k,v in self.inputs.items()},self.y.cuda(),self.lengths,0)
        loss.backward();assert_frozen(anchor)


if __name__=='__main__':unittest.main()
