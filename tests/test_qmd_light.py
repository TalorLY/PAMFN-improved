import unittest
from unittest.mock import patch

import torch

from main import parser_args, qmd_config
from models.score_qmd import ScoreQMD, reliable_score_kd, rng_state, score_kd
from tests.test_score_qmd import TinyModel
from tools.verify_sr_qmd_runtime import frozen_module


class LightQMDTest(unittest.TestCase):
    def test_soft_floor_preserves_rejected_gradient_and_detaches_teacher(self):
        student = torch.tensor([.6,.2], requires_grad=True)
        teacher = torch.tensor([.4,.5], requires_grad=True)
        target = torch.tensor([.3,.2], requires_grad=True)
        loss, info = reliable_score_kd(student,teacher,target,.5)
        loss.backward()
        self.assertTrue(torch.allclose(student.grad,torch.tensor([.1,-.075])))
        self.assertEqual(info['accepted_samples'],1)
        self.assertEqual(info['mean_gate_weight'],.75)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(target.grad)

    def test_floor_one_is_exact_score_kd(self):
        for seed in (0,1,42):
            torch.manual_seed(seed)
            student=torch.rand(31,requires_grad=True)
            teacher=torch.rand(31,requires_grad=True)
            expected=score_kd(student,teacher)
            actual,_=reliable_score_kd(student,teacher,torch.rand(31),1.)
            self.assertTrue(torch.equal(expected,actual))
            self.assertTrue(torch.equal(torch.autograd.grad(expected,student)[0],
                                        torch.autograd.grad(actual,student)[0]))

    def test_invalid_floor_and_wrong_mode(self):
        for floor in (-.1,1.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                ScoreQMD('reliable_score_qmd',0,100,score_gate_floor=floor)
        with self.assertRaises(ValueError):
            ScoreQMD('score_qmd',0,100,score_gate_floor=.5)

    def test_cli_records_floor_only_for_reliable_mode(self):
        args=['main.py','--gpu','-1','--feats','2','--action','Ball','--multi_modality',
              '--qmd_mode','reliable_score_qmd','--qmd_score_gate_floor','.5']
        with patch('sys.argv',args):
            config=qmd_config(parser_args())
        self.assertEqual(config['score_gate_floor'],.5)
        self.assertNotIn('rank_weight',config)

    def test_hard_gate_and_diagnostics_match_frozen_v2_training(self):
        frozen=frozen_module('db579d5cdc977c2193a3838628ffc10454b79543',
                             'models/score_qmd.py','frozen_v2')
        torch.set_num_threads(1)
        def run(cls,mode,**extra):
            torch.manual_seed(8)
            model=TinyModel().train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            qmd=cls(mode,0,1,warmup=0,rampup=0,**extra)
            for _ in range(4):
                inputs={m:torch.randn(4,12,4) for m in ('V','F','A')}
                target=torch.rand(4)
                before=rng_state()
                full,_=model(inputs)
                main=model.call_loss(full,target)
                aux,_=qmd.auxiliary(model,inputs,{m:[12]*4 for m in inputs},target,full,main,before,0)
                optimizer.zero_grad()
                (main+aux).backward()
                optimizer.step()
            return model.state_dict(),torch.get_rng_state(),qmd
        for mode in ('score_qmd','reliable_score_qmd'):
            expected=run(frozen.ScoreQMD,mode)
            actual=run(ScoreQMD,mode)
            self.assertEqual(expected[2].calibration,actual[2].calibration)
            self.assertTrue(torch.equal(expected[1],actual[1]))
            for key in expected[0]:
                self.assertTrue(torch.equal(expected[0][key],actual[0][key]),key)
            rows=actual[2].summary()['view_diagnostics']['rows']
            self.assertEqual(sum(r['samples'] for r in rows),16)
            self.assertEqual({r['modality'] for r in rows},{'V','F','A'})

    def test_lower_weight_scales_both_terms_without_recalibrating_beta(self):
        torch.set_num_threads(1)
        values=[]
        for weight in (1.,.1,.25):
            torch.manual_seed(9)
            model=TinyModel().train()
            inputs={m:torch.randn(4,8,4) for m in ('V','F','A')}
            target=torch.rand(4)
            qmd=ScoreQMD('score_qmd',0,1,weight=weight,warmup=0,rampup=0)
            before=rng_state()
            full,_=model(inputs)
            aux,d=qmd.auxiliary(model,inputs,{m:[8]*4 for m in inputs},target,full,
                                model.call_loss(full,target),before,0)
            values.append((float(aux.detach()),qmd.beta,d))
        for value,weight in zip(values[1:],(.1,.25)):
            self.assertAlmostEqual(value[0],values[0][0]*weight,places=6)
            self.assertEqual(value[1],values[0][1])
            self.assertEqual(value[2]['mask_gt'],values[0][2]['mask_gt'])


if __name__=='__main__':
    unittest.main()
