import copy
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader
from tests.test_qmd_modal import TinyModalAnchor,TinyData
from tools.qmd_modal import ModalBranch,make_cache
from tools.qmd_residual import freeze_anchor,state_digest,preserved_rng
from tools.qmd_supp import SupplementBranch,e_score,objective
from tools.qmd_supp_config import RECIPES,FIXED,SCALED,SHRINK,E_TAG,parameter_count,metadata
from tools.train_qmd_supp import train_branch,evaluate,statistics_digest,metrics,load,parser_args
from models.score_qmd import rng_state


class SupplementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        torch.manual_seed(32);np.random.seed(32);self.anchor=freeze_anchor(TinyModalAnchor())
        self.loader=DataLoader(TinyData(),batch_size=1)
        self.replay=(rng_state(),np.random.get_state())
        self.cache=make_cache(self.anchor,self.loader,torch.device('cpu'),self.replay)
        self.e=ModalBranch(E_TAG,6);self.e.fit_training_statistics(self.cache)
        with torch.no_grad():self.e.net[-1].weight.fill_(.2)
        freeze_anchor(self.e)
        self.es=e_score(self.e,self.cache[0],self.cache[1])
    def branch(self,tag=SHRINK):
        b=SupplementBranch(tag,6);b.fit_training_statistics(self.es,self.cache[1]);return b
    def test_private_init_and_parameter_count(self):
        before=torch.get_rng_state().clone();heads=[]
        for tag in RECIPES:
            b=SupplementBranch(tag);heads.append(state_digest(b.heads))
            self.assertEqual(sum(p.numel() for p in b.parameters()),parameter_count(tag))
        self.assertEqual(len(set(heads)),1);self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(parameter_count(SHRINK),12385)
    def test_zero_initial_exact_e(self):
        for tag in RECIPES:self.assertEqual(evaluate(self.e,self.branch(tag),self.cache)['predictions'],metrics(self.es,self.cache[2])['predictions'])
    def test_scale_calibration_and_cap(self):
        packed=torch.randn(10,12)
        for spread in (.00001,.1):
            scores=torch.linspace(-spread,spread,10,dtype=torch.double)
            b=SupplementBranch(SCALED,6);stats=b.fit_training_statistics(scores,packed)
            self.assertAlmostEqual(stats['bound'],min(.003,.1*float(scores.std(unbiased=False))),places=9)
        b=self.branch(FIXED);self.assertAlmostEqual(float(b.bound),.003,places=9)
    def test_flat_e_scale_zero_is_legitimate(self):
        b=SupplementBranch(SCALED,6);b.fit_training_statistics(torch.ones(10),torch.randn(10,12))
        with torch.no_grad():
            for h in b.heads:h[-1].weight.fill_(10.)
        self.assertTrue(torch.equal(b(torch.randn(4,12)),torch.zeros(4)))
    def test_statistics_immutable_and_no_test_refit(self):
        b=self.branch();digest=statistics_digest(b)
        evaluate(self.e,b,(self.cache[0],self.cache[1]*5,self.cache[2]+.1))
        self.assertEqual(digest,statistics_digest(b))
        with self.assertRaises(RuntimeError):b.fit_training_statistics(self.es,self.cache[1])
    def test_nonfinite_statistics_rejected(self):
        b=SupplementBranch(SCALED,6)
        with self.assertRaises(ValueError):b.fit_training_statistics(torch.full_like(self.es,float('nan')),self.cache[1])
    def test_global_gate_zero_exact_recovery_and_reopen_gradient(self):
        b=self.branch()
        with torch.no_grad():
            for h in b.heads:h[-1].weight.fill_(.3)
            b.gate.zero_()
        self.assertEqual(evaluate(self.e,b,self.cache)['predictions'],metrics(self.es,self.cache[2])['predictions'])
        loss=-b(self.cache[1]).square().sum()+b(self.cache[1]).sum()
        loss.backward();self.assertIsNotNone(b.gate.grad)
        self.assertNotEqual(float(b.gate.grad),0.)
    def test_projection_and_regularizer_can_close_gate(self):
        b=self.branch();self.assertGreater(float(b.bound),0)
        opt=torch.optim.SGD([b.gate],lr=1./float(b.bound.square()))
        opt.zero_grad();b.penalty().backward();opt.step();b.project_gate()
        self.assertEqual(float(b.gate),0.)
        with torch.no_grad():b.gate.fill_(2.)
        b.project_gate();self.assertEqual(float(b.gate),1.)
    def test_per_modality_isolation(self):
        b=self.branch();p=self.cache[1].clone()
        with torch.no_grad():
            for h in b.heads:h[-1].weight.fill_(.3)
        old=b.components(p);p[:,6:8]+=2;new=b.components(p)
        self.assertTrue(torch.equal(old[:,1:],new[:,1:]));self.assertFalse(torch.equal(old[:,0],new[:,0]))
    def test_bounds_and_fixed_mean(self):
        b=self.branch()
        with torch.no_grad():
            b.gate.fill_(.37)
            for h in b.heads:h[-1].weight.fill_(100.)
        x=self.cache[1]*100
        self.assertLessEqual(float(b(x).abs().max()),float(b.bound)*.370001)
        self.assertTrue(torch.equal(b.components(x).mean(1),b(x)))
    def test_objective_freezes_e_ordered_no_kd(self):
        import pipeline
        b=self.branch();before=(state_digest(self.anchor),state_digest(self.e));old=state_digest(b)
        inputs,y=pipeline._prepare_batch(next(iter(self.loader)),torch.device('cpu'))
        # Use a nontrivial batch for centered supervision.
        inputs={k:v.repeat(4,1,1)+torch.randn_like(v.repeat(4,1,1))*.1 for k,v in inputs.items()};y=torch.tensor([.1,.3,.6,.8])
        opt=b.optimizer();opt.zero_grad();loss,info=objective(self.anchor,self.e,b,inputs,y);loss.backward();opt.step();b.project_gate()
        self.assertEqual(before,(state_digest(self.anchor),state_digest(self.e)));self.assertNotEqual(old,state_digest(b))
        self.assertTrue(all(p.grad is None for m in (self.anchor,self.e) for p in m.parameters()))
        self.assertAlmostEqual(float(loss),info['centered_mse']+info['mean_penalty']+info['shrink_penalty'])
        self.assertNotIn('kd',info)
    def test_train_reload_three_recipes(self):
        for tag in RECIPES:
            with tempfile.TemporaryDirectory() as temp:
                out=Path(temp);rng=out/'rng.pt';torch.save(self.replay,rng)
                b=self.branch(tag);source=dict(ordered={},e={})
                stats=dict(source_split='train',samples=len(self.es))
                result=train_branch(self.anchor,self.e,b,DataLoader(TinyData(),batch_size=4),self.cache,2,tag,0,True,out,source,rng,stats)
                saved=load(out/'model.pth');restored=SupplementBranch(tag,6);restored.load_state_dict(saved['branch'])
                self.assertAlmostEqual(evaluate(self.e,restored,self.cache)['coef'],result['saved_model_srcc']*100)
                self.assertEqual(statistics_digest(restored),saved['statistics_sha256'])
    def test_metadata_has_no_kd_or_router(self):
        for tag in RECIPES:
            m=metadata('Ball',0,tag,500)
            self.assertEqual(m['kd_weight'],0);self.assertFalse(m['masked_forward']);self.assertFalse(m['sample_reliability_router']);self.assertFalse(m['epoch_zero_eligible'])
    def test_parser_rejects_formal_epoch_override(self):
        argv=['--gpu','-1','--action','Ball','--seed','0','--run_tag',SHRINK,'--teacher_model','m','--teacher_rng','r',
              '--teacher_record','t','--e_model','e','--e_record','er','--result_json','j','--save_final_model','model.pth']
        self.assertFalse(parser_args(argv).smoke)
        with self.assertRaises(SystemExit):parser_args(argv+['--epochs','3'])
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA only')
    def test_cuda_zero_parity(self):
        e=copy.deepcopy(self.e).cuda();cache=tuple(v.cuda() for v in self.cache);b=self.branch().cuda()
        self.assertEqual(evaluate(e,b,cache)['predictions'],metrics(e_score(e,cache[0],cache[1]),cache[2])['predictions'])


if __name__=='__main__':unittest.main()
