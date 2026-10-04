"""Actual pinned MCMoE CPU forward/backward and RNG compatibility tests."""
import copy
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from adapter import (TransferModel,TransferObjective,build_native,seed_all,freeze,frozen_view,
                     capture_rng,preserved_rng,state_digest,historical_inference,assert_frozen)

class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def inputs(self,n=2):
        gen=torch.Generator().manual_seed(87)
        return [torch.randn(n,10,d,generator=gen) for d in (768,768,1024)]
    def test_initialization_and_native_predictions(self):
        seed_all(8);base=build_native(True);state=torch.get_rng_state().clone()
        seed_all(8);wrapped=TransferModel(True,[.25,.5,.75],8,True)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(state_digest(base),state_digest(wrapped.backbone))
        base.eval();wrapped.eval();xs=self.inputs()
        with torch.no_grad():self.assertTrue(torch.equal(base(*xs,[1,1,1])['output'],wrapped(*xs,[1,1,1])['output']))
    def test_frozen_hook_exact_and_order(self):
        seed_all(3);model=TransferModel(False,cpu=True);model.eval();xs=self.inputs()
        with torch.no_grad():expected=model(*xs,[1,1,1])['output']
        freeze(model)
        before=state_digest(model);rng=torch.get_rng_state().clone()
        pred,joint,modal=frozen_view(model,*xs)
        self.assertTrue(torch.equal(expected,pred));self.assertEqual(before,state_digest(model))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(joint.shape,(2,256))
        self.assertEqual(modal.shape,(2,768))
        with historical_inference(model):
            expected_modal=[]
            for key,x in zip(('v','f','a'),(xs[0],xs[2],xs[1])):
                seq=model.backbone.in_proj[key](x.transpose(1,2)).transpose(1,2)
                expected_modal.append(model.backbone.transformer.encoder(seq).mean(1))
        self.assertTrue(torch.equal(torch.cat(expected_modal,1),modal))
        self.assertEqual(len(model.backbone.regressor._forward_pre_hooks),0)
        self.assertEqual(len(model.backbone.transformer.encoder._forward_hooks),0)
        assert_frozen(model)
    def test_trained_batch1_inference_metadata_exact_and_restored(self):
        # Regresses the real failure: freezing native trainable parameters
        # changed batch-1 predictions by 1 ULP even without any forward hook.
        from torch.utils.data import DataLoader
        from integration import Synthetic
        from train import train_epoch
        from worker import native_evaluate,make_cache,metrics,parity
        seed_all(0);model=TransferModel(False,cpu=True)
        from models.loss import LossFun
        loss_fun=LossFun(1.,1.)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
        train=DataLoader(Synthetic(),batch_size=4,shuffle=True)
        test=DataLoader(Synthetic(),batch_size=1)
        for epoch in range(2):
            train_epoch(epoch,model,loss_fun,train,optimizer,None,torch.device('cpu'),SimpleNamespace(score_range=130.))
        expected=native_evaluate(model,test,torch.device('cpu'))
        before=state_digest(model)
        freeze(model);freeze(model)
        with historical_inference(model):
            self.assertEqual(expected['predictions'],native_evaluate(model,test,torch.device('cpu'))['predictions'])
        cache=make_cache(model,test,torch.device('cpu'))
        parity(metrics(cache[0].tolist(),cache[3].tolist()),expected)
        self.assertTrue(all(not tensor.requires_grad for tensor in cache))
        assert_frozen(model)
        self.assertEqual(before,state_digest(model))
        with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
            with historical_inference(model):
                self.assertFalse(torch.is_grad_enabled())
                self.assertTrue(any(p.requires_grad for p in model.parameters()))
                raise RuntimeError('synthetic failure')
        assert_frozen(model)
    def test_zero_weight_native_optimizer_parity(self):
        seed_all(13);model=TransferModel(True,[.25,.5,.75],13,True)
        native=copy.deepcopy(model.backbone)
        from models.loss import LossFun
        original=LossFun(1.,1.);objective=TransferObjective(model,weight=0.)
        a=torch.optim.Adam(native.parameters(),lr=1e-4,weight_decay=1e-4)
        b=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
        xs=self.inputs();y=torch.tensor([.3,.7]);options=SimpleNamespace(score_range=130.)
        native.train();model.train();replay=capture_rng()
        for m,opt,obj in ((native,a,original),(model,b,objective)):
            with preserved_rng(replay):
                out=m(*xs,[1,1,1]);loss=obj(out['output'],y,out['embed'],out['embed2'],out['recon_loss'],options)
                opt.zero_grad();loss.backward();opt.step()
        self.assertEqual(state_digest(native),state_digest(model.backbone))
        self.assertTrue(all(p.grad is None for p in model.quality.parameters()))
    def test_ordered_gradient_reaches_native_backbone(self):
        seed_all(21);model=TransferModel(True,[.25,.5,.75],21,True);model.train()
        out=model(*self.inputs(),[1,1,1]);model.quality.loss(out['embed'].mean(1),torch.tensor([.2,.8])).backward()
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in model.backbone.parameters()))
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in model.quality.parameters()))
    def test_frozen_view_rejects_training(self):
        model=TransferModel(False,cpu=True)
        with self.assertRaises(RuntimeError):frozen_view(model,*self.inputs())

if __name__=='__main__':unittest.main()
