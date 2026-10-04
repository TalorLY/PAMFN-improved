import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader
from tests.test_qmd_modal import TinyModalAnchor, TinyData
import contract as c
import run as runner
import worker as entry
from tools import train_qmd_supp as old_s
from tools import train_qmd_modal as old_e
from tools.qmd_modal import ModalBranch, make_cache
from tools.qmd_supp import SupplementBranch, e_score
from tools.qmd_supp_config import SCALED, E_TAG
from tools.qmd_residual import freeze_anchor, state_digest, preserved_rng
from model import ControlledSupplement, ZeroE, predict_views, diagnose, control_context, raw_supervision, rgb_view, RGBBranch
from models.score_qmd import rng_state


class EvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17); np.random.seed(17)
        self.anchor=freeze_anchor(TinyModalAnchor())
        self.loader=DataLoader(TinyData(),batch_size=4)
        self.replay=(rng_state(),np.random.get_state())
        self.cache=make_cache(self.anchor,self.loader,torch.device('cpu'),self.replay)
        self.e=ModalBranch(E_TAG,6); self.e.fit_training_statistics(self.cache); freeze_anchor(self.e)

    def make(self,control=None):
        b=ControlledSupplement(control,6) if control else SupplementBranch(SCALED,6)
        b.fit_training_statistics(e_score(self.e,self.cache[0],self.cache[1]),self.cache[1])
        return b

    def test_contract_keeps_all_actions_seeds(self):
        self.assertEqual(len(c.cells()),126); self.assertEqual(len(c.cells(c.SMOKE_METHODS,True)),18)
        self.assertEqual(len(c.cells(('replay',))),18)
        self.assertEqual(set(s for a,s,m in c.cells()),{0,1,2})

    def test_controls_equal_capacity_initial_weights_rng(self):
        state=torch.get_rng_state().clone()
        branches=[SupplementBranch(SCALED,6),ControlledSupplement('fused_s',6),ControlledSupplement('unbounded_s',6)]
        self.assertEqual(len(set(state_digest(b.heads) for b in branches)),1)
        self.assertEqual(len(set(sum(p.numel() for p in b.parameters()) for b in branches)),1)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_control_context_is_scoped_and_restored_on_failure(self):
        from tools import qmd_modal
        view,loss=qmd_modal.modal_view,qmd_modal.centered_supervision
        with self.assertRaises(RuntimeError):
            with control_context('raw_e'):
                self.assertIs(qmd_modal.centered_supervision,raw_supervision)
                self.assertIs(qmd_modal.modal_view,view)
                raise RuntimeError('synthetic interruption')
        self.assertIs(qmd_modal.centered_supervision,loss)
        with control_context('rgb_e'):self.assertIs(qmd_modal.modal_view,rgb_view)
        self.assertIs(qmd_modal.modal_view,view)
        with control_context('repeat_e'):
            self.assertIs(qmd_modal.centered_supervision,loss)
            self.assertIs(qmd_modal.modal_view,view)

    def test_raw_loss_keeps_mean_penalty_and_rgb_head_capacity(self):
        y=torch.tensor([.1,.4,.7]);delta=torch.tensor([.2,.3,.4],requires_grad=True)
        loss,mse,mean=raw_supervision(y+delta,y,delta)
        self.assertTrue(torch.allclose(loss,delta.square().mean()+.1*delta.mean().square()))
        loss.backward();self.assertIsNotNone(delta.grad)
        b=RGBBranch();self.assertEqual(sum(p.numel() for p in b.parameters()),4128)

    def test_v29_active_attempts_and_assets_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            c.save(root/'completion.json',dict(experiments_complete=True,formal_cells=72))
            c.save(root/'contract.json',dict(code_commit=c.ORIGIN,seeds=list(c.SEEDS),source={'dummy':True}))
            for a in c.ACTIONS:
                for s in c.SEEDS:
                    for m in ('base','ordered',E_TAG,SCALED):
                        cell=root/'screen/july28'/a/('seed%d'%s)/m;out=cell/'attempts/0001'
                        c.save(cell/'active.json',dict(attempt='attempts/0001'))
                        for n in ('test_result.json','model.pth','model.pth.eval_rng.pt'):c.save(out/n,dict(coef=80))
                        c.save(out/'complete.json',dict(metrics=dict(action=a,seed=s,method=m,epochs=c.BUDGET[a],srcc_x100=80),
                            files={'test_result.json':c.sha(out/'test_result.json')},checkpoint_sha256=c.sha(out/'model.pth'),
                            eval_rng_sha256=c.sha(out/'model.pth.eval_rng.pt')))
            old,refs=c.build_references(root)
            self.assertEqual(len(refs),18);self.assertEqual(old,{'dummy':True})
            c.save(Path(refs[0]['base']['model']),dict(changed=True))
            with self.assertRaises(ValueError):c.build_references(root)

    def test_new_head_backup_hashes_verified_and_old_assets_not_copied(self):
        import tarfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'result';cell=('Ball',0,'base_e')
            for phase in ('smoke','formal'):
                directory=c.cell_dir(root,phase,cell);out=directory/'attempts/001'
                c.save(directory/'active.json',dict(directory=str(out)))
                c.save(out/'model.pth',dict(synthetic=True));c.save(out/'complete.json',dict(files={},checkpoint_sha256=c.sha(out/'model.pth')))
            for n in ('manifest.json','completion.json','summary.json'):c.save(root/n,dict(origin='/read-only-v29'))
            with patch.object(c,'cells',return_value=[cell]):runner.backup(root)
            receipt=c.read(root/'archive.json');self.assertTrue(receipt['assets_verified'])
            with tarfile.open(receipt['assets_path']) as tar:
                names=tar.getnames()
                self.assertEqual(sum(n.endswith('model.pth') for n in names),2)
                self.assertIn('asset_manifest.json',names)
                self.assertFalse(any('read-only-v29' in n for n in names))

    def test_controls_zero_exact_and_statistics_train_only(self):
        for control in ('fused_s','unbounded_s'):
            b=self.make(control)
            self.assertTrue(torch.equal(b(self.cache[1]),torch.zeros_like(self.cache[0])))
            with self.assertRaises(RuntimeError): b.fit_training_statistics(self.cache[0],self.cache[1])

    def test_fused_control_ignores_pre_fusion_features(self):
        b=self.make('fused_s')
        with torch.no_grad():
            for h in b.heads: h[-1].weight.fill_(.1)
        changed=self.cache[1].clone(); changed[:,6:]+=10
        self.assertTrue(torch.equal(b(changed),b(self.cache[1])))
        self.assertTrue(torch.allclose(b.feature_mean,self.cache[1][:,:6].double().mean(0).float()))

    def test_unbounded_can_exceed_bound(self):
        b=self.make('unbounded_s')
        with torch.no_grad():
            for h in b.heads:
                h[0].weight.zero_(); h[0].bias.fill_(1.); h[-1].weight.fill_(100.)
        self.assertGreater(float(b(self.cache[1]).detach().abs().max()),float(b.bound))

    def test_one_forward_preserves_exact_v17_arithmetic(self):
        import pipeline
        b=self.make(); inputs,_=pipeline._prepare_batch(next(iter(self.loader)),torch.device('cpu'))
        calls=[]; hook=self.anchor.register_forward_hook(lambda *x:calls.append(1))
        try:
            with preserved_rng(self.replay): result=predict_views(self.anchor,self.e,b,inputs)
        finally: hook.remove()
        self.assertEqual(len(calls),1)
        with preserved_rng(self.replay):
            from tools.qmd_modal import modal_view
            base,packed=modal_view(self.anchor,inputs)
            expected=e_score(self.e,base,packed)+b(packed).double()
        self.assertTrue(torch.equal(result['v17'],expected))

    def test_diagnostics_zero_and_denominator(self):
        b=self.make()
        with torch.no_grad():
            for h in b.heads: h[-1].weight.fill_(.2)
        before=state_digest(b); output=diagnose(self.e,b,self.cache)
        self.assertEqual(before,state_digest(b)); self.assertEqual(output['denominator'],3)
        frozen=e_score(self.e,self.cache[0],self.cache[1])
        self.assertEqual(output['variants']['zero']['predictions'],entry.metrics(frozen,self.cache[2])['predictions'])
        raw=b.components(self.cache[1]).detach();raw[:,0]=0
        expected=entry.metrics(frozen+raw.mean(1).double(),self.cache[2])
        self.assertEqual(expected['predictions'],output['variants']['without_V']['predictions'])

    def test_zero_e_has_no_parameters(self):
        e=freeze_anchor(ZeroE()); self.assertEqual(list(e.parameters()),[])
        self.assertTrue(torch.equal(e_score(e,self.cache[0],self.cache[1]),self.cache[0].double()))

    def test_original_s_function_is_used(self):
        self.assertIs(entry.original_s.train_branch,old_s.train_branch)
        self.assertIs(entry.original_e.train_branch,old_e.train_branch)

    def test_control_online_two_epoch_train_reload(self):
        for control in ('fused_s','unbounded_s',None):
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp); replay=out/'rng.pt'; torch.save(self.replay,replay)
                b=self.make(control); before=state_digest(self.anchor)
                old_s.train_branch(self.anchor,self.e,b,self.loader,self.cache,2,SCALED,0,True,out,{},replay,{})
                state=entry.load(out/'model.pth'); b.load_state_dict(state['branch'])
                self.assertEqual(state_digest(self.anchor),before)
                self.assertTrue(np.isfinite(old_s.evaluate(self.e,b,self.cache)['coef']))

    def test_immutable_and_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.json'; c.immutable(p,{'a':1}); c.immutable(p,{'a':1})
            with self.assertRaises(ValueError): c.immutable(p,{'a':2})
            with self.assertRaises(ValueError): c.child(tmp,'../other.json')

    def test_incomplete_methods_not_averaged_and_fisher(self):
        rows=[dict(method='m',action=a,seed=s,srcc_x100=80+s) for a in c.ACTIONS for s in c.SEEDS]
        self.assertEqual(c.aggregate(rows[:-1]),[])
        result=c.aggregate(rows)[0];self.assertAlmostEqual(result['six_macro'],81.)
        self.assertAlmostEqual(result['sample_sd_seed_means'],1.)
        self.assertAlmostEqual(result['RG_fisher'],81.); self.assertEqual(len(result['per_seed']),3)
        self.assertEqual(c.aggregate(rows+[rows[0]]),[])

    def test_resume_checks_files_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'cell'; out=directory/'attempts/001'
            c.save(directory/'active.json',dict(directory=str(out)))
            c.save(out/'test_result.json',dict(coef=80))
            c.save(out/'model.pth',dict(dummy=True))
            c.save(out/'complete.json',dict(files={'test_result.json':c.sha(out/'test_result.json')},checkpoint_sha256=c.sha(out/'model.pth')))
            self.assertTrue(runner.valid_done(directory))
            c.save(out/'model.pth',dict(dummy=False))
            with self.assertRaises(ValueError): runner.valid_done(directory)

    def test_package_excludes_weights_and_includes_failures(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'result'; c.save(root/'state.json',{'failed':True}); c.save(root/'model.pth',{})
            c.save(root/'rng.pt',{}); result=runner.package(root,True)
            with zipfile.ZipFile(result) as z:
                self.assertIn('state.json',z.namelist())
                self.assertNotIn('model.pth',z.namelist());self.assertNotIn('rng.pt',z.namelist())
                self.assertIn('evidence_code/worker.py',z.namelist())

    def test_record_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); c.save(root/'test_result.json',{'coef':1})
            c.save(root/'complete.json',dict(files={'test_result.json':c.sha(root/'test_result.json')},metrics={}))
            c.verify_record(root); c.save(root/'test_result.json',{'coef':2})
            with self.assertRaises(ValueError): c.verify_record(root)

    def test_exact_replay_rejects_same_score_changed_prediction(self):
        with self.assertRaises(ValueError): c.exact_predictions({'coef':80,'predictions':[1,2]}, {'coef':80,'predictions':[2,1]},'same score')

    def test_pid_reuse_and_zombie_not_alive(self):
        rec=dict(pid=1,host='h',start_ticks='10',boot='b',state='S')
        with patch.object(runner,'identity',return_value=dict(rec,start_ticks='11')): self.assertFalse(runner.alive(rec))
        with patch.object(runner,'identity',return_value=dict(rec,state='Z')): self.assertFalse(runner.alive(rec))
        with patch.object(runner,'identity',return_value=rec): self.assertTrue(runner.alive(rec))

    def test_failed_worker_restarts_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cell=('Ball',0,'repeat_e')
            class Failed:
                pid=99999999
                def wait(self): return 1
            with patch.object(runner.subprocess,'Popen',return_value=Failed()),patch.object(runner,'check_space'),patch.object(runner,'identity',return_value=None):
                with self.assertRaises(RuntimeError): runner.run_cell(root,'smoke',cell,'0')
                first=runner.active_dir(c.cell_dir(root,'smoke',cell)); c.save(first/'marker.json',dict(keep=True))
                with self.assertRaises(RuntimeError): runner.run_cell(root,'smoke',cell,'0')
                second=runner.active_dir(c.cell_dir(root,'smoke',cell))
                self.assertNotEqual(first,second); self.assertTrue((first/'marker.json').is_file())
                self.assertFalse((second/'complete.json').exists())

    def test_output_guard_precedes_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'not_results'
            with patch('sys.argv',['run','--root',str(target)]):
                with self.assertRaises(ValueError): runner.main()
            self.assertFalse(target.exists())

    def test_report_has_per_seed_and_paired_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); refs=[]
            for a in c.ACTIONS:
                for s in c.SEEDS:
                    refs.append(dict(action=a,seed=s,**{m:dict(test=dict(coef=80+s+i*.1)) for i,m in enumerate(('base','ordered','e','s'))}))
                    out=c.cell_dir(root,'formal',(a,s,'base_e'))/'attempts/001'
                    c.save(out.parents[1]/'active.json',dict(directory=str(out)))
                    c.save(out/'complete.json',dict(srcc_x100=80.2+s,files={}))
            c.save(root/'manifest.json',dict(references=refs,source=dict(remaining_evidence=['transfer'])))
            runner.report(root); data=c.read(root/'summary.json')
            self.assertEqual(len(data['summary']),5); self.assertEqual(len(data['rows']),90)
            self.assertTrue(data['paired_comparisons']);self.assertIn('## Seed 2',(root/'tables.md').read_text())


if __name__=='__main__': unittest.main()
