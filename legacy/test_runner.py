"""Resume/queue/selection safety, without server or GPU mutations."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import spec
import run

class RunnerTests(unittest.TestCase):
    def source_fixture(self,root):
        """Tiny weight bytes test provenance only; never used as real torch models."""
        policy=dict(dataset='FS1000',train=1000,evaluation=247,train_crop=95,test_crop='center95',
            selection=spec.POLICY['selection'],mcmoe_epochs={t:spec.epochs('mcmoe',t) for t in spec.TASKS},
            batch_size=32,loader_workers=2,test_batch_size=1,input_dimensions=[768,768,1024],
            rgb_pool='numpy float32 mean over 15 spatial tokens')
        env=dict(python='3.8.20',torch='1.12.1+cu113',numpy='1.21.6',scipy='1.10.1',cuda='11.3',available=True)
        spec.save(root/'manifest.json',dict(policy=policy,environments={'external':env},
            input_manifest_sha256=spec.sha(run.HERE/'FS1000_INPUT_SHA256SUMS')))
        spec.save(root/'completion.json',dict(status='EXPERIMENTS_COMPLETE'))
        for phase in ('smoke','formal'):
            for task in (('TES','PCS') if phase=='smoke' else spec.TASKS):
                for seed in ((0,) if phase=='smoke' else spec.SEEDS):
                    cell=root/phase/task/('seed%d'%seed)/'mcmoe';out=cell/'attempt_fixture';out.mkdir(parents=True)
                    n=32 if phase=='smoke' else 247
                    result=dict(task=task,seed=seed,method='mcmoe',phase=phase,reloaded_exact=True,
                        epochs=spec.epochs('mcmoe',task,phase=='smoke'),selected_epoch=1,
                        ids=['video_%d'%i for i in range(n)],coef=100.,cpu_test_only=False,
                        predictions={'prediction':list(range(n)),'label':list(range(n))})
                    (out/'model.pth').write_bytes(b'fixture only, not loadable')
                    (out/'model.pth.eval_rng.pt').write_bytes(b'fixture rng')
                    spec.save(out/'test_result.json',result)
                    spec.save(cell/'complete.json',dict(attempt=out.name,
                        files={p.name:spec.sha(p) for p in out.iterdir()}))
        return env

    def test_source_records_all_23_readonly_and_reject_weight_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.source_fixture(root)
            before={str(p.relative_to(root)):spec.sha(p) for p in root.rglob('*') if p.is_file()}
            manifest,records=run.source_records(root)
            self.assertEqual(len(records),23)
            self.assertEqual(sum(r['phase']=='formal' for r in records),21)
            self.assertEqual(manifest['policy']['train'],1000)
            after={str(p.relative_to(root)):spec.sha(p) for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before,after,'reference validation must not mutate old experiments')
            (Path(records[0]['directory'])/'model.pth').write_bytes(b'changed')
            with self.assertRaises(ValueError):run.source_records(root)

    def test_source_records_reject_budget_or_input_contract_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.source_fixture(root);path=root/'manifest.json'
            original=spec.read(path);altered=json.loads(json.dumps(original))
            altered['policy']['mcmoe_epochs']['TES']=500;spec.save(path,altered)
            with self.assertRaises(ValueError):run.source_records(root)
            altered=json.loads(json.dumps(original));altered['input_manifest_sha256']='bad';spec.save(path,altered)
            with self.assertRaises(ValueError):run.source_records(root)

    def test_source_records_reject_loader_or_feature_recipe_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.source_fixture(root);path=root/'manifest.json';original=spec.read(path)
            for key,value in (('batch_size',64),('loader_workers',0),('test_batch_size',32),
                              ('rgb_pool','torch mean'),('input_dimensions',[1024,768,1024])):
                with self.subTest(key=key):
                    altered=json.loads(json.dumps(original));altered['policy'][key]=value;spec.save(path,altered)
                    with self.assertRaisesRegex(ValueError,'matched data/loader mismatch'):
                        run.source_records(root)

    def test_prepare_references_reject_wrong_environment_before_new_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);source=folder/'v37';source.mkdir();current=self.source_fixture(source)
            target=folder/'v38';target.mkdir();current=dict(current,scipy='1.6.2')
            args=argparse.Namespace(source=str(source),data=str(folder/'data'))
            with self.assertRaisesRegex(ValueError,'environment differs'):
                run.prepare_references(args,target,current)
            self.assertFalse((target/'manifest.json').exists())
            self.assertFalse((target/'references').exists())

    def test_prepare_references_writes_metadata_only_and_detects_snapshot_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);source=folder/'v37';source.mkdir();current=self.source_fixture(source)
            target=folder/'v38';target.mkdir();args=argparse.Namespace(source=str(source),data=str(folder/'data'))
            run.prepare_references(args,target,current)
            self.assertFalse(list(target.rglob('*.pth')))
            self.assertFalse(list(target.rglob('*.pt')))
            self.assertEqual(len(list(target.rglob('record.json'))),23)
            snapshot=spec.reference_path(target,'formal','TES',0)/'test_result.json'
            value=spec.read(snapshot);value['coef']=0.;spec.save(snapshot,value)
            with self.assertRaisesRegex(ValueError,'snapshot changed'):
                run.prepare_references(args,target,current)

    def test_full_matrix(self):
        self.assertEqual(len(spec.cells()),105);self.assertEqual(len(set(spec.cells())),105)
        self.assertEqual(len(spec.cells(True)),10)
        self.assertEqual(spec.SEEDS,(0,1,2));self.assertEqual(len(spec.TASKS),7)
        self.assertEqual(spec.epochs('mc_ordered','PCS'),460)
        self.assertEqual(spec.epochs('mc_u','PCS'),500)
    def test_complete_integrity_and_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'attempt';out.mkdir();p=out/'model.pth';p.write_bytes(b'fixture')
            spec.save(root/'complete.json',dict(attempt='attempt',files={'model.pth':spec.sha(p)}))
            self.assertEqual(spec.completed(root,True),out)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):spec.completed(root,True)
            spec.save(root/'complete.json',dict(attempt='../other',files={}))
            with self.assertRaises(ValueError):spec.completed(root)
    def test_queue_keeps_independent_jobs_after_failure(self):
        calls=[]
        def fake(args,root,phase,t,s,m,gpu,workers):
            calls.append((t,m))
            if t=='TES' and m=='mc_ordered':raise RuntimeError('injected native failure')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(RuntimeError):run.wave(argparse.Namespace(),root,'smoke',['0','1'],runner=fake)
            result=spec.read(root/'smoke_wave.json')
            self.assertEqual(result['completed'],7)
            self.assertEqual(len(result['failures']),1);self.assertEqual(len(result['blocked']),2)
            self.assertIn(('TES','mc_u'),calls);self.assertIn(('PCS','mc_ordered_u'),calls)
    def test_queue_resume_skips_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cell=spec.cell_path(root,'smoke','TES',0,'mc_ordered');(cell/'attempt').mkdir(parents=True)
            spec.save(cell/'attempt'/'test_result.json',dict(dependencies={}))
            spec.save(cell/'complete.json',dict(attempt='attempt',files={
                'test_result.json':spec.sha(cell/'attempt'/'test_result.json')}))
            calls=[]
            run.wave(argparse.Namespace(),root,'smoke',['0'],runner=lambda a,r,p,t,s,m,g,w:calls.append((t,m)))
            self.assertEqual(len(calls),9);self.assertNotIn(('TES','mc_ordered'),calls)
    def test_queue_rejects_child_after_parent_replaced_or_missing(self):
        for missing in (False,True):
            with self.subTest(missing=missing),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);parent=spec.cell_path(root,'smoke','TES',0,'mc_ordered')
                child=spec.cell_path(root,'smoke','TES',0,'mc_ordered_e')
                for cell in (parent,child):(cell/'attempt').mkdir(parents=True)
                weight=parent/'attempt'/'model.pth';weight.write_bytes(b'original')
                old_digest=spec.sha(weight)
                spec.save(parent/'attempt'/'test_result.json',dict(dependencies={}))
                spec.save(child/'attempt'/'test_result.json',dict(dependencies={'mc_ordered':old_digest}))
                for cell in (parent,child):
                    spec.save(cell/'complete.json',dict(attempt='attempt',
                        files={p.name:spec.sha(p) for p in (cell/'attempt').iterdir()}))
                if missing:
                    (parent/'complete.json').unlink()
                else:
                    # Simulate an independent, valid re-run replacing the parent's active checkpoint.
                    weight.write_bytes(b'retrained different parent')
                    spec.save(parent/'complete.json',dict(attempt='attempt',
                        files={p.name:spec.sha(p) for p in (parent/'attempt').iterdir()}))
                calls=[]
                with self.assertRaises((ValueError,RuntimeError)):
                    run.wave(argparse.Namespace(),root,'smoke',['0'],
                        runner=lambda *args:calls.append(args))
                self.assertEqual(calls,[],'mixed dependency graphs must be rejected before any new job starts')
    def test_cpu_preflight_hides_unselected_gpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);current=dict(torch='1.12.1+cu113',numpy='1.21.6',scipy='1.10.1',available=True)
            args=argparse.Namespace(data=str(root/'data'))
            with patch.object(run,'code_check'),patch.object(run,'disk_check'),\
                 patch.object(run,'prepare_references'),\
                 patch.object(run.subprocess,'check_output',return_value=json.dumps(current)),\
                 patch.object(run.subprocess,'run') as invoke:
                run.check(args,root,4)
            self.assertEqual(invoke.call_count,2)
            for call in invoke.call_args_list:
                self.assertEqual(call.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
                self.assertEqual(call.kwargs['env']['PYTHONNOUSERSITE'],'1')
    def test_storage_excludes_existing_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cell=spec.cell_path(root,'formal','TES',0,'mc_e');(cell/'attempt').mkdir(parents=True)
            spec.save(cell/'complete.json',dict(attempt='attempt',files={}))
            with patch.object(run,'read',return_value={'checkpoint_bytes':{m:100 for m in spec.NEW_METHODS}}),\
                 patch.object(run,'storage_report',return_value={'free_bytes':10,'required_bytes':1,'pass':True}) as measure:
                run.disk_check(root,3)
                counts=measure.call_args[0][2]
                self.assertEqual(counts['mc_e'],22);self.assertEqual(counts['mc_ordered'],23)
    def test_pack_has_no_weights_and_verified_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'study';root.mkdir();spec.save(root/'runner.json',{'state':'failed'})
            (root/'model.pth').write_bytes(b'weights not returned')
            dest=run.pack(root,True)
            with zipfile.ZipFile(dest) as z:
                self.assertFalse(any(p.endswith('.pth') for p in z.namelist()))
                for row in json.loads(z.read('RETURN_CONTENTS.json')):
                    self.assertEqual(hashlib.sha256(z.read(root.name+'/'+row['path'])).hexdigest(),row['sha256'])
    def test_gpu_selection_rejects_bad_or_busy(self):
        with patch.object(run,'gpu_inventory',return_value={'0':12000,'1':3000}):
            self.assertEqual(run.selected_gpus('0'),['0'])
            for value in ('','0 0','1','2'):
                with self.assertRaises(RuntimeError):run.selected_gpus(value)
    def test_metrics_guard_does_not_fake_zero(self):
        from worker import metrics,MetricGuard
        result=metrics([.2,.2,.2],[.1,.5,.9]);self.assertIsNone(result['coef'])
        guard=MetricGuard();self.assertFalse(guard.observe(1,result))
        self.assertTrue(guard.observe(2,metrics([.2,.4,.6],[.1,.5,.9])))
        self.assertEqual(guard.finish()['undefined_epochs'],[1])
        with self.assertRaises(FloatingPointError):metrics([float('nan'),.3],[.1,.5])
        with self.assertRaises(ValueError):metrics([.2,.3],[.1,.1])
        for i in range(10):guard.observe(i+3,result)
        with self.assertRaises(FloatingPointError):guard.observe(13,result)

if __name__=='__main__':unittest.main()
