import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import run as r

HISTORY = Path(__file__).parent/'v17_history.zip'
if not HISTORY.exists(): HISTORY = Path('D:/edge_download/old/qmd_supp_v17_return.zip')


def completed(directory, score=100.):
    out=directory/'attempts/0001';out.mkdir(parents=True)
    for name in ('model.pth','model.pth.eval_rng.pt'):(out/name).write_bytes(name.encode())
    r.save(out/'test_result.json',dict(coef=score))
    r.save(out/'complete.json',dict(metrics=dict(srcc_x100=score),
        files={'test_result.json':r.sha(out/'test_result.json')},
        checkpoint_sha256=r.sha(out/'model.pth'),eval_rng_sha256=r.sha(out/'model.pth.eval_rng.pt')))
    r.save(directory/'active.json',dict(attempt='attempts/0001'))
    return out


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.hist=r.history(HISTORY)

    def test_all_72_historical_arguments(self):
        with patch.object(r,'active',return_value=Path('/new/teacher')):
            for a in r.ACTIONS:
                for s in range(3):
                    for m in r.METHODS:
                        argv=r.arguments(self.hist,Path('/new'),'screen',a,s,m,'5')[5:]
                        old=self.hist['templates']['%s/%d/%s'%(a,s,m)]
                        def strip_paths(values):
                            values=list(values)
                            for name in ('teacher_model','teacher_rng','teacher_record','e_model','e_record'):
                                if '--'+name in values:values[values.index('--'+name)+1]='DEPENDENCY_PATH'
                            return values
                        self.assertEqual(strip_paths(argv),strip_paths(old))
                        self.assertNotIn('--epochs',argv)

    def test_production_unit_module_list(self):
        self.assertEqual(r.trainer_unit_argv()[4:],list(r.TRAINER_TESTS))
        self.assertIn('tests.test_july28_qmd',r.TRAINER_TESTS)
        self.assertNotIn('tests.test_score_qmd',r.TRAINER_TESTS)

    def test_check_stage_never_trains(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'results';repo=Path(t)/'trainer'
            lock=types.ModuleType('fcntl');lock.LOCK_EX=1;lock.LOCK_NB=2;lock.flock=lambda *a:None
            def cmd(argv,repo,directory,name,visible_gpu=None):
                (directory/(name+'.log')).write_text('OK',encoding='utf-8')
            args=['run.py','--stage','check','--root',str(root),'--repo',str(repo),'--history',str(HISTORY)]
            with patch('sys.argv',args),patch.dict('sys.modules',{'fcntl':lock}),patch.object(r,'preflight',return_value=['0']),patch.object(r,'command',side_effect=cmd),patch.object(r,'wave') as wave,contextlib.redirect_stdout(io.StringIO()):
                r.main()
            wave.assert_not_called()
            self.assertEqual(r.read(root/'runner.json')['phase'],'checked')
            self.assertFalse((root/'smoke_gate.json').exists())

    def test_wrong_archive_refused(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'x.zip';path.write_bytes(b'wrong')
            with self.assertRaises(ValueError):r.history(path)

    def test_dependency_must_be_new_completed_cell(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            with self.assertRaises(ValueError):r.arguments(self.hist,root,'screen','Ball',0,r.METHODS[2],'4')
            o=completed(r.cell(root,'screen','Ball',0,'ordered'))
            argv=r.arguments(self.hist,root,'screen','Ball',0,r.METHODS[2],'4')
            self.assertIn(str(o/'model.pth'),argv)
            self.assertNotIn('/szztdata/ly/exp0_source/PAMFN-qmd-dual-v2',str(argv))

    def test_smoke_never_overrides_formal_budget(self):
        a=r.arguments(self.hist,Path('/new'),'smoke','Ball',0,'base','4')
        self.assertEqual(a[-2:],['--epochs','2'])
        self.assertEqual(r.BUDGET,dict(Ball=500,Clubs=100,Hoop=300,Ribbon=20,TES=100,PCS=100))

    def test_completed_weights_and_records_checked(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);out=completed(d);self.assertEqual(r.active(d),out)
            (out/'model.pth').write_bytes(b'changed')
            with self.assertRaises(ValueError):r.active(d)

    def test_unsafe_pointer_refused(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);r.save(d/'active.json',dict(attempt='../../other'))
            with self.assertRaises(ValueError):r.active(d)

    def test_keep_completed_never_calls_training(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);completed(r.cell(root,'screen','Ball',0,'base'))
            with patch.object(r,'command') as cmd,contextlib.redirect_stdout(io.StringIO()):
                r.run_cell(self.hist,root,root,'screen','Ball',0,'base','0')
            cmd.assert_not_called()

    def test_interrupted_attempt_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);d=r.cell(root,'smoke','Ball',0,'base');out=d/'attempts/0001'
            out.mkdir(parents=True);(out/'model.pth').write_bytes(b'old-partial')
            r.save(d/'active.json',dict(attempt='attempts/0001'))
            with patch.object(r,'command',side_effect=RuntimeError('new failure')),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):r.run_cell(self.hist,root,root,'smoke','Ball',0,'base','0')
            self.assertEqual((out/'model.pth').read_bytes(),b'old-partial')
            self.assertEqual(r.read(d/'active.json')['attempt'],'attempts/0002')

    def test_contract_change_not_silently_resumed(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'contract.json';r.immutable(p,dict(x=1))
            with self.assertRaises(ValueError):r.immutable(p,dict(x=2))
            self.assertEqual(r.read(p),dict(x=1))

    def make_results(self,out):
        run=dict(action='Ball',seed=0,tag='base',feats=2,epochs=2,eval_policy='legacy_gumbel',qmd=dict(mode='off'),ordinal_prototype_weight=0.)
        pred=dict(label=list(range(50)),prediction=list(range(50)),order='native_test_loader')
        test=dict(run=run,coef=100.,predictions=pred,time=1.)
        r.save(out/'test_result.json',test);r.save(out/'reload_result.json',dict(test,time=99.))
        r.save(out/'train_result.json',dict(run=run,saved_model_srcc=1.))
        r.save(out/'model.pth.selection.json',dict(epoch=1,coef=100.))
        (out/'model.pth.epochs.jsonl').write_text('\n'.join(json.dumps(dict(epoch=e,coef=100.)) for e in (1,2)))

    def test_reload_allows_only_wallclock_difference(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);self.make_results(out)
            self.assertEqual(r.validate(out,'Ball',0,'base','smoke')['selected_epoch'],1)
            d=r.read(out/'reload_result.json');d['predictions']['prediction'][0]=.01;r.save(out/'reload_result.json',d)
            with self.assertRaises(ValueError):r.validate(out,'Ball',0,'base','smoke')

    def test_selection_is_first_trained_maximum(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);self.make_results(out);r.save(out/'model.pth.selection.json',dict(epoch=2,coef=100.))
            with self.assertRaises(ValueError):r.validate(out,'Ball',0,'base','smoke')

    def test_complete_asset_backup_is_separate_from_return(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'result';root.mkdir()
            for stage,pairs in [('smoke',[('Ball',0),('TES',0)]),('screen',[(a,s) for a in r.ACTIONS for s in range(3)])]:
                for a,s in pairs:
                    for m in r.METHODS:completed(r.cell(root,stage,a,s,m))
            with contextlib.redirect_stdout(io.StringIO()):r.package(root)
            self.assertTrue(r.read(root/'archive.json')['ready'])
            manifest=r.read(root/'asset_manifest.json')['files']
            self.assertEqual(sum(n.endswith('/model.pth') for n in manifest),80)
            self.assertEqual(sum(n.endswith('/model.pth.eval_rng.pt') for n in manifest),80)
            with zipfile.ZipFile(root.parent/'result_return.zip') as z:
                self.assertFalse(any(n.endswith('.pth') or n.endswith('.pt') for n in z.namelist()))
            self.assertTrue((root.parent/'result_assets.tar.gz').exists())


if __name__=='__main__':unittest.main()
