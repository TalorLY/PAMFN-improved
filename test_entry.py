"""Synthetic original CLI vs new worker parity, plus all new worker paths."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader
import pipeline
from tests.test_qmd_modal import TinyModalAnchor, TinyData
from models.score_qmd import rng_state
from tools.qmd_residual import freeze_anchor
from tools.qmd_modal import make_cache
from tools.qmd_supp_config import E_TAG, SCALED
from tools import train_qmd_modal as old_e
from tools import train_qmd_supp as old_s
import worker as entry
import contract as c
from tests.test_qmd_modal import TinyModality


class Wide(TinyModalAnchor):
    def __init__(self): super().__init__(dim=256)


class Data(TinyData):
    def __init__(self,n): self.n=n
    def __len__(self): return self.n
    def __getitem__(self,i):
        v,a,f,y,lengths=super().__getitem__(i%50)
        return v,a,f,.2+((i*17)%50)*.012,lengths


def datasets(**kwargs):
    return DataLoader(Data(32),batch_size=8,shuffle=True),DataLoader(Data(12),batch_size=1)


def config(args): return {},2,{},dict(model_conf={}),{},{}


class EntryTest(unittest.TestCase):
    def test_original_cli_parity_and_all_worker_paths(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); pipeline.setup_seed(0); anchor=freeze_anchor(Wide())
            model=root/'anchor.pth'; rng=root/'anchor.pth.eval_rng.pt'
            old_s.tensor_save(model,anchor.state_dict()); replay=(rng_state(),np.random.get_state());old_s.tensor_save(rng,replay)
            cache=make_cache(anchor,datasets()[1],torch.device('cpu'),replay)
            baseline=entry.metrics(cache[0],cache[2]); source=dict(action='Ball',seed=0,checkpoint_sha256=c.sha(model),
                eval_rng_sha256=c.sha(rng),expected_srcc_x100=baseline['coef'])
            teacher_record=root/'teacher/complete.json'
            c.save(teacher_record,dict(checkpoint_sha256=c.sha(model),eval_rng_sha256=c.sha(rng),
                metrics=dict(protocol='july28',action='Ball',seed=0,method='ordered',srcc_x100=baseline['coef'])))
            shared=['--gpu','-1','--action','Ball','--seed','0','--teacher_model',str(model),
                    '--teacher_rng',str(rng),'--teacher_record',str(teacher_record),'--smoke','--epochs','2']
            refs=dict(action='Ball',seed=0)
            refs['ordered']=dict(source=source,model=str(model),rng=str(rng),test=baseline,record=str(teacher_record))
            refs['base']=refs['ordered']
            with patch.object(pipeline,'build_dataset',side_effect=datasets),patch.object(pipeline,'build_model',side_effect=lambda **k:Wide()), \
                 patch.object(old_e,'configurations',side_effect=config),patch.object(old_s,'configurations',side_effect=config), \
                 patch.object(entry,'configurations',side_effect=config):
                for module,key,tag in [(old_e,'e',E_TAG),(old_s,'s',SCALED)]:
                    out=root/key; out.mkdir()
                    extra=[] if key=='e' else ['--e_model',refs['e']['model'],'--e_record',refs['e']['record']]
                    argv=['train']+shared+extra+['--run_tag',tag]
                    with patch('sys.argv',argv+['--save_final_model',str(out/'model.pth'),'--result_json',str(out/'train_result.json')]): module.main()
                    with patch('sys.argv',argv+['--test','--ckpt_path',str(out/'model.pth'),
                               '--replay_rng_path',str(out/'model.pth.eval_rng.pt'),'--result_json',str(out/'test_result.json')]): module.main()
                    score=c.read(out/'test_result.json')
                    record=dict(checkpoint_sha256=c.sha(out/'model.pth'),eval_rng_sha256=c.sha(rng),
                        files={n:c.sha(out/n) for n in ('test_result.json','training_statistics.json')},
                        metrics=dict(protocol='july28',action='Ball',seed=0,method=tag,srcc_x100=score['coef']))
                    c.save(out/'complete.json',record)
                    refs[key]=dict(model=str(out/'model.pth'),record=str(out/'complete.json'),test=score,source=c.source_from(record))
                rgb=root/'pretrained_models/feats2/Ball_rgb_UMT.pth';rgb.parent.mkdir(parents=True)
                rgb_model=TinyModality('V',256);rgb_model.modality='V';torch.save(rgb_model,rgb)
                manifest=root/'manifest.json'; c.save(manifest,dict(references=[refs],source=dict(inputs={
                    'pretrained_models/feats2/Ball_rgb_UMT.pth':dict(sha256=c.sha(rgb))})))
                for method in ('replay',)+c.SMOKE_METHODS:
                    directory=root/'formal/Ball/seed0'/method
                    out=directory/'attempts/001'
                    c.save(directory/'active.json',dict(directory=str(out)))
                    args=argparse.Namespace(gpu='-1',manifest=str(manifest),action='Ball',seed=0,method=method,output=str(out),smoke=False,evaluate=False)
                    with patch.object(entry,'REPO',root):entry.run(args)
                    self.assertTrue((out/'test_result.json').exists())
                    if method in ('repeat_e','repeat_s'):
                        self.assertTrue(c.read(out/'parity.json')['branch_state_exact'])
                    if method=='base_e': c.save(out/'complete.json',dict(checkpoint_sha256=c.sha(out/'model.pth')))
                    if method=='replay':
                        self.assertEqual(len(c.read(out/'inference_draws.json')['rows']),10)
                        self.assertEqual(len(c.read(out/'cost.json')['measurements']),12)
                    else:
                        args.evaluate=True
                        with patch.object(entry,'REPO',root):entry.run(args)
                        self.assertEqual(c.read(out/'test_result.json')['predictions'],c.read(out/'reload_result.json')['predictions'])


if __name__=='__main__': unittest.main()
