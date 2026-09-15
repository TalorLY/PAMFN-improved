import copy
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import main as entry
import pipeline
from models.ema_shadow import EMAShadow, DECAYS, VIEWS, calibration_loader, refresh_bn, isolated_randomness
from models.score_qmd import ScoreQMD
from tests.test_qmd_head import HeadModel


class TrainingData(Dataset):
    is_train = True

    def __init__(self, n=8):
        g=torch.Generator().manual_seed(16)
        self.x=torch.randn(n,12,4,generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        # Deliberately exercise both global Python and NumPy randomness.
        value=self.x[i]+.01*np.random.random()+.01*random.random()
        return value,value*.6,value*.3,torch.tensor((i+1)/(len(self)+1)),{m:12 for m in ('V','F','A')}


class EMAUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_optimizer_value_equal(self, left, right):
        # AdamW step is a Python scalar in older PyTorch and a Tensor in newer versions.
        self.assertEqual(type(left), type(right))
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
        else:
            self.assertEqual(left, right)

    def test_optimizer_value_comparison_accepts_scalars_and_tensors_but_rejects_changes(self):
        for left, right in ((4, 4), (4., 4.), (torch.tensor(4.), torch.tensor(4.)),
                            (torch.zeros(2), torch.zeros(2))):
            with self.subTest(equal_type=type(left).__name__):
                self.assert_optimizer_value_equal(left, right)
        for left, right in ((4, 5), (4., 5.), (torch.tensor(4.), torch.tensor(5.)),
                            (torch.zeros(2), torch.ones(2)), (4, torch.tensor(4.))):
            with self.subTest(changed_type=type(left).__name__), self.assertRaises(AssertionError):
                self.assert_optimizer_value_equal(left, right)

    def test_parameter_formula_first_copy_no_alias_and_fixed_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=HeadModel()
            ema=EMAShadow(model,TrainingData(),Path(tmp)/'model.pth')
            before=torch.get_rng_state().clone()
            ema.update(model)
            old={n:p.clone() for n,p in model.named_parameters()}
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(.25)
            ema.update(model)
            for view,decay in DECAYS.items():
                for n,p in model.named_parameters():
                    self.assertTrue(torch.equal(ema.weights[view][n],old[n]+(p-old[n])*(1-decay)))
                    self.assertNotEqual(ema.weights[view][n].data_ptr(),p.data_ptr())
            self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_two_epoch_student_qmd_optimizer_rng_parity(self):
        outcomes=[]
        with tempfile.TemporaryDirectory() as tmp:
            for enabled in (False,True):
                pipeline.setup_seed(41)
                random.seed(41)
                data=TrainingData()
                loader=DataLoader(data,batch_size=4,shuffle=True,num_workers=0)
                model=HeadModel()
                opt=torch.optim.AdamW(model.parameters(),lr=.001)
                qmd=ScoreQMD('score_qmd',0,2,warmup=0,rampup=0)
                ema=EMAShadow(model,data,Path(tmp)/'model.pth') if enabled else None
                metrics=[]
                for epoch in range(2):
                    train=pipeline.train_epoch(model,opt,None,loader,epoch,qmd=qmd,ema=ema)
                    test=pipeline.test_epoch(model,loader,eval_policy='argmax')
                    if ema:
                        ema.evaluate(model,epoch+1,loader,pipeline.test_epoch,pipeline._prepare_batch,20260908)
                    train.pop('time'); test.pop('time')
                    metrics.append((train,test))
                if ema:
                    self.assertEqual(ema.finish()['updates'],4)
                    for view in VIEWS:
                        selected=json.loads((Path(tmp)/(view+'.selection.json')).read_text())
                        with isolated_randomness():
                            restored=HeadModel()
                            restored.load_state_dict(pipeline._load_state_dict(str(Path(tmp)/(view+'.pth'))))
                        # Deterministic test fixture for reload handled separately; files all present here.
                        self.assertTrue(np.isfinite(selected['coef']))
                outcomes.append((copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),
                    torch.get_rng_state().clone(),np.random.get_state(),random.getstate(),metrics,qmd.summary()))
        for n,p in outcomes[0][0].items():
            self.assertTrue(torch.equal(p,outcomes[1][0][n]),n)
        for key,state in outcomes[0][1]['state'].items():
            for name,v in state.items():
                self.assert_optimizer_value_equal(v,outcomes[1][1]['state'][key][name])
        self.assertTrue(torch.equal(outcomes[0][2],outcomes[1][2]))
        self.assertTrue(np.array_equal(outcomes[0][3][1],outcomes[1][3][1]))
        self.assertEqual(outcomes[0][3][2:],outcomes[1][3][2:])
        self.assertEqual(outcomes[0][4:],outcomes[1][4:])

    def test_bn_refresh_all_samples_dropout_off_modes_restored_and_rng_isolated(self):
        data=TrainingData(33)
        loader=calibration_loader(data)
        self.assertEqual(list(loader.batch_sampler),[list(range(33))])
        model=HeadModel().eval()
        dropout_modes=[]
        handles=[m.register_forward_pre_hook(lambda module,inputs:dropout_modes.append(module.training))
                 for m in model.modules() if isinstance(m,nn.Dropout)]
        torch_state=torch.get_rng_state().clone()
        np_state=np.random.get_state()
        py_state=random.getstate()
        report=refresh_bn(model,loader,pipeline._prepare_batch)
        self.assertEqual(report['samples'],33)
        self.assertFalse(any(dropout_modes))
        self.assertFalse(any(m.training for m in model.modules()))
        self.assertEqual(model.fc[2].momentum,.1)
        self.assertEqual(int(model.fc[2].num_batches_tracked),1)
        self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))
        self.assertTrue(np.array_equal(np_state[1],np.random.get_state()[1]))
        self.assertEqual(np_state[2:],np.random.get_state()[2:])
        self.assertEqual(py_state,random.getstate())
        for h in handles:
            h.remove()
        data.is_train=False
        with self.assertRaisesRegex(ValueError,'training dataset'):
            calibration_loader(data)

    def test_exception_restores_all_rng(self):
        before=torch.get_rng_state().clone()
        py=random.getstate(); np_state=np.random.get_state()
        with self.assertRaises(RuntimeError):
            with isolated_randomness(99):
                raise RuntimeError('test')
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(py,random.getstate())
        self.assertTrue(np.array_equal(np_state[1],np.random.get_state()[1]))

    def test_cli_default_unchanged_and_invalid_combinations_rejected(self):
        argv=['main.py','--gpu','-1','--feats','2','--action','Ball','--multi_modality',
              '--qmd_mode','score_qmd','--eval_policy','argmax']
        with patch('sys.argv',argv):
            args=entry.parser_args()
            self.assertNotIn('ema',entry._run_metadata(args,500))
        with patch('sys.argv',argv+['--ema_screen','--save_final_model','x.pth']):
            args=entry.parser_args()
            self.assertFalse(entry._run_metadata(args,500)['ema']['recipe']['feedback_to_student'])
        for extra in (['--ema_screen'],['--ema_view','ema_short'],
                      ['--ema_screen','--save_final_model','x.pth','--score_head','linear']):
            with patch('sys.argv',argv+extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                entry.parser_args()


if __name__=='__main__':
    unittest.main()
