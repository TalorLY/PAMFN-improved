import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import main as entry
import pipeline
from datasets import AQADataset
from models.score_qmd import ScoreQMD
from tools.verify_july28_qmd import frozen


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.BatchNorm1d(4),
                                      torch.nn.Dropout(.3),torch.nn.Linear(4,1),torch.nn.Sigmoid())

    def forward(self, inputs):
        result = self.fc(sum(inputs.values()).mean(1))
        # Consume eval RNG as July28 Gumbel does, so replay is actually exercised.
        return result + torch.rand_like(result)*.00001, {}

    def call_loss(self, pred, label, **kwargs):
        return (pred.flatten()-label.flatten()).square().mean()


class TinyDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        start = np.random.randint(0,4)
        x = torch.arange(48,dtype=torch.float32).view(12,4)[start:start+8]/48 + index*.1
        return x, x*.7, x*.8, float(index)/12, dict(V=8,F=8,A=8)


class July28QMDTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cli_original_ordered_and_both_qmd_modes(self):
        for mode in ('off','score_qmd','score_rank_qmd'):
            argv=['main.py','--gpu','-1','--feats','2','--action','Ball','--multi_modality',
                  '--ordinal_prototype_weight','.0001','--qmd_mode',mode]
            with patch('sys.argv',argv):
                args=entry.parser_args()
            meta=entry.run_metadata(args,500)
            self.assertEqual(meta['eval_policy'],'legacy_gumbel')
            self.assertFalse(meta['quality_aux_rng_isolated'])
            self.assertEqual(meta['ordinal_prototype_target_mode'],'discrete')

    def test_original_and_off_and_zero_match_training_rng(self):
        old=frozen('pipeline.py')
        values=[]
        for impl,zero in ((old,False),(pipeline,False),(pipeline,True)):
            pipeline.setup_seed(17)
            model=TinyModel()
            optim=torch.optim.AdamW(model.parameters(),lr=.001)
            loader=DataLoader(TinyDataset(),batch_size=4,shuffle=True)
            metrics=[]
            for epoch in range(2):
                extra=dict(qmd=ScoreQMD('score_qmd',17,2,weight=0)) if zero else {}
                metric=impl.train_epoch(model,optim,None,loader,epoch,**extra)
                metric.pop('time')
                evaluation=impl.test_epoch(model,loader)
                metrics.append((metric,evaluation['coef'],evaluation['loss_x100']))
            values.append((model.state_dict(),torch.get_rng_state(),np.random.get_state()[1],metrics))
        for actual in values[1:]:
            self.assertEqual(actual[3],values[0][3])
            self.assertTrue(torch.equal(actual[1],values[0][1]))
            self.assertTrue(np.array_equal(actual[2],values[0][2]))
            for key,value in actual[0].items():
                self.assertTrue(torch.equal(value,values[0][0][key]),key)

    def test_valid_lengths_do_not_change_original_dataset(self):
        original=frozen('datasets.py')
        for length in (4,8,12):
            data={m:{'x':np.arange(length*4,dtype=np.float32).reshape(length,4)} for m in ('rgb','flow','audio')}
            outputs=[]
            for cls in (original.AQADataset,AQADataset):
                module=original if cls is original.AQADataset else __import__('datasets')
                with patch.object(module,'read_labels',return_value=[('x',.5)]),patch.object(cls,'read_data',return_value=data):
                    dataset=cls('TES','.', '', '', '', segments=8,return_valid_lengths=True)
                np.random.seed(9)
                outputs.append(dataset[0])
            for index in range(3):
                self.assertTrue(torch.equal(outputs[0][index],outputs[1][index]))
            self.assertEqual(outputs[1][4],dict(V=min(8,length),F=min(8,length),A=min(8,length)))

    def test_active_score_and_rank_have_gradients_and_finite_epochs(self):
        for mode in ('score_qmd','score_rank_qmd'):
            pipeline.setup_seed(7)
            model=TinyModel()
            optim=torch.optim.AdamW(model.parameters(),lr=.001)
            qmd=ScoreQMD(mode,7,1,warmup=0,rampup=0,rank_gap=0)
            metric=pipeline.train_epoch(model,optim,None,DataLoader(TinyDataset(),batch_size=4),0,qmd=qmd)
            self.assertTrue(np.isfinite(metric['objective_x100']))
            self.assertEqual(qmd.active_batches,3)
            self.assertGreater(qmd.beta,0)
            self.assertEqual(int(model.fc[1].num_batches_tracked),3)

    def test_saved_stochastic_evaluation_is_exactly_replayed(self):
        pipeline.setup_seed(17)
        model=TinyModel()
        loader=DataLoader(TinyDataset(),batch_size=4)
        optim=torch.optim.AdamW(model.parameters(),lr=.001)
        with tempfile.TemporaryDirectory() as temp:
            path=str(Path(temp)/'model.pth')
            # Ensure the historical positive-SRCC save condition is met in this fixture.
            real_test=pipeline.test_epoch
            def positive_test(*args):
                result=real_test(*args)
                result['coef']=abs(result['coef'])
                return result
            with patch.object(pipeline,'test_epoch',side_effect=positive_test):
                result=pipeline.train_loop(model,optim,None,loader,loader,2,save_final_model=path,log_func=lambda x:None)
                with patch.object(pipeline,'build_dataset',return_value=(loader,loader)),patch.object(pipeline,'build_model',side_effect=lambda **kw:TinyModel()):
                    test=pipeline.test_model({}, {}, path, replay_rng_path=path+'.eval_rng.pt',log_func=lambda x:None)
            self.assertEqual(test['coef'],result[4])
            self.assertTrue(test['evaluation']['replayed_rng'])
            selection=json.loads(Path(path+'.selection.json').read_text())
            self.assertEqual(selection['coef'],test['coef'])


if __name__ == '__main__':
    unittest.main()
