import subprocess
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets import AQADataset
from models.score_qmd import (MeanSegmentMask, ScoreQMD, paired_masked_forward,
                              ramp, rng_state, score_kd)
from pipeline import train_epoch

BASE_COMMIT = 'a2544aa6b4c5ba19f97ac52a3c9f9e941d859393'


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4),
                                      torch.nn.Dropout(0.3), torch.nn.Linear(4, 1))

    def forward(self, inputs):
        features = sum(inputs.values()).mean(1)
        return self.fc(features), {}

    def call_loss(self, prediction, label, **kwargs):
        return (prediction.flatten() - label.flatten()).square().mean()


class TinyDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        # Exercise the original NumPy crop and torch dropout/shuffle streams.
        start = np.random.randint(0, 4)
        x = torch.arange(48, dtype=torch.float32).view(12, 4)[start:start+8] / 48
        return x, x * 0.7, x * 0.8, float(index) / 8, 0


class ScoreQMDTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mask_reproducible_balanced_and_preserves_padding(self):
        inputs = {m: torch.arange(80.).view(2, 10, 4) for m in ('V', 'F', 'A')}
        lengths = {m: torch.tensor([6, 9]) for m in inputs}
        a, b = MeanSegmentMask(42), MeanSegmentMask(42)
        state = torch.get_rng_state().clone()
        names = []
        for _ in range(6):
            left, lm = a(inputs, lengths)
            right, rm = b(inputs, lengths)
            self.assertEqual(lm, rm)
            names.append(lm['modality'])
            for m in inputs:
                self.assertTrue(torch.equal(left[m], right[m]))
                for i, n in enumerate(lengths[m]):
                    self.assertTrue(torch.equal(left[m][i, n:], inputs[m][i, n:]))
                if m != lm['modality']:
                    self.assertTrue(torch.equal(left[m], inputs[m]))
            for i, (start, width, count) in enumerate(lm['spans']):
                expected = inputs[lm['modality']][i, :count].mean(0)
                self.assertTrue(torch.equal(left[lm['modality']][i, start], expected))
        self.assertEqual(sorted(names), ['A', 'A', 'F', 'F', 'V', 'V'])
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_teacher_detached(self):
        teacher = torch.tensor([0.4, 0.8], requires_grad=True)
        student = torch.tensor([0.1, 0.6], requires_grad=True)
        score_kd(student, teacher).backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(student.grad.abs().sum()), 0)

    def test_paired_rng_bn_and_backward(self):
        torch.manual_seed(5)
        model = TinyModel().train()
        inputs = {m: torch.randn(4, 8, 4) for m in ('V', 'F', 'A')}
        before = rng_state()
        full, _ = model(inputs)
        after = torch.get_rng_state().clone()
        buffers = {n: x.clone() for n, x in model.named_buffers()}
        masked, _ = paired_masked_forward(model, inputs, before)
        self.assertTrue(torch.equal(full, masked))
        self.assertTrue(torch.equal(after, torch.get_rng_state()))
        for n, x in model.named_buffers():
            self.assertTrue(torch.equal(x, buffers[n]))
        (full.square().mean() + masked.square().mean()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_schedule_and_zero_weight(self):
        self.assertEqual(ramp(9, 100), 0)
        self.assertAlmostEqual(ramp(10, 100), 0.1)
        self.assertEqual(ramp(19, 100), 1)
        self.assertEqual(ScoreQMD('score_qmd', 0, 100, weight=0).scale(99), 0)

    def test_dataset_valid_lengths_do_not_change_crop_or_tensors(self):
        for n in (4, 8, 12):
            data = {m: {'x': np.arange(n*4, dtype=np.float32).reshape(n, 4)}
                    for m in ('rgb', 'audio', 'flow')}
            with patch('datasets.read_labels', return_value=[('x', 0.5)]), \
                    patch.object(AQADataset, 'read_data', return_value=data):
                common = dict(dataset_name='TES', feat_dir='.', rgb_feat='', flow_feat='',
                              audio_feat='', segments=8)
                normal = AQADataset(**common)
                enabled = AQADataset(**common, return_valid_lengths=True)
            np.random.seed(8)
            expected = normal[0]
            np.random.seed(8)
            actual = enabled[0]
            for i in range(3):
                self.assertTrue(torch.equal(expected[i], actual[i]))
            self.assertEqual(actual[4], dict(V=min(8,n), F=min(8,n), A=min(8,n)))

    def test_active_objective_has_gradients_and_one_bn_update(self):
        for mode in ('mask_gt', 'score_qmd'):
            torch.manual_seed(7)
            model = TinyModel()
            inputs = {m: torch.randn(4, 8, 4) for m in ('V', 'F', 'A')}
            labels = torch.tensor([0.1, 0.3, 0.6, 0.9])
            qmd = ScoreQMD(mode, 0, 2, warmup=0, rampup=0)
            before = rng_state()
            full, _ = model(inputs)
            main = model.call_loss(full, labels)
            aux, _ = qmd.auxiliary(model, inputs, {m:[8]*4 for m in inputs},
                                   labels, full, main, before, 0)
            (main + aux).backward()
            self.assertTrue(torch.isfinite(aux))
            self.assertEqual(int(model.fc[1].num_batches_tracked), 1)
            self.assertGreater(qmd.active_batches, 0)
            if mode == 'score_qmd':
                self.assertGreater(qmd.beta, 0)
                self.assertLessEqual(qmd.beta, 1.0)

    def test_disabled_matches_frozen_base_two_epochs(self):
        source = subprocess.check_output(['git', 'show', BASE_COMMIT + ':pipeline.py'],
                                         cwd=Path(__file__).resolve().parents[1]).decode()
        legacy = types.ModuleType('qmd_reference_pipeline')
        exec(compile(source, 'frozen_pipeline.py', 'exec'), legacy.__dict__)
        def run(function, mode):
            torch.manual_seed(17)
            np.random.seed(17)
            model = TinyModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            loader = DataLoader(TinyDataset(), batch_size=4, shuffle=True, num_workers=0)
            extra = {} if mode is None else {'qmd': ScoreQMD('score_qmd', 17, 2, weight=0)}
            metrics = []
            for epoch in range(2):
                result = function(model, optimizer, None, loader, epoch, **extra)
                metrics.append({k:v for k,v in result.items() if k != 'time'})
            return model.state_dict(), torch.get_rng_state(), metrics
        expected = run(legacy.train_epoch, None)
        for mode in (None, 'zero'):
            actual = run(train_epoch, mode)
            self.assertEqual(expected[2], actual[2])
            self.assertTrue(torch.equal(expected[1], actual[1]))
            for key in expected[0]:
                self.assertTrue(torch.equal(expected[0][key], actual[0][key]), key)

    def test_pamfn_paired_gumbel_and_checkpoint(self):
        from tests.test_coqm_rng_isolation import COQMRNGIsolationTest
        model, _ = COQMRNGIsolationTest._build_pamfn(11, eval_policy='argmax')
        model.train()
        inputs = {m: torch.randn(2, 16, 4) for m in ('V', 'F', 'A')}
        before = rng_state()
        full, _ = model(inputs)
        paired, _ = paired_masked_forward(model, inputs, before)
        self.assertTrue(torch.equal(full, paired))
        qmd = ScoreQMD('score_qmd', 11, 1, warmup=0, rampup=0)
        label = torch.tensor([0.2, 0.8])
        main = model.call_loss(full, label)
        auxiliary, _ = qmd.auxiliary(model, inputs, {m:[16,16] for m in inputs},
                                     label, full, main, before, 0)
        (main + auxiliary).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                            if p.grad is not None))
        restored, _ = COQMRNGIsolationTest._build_pamfn(12, eval_policy='argmax')
        restored.load_state_dict(model.state_dict())
        model.eval()
        restored.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(inputs)[0], restored(inputs)[0]))


if __name__ == '__main__':
    unittest.main()
