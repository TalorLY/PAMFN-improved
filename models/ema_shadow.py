"""Evaluation-only EMA shadows; never feed gradients or targets to the student."""
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.coqm import preserve_torch_rng_state

DECAYS = dict(ema_short=.9, ema_long=.99)
VIEWS = ('online_bn', 'ema_short', 'ema_long')
CALIBRATION_SEED = 20260911


def recipe():
    return dict(decays=dict(DECAYS), update='after_each_optimizer_step_first_step_copy',
                feedback_to_student=False, bn='train_only_all_bn_cumulative_one_pass',
                calibration_seed=CALIBRATION_SEED, calibration_batch_size=32,
                calibration_routing='argmax', calibration_dropout=False)


@contextmanager
def isolated_randomness(seed=None):
    py, np_state = random.getstate(), np.random.get_state()
    with preserve_torch_rng_state():
        try:
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_initialized():
                    torch.cuda.manual_seed_all(seed)
            yield
        finally:
            random.setstate(py)
            np.random.set_state(np_state)


def calibration_loader(dataset):
    if hasattr(dataset, 'is_train') and not dataset.is_train:
        raise ValueError('BN calibration must use the training dataset')
    if len(dataset) < 2:
        raise ValueError('BN calibration needs at least two training samples')
    # Merge a singleton tail, retaining every training sample (never test labels).
    batches = [list(range(i, min(i+32, len(dataset)))) for i in range(0, len(dataset), 32)]
    if len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return DataLoader(dataset, batch_sampler=batches, num_workers=0, pin_memory=False)


@torch.no_grad()
def refresh_bn(model, loader, prepare_batch):
    """Run only BN in training mode; dropout and routing remain deterministic."""
    if getattr(model, 'eval_policy', 'argmax') != 'argmax':
        raise ValueError('EMA v6 supports argmax evaluation only')
    if hasattr(loader.dataset, 'is_train') and not loader.dataset.is_train:
        raise ValueError('test data may not calibrate BN')
    modes = [(m, m.training) for m in model.modules()]
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
             and m.track_running_stats]
    momenta = [m.momentum for m in norms]
    samples = batches = 0
    try:
        model.eval()
        for m in norms:
            m.reset_running_stats()
            m.momentum = None
            m.train()
        with isolated_randomness(CALIBRATION_SEED):
            for batch in loader:
                inputs, labels = prepare_batch(batch, next(model.parameters()).device)
                model(inputs)  # Labels only count coverage; never used by the forward or loss.
                samples += len(labels)
                batches += 1
    finally:
        for m, momentum in zip(norms, momenta):
            m.momentum = momentum
        for m, training in modes:
            m.training = training
    if samples != len(loader.dataset):
        raise RuntimeError('BN calibration did not cover every training sample')
    return dict(samples=samples, batches=batches, layers=len(norms), source='train')


class EMAShadow:
    def __init__(self, model, dataset, checkpoint):
        if not checkpoint:
            raise ValueError('EMA requires --save_final_model')
        self.root = Path(checkpoint).resolve().parent
        self.root.mkdir(parents=True, exist_ok=True)
        # Deepcopy, not reinitialization: consumes no RNG, shares no student buffers.
        self.evaluator = copy.deepcopy(model).eval()
        self.evaluator.requires_grad_(False)
        self.weights = {view: {n: p.detach().clone() for n, p in model.named_parameters()}
                        for view in DECAYS}
        self.loader = calibration_loader(dataset)
        self.steps, self.epochs, self.best = 0, 0, {}

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        for view, decay in DECAYS.items():
            for name, p in model.named_parameters():
                average = self.weights[view][name]
                if self.steps == 1 or not p.requires_grad:
                    average.copy_(p)
                else:
                    # Difference form leaves unchanged/frozen parameters bit-identical.
                    average.add_(p.detach() - average, alpha=1-decay)

    @torch.no_grad()
    def evaluate(self, model, epoch, test_loader, test_epoch, prepare_batch, eval_seed):
        rows = {}
        with isolated_randomness():
            for view in VIEWS:
                # Fresh online buffers in every view; all then get identical train-only BN treatment.
                self.evaluator.load_state_dict(model.state_dict(), strict=True)
                if view in self.weights:
                    for name, p in self.evaluator.named_parameters():
                        p.copy_(self.weights[view][name])
                calibration = refresh_bn(self.evaluator, self.loader, prepare_batch)
                result = test_epoch(self.evaluator, test_loader, eval_policy='argmax', eval_seed=eval_seed)
                row = dict(epoch=epoch, updates=self.steps, view=view, bn_calibration=calibration, **result)
                with (self.root/(view+'.epochs.jsonl')).open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, sort_keys=True)+'\n')
                previous = self.best.get(view, {}).get('coef', -math.inf)
                if math.isfinite(result['coef']) and result['coef'] > previous:
                    torch.save(self.evaluator.state_dict(), str(self.root/(view+'.pth')))
                    (self.root/(view+'.selection.json')).write_text(json.dumps(row, indent=2, sort_keys=True), encoding='utf-8')
                    self.best[view] = row
                rows[view] = dict(coef=result['coef'], best=self.best.get(view, {}).get('coef'))
        self.epochs = epoch
        return rows

    def finish(self):
        if set(self.best) != set(VIEWS):
            raise RuntimeError('an EMA/BN view has no finite selected checkpoint; retain logs for analysis')
        report = dict(recipe=recipe(), updates=self.steps, epochs=self.epochs, best=self.best)
        (self.root/'ema_training.json').write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
        return report
