"""Real Ball two-epoch native July28 parity and GT/KD update equivalence.

The frozen v8 class, current both and current KD guard must be bit-identical.
GT-only must match the old mask_gt path; KD-only the old zero-GT path.
"""
import argparse
import gc
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import types

BASE = '4d95c157f0efd745d534d604d9502895bb586c76'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--kd_weights', type=float, nargs='*', choices=(.25, .5), default=[],
                        help='Also verify the v11 fixed pure-KD strengths against frozen zero-GT updates')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import numpy as np
    import torch
    import main as entry
    import pipeline
    from models.score_qmd import ScoreQMD
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for server parity')
    torch.set_num_threads(1)
    old = types.ModuleType('frozen_split_reference')
    code = subprocess.check_output(['git', 'show', BASE + ':models/score_qmd.py']).decode()
    exec(compile(code, BASE + '/models/score_qmd.py', 'exec'), old.__dict__)
    sys.argv = ['main.py', '--gpu', args.gpu, '--feats', '2', '--action', 'Ball', '--multi_modality',
                '--seed', '0', '--qmd_mode', 'score_qmd', '--ordinal_prototype_weight', '.0001']
    config = entry.parser_args()
    current, epochs, data_conf, model_conf, optim_conf, _ = entry.build_configs(config, entry.load_config('Ball', 2))

    def fingerprint(value):
        digest = hashlib.sha256()
        def visit(x):
            if torch.is_tensor(x):
                digest.update(str((x.dtype, tuple(x.shape))).encode())
                digest.update(x.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(x, dict):
                for k, v in x.items():
                    digest.update(str(k).encode()); visit(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    visit(v)
            else:
                digest.update(pickle.dumps(x, protocol=4))
        visit(value)
        return digest.hexdigest()

    pairs = [
        ('both', 'score_qmd', {}, {}),
        ('guard', 'score_qmd', {'grad_mode': 'kd'}, {'grad_mode': 'kd'}),
        ('gt', 'mask_gt', {}, {'terms': 'gt'}),
        ('kd', 'score_qmd', {'mask_gt_weight': 0}, {'terms': 'kd'}),
    ]
    pairs += [('kd_weight_' + str(w), 'score_qmd', {'mask_gt_weight': 0, 'weight': w},
               {'terms': 'kd', 'weight': w}) for w in args.kd_weights]
    results = {}
    for name, old_mode, old_kw, new_kw in pairs:
        pair = []
        for cls, mode, kw in ((old.ScoreQMD, old_mode, old_kw), (ScoreQMD, 'score_qmd', new_kw)):
            pipeline.setup_seed(0); random.seed(0)
            loader, test_loader = pipeline.build_dataset(**data_conf)
            del test_loader
            model = pipeline.build_model(**model_conf)
            initial = fingerprint(model.state_dict())
            optimizer = pipeline.build_optimizer(model, model_name=model_conf['model_name'], **optim_conf)
            model.to(torch.device('cuda'))
            qmd = cls(mode, 0, epochs, warmup=0, rampup=0, **kw)
            metrics = []
            for epoch in range(2):
                row = pipeline.train_epoch(model, optimizer, None, islice(loader, 3), epoch,
                                           clip_grad=current['clip_grad'], qmd=qmd)
                row.pop('time'); metrics.append(row)
            pair.append(dict(initial=initial, final=fingerprint(model.state_dict()),
                optimizer=fingerprint(optimizer.state_dict()),
                rng=fingerprint((torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state(), random.getstate())),
                metrics=metrics))
            del model, optimizer, qmd, loader
            gc.collect(); torch.cuda.empty_cache()
        results[name] = dict(passed=pair[0] == pair[1], frozen=pair[0], current=pair[1])
        print('PARITY ' + name + ' ' + str(results[name]['passed']), flush=True)
    passed = all(r['passed'] for r in results.values())
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(decision='QMD_SPLIT_PARITY_PASS' if passed else 'QMD_SPLIT_PARITY_FAIL',
        base=BASE, epochs=2, batches_per_epoch=3, results=results), indent=2), encoding='utf-8')
    if not passed:
        raise RuntimeError('frozen/default/split update parity failed; do not start screen')


if __name__ == '__main__':
    main()
