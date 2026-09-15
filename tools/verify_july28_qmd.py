"""Full real-data epoch: frozen July28 Ordered versus QMD-off/weight-zero."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

COMMIT = '53716f61c117376154c9335b1a689bc667404ce2'


def frozen(path):
    module = types.ModuleType('frozen_july28_' + path.replace('/', '_'))
    source = subprocess.check_output(['git', 'show', COMMIT + ':' + path]).decode()
    exec(compile(source, COMMIT + '/' + path, 'exec'), module.__dict__)
    return module


def fingerprint(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--action', default='Ball', choices=('Ball','TES'))
    p.add_argument('--max_batches', type=int, default=0)
    p.add_argument('--base', action='store_true', help='Check original PAMFN MSE without Ordered')
    args = p.parse_args()
    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import numpy as np
    import torch
    import main as entry
    import pipeline
    from models.score_qmd import ScoreQMD
    torch.set_num_threads(1)
    if args.gpu != '-1' and not torch.cuda.is_available():
        raise RuntimeError('requested GPU unavailable')
    original = frozen('pipeline.py')
    original.AQADataset = frozen('datasets.py').AQADataset
    sys.argv = ['main.py','--gpu',args.gpu,'--feats','2','--action',args.action,
                '--seed','0','--multi_modality','--ordinal_prototype_weight','0' if args.base else '.0001']
    config = entry.parser_args()
    current, epochs, data, model_conf, optim_conf, sched_conf = entry.build_configs(config, entry.load_config(args.action,2))
    if os.name == 'nt':
        data['num_workers'] = 0
    results = {}
    for mode in ('frozen','off','zero'):
        implementation = original if mode == 'frozen' else pipeline
        pipeline.setup_seed(0)
        train, test = implementation.build_dataset(**data)
        model = pipeline.build_model(**model_conf)
        initial = fingerprint(model)
        optimizer = pipeline.build_optimizer(model, model_name=model_conf['model_name'], **optim_conf)
        scheduler = pipeline.build_lr_scheduler(optimizer, **sched_conf)
        model.to('cpu' if args.gpu == '-1' else 'cuda')
        loader = train
        if args.max_batches:
            from itertools import islice
            loader = islice(train, args.max_batches)
        extra = dict(qmd=ScoreQMD('score_qmd',0,epochs,weight=0)) if mode == 'zero' else {}
        metrics = implementation.train_epoch(model,optimizer,scheduler,loader,0,clip_grad=current['clip_grad'],**extra)
        metrics.pop('time')
        evaluation = implementation.test_epoch(model,test)
        evaluation = {key:evaluation[key] for key in ('coef','loss_x100')}
        state = torch.get_rng_state().numpy().tobytes()
        if torch.cuda.is_initialized():
            state += b''.join(x.cpu().numpy().tobytes() for x in torch.cuda.get_rng_state_all())
        results[mode] = dict(initial=initial, final=fingerprint(model), metrics=metrics, evaluation=evaluation,
                             rng=hashlib.sha256(state).hexdigest(), numpy_next=np.random.randint(0,1000000,size=4).tolist())
        print(mode, json.dumps(results[mode],sort_keys=True),flush=True)
        del model,optimizer,scheduler,train,test,loader
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    passed = results['frozen'] == results['off'] == results['zero']
    report = dict(decision='JULY28_PARITY_PASS' if passed else 'JULY28_PARITY_FAIL',
                  commit=COMMIT, action=args.action, base=args.base, full_epoch=args.max_batches==0,
                  device=args.gpu, results=results)
    path = Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
    if not passed:
        raise SystemExit('July28 disabled-path regression failed')


if __name__ == '__main__':
    main()
