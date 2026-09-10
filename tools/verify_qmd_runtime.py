"""Real-data first-epoch regression against the frozen Base implementation."""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import types
from pathlib import Path

BASE_COMMIT = 'a2544aa6b4c5ba19f97ac52a3c9f9e941d859393'


def fingerprint(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--action', default='Ball')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_batches', type=int, default=0,
                        help='diagnostic only; nonzero cannot authorize baseline reuse')
    args = parser.parse_args()
    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import sys
    import numpy as np
    import torch
    import main as entry
    import pipeline
    from models.score_qmd import ScoreQMD
    torch.set_num_threads(1)
    if args.gpu != '-1' and not torch.cuda.is_available():
        raise RuntimeError('requested GPU is unavailable')
    sys.argv = ['main.py', '--gpu', args.gpu, '--action', args.action, '--feats', '2',
                '--multi_modality', '--eval_policy', 'argmax']
    run_args = entry.parser_args()
    current, epochs, data_conf, model_conf, optim_conf, sched_conf = entry.build_configs(
        run_args, entry.load_config(args.action, 2))
    if os.name == 'nt' and args.max_batches:
        data_conf['num_workers'] = 0  # bounded CPU probe only; not the server reuse gate
    reference = types.ModuleType('frozen_qmd_pipeline')
    original_data = types.ModuleType('frozen_qmd_datasets')
    for filename, module in [('pipeline.py', reference), ('datasets.py', original_data)]:
        source = subprocess.check_output(['git', 'show', BASE_COMMIT + ':' + filename]).decode()
        exec(compile(source, BASE_COMMIT + '/' + filename, 'exec'), module.__dict__)
    reference.AQADataset = original_data.AQADataset
    results = {}
    device = torch.device('cuda' if args.gpu != '-1' else 'cpu')
    for mode in ('reference', 'off', 'zero'):
        implementation = reference if mode == 'reference' else pipeline
        pipeline.setup_seed(0)
        loader, test_loader = implementation.build_dataset(**data_conf)
        del test_loader
        model = pipeline.build_model(**model_conf)
        initial = fingerprint(model)
        optimizer = pipeline.build_optimizer(model, model_name=model_conf['model_name'], **optim_conf)
        scheduler = (None if sched_conf['scheduler'] == 'none'
                     else pipeline.build_lr_scheduler(optimizer, **sched_conf))
        model.to(device)
        data = loader
        if args.max_batches:
            from itertools import islice
            data = islice(loader, args.max_batches)
        kwargs = {'qmd': ScoreQMD('score_qmd', 0, epochs, weight=0)} if mode == 'zero' else {}
        metrics = implementation.train_epoch(model, optimizer, scheduler, data, 0,
                                               clip_grad=current['clip_grad'], **kwargs)
        metrics.pop('time')
        state = torch.get_rng_state().numpy().tobytes()
        if torch.cuda.is_initialized():
            for value in torch.cuda.get_rng_state_all():
                state += value.cpu().numpy().tobytes()
        results[mode] = dict(initial=initial, final=fingerprint(model), metrics=metrics,
                             rng=hashlib.sha256(state).hexdigest(),
                             numpy_next=np.random.randint(0, 1000000, size=4).tolist())
        print(mode, json.dumps(results[mode], sort_keys=True), flush=True)
        del model, optimizer, scheduler, loader, data
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    passed = results['reference'] == results['off'] == results['zero']
    report = dict(decision='QMD_BASE_REGRESSION_PASS' if passed else 'QMD_BASE_REGRESSION_FAIL',
                  full_epoch=args.max_batches == 0, base_commit=BASE_COMMIT,
                  action=args.action, seed=0, results=results,
                  device=str(device), torch_version=torch.__version__)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    if not passed:
        raise SystemExit('disabled QMD does not match frozen Base')


if __name__ == '__main__':
    main()
