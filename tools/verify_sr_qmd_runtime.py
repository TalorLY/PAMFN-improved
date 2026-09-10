"""Real-data active Score-QMD v1 parity, including zero Rank-KD ablation."""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

from tools.verify_qmd_runtime import fingerprint

SCORE_COMMIT = 'd6b25b7f2bef1fe6e67e906046ea51cb3138896f'


def frozen_module(commit, path, name):
    module = types.ModuleType(name)
    code = subprocess.check_output(['git', 'show', commit+':'+path]).decode()
    exec(compile(code, commit+'/'+path, 'exec'), module.__dict__)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max_batches', type=int, default=0)
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
        raise RuntimeError('requested GPU is unavailable')
    sys.argv = ['main.py','--gpu',args.gpu,'--feats','2','--action','Ball','--multi_modality',
                '--eval_policy','argmax','--qmd_mode','score_qmd','--qmd_warmup','0','--qmd_rampup','0']
    config = entry.parser_args()
    current, epochs, data_conf, model_conf, optim_conf, sched_conf = entry.build_configs(config,entry.load_config('Ball',2))
    if os.name == 'nt':
        data_conf['num_workers'] = 0
    old_pipe = frozen_module(SCORE_COMMIT, 'pipeline.py', 'frozen_score_pipeline')
    old_qmd = frozen_module(SCORE_COMMIT, 'models/score_qmd.py', 'frozen_score_qmd')
    results = {}
    device = torch.device('cuda' if args.gpu != '-1' else 'cpu')
    for mode in ('v1_reference','current_score','zero_rank'):
        pipeline.setup_seed(0)
        loader, test_loader = pipeline.build_dataset(**data_conf)
        del test_loader
        model = pipeline.build_model(**model_conf)
        initial = fingerprint(model)
        optimizer = pipeline.build_optimizer(model, model_name=model_conf['model_name'], **optim_conf)
        model.to(device)
        implementation = old_pipe if mode == 'v1_reference' else pipeline
        cls = old_qmd.ScoreQMD if mode == 'v1_reference' else ScoreQMD
        extra = dict(rank_weight=0.) if mode == 'zero_rank' else {}
        qmd = cls('score_rank_qmd' if mode=='zero_rank' else 'score_qmd',0,epochs,warmup=0,rampup=0,**extra)
        data = loader
        if args.max_batches:
            from itertools import islice
            data = islice(loader,args.max_batches)
        metrics = implementation.train_epoch(model,optimizer,None,data,0,clip_grad=current['clip_grad'],qmd=qmd)
        metrics.pop('time')
        state = torch.get_rng_state().numpy().tobytes()
        if torch.cuda.is_initialized():
            state += b''.join(x.cpu().numpy().tobytes() for x in torch.cuda.get_rng_state_all())
        results[mode] = dict(initial=initial, final=fingerprint(model), metrics=metrics,
                             rng=hashlib.sha256(state).hexdigest(), calibration=qmd.calibration,
                             active_batches=qmd.active_batches, modalities=qmd.modality_counts,
                             numpy_next=np.random.randint(0,1000000,size=4).tolist())
        print(mode, json.dumps(results[mode],sort_keys=True),flush=True)
        del model, optimizer, loader, data, qmd
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    passed = results['v1_reference'] == results['current_score'] == results['zero_rank']
    report = dict(decision='SR_QMD_V1_PARITY_PASS' if passed else 'SR_QMD_V1_PARITY_FAIL',
                  score_commit=SCORE_COMMIT, full_epoch=args.max_batches==0, device=str(device), results=results)
    path = Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
    if not passed:
        raise SystemExit('active Score-QMD parity failed')


if __name__ == '__main__':
    main()
