"""Optional CPU-only, full one-epoch train/reload smoke (not a server gate)."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import gc
import json
import os
from pathlib import Path
import sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', choices=('modern','july28'), required=True)
    p.add_argument('--root', required=True)
    args = p.parse_args()
    import torch
    import main as entry
    import pipeline
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError('local helper is CPU-only; use run_qmd_dual for server GPUs')
    original = pipeline.build_dataset
    def loaders(**kwargs):
        kwargs['num_workers'] = 0
        return original(**kwargs)
    pipeline.build_dataset = loaders
    methods = (('score_qmd','ordered_score_qmd','reliable_score_qmd','ordered_reliable_score_qmd')
               if args.protocol == 'modern' else ('ordered','ordered_score_qmd','ordered_sr_qmd'))
    root = Path(args.root).resolve()
    rows = []
    for action in ('Ball','TES'):
        for method in methods:
            out = root/args.protocol/action/'seed0'/method
            out.mkdir(parents=True,exist_ok=True)
            if any(out.iterdir()):
                raise ValueError('use a clean local diagnostic root')
            mode = ('off' if method == 'ordered' else 'score_rank_qmd' if method == 'ordered_sr_qmd'
                    else 'reliable_score_qmd' if 'reliable' in method else 'score_qmd')
            argv = ['main.py','--gpu','-1','--feats','2','--action',action,'--seed','0',
                    '--multi_modality','--epochs','1','--qmd_mode',mode,'--qmd_warmup','0',
                    '--qmd_rampup','0','--run_tag',method]
            if args.protocol == 'modern':
                argv += ['--eval_policy','argmax']
            if method.startswith('ordered'):
                argv += ['--ordinal_prototype_weight','.0001']
                if args.protocol == 'modern':
                    argv += ['--ordinal_prototype_target_mode','continuous','--quality_aux_rng_isolated']
            with (out/'train.log').open('w',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):
                sys.argv = argv + ['--save_final_model',str(out/'model.pth'),'--result_json',str(out/'train_result.json')]
                entry.main()
            with (out/'test.log').open('w',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):
                sys.argv = argv + ['--test','--ckpt_path',str(out/'model.pth'),'--result_json',str(out/'test_result.json')]
                if args.protocol == 'july28':
                    sys.argv += ['--replay_rng_path',str(out/'model.pth.eval_rng.pt')]
                entry.main()
            train = json.loads((out/'train_result.json').read_text())
            test = json.loads((out/'test_result.json').read_text())
            if abs(train['saved_model_srcc']*100-test['coef']) > 1e-6:
                raise AssertionError('checkpoint mismatch')
            row = dict(protocol=args.protocol,action=action,method=method,coef=test['coef'],
                       reload_match=True,diagnostic_only=True)
            if mode != 'off':
                qmd = json.loads((out/'qmd_training.json').read_text())
                row['qmd'] = {k:v for k,v in qmd.items() if k != 'epochs'}
            rows.append(row)
            print(json.dumps(row,sort_keys=True),flush=True)
            gc.collect()
    (root/(args.protocol+'_local_training.json')).write_text(json.dumps(rows,indent=2,sort_keys=True),encoding='utf-8')

if __name__ == '__main__':
    main()
