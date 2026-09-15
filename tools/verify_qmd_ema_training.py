"""Real-data CPU two-epoch main.py training and four independent checkpoint reloads."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import gc
from pathlib import Path
import sys

from tools import run_qmd_ema as runner


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    args=p.parse_args()
    import torch
    import main as entry
    import pipeline
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError('CPU diagnostic only; use server runner for GPU smoke')
    original=pipeline.build_dataset
    def loaders(**kwargs):
        kwargs['num_workers']=0
        return original(**kwargs)
    pipeline.build_dataset=loaders
    root=Path(args.root).resolve()
    rows=[]
    for cell in runner.smoke_cells(runner.matrix()):
        out=runner.core.cell_path(root,cell)
        out.mkdir(parents=True,exist_ok=True)
        if any(out.iterdir()):
            raise ValueError('use a new local diagnostic root')
        argv=runner.cell_args(cell,'-1',True)[2:]
        with (out/'train.log').open('w',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):
            sys.argv=argv+['--save_final_model',str(out/'model.pth'),'--result_json',str(out/'train_result.json')]
            entry.main()
        for view in ('online',)+runner.VIEWS:
            prefix='model' if view=='online' else view
            result='test_result.json' if view=='online' else view+'.test_result.json'
            with (out/(view+'.test.log')).open('w',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):
                sys.argv=argv+['--test','--ema_view',view,'--ckpt_path',str(out/(prefix+'.pth')),'--result_json',str(out/result)]
                entry.main()
        row=runner.validate_cell(out,cell,True)
        if row['warnings']:
            raise RuntimeError(str(row))
        rows.append(row)
        print('CPU EMA TRAIN/FOUR RELOADS PASS:',cell,flush=True)
        gc.collect()
    runner.core.save(root/'verification.json',dict(device='cpu',diagnostic_only=True,epochs=2,rows=rows))


if __name__=='__main__':
    main()
