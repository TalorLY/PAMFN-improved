"""CPU-only real-data two-epoch train/reload checks, not server GPU authorization."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import gc
from pathlib import Path
import sys

from tools import run_qmd_cal as cal


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    args = p.parse_args()
    import torch
    import main as entry
    import pipeline
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError('CPU diagnostic only; run server --stage all for GPU smoke')
    original = pipeline.build_dataset
    def loaders(**kwargs):
        kwargs['num_workers'] = 0
        return original(**kwargs)
    pipeline.build_dataset = loaders
    root = Path(args.root).resolve()
    results = []
    for cell in cal.smoke_cells(cal.matrix()):
        out = cal.core.cell_path(root, cell)
        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            raise ValueError('use a new local diagnostic root')
        argv = cal.cell_args(cell, '-1', True)[2:]
        with (out/'train.log').open('w', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
            sys.argv = argv + ['--save_final_model', str(out/'model.pth'), '--result_json', str(out/'train_result.json')]
            entry.main()
        with (out/'test.log').open('w', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
            sys.argv = argv + ['--test', '--ckpt_path', str(out/'model.pth'), '--result_json', str(out/'test_result.json')]
            entry.main()
        row = cal.validate_cell(out, cell, True)
        if row['warnings']:
            raise RuntimeError(str(row))
        results.append(row)
        print('CPU TRAIN/RELOAD PASS:', row, flush=True)
        gc.collect()
    cal.core.save(root/'verification.json', dict(device='cpu', diagnostic_only=True, epochs=2, rows=results))


if __name__ == '__main__':
    main()
