"""Optional CPU real-data probe; server training uses the regular smoke runner."""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

from tools.run_sr_qmd_experiments import cell_command
from tools.summarize_sr_qmd import METHODS, validate_cell


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--action',choices=('Ball','TES'),default='Ball')
    args=p.parse_args()
    import torch
    import main as entry
    import pipeline
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError('this diagnostic is CPU-only; use regular server smoke for GPU')
    original=pipeline.build_dataset
    def build(**kwargs):
        kwargs['num_workers']=0
        return original(**kwargs)
    root=Path(args.root).resolve()
    for method in METHODS:
        out=root/args.action/'seed0'/method
        out.mkdir(parents=True,exist_ok=True)
        if any(out.iterdir()):
            raise ValueError('local probe output must be empty: '+str(out))
        argv=cell_command(args.action,0,method,'-1',True)[2:]
        with patch.object(pipeline,'build_dataset',side_effect=build):
            with patch.object(sys,'argv',argv+['--save_final_model',str(out/'model.pth'),
                                               '--result_json',str(out/'train_result.json')]):
                entry.main()
            with patch.object(sys,'argv',argv+['--test','--ckpt_path',str(out/'model.pth'),
                                               '--result_json',str(out/'test_result.json')]):
                entry.main()
        metrics=validate_cell(out,args.action,0,method,True)
        (out/'complete.json').write_text(json.dumps(dict(metrics=metrics,diagnostic_only=True),indent=2),encoding='utf-8')
        print('LOCAL_TRAIN_RELOAD_PASS '+method,flush=True)


if __name__=='__main__':
    main()
