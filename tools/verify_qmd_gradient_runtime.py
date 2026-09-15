"""Server-native frozen/off/observe parity on real Ball data, two x three batches."""
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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',choices=('modern','july28'),required=True)
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    import numpy as np
    import torch
    import main as entry
    import pipeline
    from models.score_qmd import ScoreQMD
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for server parity')
    torch.set_num_threads(1)
    base=('09a38c6009c17f5c1df3e25f5ad1d8a5aac8750f' if args.protocol=='modern'
          else 'ad13082a15b36041268f0eb9685978a0c87e346b')
    old=[]
    for path in ('pipeline.py','models/score_qmd.py'):
        module=types.ModuleType('frozen_'+path.replace('/','_'))
        code=subprocess.check_output(['git','show',base+':'+path]).decode()
        exec(compile(code,base+'/'+path,'exec'),module.__dict__)
        old.append(module)
    argv=['main.py','--gpu',args.gpu,'--feats','2','--action','Ball','--multi_modality',
          '--seed','0','--qmd_mode','score_qmd','--qmd_warmup','0','--qmd_rampup','0']
    if args.protocol=='modern': argv+=['--eval_policy','argmax']
    else: argv+=['--ordinal_prototype_weight','.0001','--ordinal_prototype_dim','128',
                 '--ordinal_prototype_temperature','.1','--ordinal_prototype_order_margin','.1','--ordinal_prototype_order_scale','.1']
    sys.argv=argv
    config=entry.parser_args()
    current,epochs,data_conf,model_conf,optim_conf,_=entry.build_configs(config,entry.load_config('Ball',2))

    def fingerprint(value):
        h=hashlib.sha256()
        def visit(x):
            if torch.is_tensor(x):
                h.update(str((x.dtype,tuple(x.shape))).encode())
                h.update(x.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(x,dict):
                for k,v in x.items(): h.update(str(k).encode()); visit(v)
            elif isinstance(x,(tuple,list)):
                for v in x: visit(v)
            else: h.update(pickle.dumps(x,protocol=4))
        visit(value)
        return h.hexdigest()
    results={}
    for mode in ('frozen','off','observe'):
        pipeline.setup_seed(0)
        random.seed(0)
        loader,test_loader=pipeline.build_dataset(**data_conf)
        del test_loader
        model=pipeline.build_model(**model_conf)
        initial=fingerprint(model.state_dict())
        optimizer=pipeline.build_optimizer(model,model_name=model_conf['model_name'],**optim_conf)
        model.to(torch.device('cuda'))
        cls=old[1].ScoreQMD if mode=='frozen' else ScoreQMD
        extra={} if mode=='frozen' else dict(grad_mode=mode)
        qmd=cls('score_qmd',0,epochs,warmup=0,rampup=0,**extra)
        metrics=[]
        for epoch in range(2):
            value=(old[0] if mode=='frozen' else pipeline).train_epoch(model,optimizer,None,islice(loader,3),epoch,
                                  clip_grad=current['clip_grad'],qmd=qmd)
            value.pop('time'); metrics.append(value)
        report=qmd.summary(); report.pop('gradient_guard',None)
        results[mode]=dict(initial=initial,final=fingerprint(model.state_dict()),optimizer=fingerprint(optimizer.state_dict()),
                          rng=fingerprint((torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state(),random.getstate())),
                          metrics=metrics,qmd=report)
        print('PARITY '+mode+' '+results[mode]['final'],flush=True)
        del model,optimizer,qmd,loader
        gc.collect(); torch.cuda.empty_cache()
    passed=all(v==results['frozen'] for v in results.values())
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(decision='QMD_GRAD_PARITY_PASS' if passed else 'QMD_GRAD_PARITY_FAIL',
                         protocol=args.protocol,base=base,epochs=2,batches_per_epoch=3,results=results),indent=2),encoding='utf-8')
    if not passed: raise RuntimeError('frozen/off/observe parity failed; do not run screen')


if __name__=='__main__': main()
