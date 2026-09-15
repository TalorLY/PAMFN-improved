"""Real-data, two-epoch, active-QMD parity with EMA evaluation enabled/disabled."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys

from tools.verify_qmd_runtime import fingerprint


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='' if args.gpu=='-1' else args.gpu
    import numpy as np
    import torch
    import main as entry
    import pipeline
    from models.ema_shadow import EMAShadow, isolated_randomness
    from models.score_qmd import ScoreQMD
    torch.set_num_threads(1)
    if args.gpu!='-1' and not torch.cuda.is_available():
        raise RuntimeError('requested GPU unavailable')
    output=Path(args.output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    argv=['main.py','--gpu',args.gpu,'--feats','2','--action','Ball','--multi_modality',
          '--eval_policy','argmax','--qmd_mode','score_qmd','--qmd_warmup','0','--qmd_rampup','0']
    sys.argv=argv
    config=entry.parser_args()
    current,epochs,data_conf,model_conf,optim_conf,sched_conf=entry.build_configs(config,entry.load_config('Ball',2))
    if os.name=='nt':
        data_conf['num_workers']=0
    outcomes={}
    for enabled in (False,True):
        pipeline.setup_seed(0)
        random.seed(0)
        train,test=pipeline.build_dataset(**data_conf)
        model=pipeline.build_model(**model_conf)
        initial=fingerprint(model)
        optimizer=pipeline.build_optimizer(model,model_name=model_conf['model_name'],**optim_conf)
        model.to(torch.device('cpu' if args.gpu=='-1' else 'cuda'))
        qmd=ScoreQMD('score_qmd',0,epochs,warmup=0,rampup=0)
        ema=EMAShadow(model,train.dataset,output.parent/'parity_shadows/model.pth') if enabled else None
        metrics=[]
        for epoch in range(2):
            tr=pipeline.train_epoch(model,optimizer,None,train,epoch,clip_grad=current['clip_grad'],qmd=qmd,ema=ema)
            te=pipeline.test_epoch(model,test,eval_policy='argmax')
            if ema:
                ema.evaluate(model,epoch+1,test,pipeline.test_epoch,pipeline._prepare_batch,20260908)
            tr.pop('time');te.pop('time')
            metrics.append(dict(train=tr,test=te))
            print('EMA PARITY enabled=%s epoch=%d/2'%(enabled,epoch+1),flush=True)
        state=torch.get_rng_state().numpy().tobytes()
        if torch.cuda.is_initialized():
            state+=b''.join(s.cpu().numpy().tobytes() for s in torch.cuda.get_rng_state_all())
        outcomes[str(enabled)]=dict(initial=initial,final=fingerprint(model),metrics=metrics,qmd=qmd.summary(),
            rng=hashlib.sha256(state).hexdigest(),numpy_next=np.random.randint(0,1000000,size=4).tolist(),python_next=random.random())
        if ema:
            report=ema.finish()
            with isolated_randomness():
                for view,selected in report['best'].items():
                    ema.evaluator.load_state_dict(pipeline._load_state_dict(str(ema.root/(view+'.pth'))))
                    reload=pipeline.test_epoch(ema.evaluator,test,eval_policy='argmax')
                    for key in ('coef','loss_x100','prediction_std'):
                        if not np.isfinite(reload[key]) or abs(reload[key]-selected[key])>1e-6:
                            raise RuntimeError('EMA parity checkpoint reload mismatch')
        del model,optimizer,train,test,qmd,ema
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    passed=outcomes['False']==outcomes['True']
    output.write_text(json.dumps(dict(decision='QMD_EMA_PARITY_PASS' if passed else 'QMD_EMA_PARITY_FAIL',
        two_full_epochs=True,device='cpu' if args.gpu=='-1' else 'cuda',results=outcomes),indent=2,sort_keys=True),encoding='utf-8')
    if not passed:
        raise SystemExit('EMA changed the student trajectory')
    print('QMD_EMA_PARITY_PASS',flush=True)


if __name__=='__main__':
    main()
