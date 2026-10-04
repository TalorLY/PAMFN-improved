"""Run one new control while keeping all v38 assets read-only."""
import argparse
from pathlib import Path
import time
import bootstrap
import torch
from adapter import seed_all
from contract import POLICY,reference,save,sha
from controls import control_engine

def execute(args,train=None,test=None):
    seed_all(args.seed);device=torch.device('cpu' if args.cpu else 'cuda')
    if train is None:
        from fs_data import loaders
        train,test=loaders(args.data,args.task,args.phase=='smoke',external=True,workers=args.workers)
    keys=['mcmoe']+(['mc_e'] if args.method=='mc_fused_u' else [])
    deps={m:reference(args.root,args.phase,args.task,args.seed,m) for m in keys}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);start=time.time()
    with control_engine(args.method) as engine:
        result=engine.branch_train(args,train,test,device,deps)
    result.update(task=args.task,seed=args.seed,method=args.method,phase=args.phase,
        ids=[r[0] for r in test.dataset.rows],wall_seconds=time.time()-start,
        dependencies={k:sha(v/'model.pth') for k,v in deps.items()},selection=POLICY['selection'],
        study_version=39,synthetic_cpu_test_only=args.cpu,engine_checkpoint_policy_version=38)
    save(out/'test_result.json',result)
    save(out/'control_contract.json',dict(policy=POLICY,method=args.method,source='byte-preserved v38 branch training engine'))
    print('V39_CONTROL_OK',args.method,result['coef'],flush=True)
    return result

def main():
    p=argparse.ArgumentParser()
    for name in ('root','phase','task','method','out','data'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',required=True,type=int);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--cpu',action='store_true');execute(p.parse_args())
if __name__=='__main__':main()
