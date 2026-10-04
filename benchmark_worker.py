"""Bitwise replay first, then serial, order-balanced paired inference timing."""
import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import time
import bootstrap
import numpy as np
import torch
from adapter import (freeze,load,frozen_view,preserved_rng,historical_inference,state_digest,seed_all)
from modules import EBranch,UBranch
from worker import load_anchor,make_cache,branch_evaluate,metrics,parity
from contract import reference,read,save,sha
from deployment import InferenceSession,e_score,u_delta

def gpu_processes(physical_gpu):
    try:
        uuid=subprocess.check_output(['nvidia-smi','-i',str(physical_gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        pids=[int(row.split(',')[1].strip()) for row in raw.splitlines() if row.split(',')[0].strip()==uuid]
        return dict(query_ok=True,physical_gpu=str(physical_gpu),uuid=uuid,other_pids=sorted(set(pids)-{os.getpid()}))
    except Exception as e:return dict(query_ok=False,physical_gpu=str(physical_gpu),error=repr(e),other_pids=None)

def exact(a,b,name):
    if a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a,b):
        raise ValueError('Bitwise inference mismatch: '+name)

def replay(model,e,u,test,device,paths):
    rng=load(paths['anchor']/'model.pth.eval_rng.pt')
    cache=make_cache(model,test,device,rng)
    parity(metrics(cache[0].cpu().tolist(),cache[3].cpu().tolist()),read(paths['anchor']/'test_result.json'))
    parity(branch_evaluate(cache,e),read(paths['e']/'test_result.json'))
    parity(branch_evaluate(cache,e,u),read(paths['u']/'test_result.json'))
    values=[[],[],[],[]];batch1_legacy=[];batch1_fast=[]
    with preserved_rng(rng),InferenceSession(model,e,u) as session:
        for v,a,f,y in test:
            xs=[x.to(device) for x in (v,a,f)]
            p,j,m=session.features(*xs)
            score=e_score(e,p,j)+u_delta(u,m).double()
            for dest,value in zip(values,(p,j,m,y.to(device).reshape(-1))):dest.append(value)
            # Same batch and same parent features: direct formula versus original branches.
            legacy=e.evaluation_score(p,j)+u(m).double()
            exact(score,legacy,'batch1 residual evaluation')
            batch1_legacy.extend(legacy.cpu().tolist());batch1_fast.extend(score.cpu().tolist())
    fast=tuple(torch.cat(v) for v in values)
    for i,(a,b) in enumerate(zip(cache,fast)):exact(a,b,'full-dataset feature channel %d'%i)
    with torch.no_grad():
        deployed=e_score(e,fast[0],fast[1])+u_delta(u,fast[2]).double()
        expected=e.evaluation_score(cache[0],cache[1])+u(cache[2]).double()
        exact(deployed,expected,'reference-batch full prediction')
    # Native no-hook and E-only session dispatch are checked too.
    xs=[x.to(device) for x in next(iter(test))[:3]]
    with historical_inference(model):p=model(*xs,[1,1,1])['output'].reshape(-1)
    with InferenceSession(model) as session:exact(session(*xs),p,'anchor deployment')
    legacy_p,j,_=frozen_view(model,*xs)
    with torch.no_grad():legacy_e=e.evaluation_score(legacy_p,j)
    with InferenceSession(model,e) as session:exact(session(*xs),legacy_e,'E-only deployment')
    batch1=metrics(batch1_fast,cache[3].cpu().tolist())
    return dict(status='PASS',samples=len(test.dataset),same_shape_bitwise=True,
        reference_batch_bitwise=True,all_features_bitwise=True,
        branch_evaluation_note='Reference scores use the preserved full-evaluation feature cache; batch1 timing uses original versus optimized batch1. Batch-shape roundoff is separately disclosed.',
        batch1_vs_reference_max_abs=float(np.max(np.abs(np.asarray(batch1_fast)-expected.cpu().numpy()))),
        batch1_coef=batch1['coef'],reference_coef=read(paths['u']/'test_result.json')['coef']),xs

def measure(model,e,u,xs,physical_gpu):
    device=xs[0].device
    if device.type!='cuda':return dict(status='not_measured',reason='CPU integration only')
    modes=('legacy_anchor','legacy_e','legacy_u','deploy_anchor','deploy_e','deploy_u')
    rows={m:dict(cuda_ms=[],wall_ms=[],peak_allocated_bytes=[],temporary_peak_bytes=[]) for m in modes}
    observations=[gpu_processes(physical_gpu)]
    # Each complete block changes the first timed variant; all variants share inputs/device.
    with preserved_rng(),torch.no_grad():
        for block in range(6):
            order=modes[block:]+modes[:block]
            for mode in order:
                deploy=mode.startswith('deploy');component=mode.split('_')[1]
                context=InferenceSession(model,e if component!='anchor' else None,u if component=='u' else None) if deploy else nullcontext()
                with context as session:
                    def forward():
                        if deploy:return session(*xs)
                        if component=='anchor':
                            with historical_inference(model):return model(*xs,[1,1,1])['output'].reshape(-1)
                        p,j,m=frozen_view(model,*xs);v=e.evaluation_score(p,j)
                        return v if component=='e' else v+u(m).double()
                    for _ in range(10):forward()
                    torch.cuda.synchronize();start_bytes=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    for _ in range(10):
                        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize();t=time.perf_counter();a.record();value=forward();b.record();torch.cuda.synchronize()
                        rows[mode]['wall_ms'].append((time.perf_counter()-t)*1000)
                        rows[mode]['cuda_ms'].append(a.elapsed_time(b))
                    rows[mode]['peak_allocated_bytes'].append(torch.cuda.max_memory_allocated())
                    rows[mode]['temporary_peak_bytes'].append(max(0,torch.cuda.max_memory_allocated()-start_bytes))
                    if not bool(torch.isfinite(value).all()):raise FloatingPointError('Nonfinite benchmark output')
            observations.append(gpu_processes(physical_gpu))
    exclusive=all(v['query_ok'] and not v['other_pids'] for v in observations)
    for row in rows.values():
        for key in ('cuda_ms','wall_ms'):row[key+'_median']=float(np.median(row[key]))
    return dict(status='measured',variants=rows,observed_exclusive=exclusive,process_observations=observations,
        device=torch.cuda.get_device_name(),torch=str(torch.__version__),physical_gpu=str(physical_gpu),
        scope='Cached input feature forward, batch1 x95 clips. No disk IO or raw-video encoders. Resident model/E/U are held for all variants.',
        repetitions=60,blocks=6,warmup_per_block=10,
        exclusivity_limit='No other compute PID observed before or between blocks; not a hardware reservation or guarantee against unobserved transient use.',
        session_setup_timed=False,session_setup='Persistent session setup is separately measured by caller; legacy variants retain per-call setup.')

def execute(args,test=None):
    seed_all(args.seed);device=torch.device('cpu' if args.cpu else 'cuda')
    phase=args.phase
    names=('mcmoe','mc_e','mc_u') if args.parent=='mcmoe' else ('mc_ordered','mc_ordered_e','mc_ordered_u')
    paths={k:reference(args.root,phase,args.task,args.seed,m) for k,m in zip(('anchor','e','u'),names)}
    hashes={k:sha(v/'model.pth') for k,v in paths.items()}
    if args.require_exclusive and not args.cpu:
        state=gpu_processes(args.physical_gpu)
        if not state['query_ok'] or state['other_pids']:raise RuntimeError('Exclusive benchmark requested; selected GPU is busy or process query unavailable: '+str(state))
    model=load_anchor(paths['anchor'],args.parent=='mc_ordered',args.seed,device)
    e=EBranch(256,args.seed).to(device);e.load_state_dict(load(paths['e']/'model.pth')['branch']);freeze(e)
    u=UBranch(768,args.seed).to(device);u.load_state_dict(load(paths['u']/'model.pth')['branch']);freeze(u)
    before=[state_digest(v) for v in (model,e,u)]
    if test is None:
        from fs_data import loaders
        _,test=loaders(args.data,args.task,phase=='smoke',external=True,workers=2)
    checked,xs=replay(model,e,u,test,device,paths)
    if device.type=='cuda':torch.cuda.synchronize()
    t=time.perf_counter()
    with InferenceSession(model,e,u):pass
    if device.type=='cuda':torch.cuda.synchronize()
    setup_ms=(time.perf_counter()-t)*1000
    cost=measure(model,e,u,xs,args.physical_gpu)
    if args.require_exclusive and not args.cpu and not cost.get('observed_exclusive'):
        raise RuntimeError('Another compute process appeared during required-exclusive benchmark; previous results retained')
    if before!=[state_digest(v) for v in (model,e,u)]:raise ValueError('Benchmark changed model state')
    if hashes!={k:sha(v/'model.pth') for k,v in paths.items()}:raise ValueError('Reference file changed')
    result=dict(task=args.task,seed=args.seed,parent=args.parent,phase=phase,parity=checked,cost=cost,
        dependencies=hashes,session_setup_wall_ms=setup_ms,parameters=dict(anchor=sum(p.numel() for p in model.backbone.parameters()),e=sum(p.numel() for p in e.parameters()),u=sum(p.numel() for p in u.parameters())),
        state_unchanged=True,synthetic_cpu_test_only=args.cpu)
    save(Path(args.out)/'benchmark.json',result)
    print('BENCHMARK_PARITY_PASS',args.task,args.seed,args.parent,'timing',cost['status'],flush=True)
    return result

def main():
    p=argparse.ArgumentParser()
    for name in ('root','phase','task','parent','data','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--physical-gpu',default='4')
    p.add_argument('--cpu',action='store_true');p.add_argument('--require-exclusive',action='store_true')
    execute(p.parse_args())
if __name__=='__main__':main()
