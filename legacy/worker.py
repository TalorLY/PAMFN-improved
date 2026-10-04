"""One v38 stage. Native MCMoE training or frozen E/U; never writes v37 references."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
from scipy.stats import spearmanr
from adapter import (TransferModel,TransferObjective,seed_all,freeze,assert_frozen,frozen_view,
                     capture_rng,preserved_rng,state_digest,tensor_save,load,historical_inference)
from modules import EBranch,UBranch,e_objective,u_objective
from spec import (HERE,POLICY,read,save,sha,epochs,artifact,dependencies,anchor_method,RANGES)

def metrics(pred,label):
    p=np.asarray(pred,dtype=np.float64).reshape(-1); y=np.asarray(label,dtype=np.float64).reshape(-1)
    if p.shape!=y.shape or len(p)<2 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise FloatingPointError('Invalid or nonfinite evaluation values')
    if np.ptp(y)==0: raise ValueError('Evaluation labels are constant')
    coef=None if np.ptp(p)==0 else float(spearmanr(p,y)[0]*100)
    if coef is not None and not np.isfinite(coef): raise FloatingPointError('Undefined nonconstant SRCC')
    return dict(coef=coef,mse=float(np.mean((p-y)**2)),predictions=dict(prediction=p.tolist(),label=y.tolist()),
                valid_srcc=coef is not None,undefined_reason='constant finite prediction' if coef is None else None)

class MetricGuard:
    def __init__(self): self.epochs=[]; self.streak=0
    def observe(self,epoch,result):
        if result['coef'] is None:
            self.epochs.append(epoch); self.streak+=1
            if self.streak>10 or len(self.epochs)>20: raise FloatingPointError('Persistent constant prediction; not eligible for checkpoint selection')
            return False
        self.streak=0; return True
    def finish(self):
        if self.streak: raise FloatingPointError('Final epoch still has undefined SRCC; recovery not confirmed')
        return dict(undefined_epochs=self.epochs,recovered=self.streak==0,total_undefined=len(self.epochs))

def parity(result,reference):
    if result['predictions']!=reference['predictions'] or result['coef'] is None or abs(result['coef']-reference['coef'])>1e-8:
        a=np.asarray(result['predictions']['prediction']); b=np.asarray(reference['predictions']['prediction'])
        raise ValueError('Parent prediction replay mismatch (max_abs=%s); refuse substitute baseline'%
                         (float(np.max(np.abs(a-b))) if a.shape==b.shape else 'shape'))

def append_epoch(out,value):
    with (out/'epochs.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,allow_nan=False)+'\n')

def native_evaluate(model,loader,device):
    model.eval(); ps=[]; ys=[]
    with torch.no_grad():
        for v,a,f,y in loader:
            p=model(v.to(device),a.to(device),f.to(device),[1,1,1])['output']
            ps.extend(p.cpu().reshape(-1).tolist()); ys.extend(y.reshape(-1).tolist())
    return metrics(ps,ys)

def load_anchor(path,ordered,seed,device):
    saved=load(path/'model.pth')
    thresholds=saved['thresholds'] if ordered else None
    model=TransferModel(ordered,thresholds,seed,device.type=='cpu')
    if ordered:
        model.load_state_dict(saved['model'],strict=True)
    else:
        model.backbone.load_state_dict(saved,strict=True)
    return freeze(model.to(device))

def make_cache(model,loader,device,replay=None):
    values=[[],[],[],[]]
    with preserved_rng(replay):
        for v,a,f,y in loader:
            p,j,m=frozen_view(model,v.to(device),a.to(device),f.to(device))
            for store,value in zip(values,(p,j,m,y.to(device).reshape(-1))):store.append(value)
    return tuple(torch.cat(v) for v in values)

def branch_evaluate(cache,e,u=None):
    with torch.no_grad():
        corrected=e.evaluation_score(cache[0],cache[1])
        pred=corrected if u is None else corrected+u(cache[2]).double()
        result=metrics(pred.cpu().tolist(),cache[3].cpu().tolist())
        delta=e(cache[1]).double() if u is None else u(cache[2]).double()
        result.update(correction_mean=float(delta.mean()),correction_std=float(delta.std(unbiased=False)),
                      correction_max_abs=float(delta.abs().max()),global_bias=float(e.global_bias))
        return result

def main_train(args,train,test,device):
    out=Path(args.out); budget=epochs(args.method,args.task,args.phase=='smoke')
    thresholds=np.quantile([r[1] for r in train.dataset.rows],[.25,.5,.75]).tolist()
    if not np.all(np.diff(thresholds)>0):raise ValueError('Training quartiles are not distinct')
    model=TransferModel(True,thresholds,args.seed,device.type=='cpu').to(device)
    from train import train_epoch
    objective=TransferObjective(model)
    optim=torch.optim.Adam(model.parameters(),lr=.0001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=budget,eta_min=.000001)
    best=-float('inf'); selected=None; started=time.time(); guard=MetricGuard()
    for epoch in range(budget):
        loss,_=train_epoch(epoch,model,objective,train,optim,None,device,SimpleNamespace(score_range=RANGES[args.task]))
        scalar_loss=float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        if not np.isfinite(scalar_loss) or any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite native loss or model parameters')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite native gradients')
        scheduler.step(); replay=capture_rng(); result=native_evaluate(model,test,device)
        valid=guard.observe(epoch+1,result)
        append_epoch(out,dict(epoch=epoch+1,coef=result['coef'],valid_srcc=valid,loss=scalar_loss,
                             ordinal_loss_last_batch=objective.last_ordinal,eligible=valid))
        if valid and result['coef']>best:
            best=result['coef']; selected=epoch+1
            tensor_save(out/'model.pth',dict(model=model.state_dict(),thresholds=thresholds,method=args.method,policy_version=38))
            tensor_save(out/'model.pth.eval_rng.pt',replay)
            save(out/'selected_result.json',result);save(out/'selection.json',dict(epoch=selected,coef=best))
        print('Epoch[%d/%d] SRCC=%s selected=%s'%(epoch+1,budget,result['coef'],selected),flush=True)
    guard_result=guard.finish()
    if selected is None:raise FloatingPointError('No trained finite checkpoint')
    save(out/'last_epoch_result.json',result)
    model.load_state_dict(load(out/'model.pth')['model'],strict=True)
    with preserved_rng(load(out/'model.pth.eval_rng.pt')):final=native_evaluate(model,test,device)
    parity(final,read(out/'selected_result.json'))
    save(out/'numerical_guard.json',guard_result)
    final.update(selected_epoch=selected,epochs=budget,training_seconds=time.time()-started,
        parameters=sum(p.numel() for p in model.parameters()),total_parameters=sum(p.numel() for p in model.backbone.parameters()),
        training_only_parameters=sum(p.numel() for p in model.quality.parameters()),reloaded_exact=True)
    return final

def pair_counts(base,delta,y):
    b=base.detach().double().reshape(-1); d=delta.detach().double().reshape(-1); y=y.detach().double().reshape(-1)
    gap=y[:,None]-y[None,:]; valid=torch.triu(torch.ones_like(gap,dtype=torch.bool),diagonal=1)&(gap.abs()>.02)
    before=(b[:,None]-b[None,:])*gap.sign(); after=((b+d)[:,None]-(b+d)[None,:])*gap.sign()
    wrong=before<=0; now=after<=0
    return dict(reliable_pairs=int(valid.sum()),repaired_pairs=int((valid&wrong&~now).sum()),
                broken_pairs=int((valid&~wrong&now).sum()))

def mechanism(cache,e,u=None):
    with torch.no_grad():
        base=cache[0].double() if u is None else e.evaluation_score(cache[0],cache[1])
        delta=e(cache[1]).double() if u is None else u(cache[2]).double()
        bias=e.global_bias.double() if u is None else 0.
        variants={}
        for name,scale in [('zero',0.),('half',.5),('full',1.),('double',2.)]:
            value=delta*scale
            variants[name]=dict(metrics((base+bias+value).cpu().tolist(),cache[3].cpu().tolist()),**pair_counts(base,value,cache[3]))
        if u is not None:
            components=u.components(cache[2])
            for i,modality in enumerate(('V','F','A')):
                removed=components.clone();removed[:,i]=0
                value=removed.mean(1).double()
                variants['without_'+modality]=dict(metrics((base+value).cpu().tolist(),cache[3].cpu().tolist()),**pair_counts(base,value,cache[3]))
        return dict(kind='fixed checkpoint intervention; not a retrained model or selection candidate',
            label_gap=.02,variants=variants,global_bias=float(bias),sample_delta=delta.cpu().tolist())

def benchmark(model,e,u,test,device):
    if device.type!='cuda':return dict(status='not_measured',reason='CPU test only')
    xs=[x.to(device) for x in next(iter(test))[:3]]
    def forward(mode):
        if mode=='anchor':
            with historical_inference(model):return model(*xs,[1,1,1])['output']
        p,j,m=frozen_view(model,*xs);value=e.evaluation_score(p,j)
        return value if mode=='e' else value+u(m).double()
    rows={}
    with preserved_rng(),torch.no_grad():
        for mode in ('anchor','e','u'):
            for _ in range(10):forward(mode)
            torch.cuda.synchronize();values=[]
            for _ in range(40):
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record();forward(mode);b.record();torch.cuda.synchronize();values.append(a.elapsed_time(b))
            rows[mode]=dict(median_ms=float(np.median(values)),mean_ms=float(np.mean(values)),
                           repetitions_ms=values)
    return dict(status='measured',gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),
        scope='shared GPU, cached feature forward batch1/95clips; excludes extraction and IO',
        warmup=10,repetitions=40,variants=rows,exclusive_gpu=False)

def branch_train(args,train,test,device,deps):
    out=Path(args.out); parent=anchor_method(args.method); anchor_path=deps[parent]
    anchor=load_anchor(anchor_path,parent=='mc_ordered',args.seed,device)
    frozen_before=state_digest(anchor); budget=epochs(args.method,args.task,args.phase=='smoke')
    e=EBranch(input_dim=256,seed=args.seed).to(device); supplement=args.method.endswith('_u')
    if supplement:
        ekey='mc_ordered_e' if parent=='mc_ordered' else 'mc_e'
        e.load_state_dict(load(deps[ekey]/'model.pth')['branch'],strict=True);freeze(e)
        branch=UBranch(input_dim=768,seed=args.seed).to(device); e_before=state_digest(e)
    else:branch=e
    training_cache=make_cache(anchor,train,device)
    if supplement:
        with torch.no_grad():e_scores=e.evaluation_score(training_cache[0],training_cache[1])
        stats=branch.fit(e_scores,training_cache[2])
    else:stats=branch.fit(training_cache[0],training_cache[1],training_cache[3])
    del training_cache
    initial_state=state_digest(branch)
    buffers_before={k:v.detach().clone() for k,v in branch.named_buffers()}
    replay=load(anchor_path/'model.pth.eval_rng.pt')
    cache=make_cache(anchor,test,device,replay)
    native=metrics(cache[0].cpu().tolist(),cache[3].cpu().tolist())
    parity(native,read(anchor_path/'test_result.json'))
    if supplement:parity(branch_evaluate(cache,e),read(deps[ekey]/'test_result.json'))
    initial=branch_evaluate(cache,e,branch if supplement else None)
    zero_base=(e.evaluation_score(cache[0],cache[1]) if supplement else cache[0].double()+e.global_bias.double())
    if initial['predictions']['prediction']!=zero_base.cpu().tolist():raise ValueError('Zero branch changed parent')
    save(out/'teacher_check.json',dict(parent=parent,native=native,zero_branch_exact=True,initial=initial,
        baseline_predicted_exact=True,initial_frozen_sha256=frozen_before,epoch_zero_eligible=False))
    save(out/'training_statistics.json',stats)
    optimizer=branch.optimizer() if supplement else torch.optim.AdamW(branch.parameters(),lr=.0001,weight_decay=.01)
    best=-float('inf'); selected=None; started=time.time();guard=MetricGuard();positive=0
    for epoch in range(budget):
        branch.train();assert_frozen(anchor)
        if supplement:assert_frozen(e)
        losses=[]; norms=[]
        for v,a,f,y in train:
            p,j,m=frozen_view(anchor,v.to(device),a.to(device),f.to(device)); y=y.to(device)
            optimizer.zero_grad()
            if supplement:
                with torch.no_grad():frozen_e=e.evaluation_score(p,j)
                loss,detail=u_objective(branch,frozen_e,m,y)
            else:loss,detail=e_objective(branch,p,j,y)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite correction objective')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in branch.parameters()):
                raise FloatingPointError('Nonfinite correction gradients')
            norm=sum(float(p.grad.detach().double().square().sum()) for p in branch.parameters() if p.grad is not None)**.5
            optimizer.step(); losses.append(float(loss.detach()));norms.append(norm);positive+=int(norm>0)
        if not losses:raise ValueError('Empty training loader')
        branch.eval(); result=branch_evaluate(cache,e,branch if supplement else None)
        valid=guard.observe(epoch+1,result)
        append_epoch(out,dict(epoch=epoch+1,coef=result['coef'],loss=float(np.mean(losses)),
            gradient_norm_mean=float(np.mean(norms)),valid_srcc=valid,eligible=valid,
            correction_std=result['correction_std'],correction_max_abs=result['correction_max_abs']))
        if valid and result['coef']>best:
            best=result['coef'];selected=epoch+1
            tensor_save(out/'model.pth',dict(branch=branch.state_dict(),method=args.method,parent=parent,
                source_sha256=sha(anchor_path/'model.pth'),policy_version=38))
            save(out/'selected_result.json',result);save(out/'selection.json',dict(epoch=selected,coef=best))
        print('Epoch[%d/%d] SRCC=%s selected=%s'%(epoch+1,budget,result['coef'],selected),flush=True)
    guard_result=guard.finish()
    if selected is None:raise FloatingPointError('No valid correction checkpoint')
    elapsed=time.time()-started
    final_state=state_digest(branch)
    if positive==0 or initial_state==final_state:raise RuntimeError('Correction branch did not learn; no positive gradient/update')
    frozen_after=state_digest(anchor)
    if frozen_before!=frozen_after or (supplement and e_before!=state_digest(e)):raise RuntimeError('Frozen parent changed')
    if any(not torch.equal(v,dict(branch.named_buffers())[k]) for k,v in buffers_before.items()):raise RuntimeError('Training statistics changed')
    save(out/'last_epoch_result.json',result)
    branch.load_state_dict(load(out/'model.pth')['branch'],strict=True);branch.eval()
    final=branch_evaluate(cache,e,branch if supplement else None);parity(final,read(out/'selected_result.json'))
    clone=(UBranch(768,args.seed) if supplement else EBranch(256,args.seed)).to(device)
    clone.load_state_dict(load(out/'model.pth')['branch'],strict=True);clone.eval()
    if state_digest(clone)==initial_state:raise RuntimeError('Selected checkpoint still equals untrained branch')
    parity(branch_evaluate(cache,e,clone) if supplement else branch_evaluate(cache,clone),final)
    save(out/'qmd_training.json',dict(frozen_before=frozen_before,frozen_after=frozen_after,anchor_unchanged=True,
        statistics_unchanged=True,branch_updated=initial_state!=final_state,positive_gradient_batches=positive,
        epochs=budget,training_seconds=elapsed,kd_weight=0.,no_disk_feature_cache=True))
    save(out/'numerical_guard.json',guard_result)
    save(out/'mechanism.json',mechanism(cache,e,branch if supplement else None))
    if supplement:save(out/'cost.json',benchmark(anchor,e,branch,test,device))
    final.update(selected_epoch=selected,epochs=budget,training_seconds=elapsed,
        parameters=sum(p.numel() for p in branch.parameters()),
        total_parameters=sum(p.numel() for p in anchor.backbone.parameters())+sum(p.numel() for p in e.parameters())+
                         (sum(p.numel() for p in branch.parameters()) if supplement else 0),reloaded_exact=True)
    return final

def execute(args,train=None,test=None):
    seed_all(args.seed);device=torch.device('cpu' if args.cpu else 'cuda')
    if not args.cpu and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if train is None:
        from fs_data import loaders
        train,test=loaders(args.data,args.task,args.phase=='smoke',external=True,workers=args.workers)
    deps={m:artifact(args.root,args.phase,args.task,args.seed,m) for m in dependencies(args.method)}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);started=time.time()
    result=main_train(args,train,test,device) if args.method=='mc_ordered' else branch_train(args,train,test,device,deps)
    result.update(task=args.task,seed=args.seed,method=args.method,phase=args.phase,
        ids=[r[0] for r in test.dataset.rows],wall_seconds=time.time()-started,
        selection=POLICY['selection'],synthetic_cpu_test_only=args.cpu,
        dependencies={k:sha(v/'model.pth') for k,v in deps.items()})
    save(out/'test_result.json',result)
    print('TRANSFER_CELL_OK SRCC=%.8f'%result['coef'],flush=True)
    return result

def main():
    p=argparse.ArgumentParser()
    for k in ('root','phase','task','method','out','data'):p.add_argument('--'+k,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--cpu',action='store_true')
    execute(p.parse_args())

if __name__=='__main__':main()
