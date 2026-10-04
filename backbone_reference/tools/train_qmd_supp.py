"""Train only a supplement to hash-verified frozen E; no KD, no test-time routing."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from tools.train_qmd_residual import read,save,sha,load,tensor_save,verify_teacher,configurations,metrics
from tools.qmd_residual import freeze_anchor,assert_frozen,state_digest
from tools.qmd_modal import ModalBranch,make_cache
from tools.train_qmd_modal import statistics_digest
from tools.qmd_pair import pair_metrics
from tools.qmd_supp import SupplementBranch,e_score,objective
from tools.qmd_supp_config import RECIPES,E_TAG,metadata
from models.score_qmd import rng_state


def load_e(args,ordered,device):
    record=read(args.e_record);row=record['metrics']
    if (row['protocol'],row['action'],row['seed'],row['method'])!=('july28',args.action,args.seed,E_TAG):
        raise ValueError('wrong frozen E action/seed/tag')
    if sha(args.e_model)!=record['checkpoint_sha256'] or record['eval_rng_sha256']!=ordered['eval_rng_sha256']:
        raise ValueError('E model or Ordered RNG hash mismatch')
    parent=Path(args.e_record).parent
    for name in ('test_result.json','training_statistics.json'):
        if sha(parent/name)!=record['files'][name]:raise ValueError('E reference file changed: '+name)
    saved=load(args.e_model);e=ModalBranch(E_TAG,seed=args.seed).to(device)
    if saved['source']!=ordered or saved['tag']!=E_TAG or saved['config']!=e.config:raise ValueError('E checkpoint metadata changed')
    e.load_state_dict(saved['branch'],strict=True);freeze_anchor(e)
    if statistics_digest(e)!=saved['statistics_sha256']:raise ValueError('E statistics mismatch')
    reference=read(parent/'test_result.json')
    if reference['source']!=ordered:raise ValueError('E trained on a different Ordered checkpoint')
    return e,dict(checkpoint_sha256=record['checkpoint_sha256'],eval_rng_sha256=record['eval_rng_sha256'],
        expected_srcc_x100=row['srcc_x100'],action=args.action,seed=args.seed),reference


def evaluate(e,branch,cache):
    with torch.no_grad():
        frozen=e_score(e,cache[0],cache[1]);delta=branch(cache[1]).double()
        result=metrics(frozen+delta,cache[2]);result.update(pair_metrics(frozen,delta,cache[2]))
        bound=float(branch.bound);gate=float(branch.gate)
        result.update(e_srcc_x100=metrics(frozen,cache[2])['coef'],gate=gate,bound=bound,
            correction_mean=float(delta.mean()),correction_std=float(delta.std(unbiased=False)),
            correction_max_abs=float(delta.abs().max()),
            relative_correction_std=float(delta.std(unbiased=False)/frozen.std(unbiased=False)) if frozen.std(unbiased=False)>0 else None,
            expert_metrics={},expert_predictions={})
        for i,key in enumerate(('V','F','A')):
            component=branch.components(cache[1])[:,i].double()
            data=metrics(frozen+component,cache[2]);result['expert_predictions'][key]=data.pop('predictions')
            data.update(pair_metrics(frozen,component,cache[2]));result['expert_metrics'][key]=data
        return result


def train_branch(anchor,e,branch,loader,cache,epochs,tag,seed,smoke,out,source,replay_path,statistics):
    import pipeline
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    frozen_before=dict(ordered=state_digest(anchor),e=state_digest(e))
    initial_state=state_digest(branch);stat_hash=statistics_digest(branch)
    baseline=metrics(e_score(e,cache[0],cache[1]),cache[2]);initial=evaluate(e,branch,cache)
    if initial['predictions']!=baseline['predictions'] or initial['correction_max_abs']!=0:raise RuntimeError('zero supplement changed E')
    save(out/'teacher_check.json',dict(source=source,baseline=baseline,ordered=metrics(cache[0],cache[2]),
        zero_branch_exact=True,initial_frozen_sha256=frozen_before,initial_branch_sha256=initial_state,
        initial_heads_sha256=state_digest(branch.heads),training_statistics_sha256=stat_hash,
        trainable_parameters=sum(p.numel() for p in branch.parameters()),branch_config=branch.config))
    save(out/'training_statistics.json',dict(statistics,statistics_sha256=stat_hash))
    optimizer=branch.optimizer();reports=[];best=-float('inf');selected=None;started=time.time()
    for epoch in range(epochs):
        branch.train();assert_frozen(anchor);assert_frozen(e);rows=[]
        for batch in loader:
            inputs,y=pipeline._prepare_batch(batch,next(branch.parameters()).device);optimizer.zero_grad()
            loss,info=objective(anchor,e,branch,inputs,y)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite supplement loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in branch.parameters()):raise FloatingPointError('nonfinite gradient')
            info['gradient_norm']=sum(float(p.grad.detach().double().square().sum()) for p in branch.parameters() if p.grad is not None)**.5
            optimizer.step();branch.project_gate();rows.append(info)
        if not rows:raise RuntimeError('empty training loader')
        assert_frozen(anchor);assert_frozen(e)
        report=dict(epoch=epoch+1,batches=len(rows),positive_gradient_batches=sum(v['gradient_norm']>0 for v in rows),
            **{k+'_mean':float(np.mean([v[k] for v in rows])) for k in rows[0]})
        reports.append(report);result=evaluate(e,branch,cache)
        if not math.isfinite(result['coef']):raise FloatingPointError('nonfinite evaluation SRCC')
        compact={k:v for k,v in result.items() if k not in ('predictions','expert_predictions')}
        with (out/'model.pth.epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(epoch=epoch+1,**compact),sort_keys=True)+'\n')
        # Only trained models compete; zero-output recovery must be learned, not test-selected.
        if result['coef']>best:
            best=result['coef'];selected=epoch+1
            tensor_save(out/'model.pth',dict(branch=branch.state_dict(),config=branch.config,source=source,
                tag=tag,statistics=statistics,statistics_sha256=stat_hash))
            shutil.copyfile(replay_path,out/'model.pth.eval_rng.pt')
            save(out/'model.pth.selection.json',dict(epoch=selected,**compact))
        print('Epoch[%d/%d] Coef=%.6f Best=%.6f E=%.6f GT=%.6f Gate=%.6f Bound=%.8f Repairs/Broken=%d/%d' %
            (epoch+1,epochs,result['coef'],best,baseline['coef'],report['centered_mse_mean'],result['gate'],
             result['bound'],result['repaired_pairs'],result['broken_pairs']),flush=True)
    after=dict(ordered=state_digest(anchor),e=state_digest(e))
    if after!=frozen_before or statistics_digest(branch)!=stat_hash:raise RuntimeError('frozen E/Ordered/statistics changed')
    final_state=state_digest(branch)
    save(out/'qmd_training.json',dict(mode='frozen_E_supplement',recipe=RECIPES[tag],epochs=reports,
        frozen_before=frozen_before,frozen_after=after,frozen_unchanged=True,frozen_gradients_none=True,
        initial_branch_sha256=initial_state,final_branch_sha256=final_state,branch_updated=initial_state!=final_state,
        statistics_before=stat_hash,statistics_after=statistics_digest(branch),statistics_unchanged=True,
        positive_gradient_batches=sum(v['positive_gradient_batches'] for v in reports),
        kd_weight=0.,weighted_kd_sum=0.,positive_kd_gradient_batches=0,rank_weight=0.,
        epoch_zero_eligible=False,extra_stage_epochs=epochs,training_seconds=time.time()-started))
    save(out/'last_epoch_result.json',dict(result,epoch=epochs,selection='last epoch diagnostic only'))
    return dict(source=source,saved_model_srcc=best/100,best_srcc=best/100,selected_epoch=selected)


def parser_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',required=True);p.add_argument('--action',required=True,choices=('TES','PCS','Ball','Clubs','Hoop','Ribbon'))
    p.add_argument('--seed',type=int,choices=(0,1,2),required=True);p.add_argument('--run_tag',choices=tuple(RECIPES),required=True)
    for key in ('teacher_model','teacher_rng','teacher_record','e_model','e_record','result_json'):p.add_argument('--'+key,required=True)
    p.add_argument('--epochs',type=int);p.add_argument('--smoke',action='store_true');p.add_argument('--save_final_model',default='')
    p.add_argument('--test',action='store_true');p.add_argument('--ckpt_path',default='');p.add_argument('--replay_rng_path',default='')
    args=p.parse_args(argv)
    if args.smoke!=(args.epochs==2) or (args.epochs is not None and not args.smoke):p.error('only smoke may override native epochs, exactly 2')
    if args.test and not (args.ckpt_path and args.replay_rng_path):p.error('test needs checkpoint and RNG')
    if not args.test and Path(args.save_final_model).name!='model.pth':p.error('output must be model.pth')
    return args


def main():
    args=parser_args();os.environ['CUDA_VISIBLE_DEVICES']=args.gpu if args.gpu!='-1' else ''
    if args.gpu!='-1' and not torch.cuda.is_available():raise RuntimeError('requested GPU unavailable')
    import pipeline
    ordered=verify_teacher(args);pipeline.setup_seed(args.seed)
    _,epochs,dataset_cfg,model_cfg,_,_=configurations(args)
    loader,test_loader=pipeline.build_dataset(**dataset_cfg);anchor=pipeline.build_model(**model_cfg)
    anchor.load_state_dict(pipeline._load_state_dict(args.teacher_model),strict=True)
    device=torch.device('cuda' if args.gpu!='-1' else 'cpu');freeze_anchor(anchor);anchor.to(device)
    e,e_source,e_reference=load_e(args,ordered,device);source=dict(ordered=ordered,e=e_source)
    branch=SupplementBranch(args.run_tag,seed=args.seed).to(device)
    if args.test:
        saved=load(args.ckpt_path)
        if saved['source']!=source or saved['tag']!=args.run_tag or saved['config']!=branch.config:raise ValueError('supplement checkpoint mismatch')
        if sha(args.replay_rng_path)!=ordered['eval_rng_sha256']:raise ValueError('replay RNG mismatch')
        branch.load_state_dict(saved['branch'],strict=True)
        if statistics_digest(branch)!=saved['statistics_sha256']:raise ValueError('supplement statistics changed')
    else:
        replay=(rng_state(),np.random.get_state())
        train_cache=make_cache(anchor,loader,device,replay)
        statistics=branch.fit_training_statistics(e_score(e,train_cache[0],train_cache[1]),train_cache[1])
        del train_cache
    cache=make_cache(anchor,test_loader,device,load(args.teacher_rng));baseline=metrics(e_score(e,cache[0],cache[1]),cache[2])
    if baseline['predictions']!=e_reference['predictions'] or abs(baseline['coef']-e_source['expected_srcc_x100'])>1e-6:
        raise RuntimeError('frozen E did not replay exactly; do not train a substitute')
    run=metadata(args.action,args.seed,args.run_tag,epochs,args.smoke)
    if args.test:
        result=dict(evaluate(e,branch,cache),run=run,source=source,evaluation=dict(policy=run['eval_policy'],replayed_rng=True),
            training_statistics_sha256=statistics_digest(branch))
    else:
        result=train_branch(anchor,e,branch,loader,cache,epochs,args.run_tag,args.seed,args.smoke,
            Path(args.save_final_model).parent,source,args.teacher_rng,statistics);result['run']=run
    save(args.result_json,result);print('SUPP_RESULT '+json.dumps({k:result[k] for k in ('coef','saved_model_srcc') if k in result}),flush=True)


if __name__=='__main__':main()
