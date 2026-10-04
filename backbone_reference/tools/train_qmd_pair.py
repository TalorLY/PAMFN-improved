"""v15: frozen-anchor pair repair and selective pair-residual distillation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch

from tools.qmd_residual import freeze_anchor,assert_frozen,state_digest
from models.score_qmd import rng_state
from tools.train_qmd_residual import (read,save,sha,load,tensor_save,verify_teacher,configurations,metrics,make_cache)
from tools.qmd_pair import PairBranch,PairObjective,pair_metrics
from tools.train_qmd_reskd import epoch_report as base_epoch_report
from tools.qmd_pair_config import RECIPES,LR,metadata


def statistics_digest(branch):
    h=hashlib.sha256()
    for name,value in sorted(branch.named_buffers()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate(branch,cache):
    with torch.no_grad():
        prediction=branch.evaluation_score(cache[0],cache[1])
        result=metrics(prediction,cache[2]);delta=branch(cache[1]).double()
        result.update(global_bias=float(branch.global_bias),correction_mean=float(delta.mean()),
            correction_std=float(delta.std(unbiased=False)),correction_max_abs=float(delta.abs().max()),
            correction_saturation=float((delta.abs()>=.95*branch.config['bound']).double().mean()))
        result.update(pair_metrics(cache[0],delta,cache[2]))
        return result


def epoch_report(epoch,rows):
    report=base_epoch_report(epoch,rows)
    for key in ('rank','rank_beta','rank_coefficient','rank_ratio','rank_norm','weighted_rank','repair_loss','protect_loss'):
        report[key+'_mean']=float(np.mean([r[key] for r in rows]))
    for key in ('reliable_pairs','hard_pairs','protected_pairs','unreachable_pairs','accepted_pairs','anchor_wrong_pairs',
                'corrected_wrong_pairs','repaired_pairs','broken_pairs'):
        report[key]=sum(r[key] for r in rows)
    report['gate_open_batches']=sum(r['accepted_pairs']>0 for r in rows)
    report['rank_ratio_max']=max(r['rank_ratio'] for r in rows)
    report['positive_rank_batches']=sum(r['rank_coefficient']*r['rank_norm']>0 for r in rows)
    report['weighted_rank_sum']=sum(r['weighted_rank'] for r in rows)
    rank_cos=[r['rank_cosine'] for r in rows if r['rank_cosine'] is not None]
    report['rank_conflicts']=sum(v<0 for v in rank_cos);report['rank_cosine_batches']=len(rank_cos)
    return report


def train_branch(anchor,branch,train_loader,cache,epochs,tag,seed,smoke,out,source,replay_path,statistics):
    import pipeline
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    frozen_before=state_digest(anchor);start_state=state_digest(branch);stats_before=statistics_digest(branch)
    baseline=metrics(cache[0],cache[2]);bias_only=metrics(cache[0].double()+branch.global_bias.double(),cache[2])
    initial=evaluate(branch,cache)
    if initial['predictions']!=bias_only['predictions'] or initial['correction_max_abs']!=0:
        raise RuntimeError('zero residual changed bias-only predictions')
    if (not all(math.isfinite(v) for v in (initial['coef'],baseline['coef'],source['expected_srcc_x100'])) or
        abs(initial['coef']-baseline['coef'])>1e-6 or abs(baseline['coef']-source['expected_srcc_x100'])>1e-6):
        raise RuntimeError('bias-only initialization did not preserve Ordered ranking')
    save(out/'training_statistics.json',dict(statistics,statistics_sha256=stats_before))
    save(out/'teacher_check.json',dict(source=source,baseline=baseline,bias_only=bias_only,
        zero_branch_exact=True,bias_only_rank_preserved=True,initial_branch_sha256=start_state,
        initial_anchor_sha256=frozen_before,training_statistics_sha256=stats_before,
        branch_parameters=sum(p.numel() for p in branch.parameters()),branch_config=branch.config))
    optimizer=torch.optim.AdamW(branch.parameters(),lr=LR,weight_decay=.01)
    objective=PairObjective(seed,tag,smoke);reports=[];history=[];best=-float('inf');selected=None
    device=next(branch.parameters()).device;started=time.time()
    for epoch in range(epochs):
        branch.train();assert_frozen(anchor);rows=[]
        for batch in train_loader:
            inputs,label=pipeline._prepare_batch(batch,device);optimizer.zero_grad()
            loss,info=objective.losses(anchor,branch,inputs,label,batch[4],epoch)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite complementary objective')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in branch.parameters()):
                raise FloatingPointError('nonfinite complementary gradients')
            optimizer.step();rows.append(info)
        if not rows:raise RuntimeError('empty training loader')
        assert_frozen(anchor)
        result=evaluate(branch,cache);compact={k:v for k,v in result.items() if k!='predictions'}
        history.append(dict(epoch=epoch+1,**compact))
        with (out/'model.pth.epochs.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(history[-1],sort_keys=True)+'\n')
        report=epoch_report(epoch+1,rows);reports.append(report)
        # Only trained epochs compete; never replace a losing cell with Ordered.
        if math.isfinite(result['coef']) and result['coef']>best:
            best=result['coef'];selected=epoch+1
            tensor_save(out/'model.pth',dict(branch=branch.state_dict(),config=branch.config,
                source=source,tag=tag,statistics=statistics,statistics_sha256=stats_before))
            shutil.copyfile(replay_path,out/'model.pth.eval_rng.pt')
            save(out/'model.pth.selection.json',dict(epoch=selected,**compact))
        print('Epoch[%d/%d] Coef=%.6f Best=%.6f GT=%.6f Rank=%.6f KD=%.6f RankRatio=%.5f KDRatio=%.5f GatePairs=%d/%d Repaired/Broken=%d/%d' % (
            epoch+1,epochs,result['coef'],best,report['gt_mean'],report['rank_mean'],report['kd_mean'],
            report['rank_ratio_mean'],report['kd_ratio_mean'],report['accepted_pairs'],report['reliable_pairs'],
            result['repaired_pairs'],result['broken_pairs']),flush=True)
    if selected is None:raise RuntimeError('all trained checkpoint SRCCs are undefined')
    frozen_after=state_digest(anchor);final_branch=state_digest(branch);stats_after=statistics_digest(branch)
    if frozen_before!=frozen_after:raise RuntimeError('Ordered weights/buffers changed')
    if stats_before!=stats_after:raise RuntimeError('fixed training statistics changed')
    if start_state==final_branch:raise RuntimeError('correction branch never updated')
    summary=dict(mode='selective_pair_correction',recipe=RECIPES[tag],epochs=reports,calibration=objective.calibration_summary(),
        active_batches=sum(r['active_batches'] for r in reports),
        weighted_kd_sum=sum(r['weighted_kd_sum'] for r in reports),
        positive_kd_gradient_batches=sum(r['positive_kd_gradient_batches'] for r in reports),
        conflicts=sum(r['conflict_batches'] for r in reports),cosine_batches=sum(r['cosine_batches'] for r in reports),
        anchor_unchanged=True,anchor_gradients_none=True,anchor_state_before=frozen_before,anchor_state_after=frozen_after,
        initial_branch_sha256=start_state,final_branch_sha256=final_branch,branch_updated=True,
        statistics_before=stats_before,statistics_after=stats_after,statistics_unchanged=True,
        training_seconds=time.time()-started,lr=LR,optimizer='AdamW',weight_decay=.01,
        evaluation='fixed original Ordered test RNG; double sum of anchor, bias and residual',
        epoch_zero_eligible=False,extra_stage_epochs=epochs,kd_first_epoch=1)
    summary.update(weighted_rank_sum=sum(r['weighted_rank_sum'] for r in reports),
        positive_rank_batches=sum(r['positive_rank_batches'] for r in reports),
        accepted_pairs=sum(r['accepted_pairs'] for r in reports),reliable_pairs=sum(r['reliable_pairs'] for r in reports))
    save(out/'qmd_training.json',summary)
    save(out/'last_epoch_result.json',dict(history[-1],selection='diagnostic only; not final selected model'))
    return dict(saved_model_srcc=best/100,best_srcc=best/100,selected_epoch=selected,source=source)


def parser_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',required=True)
    p.add_argument('--action',required=True,choices=('TES','PCS','Ball','Clubs','Hoop','Ribbon'))
    p.add_argument('--seed',required=True,type=int,choices=(0,1,2))
    p.add_argument('--run_tag',required=True,choices=tuple(RECIPES))
    p.add_argument('--teacher_model',required=True);p.add_argument('--teacher_rng',required=True)
    p.add_argument('--teacher_record',required=True)
    p.add_argument('--epochs',type=int);p.add_argument('--smoke',action='store_true')
    p.add_argument('--save_final_model',default='');p.add_argument('--result_json',required=True)
    p.add_argument('--test',action='store_true');p.add_argument('--ckpt_path',default='')
    p.add_argument('--replay_rng_path',default='')
    args=p.parse_args(argv)
    if args.smoke!=(args.epochs==2):p.error('smoke requires exactly 2 epochs')
    if args.epochs is not None and not args.smoke:p.error('formal budget overrides not allowed')
    if args.test and not (args.ckpt_path and args.replay_rng_path):p.error('test needs checkpoint and RNG')
    if not args.test and Path(args.save_final_model).name!='model.pth':p.error('output must be model.pth')
    return args


def main():
    args=parser_args();os.environ['CUDA_VISIBLE_DEVICES']=args.gpu if args.gpu!='-1' else ''
    if args.gpu!='-1' and not torch.cuda.is_available():raise RuntimeError('requested GPU unavailable')
    import pipeline
    source=verify_teacher(args);pipeline.setup_seed(args.seed)
    _,epochs,dataset_cfg,model_cfg,_,_=configurations(args)
    train_loader,test_loader=pipeline.build_dataset(**dataset_cfg)
    anchor=pipeline.build_model(**model_cfg)
    anchor.load_state_dict(pipeline._load_state_dict(args.teacher_model),strict=True)
    device=torch.device('cuda' if args.gpu!='-1' else 'cpu');freeze_anchor(anchor);anchor.to(device)
    branch=PairBranch(args.run_tag,seed=args.seed).to(device)
    if args.test:
        saved=load(args.ckpt_path)
        if saved['source']!=source or saved['config']!=branch.config or saved['tag']!=args.run_tag:
            raise ValueError('branch checkpoint/source/recipe mismatch')
        if sha(args.replay_rng_path)!=source['eval_rng_sha256']:raise ValueError('branch evaluation RNG mismatch')
        branch.load_state_dict(saved['branch'],strict=True)
        if statistics_digest(branch)!=saved['statistics_sha256']:raise ValueError('checkpoint statistics changed')
    else:
        # No test-loader access before train-only calibration. RNG is restored,
        # so this pass does not advance any method's training random trajectory.
        replay=(rng_state(),np.random.get_state())
        training_cache=make_cache(anchor,train_loader,device,replay)
        statistics=branch.fit_training_statistics(training_cache)
        del training_cache
    cache=make_cache(anchor,test_loader,device,load(args.teacher_rng))
    baseline=metrics(cache[0],cache[2])
    if not math.isfinite(baseline['coef']) or abs(baseline['coef']-source['expected_srcc_x100'])>1e-6:
        raise RuntimeError('Ordered replay mismatch')
    run=metadata(args.action,args.seed,args.run_tag,epochs,args.smoke)
    if args.test:
        result=dict(evaluate(branch,cache),run=run,source=source,
            evaluation=dict(policy=run['eval_policy'],replayed_rng=True),baseline_srcc_x100=baseline['coef'],
            training_statistics_sha256=statistics_digest(branch))
    else:
        out=Path(args.save_final_model).parent
        result=train_branch(anchor,branch,train_loader,cache,epochs,args.run_tag,args.seed,args.smoke,
            out,source,args.teacher_rng,statistics)
        result['run']=run
    save(args.result_json,result)
    print('PAIR_RESULT '+json.dumps({k:result[k] for k in ('coef','saved_model_srcc') if k in result}),flush=True)


if __name__=='__main__':main()
