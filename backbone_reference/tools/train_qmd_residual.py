"""Train only a lightweight residual branch on a hash-verified Ordered model."""
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
from scipy.stats import spearmanr

from tools.qmd_residual import (ResidualBranch, ResidualObjective, anchor_view,
    assert_frozen, freeze_anchor, preserved_rng, state_digest)

WEIGHTS = {'ordered_residual_gt': 0., 'ordered_residual_kd_w050': .5, 'ordered_residual_kd_w100': 1.}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def load(path):
    try: return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError: return torch.load(path, map_location='cpu')


def tensor_save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temp); temp.replace(path)


def verify_teacher(args):
    record = read(args.teacher_record)
    if record['checkpoint_sha256'] != sha(args.teacher_model) or record['eval_rng_sha256'] != sha(args.teacher_rng):
        raise ValueError('Ordered checkpoint/RNG hash mismatch: do not substitute another model')
    row = record['metrics']
    if (row['protocol'], row['action'], row['seed'], row['method']) != ('july28',args.action,args.seed,'ordered'):
        raise ValueError('wrong Ordered teacher action/seed/protocol')
    return dict(checkpoint_sha256=record['checkpoint_sha256'], eval_rng_sha256=record['eval_rng_sha256'],
                expected_srcc_x100=row['srcc_x100'], action=args.action, seed=args.seed)


def configurations(args):
    import main as original
    cfg = argparse.Namespace(action=args.action, feats=2, multi_modality=True, epochs=args.epochs,
        feat_dir='./data/features', qmd_mode='off', qmd_weight=0., test=False,
        shared_rank_weight=0.,shared_rank_margin=.02,shared_rank_temperature=.05,
        shared_list_weight=0.,shared_list_margin=.02,shared_list_temperature=.05,
        ordinal_prototype_weight=.0001, ordinal_prototype_dim=128,
        ordinal_prototype_temperature=.1,ordinal_prototype_order_margin=.1,ordinal_prototype_order_scale=.1)
    values=original.build_configs(cfg, original.load_config(args.action,2))
    values[2]['return_valid_lengths']=True
    return values


def metrics(prediction, labels):
    pred, y = [v.detach().cpu().double().numpy().reshape(-1) for v in (prediction,labels)]
    if not np.isfinite(pred).all() or not np.isfinite(y).all():
        raise FloatingPointError('nonfinite predicted scores/labels')
    return dict(coef=float(spearmanr(y,pred)[0])*100, loss_x100=float(np.mean((pred-y)**2))*100,
                prediction_std=float(pred.std()), prediction_std_ratio=float(pred.std()/y.std()) if y.std()>0 else 0.,
                predictions=dict(label=y.tolist(),prediction=pred.tolist(),order='native_test_loader'))


def make_cache(anchor, loader, device, replay):
    import pipeline
    scores, features, labels = [], [], []
    with preserved_rng(replay):
        for batch in loader:
            inputs,y=pipeline._prepare_batch(batch,device)
            score,feature=anchor_view(anchor,inputs)
            scores.append(score); features.append(feature); labels.append(y.reshape(-1))
    return torch.cat(scores),torch.cat(features),torch.cat(labels)


def evaluate(branch, cache):
    with torch.no_grad():
        corrected=branch.score(cache[0],cache[1])
        result=metrics(corrected,cache[2])
        delta=corrected-cache[0]
        result.update(correction_mean=float(delta.mean()),correction_std=float(delta.std(unbiased=False)),
                      correction_max_abs=float(delta.abs().max()))
        return result


def train_branch(anchor, branch, train_loader, cache, epochs, weight, seed, lr, smoke, out, source, replay_path):
    import pipeline
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    frozen_before=state_digest(anchor)
    start_state=state_digest(branch)
    initial=evaluate(branch,cache)
    baseline=metrics(cache[0],cache[2])
    if initial['predictions'] != baseline['predictions']:
        raise RuntimeError('zero branch changed Ordered predictions')
    if abs(baseline['coef']-source['expected_srcc_x100'])>1e-6:
        raise RuntimeError('Ordered replay score mismatch')
    save(out/'teacher_check.json',dict(source=source, baseline=baseline, zero_branch_exact=True,
        initial_branch_sha256=start_state, initial_anchor_sha256=frozen_before,
        branch_parameters=sum(p.numel() for p in branch.parameters()),
        anchor_parameters=sum(p.numel() for p in anchor.parameters()), branch_config=branch.config))
    # This is an additional second-stage budget, not a retraining of Ordered.
    optimizer=torch.optim.AdamW(branch.parameters(),lr=lr,weight_decay=.01)
    objective=ResidualObjective(seed,epochs,weight,smoke)
    best=-float('inf'); selected=None; history=[]; reports=[]
    device=next(branch.parameters()).device
    started=time.time()
    for epoch in range(epochs):
        branch.train(); assert_frozen(anchor)
        rows=[]
        for batch in train_loader:
            inputs,label=pipeline._prepare_batch(batch,device)
            optimizer.zero_grad()
            loss,info=objective.losses(anchor,branch,inputs,label,batch[4],epoch)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite branch objective')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in branch.parameters()):
                raise FloatingPointError('nonfinite branch optimizer gradient')
            optimizer.step(); rows.append(info)
        assert_frozen(anchor)
        result=evaluate(branch,cache)
        compact={k:v for k,v in result.items() if k!='predictions'}
        history.append(dict(epoch=epoch+1,**compact))
        with (out/'model.pth.epochs.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(history[-1],sort_keys=True)+'\n')
        active=[r for r in rows if r['active']]
        cos=[r['cosine'] for r in active if r['cosine'] is not None]
        report=dict(epoch=epoch+1,batches=len(rows),active_batches=len(active),
            gt_mean=float(np.mean([r['gt'] for r in rows])),kd_mean=float(np.mean([r['kd'] for r in rows])),
            weighted_kd_sum=sum(r['weighted_kd'] for r in rows),
            positive_kd_gradient_batches=sum(r['coefficient']*r['kd_norm']>0 for r in rows),
            conflict_batches=sum(v<0 for v in cos),cosine_batches=len(cos),
            cosine_mean=float(np.mean(cos)) if cos else None,
            gt_gradient_norm_mean=float(np.mean([r['gt_norm'] for r in active])) if active else 0.,
            kd_gradient_norm_mean=float(np.mean([r['kd_norm'] for r in active])) if active else 0.)
        report['by_modality']={}
        for modality in ('V','F','A'):
            group=[r for r in active if r['modality']==modality]
            valid=[r['cosine'] for r in group if r['cosine'] is not None]
            report['by_modality'][modality]=dict(batches=len(group),conflicts=sum(v<0 for v in valid),
                cosine_mean=float(np.mean(valid)) if valid else None,
                anchor_disagreement_mean=float(np.mean([r['anchor_disagreement'] for r in group])) if group else 0.)
        reports.append(report)
        # Epoch 0 is reported but NEVER offered as a test-score fallback.
        if math.isfinite(result['coef']) and result['coef'] > best:
            best=result['coef']; selected=epoch+1
            tensor_save(out/'model.pth',dict(branch=branch.state_dict(),config=branch.config,source=source))
            shutil.copyfile(replay_path,out/'model.pth.eval_rng.pt')
            save(out/'model.pth.selection.json',dict(epoch=selected,**compact))
        print('Epoch[%d/%d] Coef=%.6f Best=%.6f GT=%.6f KD=%.6f conflict=%d/%d' % (
            epoch+1,epochs,result['coef'],best,report['gt_mean'],report['kd_mean'],report['conflict_batches'],len(cos)),flush=True)
    if selected is None: raise RuntimeError('all trained checkpoints have undefined SRCC')
    frozen_after=state_digest(anchor); final_branch=state_digest(branch)
    if frozen_after!=frozen_before: raise RuntimeError('frozen Ordered state/buffers changed')
    if final_branch==start_state: raise RuntimeError('residual branch never updated')
    summary=dict(mode='residual_kd',weight=weight,epochs=reports,calibration=objective.calibration,
        active_batches=sum(r['active_batches'] for r in reports),
        weighted_kd_sum=sum(r['weighted_kd_sum'] for r in reports),
        positive_kd_gradient_batches=sum(r['positive_kd_gradient_batches'] for r in reports),
        conflicts=sum(r['conflict_batches'] for r in reports),cosine_batches=sum(r['cosine_batches'] for r in reports),
        anchor_unchanged=True,anchor_gradients_none=True,anchor_state_before=frozen_before,anchor_state_after=frozen_after,
        initial_branch_sha256=start_state,final_branch_sha256=final_branch,branch_updated=True,
        training_seconds=time.time()-started,lr=lr,optimizer='AdamW',weight_decay=.01,
        evaluation='fixed original Ordered checkpoint test RNG; cached detached anchor features',
        epoch_zero_eligible=False,extra_stage_epochs=epochs)
    save(out/'qmd_training.json',summary)
    return dict(saved_model_srcc=best/100,best_srcc=best/100,selected_epoch=selected,source=source)


def parser_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',required=True)
    p.add_argument('--action',required=True,choices=('TES','PCS','Ball','Clubs','Hoop','Ribbon'))
    p.add_argument('--seed',required=True,type=int,choices=(0,1,2))
    p.add_argument('--run_tag',required=True,choices=tuple(WEIGHTS))
    p.add_argument('--teacher_model',required=True); p.add_argument('--teacher_rng',required=True)
    p.add_argument('--teacher_record',required=True)
    p.add_argument('--epochs',type=int);p.add_argument('--smoke',action='store_true')
    p.add_argument('--save_final_model',default='');p.add_argument('--result_json',required=True)
    p.add_argument('--test',action='store_true');p.add_argument('--ckpt_path',default='')
    p.add_argument('--replay_rng_path',default='')
    args=p.parse_args(argv)
    if args.smoke != (args.epochs == 2): p.error('smoke requires exactly 2 epochs; formal uses native budgets')
    if args.epochs is not None and not args.smoke: p.error('formal budget overrides are not allowed')
    if args.test and not (args.ckpt_path and args.replay_rng_path): p.error('test needs checkpoint and replay RNG')
    if not args.test and Path(args.save_final_model).name!='model.pth': p.error('output must be named model.pth')
    return args


def main():
    args=parser_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu if args.gpu!='-1' else ''
    if args.gpu!='-1' and not torch.cuda.is_available(): raise RuntimeError('requested GPU unavailable')
    import pipeline
    source=verify_teacher(args)
    pipeline.setup_seed(args.seed)
    current,epochs,dataset_cfg,model_cfg,_,_=configurations(args)
    train_loader,test_loader=pipeline.build_dataset(**dataset_cfg)
    anchor=pipeline.build_model(**model_cfg)
    anchor.load_state_dict(pipeline._load_state_dict(args.teacher_model),strict=True)
    device=torch.device('cuda' if args.gpu!='-1' else 'cpu')
    freeze_anchor(anchor);anchor.to(device)
    cache=make_cache(anchor,test_loader,device,load(args.teacher_rng))
    baseline=metrics(cache[0],cache[2])
    if abs(baseline['coef']-source['expected_srcc_x100'])>1e-6:
        raise RuntimeError('Ordered replay differs from its frozen reference: %.9f vs %.9f' % (baseline['coef'],source['expected_srcc_x100']))
    branch=ResidualBranch(cache[1].shape[1],seed=args.seed).to(device)
    metadata=dict(action=args.action,seed=args.seed,tag=args.run_tag,epochs=epochs,feats=2,
        protocol='july28_frozen_residual',weight=WEIGHTS[args.run_tag],branch=branch.config,
        anchor='July28 Ordered frozen checkpoint',eval_policy='fixed_ordered_eval_rng',
        mask_ratio=.15,warmup=0. if args.smoke else .1,rampup=0. if args.smoke else .1,
        target_ratio=.075,beta_cap=1.,gt='full corrected score',kd_target='frozen full-input Ordered score')
    if args.test:
        saved=load(args.ckpt_path)
        if saved['source']!=source or saved['config']!=branch.config: raise ValueError('branch checkpoint/source mismatch')
        if sha(args.replay_rng_path)!=source['eval_rng_sha256']: raise ValueError('branch eval RNG mismatch')
        branch.load_state_dict(saved['branch'],strict=True)
        result=dict(evaluate(branch,cache),run=metadata,source=source,
            evaluation=dict(policy='fixed_ordered_eval_rng',replayed_rng=True),baseline_srcc_x100=baseline['coef'])
    else:
        out=Path(args.save_final_model).parent
        result=train_branch(anchor,branch,train_loader,cache,epochs,WEIGHTS[args.run_tag],args.seed,
            current['lr'],args.smoke,out,source,args.teacher_rng)
        result['run']=metadata
    save(args.result_json,result)
    print('RESIDUAL_RESULT '+json.dumps({k:result[k] for k in ('coef','saved_model_srcc') if k in result}),flush=True)


if __name__=='__main__': main()
