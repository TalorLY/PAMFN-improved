"""v14: six fixed methods x six actions x three seeds; no score filtering."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

from tools import run_qmd_residual as previous
from tools.reuse_qmd_gradient_v8 import copy_unchanged
from tools.qmd_complement_config import metadata as v13_metadata
from tools import run_qmd_dual as core
from tools import run_qmd_cal as cal
from tools import run_qmd_full as full
from tools import run_qmd_split as split
from tools import run_qmd_strength as strength
from tools.qmd_process import alive,identity
from tools.qmd_reskd_config import (RECIPES,GT,SCORE,RESIDUAL,ENERGY_GT,ENERGY_SCORE,ENERGY_RESIDUAL,
    V13_GT,V13_KD,LR,TARGET_RATIO,BETA_CAP,metadata,parameter_count,calibration_contract)

REPO=Path(__file__).resolve().parents[1]
BASE,ORDERED=previous.BASE,previous.ORDERED
REFERENCES=(BASE,ORDERED,V13_GT,V13_KD)
V13_COMMITS=dict(modern='d4e598fff278e350c1cb164213812d108f60aaa2',july28='dbb6959634a863f09b9c17c31a7705c45c172bfd')
METHODS=REFERENCES+tuple(RECIPES)
SEEDS=(0,1,2)
def reference_cells():return [('july28',a,s,m) for a in core.ACTIONS for s in SEEDS for m in REFERENCES]
resolve_teachers=previous.resolve_teachers
teacher_info=previous.teacher_info
unit_check=previous.unit_check


def validate_reference(out,cell):
    if cell[3] in (BASE,ORDERED):return previous.validate_reference(out,cell)
    if cell[3] not in (V13_GT,V13_KD):raise ValueError('unknown retained reference')
    import numpy as np
    from scipy.stats import spearmanr
    record=split.verify_record(out)
    train,test,check,q,stats=[core.read(out/n) for n in ('train_result.json','test_result.json',
        'teacher_check.json','qmd_training.json','training_statistics.json')]
    root=out.parents[4]
    teacher=next(t for t in core.read(root/'reference_teacher_manifest.json')['records']
        if (t['action'],t['seed'])==cell[1:3])
    source={k:teacher[k] for k in ('action','seed','checkpoint_sha256','eval_rng_sha256','expected_srcc_x100')}
    epochs=core.BUDGET[cell[1]];meta=v13_metadata(cell[1],cell[2],cell[3],epochs)
    if train['run']!=meta or test['run']!=meta or any(x['source']!=source for x in (train,test,check)):
        raise ValueError('retained v13 recipe/source changed')
    # Records retain original absolute paths; only the result-root prefix may move.
    args=core.read(out/'run_config.json')
    argv=args['argv'];position=argv.index('--teacher_record');teacher_record=argv[position+1].replace('\\','/')
    suffix='/reference/july28/%s/seed%d/ordered/complete.json' % cell[1:3]
    expected=['--action',cell[1],'--seed',str(cell[2]),'--run_tag',cell[3],
        '--teacher_model',teacher['model'],'--teacher_rng',teacher['rng'],'--teacher_record',argv[position+1]]
    if args['protocol']!='july28' or argv!=expected or not teacher_record.endswith(suffix):
        raise ValueError('retained v13 arguments changed')
    original=split.verify_record(core.cell_path(root/'reference',cell[:3]+(ORDERED,)))
    if any(original[k]!=source[k] for k in ('checkpoint_sha256','eval_rng_sha256')):
        raise ValueError('retained v13 anchor differs from Ordered')
    if record['eval_rng_sha256']!=source['eval_rng_sha256'] or test['evaluation']!=dict(policy=meta['eval_policy'],replayed_rng=True):
        raise ValueError('retained v13 replay changed')
    y,p=[np.asarray(test['predictions'][k]) for k in ('label','prediction')]
    if len(y)!=(100 if cell[1] in ('TES','PCS') else 50) or p.shape!=y.shape or not np.isfinite(p).all():
        raise ValueError('retained v13 predictions missing')
    score=float(spearmanr(y,p)[0])*100
    lines=[json.loads(v) for v in (out/'model.pth.epochs.jsonl').read_text().splitlines()]
    selection=core.read(out/'model.pth.selection.json');epoch=selection['epoch']
    if [v['epoch'] for v in lines]!=list(range(1,epochs+1)) or not 1<=epoch<=epochs:
        raise ValueError('retained v13 incomplete epochs')
    for value in (test['coef'],train['saved_model_srcc']*100,record['metrics']['srcc_x100'],selection['coef'],
        lines[epoch-1]['coef'],max(v['coef'] for v in lines if math.isfinite(v['coef']))):
        if not math.isfinite(value) or abs(score-value)>1e-6:raise ValueError('retained v13 score mismatch')
    if abs(np.mean((y-p)**2)*100-test['loss_x100'])>1e-6:raise ValueError('retained v13 MSE mismatch')
    if not q['anchor_unchanged'] or not q['anchor_gradients_none'] or not q['statistics_unchanged'] or not q['branch_updated']:
        raise ValueError('retained v13 frozen-state check failed')
    if q['anchor_state_before']!=q['anchor_state_after'] or q['anchor_state_before']!=check['initial_anchor_sha256']:
        raise ValueError('retained v13 anchor state changed')
    if (stats['source_split']!='train' or len({stats['statistics_sha256'],q['statistics_before'],q['statistics_after'],
        test['training_statistics_sha256'],check['training_statistics_sha256']})!=1):
        raise ValueError('retained v13 training statistics changed')
    if q['recipe']!=meta['recipe'] or q['epoch_zero_eligible'] or q['extra_stage_epochs']!=epochs:
        raise ValueError('retained v13 objective/budget mismatch')
    if (q['weighted_kd_sum']>0)!=(meta['recipe']['weight']>0):raise ValueError('retained v13 KD mismatch')
    return dict(record['metrics'],origin='frozen_v13_reference')


def checked_v13(reference,source,root):
    full.check_result_roots(root,(reference,))
    old,complete=[core.read(reference/n) for n in ('source.json','completion.json')]
    if (old['commits']!=V13_COMMITS or complete.get('source')!=old or not complete.get('experiments_complete')
        or complete.get('new_cells')!=108 or complete.get('reference_cells')!=72 or complete.get('total_rows')!=180):
        raise ValueError('requires completed frozen v13 results')
    if any(old[k]!=source[k] for k in ('inputs','environment')):raise ValueError('v13 data/environment mismatch')
    core.immutable(root/'reference_teacher_manifest.json',core.read(reference/'teacher_manifest.json'))
    for c in reference_cells():
        origin=core.cell_path(reference/('screen' if c[3] in (V13_GT,V13_KD) else 'reference'),c)
        record=split.verify_record(origin);dest=core.cell_path(root/'reference',c)
        for name in list(record['files'])+['complete.json']:copy_unchanged(origin/name,dest/name)
        for name in ('train.log','test.log'):
            if (origin/name).exists():copy_unchanged(origin/name,dest/name)
    rows=[validate_reference(core.cell_path(root/'reference',c),c) for c in reference_cells()]
    core.immutable(root/'reference_manifest.json',dict(source=old,records=rows,source_path=str(reference)))
    return rows


def validate_calibration(qmd,weight,smoke):
    c=qmd['calibration'];reports=qmd['epochs'];steps=sum(r['batches'] for r in reports)
    if any(c.get(k)!=v for k,v in calibration_contract().items()) or c['steps']!=steps:
        raise ValueError('per-batch calibration contract changed')
    for name in ('beta_min','beta_mean','beta_max'):
        if not math.isfinite(c[name]) or not 0<=c[name]<=BETA_CAP:raise ValueError('invalid calibration beta')
    if not c['beta_min']<=c['beta_mean']+1e-8<=c['beta_max']+1e-8:raise ValueError('calibration range mismatch')
    if abs(c['beta_mean']-sum(r['beta_mean']*r['batches'] for r in reports)/steps)>1e-8:
        raise ValueError('calibration mean mismatch')
    for key in ('zero_gradient_batches','capped_batches'):
        if c[key]!=sum(r[key] for r in reports) or not 0<=c[key]<=steps:raise ValueError('calibration counts mismatch')
    positive=sum(r['positive_kd_gradient_batches'] for r in reports)
    if qmd['positive_kd_gradient_batches']!=positive:raise ValueError('positive KD batch count mismatch')
    if (c['first_positive'] is not None)!=(positive>0):raise ValueError('first effective KD mismatch')
    if c['first_positive'] is not None:
        first=c['first_positive'];e=first['epoch'];prior=sum(r['batches'] for r in reports if r['epoch']<e)
        if not 1<=e<=len(reports) or not prior<first['step']<=prior+reports[e-1]['batches']:
            raise ValueError('first effective KD outside history')
    maxima=[]
    for r in reports:
        ceiling=weight*TARGET_RATIO*min(1.,r['epoch']/(1 if smoke else 5))
        if not 0<=r['kd_ratio_mean']<=r['kd_ratio_max']+1e-8 or r['kd_ratio_max']>ceiling+1e-8:
            raise ValueError('KD gradient exceeded bounded ratio')
        if not 0<=r['beta_min']<=r['beta_mean']+1e-8<=r['beta_max']+1e-8 or r['beta_max']>BETA_CAP:
            raise ValueError('epoch calibration range mismatch')
        energy=qmd['recipe']['energy']*r['energy_mean']
        if abs(r['weighted_energy_mean']-energy)>1e-8 or abs(r['supervised_mean']-r['gt_mean']-energy)>1e-7:
            raise ValueError('energy objective mismatch')
        if not all(math.isfinite(r[k]) and r[k]>=0 for k in ('energy_mean','weighted_energy_mean','kd_ratio_max','supervised_mean')):
            raise ValueError('nonfinite energy or ratio')
        maxima.append(r['kd_ratio_max'])
    if abs(c['max_effective_ratio']-max(maxima))>1e-8:raise ValueError('maximum KD ratio mismatch')


def matrix():return [('july28',a,s,m) for a in core.ACTIONS for s in SEEDS for m in RECIPES]
def smoke_cells(jobs):return [c for c in matrix() if c[2]==0] if jobs else []


def cell_args(root,cell,gpu,smoke=False):
    if cell not in matrix():raise ValueError('cell outside fixed v14 grid')
    teacher=teacher_info(root,cell)
    argv=[os.sys.executable,'-u','tools/train_qmd_reskd.py','--gpu',str(gpu),
        '--action',cell[1],'--seed',str(cell[2]),'--run_tag',cell[3],
        '--teacher_model',teacher['model'],'--teacher_rng',teacher['rng'],
        '--teacher_record',str(core.cell_path(root/'reference',cell[:3]+(ORDERED,))/'complete.json')]
    if smoke:argv+=['--smoke','--epochs','2']
    return argv


def validate_cell(out,cell,smoke=False):
    import numpy as np
    from scipy.stats import spearmanr
    root=out.parents[4];epochs=2 if smoke else core.BUDGET[cell[1]]
    if core.read(out/'run_config.json')!=dict(argv=cell_args(root,cell,'0',smoke)[5:],protocol=cell[0]):
        raise ValueError('v14 arguments changed')
    train,test,check,qmd,stats=[core.read(out/n) for n in
        ('train_result.json','test_result.json','teacher_check.json','qmd_training.json','training_statistics.json')]
    teacher=teacher_info(root,cell)
    source={k:teacher[k] for k in ('checkpoint_sha256','eval_rng_sha256','expected_srcc_x100','action','seed')}
    expected=metadata(cell[1],cell[2],cell[3],epochs,smoke)
    if any(d['run']!=expected or d['source']!=source for d in (train,test)):
        raise ValueError('v14 recipe/source mismatch')
    if check['source']!=source or not check['zero_branch_exact'] or not check['bias_only_rank_preserved']:
        raise ValueError('zero residual or anchor parity failed')
    if check['branch_config']!=expected['branch'] or check['branch_parameters']!=parameter_count(expected['branch']):
        raise ValueError('unexpected low-capacity architecture')
    if stats['source_split']!='train' or stats['samples']<2 or stats['passes']!=1 or stats['feature_dim']!=768:
        raise ValueError('statistics must come only from training data')
    if not math.isfinite(stats['global_bias']) or abs(stats['global_bias']-(stats['mean_label']-stats['mean_anchor']))>1e-6:
        raise ValueError('invalid training-only bias')
    if not (.009999<=stats['feature_scale_min']<=stats['feature_scale_max']<float('inf')):
        raise ValueError('invalid fixed feature normalization')
    if not qmd['statistics_unchanged'] or len({stats['statistics_sha256'],qmd['statistics_before'],qmd['statistics_after'],
        check['training_statistics_sha256'],test['training_statistics_sha256']})!=1:
        raise ValueError('training statistics changed or were refitted on test')
    baseline=check['baseline']['coef']
    if abs(baseline-source['expected_srcc_x100'])>1e-6 or abs(test['baseline_srcc_x100']-baseline)>1e-6:
        raise ValueError('original Ordered score not recovered')
    if abs(check['bias_only']['coef']-baseline)>1e-6:raise ValueError('global offset changed anchor ranking')
    if test['evaluation']!=dict(policy=expected['eval_policy'],replayed_rng=True):raise ValueError('fixed eval replay missing')
    for data in (test,check['baseline'],check['bias_only']):
        y,p=[np.asarray(data['predictions'][k]) for k in ('label','prediction')]
        n=100 if cell[1] in ('TES','PCS') else 50
        if y.shape!=(n,) or p.shape!=y.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError('missing/nonfinite predictions')
        coef=spearmanr(y,p)[0]*100
        if not math.isfinite(coef) or abs(coef-data['coef'])>1e-6 or abs(np.mean((y-p)**2)*100-data['loss_x100'])>1e-6:
            raise ValueError('prediction metrics disagree')
        if abs(np.std(p)-data['prediction_std'])>1e-8:raise ValueError('prediction spread mismatch')
        if data['predictions']['label']!=check['baseline']['predictions']['label']:raise ValueError('test label order changed')
    base=np.asarray(check['baseline']['predictions']['prediction'])
    corrected=np.asarray(test['predictions']['prediction'])
    if not np.allclose(np.asarray(check['bias_only']['predictions']['prediction']),base+stats['global_bias'],rtol=0,atol=1e-12):
        raise ValueError('bias-only prediction mismatch')
    delta=corrected-base-stats['global_bias'];bound=expected['branch']['bound']
    if abs(test['global_bias']-stats['global_bias'])>1e-8 or abs(delta).max()>bound+1e-7:
        raise ValueError('bias or sample correction exceeded contract')
    for key,value in [('correction_mean',delta.mean()),('correction_std',delta.std()),('correction_max_abs',abs(delta).max())]:
        if abs(test[key]-value)>1e-8:raise ValueError('correction diagnostic mismatch')
    lines=[json.loads(v) for v in (out/'model.pth.epochs.jsonl').read_text().splitlines()]
    selected=core.read(out/'model.pth.selection.json');score=test['coef'];epoch=selected['epoch']
    if [v['epoch'] for v in lines]!=list(range(1,epochs+1)):raise ValueError('incomplete/duplicate epochs')
    if not 1<=epoch<=epochs:raise ValueError('epoch zero cannot be selected')
    for value in (train['saved_model_srcc']*100,selected['coef'],lines[epoch-1]['coef'],max(v['coef'] for v in lines if math.isfinite(v['coef']))):
        if not math.isfinite(value) or abs(value-score)>1e-6:raise ValueError('checkpoint selection/reload mismatch')
    if (not qmd['anchor_unchanged'] or not qmd['anchor_gradients_none'] or
        qmd['anchor_state_before']!=qmd['anchor_state_after'] or qmd['anchor_state_before']!=check['initial_anchor_sha256']):
        raise ValueError('Ordered state or gradients changed')
    if not qmd['branch_updated'] or qmd['initial_branch_sha256']==qmd['final_branch_sha256']:
        raise ValueError('branch never updated')
    if qmd['initial_branch_sha256']!=check['initial_branch_sha256']:raise ValueError('initialization changed')
    if qmd['mode']!='residual_space_kd' or qmd['recipe']!=RECIPES[cell[3]] or qmd['epoch_zero_eligible']:
        raise ValueError('wrong objective or artificial baseline fallback')
    if qmd['lr']!=LR or qmd['weight_decay']!=.01 or qmd['optimizer']!='AdamW':raise ValueError('optimizer changed')
    if [v['epoch'] for v in qmd['epochs']]!=list(range(1,epochs+1)) or qmd['extra_stage_epochs']!=epochs:
        raise ValueError('incomplete auxiliary history')
    if qmd['kd_first_epoch']!=1:raise ValueError('KD must be available in epoch one')
    if qmd['active_batches']!=sum(v['active_batches'] for v in qmd['epochs']) or qmd['active_batches']<=0:
        raise ValueError('KD diagnostics never activated')
    weighted=qmd['weighted_kd_sum'];weight=RECIPES[cell[3]]['weight']
    if not math.isfinite(weighted) or abs(weighted-sum(v['weighted_kd_sum'] for v in qmd['epochs']))>1e-8:
        raise ValueError('KD loss summary mismatch')
    if weight==0 and (weighted!=0 or qmd['positive_kd_gradient_batches']!=0):raise ValueError('GT-only contains KD update')
    calibration=qmd['calibration'];beta=calibration['beta_mean']
    validate_calibration(qmd,weight,smoke)
    warnings=[]
    if weight>0 and (weighted<=0 or qmd['positive_kd_gradient_batches']<=0):warnings.append('no_effective_branch_kd')
    last=core.read(out/'last_epoch_result.json')
    for key,value in lines[-1].items():
        same_nan=isinstance(value,float) and math.isnan(value) and isinstance(last.get(key),float) and math.isnan(last[key])
        if last.get(key)!=value and not same_nan:raise ValueError('last-epoch diagnostics mismatch')
    return dict(protocol=cell[0],action=cell[1],seed=cell[2],method=cell[3],srcc_x100=score,
        mse=test['loss_x100']/100,selected_epoch=epoch,prediction_std=test['prediction_std'],
        nan_eval_epochs=sum(not math.isfinite(v['coef']) for v in lines),active_batches=qmd['active_batches'],
        beta=beta,weight=weight,target=RECIPES[cell[3]]['target'],energy_weight=RECIPES[cell[3]]['energy'],
        max_kd_ratio=calibration['max_effective_ratio'],first_positive_kd=calibration['first_positive'],
        branch_conflict_rate=qmd['conflicts']/max(1,qmd['cosine_batches']),anchor_unchanged=True,
        branch_parameters=check['branch_parameters'],selected_before_aux=bool(weight and
            (calibration['first_positive'] is None or epoch<calibration['first_positive']['epoch'])),extra_stage_epochs=epochs,
        weighted_kd_sum=weighted,global_bias=stats['global_bias'],correction_std=test['correction_std'],
        correction_saturation=test['correction_saturation'],last_srcc_x100=last['coef'],
        last_correction_std=last['correction_std'],warnings=warnings)


def run_cell(root,repos,cell,gpu,smoke):
    return core.run_cell(root,repos,cell,gpu,smoke,argv=cell_args(root.parent,cell,gpu,smoke),
        validator=validate_cell,extra_required=('teacher_check.json','training_statistics.json','last_epoch_result.json'))


def run_wave(root,repos,jobs,gpus,smoke=False):
    return core.run_wave(root,repos,jobs,gpus,smoke,runner=run_cell,
        ordering=lambda cells:sorted(cells,key=lambda c:(-core.BUDGET[c[1]],c[2],c[3])))


def check_pairing(root,seeds=SEEDS):
    for a in core.ACTIONS:
        for s in seeds:
            paths=[core.cell_path(root,('july28',a,s,m)) for m in RECIPES]
            checks=[core.read(p/'teacher_check.json') for p in paths]
            stats=[core.read(p/'training_statistics.json') for p in paths]
            if any(v!=stats[0] for v in stats[1:]):raise ValueError('paired training statistics differ')
            if any(c['baseline']!=checks[0]['baseline'] or c['bias_only']!=checks[0]['bias_only'] for c in checks[1:]):
                raise ValueError('paired Ordered evaluation differs')
            if len({c['initial_branch_sha256'] for c in checks})!=1:
                raise ValueError('matched-capacity initialization differs')


def summarize(root):
    rows=[validate_reference(core.cell_path(root/'reference',c),c) for c in reference_cells()]
    for c in matrix():
        out=core.cell_path(root/'screen',c);record=split.verify_record(out);row=validate_cell(out,c)
        if abs(row['srcc_x100']-record['metrics']['srcc_x100'])>1e-6:raise ValueError('recorded result changed')
        rows.append(dict(row,origin='new_v14_residual_space_kd'))
    check_pairing(root/'screen')
    result=full.aggregate(rows,methods=METHODS,controls=((BASE,'base'),(ORDERED,'ordered'),
        (GT,'gt'),(ENERGY_GT,'energy_gt'),(SCORE,'score_kd'),(RESIDUAL,'residual_kd'),
        (V13_GT,'v13_gt'),(V13_KD,'v13_kd')))
    result['note']=('Six actions x three seeds; 108 new two-stage trainings and 72 frozen references. '
        'Bias/feature statistics use training data only; original Ordered remains frozen. '
        'MLP16 12320 parameters; energy 0/1 x off/score/residual KD. Matched energy GT controls.  '
        'Test-best retained; extra stage budget; no epoch-zero or per-action/seed fallback. '
        'Float64 evaluation accumulation avoids tie changes from global bias addition.')
    result['v13_gt_replay']=[]
    for a in core.ACTIONS:
        for s in SEEDS:
            old=core.read(core.cell_path(root/'reference',('july28',a,s,V13_GT))/'test_result.json')
            new=core.read(core.cell_path(root/'screen',('july28',a,s,GT))/'test_result.json')
            delta=max(abs(x-y) for x,y in zip(old['predictions']['prediction'],new['predictions']['prediction']))
            result['v13_gt_replay'].append(dict(action=a,seed=s,score_delta=new['coef']-old['coef'],max_prediction_delta=delta))
    return result


def write_report(root):
    report=summarize(root);core.save(root/'summary.json',report)
    strength.write_csv(root/'scores.csv',report['raw'])
    for s in SEEDS:strength.write_csv(root/('scores_seed%d.csv'%s),[r for r in report['raw'] if r['seed']==s])
    for key,column in (('per_seed','seed'),('per_action','action')):
        strength.write_csv(root/(key+'.csv'),[dict(method=g['method'],**{column:k},**v) for g in report['methods'] for k,v in g[key].items()])


def execute(args,root):
    if args.stage=='package':
        if (root/'completion.json').exists():write_report(root)
        return
    core.immutable(root/'manifest.json',dict(study='full',jobs=matrix(),total_new_cells=108,reference_cells=72,total_rows=180,
        actions=core.ACTIONS,seeds=SEEDS,recipes=RECIPES,epochs=core.BUDGET,smoke_epochs=2,smoke_cells=36,
        selection='fixed Ordered eval draw; trained-epoch test-best; no epoch-zero fallback',
        budgets='additional branch stage; same fixed LR and native epochs for all candidate arms'))
    gpus=args.gpus.split()
    if not gpus or len(gpus)!=len(set(gpus)) or any(not g.isdigit() for g in gpus):raise ValueError('provide distinct GPU indices')
    cal.check_resources(root,gpus)
    repos=dict(modern=REPO,july28=Path(args.july28_repo).resolve())
    print('PREFLIGHT: code/input checks, 72 v13 Base/Ordered/E/F reference rows, 18 original Ordered checkpoint/RNG pairs',flush=True)
    previous.verify_training_contract(repos['july28'])
    if core.sha(REPO/'tools/qmd_reskd_config.py')!=core.sha(repos['july28']/'tools/qmd_reskd_config.py'):
        raise ValueError('coordinator/trainer experiment contracts differ')
    source=core.provenance(repos);core.immutable(root/'source.json',source)
    checked_v13(Path(args.reference_root).resolve(),source,root)
    teachers=resolve_teachers(root,Path(args.teacher_root).resolve())
    print('TEACHERS READY: 18/18 hash-verified',flush=True)
    gate=root/'smoke/gate.json'
    expected=dict(decision='QMD_RESKD_SMOKE_PASS',cells=36,source=source,teachers=teachers)
    if args.stage in ('all','smoke') and not gate.exists():
        unit_check(REPO,root/'smoke/unit_tools.log',['tests.test_qmd_reskd_tools','tests.test_qmd_complement_tools','tests.test_qmd_residual_tools'],gpus[0])
        unit_check(repos['july28'],root/'smoke/unit_july28.log',['tests.test_qmd_reskd','tests.test_qmd_complement','tests.test_qmd_residual',
            'tests.test_qmd_full','tests.test_qmd_split','tests.test_july28_qmd','tests.test_qmd_gradient'],gpus[0])
        run_wave(root/'smoke',repos,smoke_cells(matrix()),[gpus[0]],True)
        checks=[validate_cell(core.cell_path(root/'smoke',c),c,True) for c in smoke_cells(matrix())]
        check_pairing(root/'smoke',seeds=(0,))
        core.save(root/'smoke/summary.json',dict(raw=checks,note='Functionality only; no score filtering'))
        if any(r['warnings'] for r in checks):raise RuntimeError('smoke found ineffective KD; inspect diagnostics')
        core.save(gate,expected)
    if not gate.exists() or core.read(gate)!=expected:raise RuntimeError('matching smoke PASS required')
    if args.stage=='smoke':core.package(root,'_smoke');return
    run_wave(root/'screen',repos,matrix(),gpus);write_report(root)
    core.save(root/'completion.json',dict(experiments_complete=True,new_cells=108,reference_cells=72,total_rows=180,
        source=source,completed_at=time.time()))


def status(root):
    cal.status(root,smoke_selector=smoke_cells,smoke_epochs=2)
    for m in RECIPES:
        done=sum((core.cell_path(root/'screen',c)/'complete.json').exists() for c in matrix() if c[3]==m)
        print('%s: %d/18' % (m,done))
    print('108 new = 6 actions x 3 seeds x 6 arms; completed cells are kept on rerun.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('all','smoke','screen','status','package'),default='all')
    p.add_argument('--root',default='results/qmd_reskd_v14');p.add_argument('--gpus',default='0 1 3 4')
    p.add_argument('--reference_root',default='../PAMFN-qmd-complement-v13/results/qmd_complement_v13')
    p.add_argument('--teacher_root',default='../PAMFN-qmd-dual-v2/results/qmd_dual_v2')
    p.add_argument('--july28_repo',default='../PAMFN-july28-qmd-reskd-v14')
    args=p.parse_args();root=Path(args.root).resolve()
    if args.stage=='status':status(root);return
    full.check_result_roots(root,(args.reference_root,args.teacher_root));root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (root/'.runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for path in root.rglob('*.process.json'):
            if not {'reference','attempts'}.intersection(path.relative_to(root).parts) and alive(core.read(path).get('identity')):
                raise RuntimeError('live child remains: '+str(path))
        record=dict(pid=os.getpid(),identity=identity(os.getpid()),state='running',stage=args.stage,started_at=time.time())
        core.save(root/'runner.json',record)
        try:
            execute(args,root);record.update(state='finished',ended_at=time.time());core.save(root/'runner.json',record)
            if (root/'completion.json').exists():
                core.package(root,'');print('QMD_RESKD_COMPLETE: 108/108 new; 180 rows; final ZIP ready.',flush=True)
            elif args.stage=='package':core.package(root,'_partial')
        except Exception as error:
            record.update(state='failed',ended_at=time.time(),error=repr(error));core.save(root/'runner.json',record);raise


if __name__=='__main__':main()
