"""v17: 54 new full-action E-supplement runs, 90 immutable reference rows."""
import argparse
import json
import math
import os
from pathlib import Path
import time
from tools import run_qmd_modal as prior
from tools import run_qmd_dual as core
from tools import run_qmd_cal as cal
from tools import run_qmd_full as full
from tools import run_qmd_split as split
from tools import run_qmd_strength as strength
from tools.run_qmd_reskd import copy_unchanged
from tools.qmd_process import alive,identity
from tools.qmd_supp_config import RECIPES,FIXED,SCALED,SHRINK,SEEDS,E_TAG,REFERENCE_E,metadata,parameter_count

REPO=Path(__file__).resolve().parents[1]
REFERENCES=prior.REFERENCES
METHODS=REFERENCES+tuple(RECIPES)
PINS=dict(modern='ac27d43eae4ce3c93d4746bff63642cfb15639ac',july28='cb1e82049ad760945cd21044da73d8c5871246a5')


def matrix():return [('july28',a,s,m) for a in core.ACTIONS for s in SEEDS for m in RECIPES]
def smoke_cells(jobs):return [c for c in matrix() if c[2]==0] if jobs else []
def info(root,cell,name):return next(v for v in core.read(root/name)['records'] if (v['action'],v['seed'])==cell[1:3])


def checked_v16(origin,source,root):
    """Read-only import; weight availability is checked separately before GPU jobs."""
    full.check_result_roots(root,(origin,))
    old,done=[core.read(origin/n) for n in ('source.json','completion.json')]
    if old['commits']!=PINS or done['source']!=old or not done['experiments_complete'] or [done[k] for k in ('new_cells','reference_cells','total_rows')]!=[72,90,162]:
        raise ValueError('requires complete pinned v16 results')
    if any(old[k]!=source[k] for k in ('inputs','environment')):raise ValueError('v16 environment/data mismatch')
    copy_unchanged(origin/'reference_teacher_manifest.json',root/'reference_teacher_manifest.json')
    for c in prior.reference_cells():
        src=core.cell_path(origin/'reference',c);dest=core.cell_path(root/'reference',c);record=split.verify_record(src)
        for f in list(record['files'])+['complete.json']:copy_unchanged(src/f,dest/f)
    rows=[prior.validate_reference(core.cell_path(root/'reference',c),c) for c in prior.reference_cells()]
    teachers=[]
    for a in core.ACTIONS:
        for s in SEEDS:
            c=('july28',a,s,E_TAG);src=core.cell_path(origin/'screen',c);rec=split.verify_record(src)
            dest=core.cell_path(root/'e_reference',c)
            for f in list(rec['files'])+['complete.json']:copy_unchanged(src/f,dest/f)
            test=core.read(src/'test_result.json');e=core.read(core.cell_path(root/'reference',('july28',a,s,REFERENCE_E))/'test_result.json')
            if test['predictions']!=e['predictions'] or test['training_statistics_sha256']!=e['training_statistics_sha256']:
                raise ValueError('v16 FusedGT did not reproduce E')
            if rec['metrics']['method']!=E_TAG or abs(rec['metrics']['srcc_x100']-e['coef'])>1e-6:raise ValueError('wrong E reference')
            teachers.append(dict(action=a,seed=s,model=str((src/'model.pth').resolve()),record=str((dest/'complete.json').resolve()),
                checkpoint_sha256=rec['checkpoint_sha256'],eval_rng_sha256=rec['eval_rng_sha256'],expected_srcc_x100=e['coef']))
    core.immutable(root/'e_manifest.json',dict(records=teachers,source_path=str(origin)))
    core.immutable(root/'reference_manifest.json',dict(source=old,records=rows,source_path=str(origin)))
    return rows


def resolve_e(root):
    teachers=core.read(root/'e_manifest.json')['records'];missing=[]
    for e in teachers:
        model=Path(e['model']);ordered=info(root,('july28',e['action'],e['seed'],FIXED),'teacher_manifest.json')
        if not model.is_file() or core.sha(model)!=e['checkpoint_sha256']:missing.append(str(model))
        if e['eval_rng_sha256']!=ordered['eval_rng_sha256']:raise ValueError('E and Ordered RNG differ')
    if missing:raise FileNotFoundError('Keep the v16 FusedGT weights; missing/hash-mismatched E models:\n'+'\n'.join(missing))
    return teachers


def cell_args(root,cell,gpu,smoke=False):
    if cell not in matrix():raise ValueError('outside fixed v17 matrix')
    t=info(root,cell,'teacher_manifest.json');e=info(root,cell,'e_manifest.json')
    argv=[os.sys.executable,'-u','tools/train_qmd_supp.py','--gpu',str(gpu),
        '--action',cell[1],'--seed',str(cell[2]),'--run_tag',cell[3],
        '--teacher_model',t['model'],'--teacher_rng',t['rng'],
        '--teacher_record',str(core.cell_path(root/'reference',cell[:3]+(prior.ORDERED,))/'complete.json'),
        '--e_model',e['model'],'--e_record',e['record']]
    return argv+(['--smoke','--epochs','2'] if smoke else [])


def check_predictions(data,baseline,action,bound=None):
    import numpy as np
    from scipy.stats import spearmanr
    y,p=[np.asarray(data['predictions'][k],dtype=np.float64) for k in ('label','prediction')]
    b=np.asarray(baseline['predictions']['prediction'])
    if y.shape!=((100 if action in ('TES','PCS') else 50),) or p.shape!=y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('invalid predictions')
    if data['predictions']['label']!=baseline['predictions']['label']:raise ValueError('test order changed')
    for observed,expected in [(data['coef'],spearmanr(y,p)[0]*100),(data['loss_x100'],np.mean((y-p)**2)*100),(data['prediction_std'],p.std())]:
        if not math.isfinite(observed) or not math.isfinite(expected) or abs(observed-expected)>1e-6:raise ValueError('prediction metric mismatch')
    if bound is None:return
    delta=p-b
    if not 0<=data['gate']<=1 or abs(delta).max()>bound*data['gate']+1e-8:raise ValueError('supplement exceeded gate/bound')
    for k,v in [('correction_std',delta.std()),('correction_mean',delta.mean()),('correction_max_abs',abs(delta).max())]:
        if abs(data[k]-v)>1e-8:raise ValueError('supplement diagnostic mismatch')
    i,j=np.triu_indices(len(y),1);keep=abs(y[i]-y[j])>=.02;i,j=i[keep],j[keep];sign=np.sign(y[i]-y[j])
    ow=sign*(b[i]-b[j])<=0;nw=sign*(p[i]-p[j])<=0
    if data['repaired_pairs']!=int((ow&~nw).sum()) or data['broken_pairs']!=int((~ow&nw).sum()):raise ValueError('pair counts mismatch')
    if set(data['expert_metrics'])!={'V','F','A'} or set(data['expert_predictions'])!={'V','F','A'}:raise ValueError('expert keys mismatch')
    values=[]
    for m in ('V','F','A'):
        v=dict(data['expert_metrics'][m],predictions=data['expert_predictions'][m]);check_predictions(v,baseline,action)
        p1=np.asarray(v['predictions']['prediction']);values.append(p1)
        if abs(p1-b).max()>bound*data['gate']+1e-8:raise ValueError('expert exceeds bound')
    if not np.allclose(np.mean(values,axis=0),p,atol=1e-8,rtol=0):raise ValueError('not a fixed expert mean')


def validate_cell(out,cell,smoke=False):
    root=out.parents[4];epochs=2 if smoke else core.BUDGET[cell[1]]
    if core.read(out/'run_config.json')!=dict(argv=cell_args(root,cell,'0',smoke)[5:],protocol='july28'):raise ValueError('arguments changed')
    t=info(root,cell,'teacher_manifest.json');e=info(root,cell,'e_manifest.json')
    fields=('checkpoint_sha256','eval_rng_sha256','expected_srcc_x100','action','seed')
    source=dict(ordered={k:t[k] for k in fields},e={k:e[k] for k in fields})
    train,test,check,stats,q,last=[core.read(out/(n+'.json')) for n in ('train_result','test_result','teacher_check','training_statistics','qmd_training','last_epoch_result')]
    meta=metadata(cell[1],cell[2],cell[3],epochs,smoke)
    if train['source']!=source or test['source']!=source or check['source']!=source or train['run']!=meta or test['run']!=meta:raise ValueError('source/recipe mismatch')
    if not check['zero_branch_exact'] or check['branch_config']!=meta['branch'] or check['trainable_parameters']!=parameter_count(cell[3]):raise ValueError('initial E parity or architecture mismatch')
    original=core.read(Path(e['record']).parent/'test_result.json')
    if check['baseline']['predictions']!=original['predictions'] or abs(check['baseline']['coef']-e['expected_srcc_x100'])>1e-6:raise ValueError('E output changed')
    if abs(check['ordered']['coef']-t['expected_srcc_x100'])>1e-6:raise ValueError('Ordered replay mismatch')
    if stats['source_split']!='train' or stats['samples']!=(400 if cell[1] in ('TES','PCS') else 200) or stats['passes']!=1 or stats['new_global_bias']:
        raise ValueError('invalid training-only calibration')
    if not math.isfinite(stats['e_prediction_std']) or stats['e_prediction_std']<0:raise ValueError('nonfinite E training spread')
    bound=min(.003,.1*stats['e_prediction_std']) if RECIPES[cell[3]]['relative'] else .003
    if not math.isfinite(bound) or bound<0 or abs(bound-stats['bound'])>1e-9:raise ValueError('invalid calibrated bound')
    if not (.009999<=stats['feature_scale_min']<=stats['feature_scale_max']<float('inf')):raise ValueError('invalid feature scale')
    if len({stats['statistics_sha256'],q['statistics_before'],q['statistics_after'],test['training_statistics_sha256'],check['training_statistics_sha256']})!=1 or not q['statistics_unchanged']:raise ValueError('statistics changed')
    if not q['frozen_unchanged'] or not q['frozen_gradients_none'] or q['frozen_before']!=q['frozen_after'] or q['frozen_before']!=check['initial_frozen_sha256']:raise ValueError('E/Ordered updated')
    if q['initial_branch_sha256']!=check['initial_branch_sha256']:raise ValueError('initialization mismatch')
    if q['branch_updated']!=(q['initial_branch_sha256']!=q['final_branch_sha256']):raise ValueError('branch update report mismatch')
    if q['mode']!='frozen_E_supplement' or q['recipe']!=RECIPES[cell[3]] or q['epoch_zero_eligible'] or q['extra_stage_epochs']!=epochs:raise ValueError('objective/epoch mismatch')
    if any(q[k]!=0 for k in ('kd_weight','weighted_kd_sum','positive_kd_gradient_batches','rank_weight')):raise ValueError('KD/rank must be disabled')
    if test['evaluation']!=dict(policy=meta['eval_policy'],replayed_rng=True):raise ValueError('wrong evaluation')
    for v in (check['baseline'],check['ordered']):check_predictions(v,check['baseline'],cell[1])
    for v in (test,last):
        check_predictions(v,check['baseline'],cell[1],bound)
        if abs(v['bound']-bound)>1e-9 or abs(v['e_srcc_x100']-e['expected_srcc_x100'])>1e-6:raise ValueError('E/bound mismatch')
        if cell[3]!=SHRINK and v['gate']!=1:raise ValueError('fixed gate changed')
    history=[json.loads(v) for v in (out/'model.pth.epochs.jsonl').read_text().splitlines()]
    selection=core.read(out/'model.pth.selection.json');epoch=selection['epoch']
    if [v['epoch'] for v in history]!=list(range(1,epochs+1)) or [v['epoch'] for v in q['epochs']]!=list(range(1,epochs+1)):raise ValueError('incomplete history')
    if not 1<=epoch<=epochs or any(not math.isfinite(v['coef']) for v in history):raise ValueError('invalid checkpoint history')
    for score in (selection['coef'],history[epoch-1]['coef'],max(v['coef'] for v in history),train['saved_model_srcc']*100):
        if abs(test['coef']-score)>1e-6:raise ValueError('selection/reload mismatch')
    if any(last.get(k)!=v for k,v in history[-1].items()):raise ValueError('last epoch mismatch')
    if q['positive_gradient_batches']!=sum(v['positive_gradient_batches'] for v in q['epochs']):raise ValueError('gradient counts mismatch')
    for v in q['epochs']:
        if not 0<=v['positive_gradient_batches']<=v['batches'] or not 0<=v['gate_mean']<=1:raise ValueError('invalid training diagnostics')
        if not all(math.isfinite(v[k]) and v[k]>=0 for k in ('loss_mean','centered_mse_mean','mean_penalty_mean','shrink_penalty_mean')):raise ValueError('invalid losses')
        if abs(v['loss_mean']-v['centered_mse_mean']-v['mean_penalty_mean']-v['shrink_penalty_mean'])>1e-7:raise ValueError('loss decomposition mismatch')
        if cell[3]!=SHRINK and v['shrink_penalty_mean']!=0:raise ValueError('unexpected shrink penalty')
    warnings=[]
    if not q['branch_updated'] or not q['positive_gradient_batches']:warnings.append('zero_bound_or_no_effective_update')
    return dict(protocol='july28',action=cell[1],seed=cell[2],method=cell[3],srcc_x100=test['coef'],mse=test['loss_x100']/100,
        selected_epoch=epoch,nan_eval_epochs=0,prediction_std=test['prediction_std'],gate=test['gate'],bound=bound,
        e_train_prediction_std=stats['e_prediction_std'],correction_std=test['correction_std'],
        repaired_pairs=test['repaired_pairs'],broken_pairs=test['broken_pairs'],last_srcc_x100=last['coef'],last_gate=last['gate'],
        positive_gradient_batches=q['positive_gradient_batches'],trainable_parameters=parameter_count(cell[3]),warnings=warnings)


def run_cell(root,repos,cell,gpu,smoke):
    return core.run_cell(root,repos,cell,gpu,smoke,argv=cell_args(root.parent,cell,gpu,smoke),validator=validate_cell,
        extra_required=('teacher_check.json','training_statistics.json','last_epoch_result.json'))


def run_wave(root,repos,jobs,gpus,smoke=False):
    return core.run_wave(root,repos,jobs,gpus,smoke,runner=run_cell,ordering=lambda cs:sorted(cs,key=lambda c:(-core.BUDGET[c[1]],c[2],c[3])))


def check_pairing(root,seeds=SEEDS):
    for a in core.ACTIONS:
        for s in seeds:
            checks=[core.read(core.cell_path(root,('july28',a,s,m))/'teacher_check.json') for m in RECIPES]
            for c in checks[1:]:
                if any(c[k]!=checks[0][k] for k in ('baseline','ordered','initial_frozen_sha256','initial_heads_sha256')):raise ValueError('paired E/initial heads differ')
            stats=[core.read(core.cell_path(root,('july28',a,s,m))/'training_statistics.json') for m in RECIPES]
            for key in ('e_prediction_std','feature_scale_min','feature_scale_max'):
                if any(t[key]!=stats[0][key] for t in stats):raise ValueError('training cache mismatch')


def write_report(root):
    rows=[prior.validate_reference(core.cell_path(root/'reference',c),c) for c in prior.reference_cells()]
    for c in matrix():
        out=core.cell_path(root/'screen',c);record=split.verify_record(out);row=validate_cell(out,c)
        if abs(row['srcc_x100']-record['metrics']['srcc_x100'])>1e-6:raise ValueError('recorded score changed')
        rows.append(dict(row,origin='new_v17_E_supplement'))
    check_pairing(root/'screen')
    result=full.aggregate(rows,methods=METHODS,controls=((prior.BASE,'base'),(prior.ORDERED,'ordered'),(REFERENCE_E,'e'),(prior.V13_KD,'v13_kd'),(prior.V15_GATE,'v15_gate'),(FIXED,'fixed'),(SCALED,'scaled')))
    result['note']='54 extra-stage trainings and 90 frozen references. Frozen Ordered and E. No KD/rank/router; trained-epoch test-best, no score fallback.'
    core.save(root/'summary.json',result);strength.write_csv(root/'scores.csv',result['raw'])
    for s in SEEDS:strength.write_csv(root/('scores_seed%d.csv'%s),[r for r in result['raw'] if r['seed']==s])
    for key,column in (('per_seed','seed'),('per_action','action')):
        strength.write_csv(root/(key+'.csv'),[dict(method=g['method'],**{column:k},**v) for g in result['methods'] for k,v in g[key].items()])


def execute(args,root):
    if args.stage=='package':
        if (root/'completion.json').exists():write_report(root)
        return
    core.immutable(root/'manifest.json',dict(study='full',jobs=matrix(),total_new_cells=54,reference_cells=90,total_rows=144,
        actions=core.ACTIONS,seeds=SEEDS,recipes=RECIPES,epochs=core.BUDGET,smoke_epochs=2,smoke_cells=18,
        selection='trained-epoch test-best; no epoch zero',frozen='Ordered plus E'))
    gpus=args.gpus.split()
    if not gpus or len(set(gpus))!=len(gpus) or any(not g.isdigit() for g in gpus):raise ValueError('provide distinct physical GPU indices')
    cal.check_resources(root,gpus);repos=dict(modern=REPO,july28=Path(args.july28_repo).resolve())
    prior.previous.verify_training_contract(repos['july28'])
    if core.sha(REPO/'tools/qmd_supp_config.py')!=core.sha(repos['july28']/'tools/qmd_supp_config.py'):raise ValueError('coordinator/trainer contract mismatch')
    print('PREFLIGHT: input/code hashes, v16 references, original Ordered and E checkpoint pairs',flush=True)
    source=core.provenance(repos);core.immutable(root/'source.json',source)
    checked_v16(Path(args.modal_root).resolve(),source,root)
    ordered=prior.resolve_teachers(root,Path(args.teacher_root).resolve());e=resolve_e(root)
    print('TEACHERS READY: 18 Ordered + 18 frozen E',flush=True)
    gate=root/'smoke/gate.json';expected=dict(decision='QMD_SUPP_SMOKE_PASS',cells=18,source=source,ordered=ordered,e=e)
    if args.stage in ('all','smoke') and not gate.exists():
        prior.unit_check(REPO,root/'smoke/unit_tools.log',['tests.test_qmd_supp_tools','tests.test_qmd_modal_tools','tests.test_qmd_residual_tools'],gpus[0])
        prior.unit_check(repos['july28'],root/'smoke/unit_july28.log',['tests.test_qmd_supp','tests.test_qmd_modal','tests.test_qmd_complement','tests.test_qmd_residual','tests.test_qmd_gradient'],gpus[0])
        run_wave(root/'smoke',repos,smoke_cells(matrix()),[gpus[0]],True)
        checks=[validate_cell(core.cell_path(root/'smoke',c),c,True) for c in smoke_cells(matrix())]
        check_pairing(root/'smoke',seeds=(0,));core.save(root/'smoke/summary.json',dict(raw=checks,note='functionality only, no score gate'))
        core.save(gate,expected)
    if not gate.exists() or core.read(gate)!=expected:raise RuntimeError('matching smoke PASS required')
    if args.stage=='smoke':core.package(root,'_smoke');return
    run_wave(root/'screen',repos,matrix(),gpus);write_report(root)
    core.save(root/'completion.json',dict(experiments_complete=True,new_cells=54,reference_cells=90,total_rows=144,source=source,completed_at=time.time()))


def status(root):
    cal.status(root,smoke_selector=smoke_cells,smoke_epochs=2)
    for m in RECIPES:print('%s: %d/18'%(m,sum((core.cell_path(root/'screen',c)/'complete.json').exists() for c in matrix() if c[3]==m)))
    print('54 new = 6 actions x 3 seeds x 3 arms. Completed cells are kept; unfinished training restarts its cell.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('all','smoke','screen','status','package'),default='all')
    p.add_argument('--root',default='results/qmd_supp_v17');p.add_argument('--gpus',required=False,default='0 1 3 4')
    p.add_argument('--modal_root',default='../PAMFN-qmd-modal-v16/results/qmd_modal_v16')
    p.add_argument('--teacher_root',default='../PAMFN-qmd-dual-v2/results/qmd_dual_v2')
    p.add_argument('--july28_repo',default='../PAMFN-july28-qmd-supp-v17');args=p.parse_args();root=Path(args.root).resolve()
    if args.stage=='status':status(root);return
    full.check_result_roots(root,(args.modal_root,args.teacher_root));root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (root/'.runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for path in root.rglob('*.process.json'):
            if not {'reference','e_reference','attempts'}.intersection(path.relative_to(root).parts) and alive(core.read(path).get('identity')):raise RuntimeError('live child remains: '+str(path))
        record=dict(pid=os.getpid(),identity=identity(os.getpid()),state='running',stage=args.stage,started_at=time.time());core.save(root/'runner.json',record)
        try:
            execute(args,root);record.update(state='finished',ended_at=time.time());core.save(root/'runner.json',record)
            if (root/'completion.json').exists():core.package(root,'');print('QMD_SUPP_COMPLETE: 54/54 new; 144 rows; final ZIP ready.',flush=True)
            elif args.stage=='package':core.package(root,'_partial')
        except Exception as error:
            record.update(state='failed',ended_at=time.time(),error=repr(error));core.save(root/'runner.json',record);raise


if __name__=='__main__':main()
