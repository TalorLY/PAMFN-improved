"""v12: 54 fixed full-budget branch trainings, 72 frozen comparison rows.

Uses original v2 Ordered weights read-only and v11 result records. Existing
training entry points/models are unchanged. No action or seed filtering.
"""
import argparse
import math
import os
from pathlib import Path
import subprocess
import time

from tools import run_qmd_dual as core
from tools import run_qmd_cal as cal
from tools import run_qmd_full as full
from tools import run_qmd_split as split
from tools import run_qmd_strength as strength
from tools.qmd_process import alive, identity
from tools.reuse_qmd_gradient_v8 import copy_unchanged

REPO=Path(__file__).resolve().parents[1]
BASE,ORDERED,KD=full.BASE,full.ORDERED,full.KD
HALF='ordered_score_qmd_split_kd_w050'
GT='ordered_residual_gt'
WEIGHTS={GT:0.,'ordered_residual_kd_w050':.5,'ordered_residual_kd_w100':1.}
REFERENCES=(BASE,ORDERED,KD,HALF)
METHODS=REFERENCES+tuple(WEIGHTS)
SEEDS=(0,1,2)
SMOKE_EPOCHS=2
V11_COMMITS=dict(modern='917765536bb9c535c818faec847be7ed684e7098',july28='12b03b561a122ebe23c0a40b6103f7071bbde298')


def matrix():return [('july28',a,s,m) for a in core.ACTIONS for s in SEEDS for m in WEIGHTS]
def reference_cells():return [('july28',a,s,m) for a in core.ACTIONS for s in SEEDS for m in REFERENCES]
def smoke_cells(jobs):return [c for c in matrix() if c[2]==0] if jobs else []


def teacher_info(root,cell):
    return next(r for r in core.read(root/'teacher_manifest.json')['records']
                if r['action']==cell[1] and r['seed']==cell[2])


def cell_args(root,cell,gpu,smoke=False):
    if cell not in matrix():raise ValueError('cell outside fixed residual grid')
    teacher=teacher_info(root,cell)
    argv=[os.sys.executable,'-u','tools/train_qmd_residual.py','--gpu',str(gpu),
        '--action',cell[1],'--seed',str(cell[2]),'--run_tag',cell[3],
        '--teacher_model',teacher['model'],'--teacher_rng',teacher['rng'],
        '--teacher_record',str(core.cell_path(root/'reference',cell[:3]+(ORDERED,))/'complete.json')]
    if smoke:argv+=['--smoke','--epochs','2']
    return argv


def validate_reference(out,cell):
    record=split.verify_record(out)
    if cell[3]==HALF:row=strength.validate_cell(out,cell)
    elif cell[3] in (BASE,KD):
        full.check_args(out,cell);row=full.validate_cell(out,cell)
    else:
        expected=dict(argv=core.cell_args(*cell,'0')[5:],protocol=cell[0])
        if core.read(out/'run_config.json')!=expected:raise ValueError('Ordered reference arguments changed')
        row=core.validate_cell(out,cell)
    if abs(record['metrics']['srcc_x100']-row['srcc_x100'])>1e-6:raise ValueError('reference score changed')
    return dict(row,origin='frozen_v11_reference')


def checked_v11(reference,source,root):
    old,complete=[core.read(reference/n) for n in ('source.json','completion.json')]
    if (old['commits']!=V11_COMMITS or complete.get('source')!=old or not complete.get('experiments_complete')
            or complete.get('new_cells')!=36 or complete.get('total_rows')!=108):
        raise ValueError('requires completed frozen v11 results')
    if any(old[k]!=source[k] for k in ('inputs','environment')):raise ValueError('v11 data/environment mismatch')
    strength.summarize(reference)
    for c in reference_cells():
        origin=core.cell_path(reference/('screen' if c[3]==HALF else 'reference'),c)
        record=split.verify_record(origin);dest=core.cell_path(root/'reference',c)
        for name in list(record['files'])+['complete.json']:
            copy_unchanged(origin/name,dest/name)
        for name in ('train.log','test.log'):
            if (origin/name).exists():copy_unchanged(origin/name,dest/name)
    rows=[validate_reference(core.cell_path(root/'reference',c),c) for c in reference_cells()]
    core.immutable(root/'reference_manifest.json',dict(source=old,records=rows,source_path=str(reference)))
    return rows


def resolve_teachers(root,teacher_root):
    records=[]; missing=[]
    for a in core.ACTIONS:
        for s in SEEDS:
            c=('july28',a,s,ORDERED)
            record=split.verify_record(core.cell_path(root/'reference',c))
            candidates=[core.cell_path(teacher_root/phase,c) for phase in ('screen','reference','')]
            match=None
            for path in candidates:
                model,rng=path/'model.pth',path/'model.pth.eval_rng.pt'
                if not model.is_file() or not rng.is_file():continue
                if core.sha(model)==record['checkpoint_sha256'] and core.sha(rng)==record['eval_rng_sha256']:
                    match=dict(action=a,seed=s,model=str(model.resolve()),rng=str(rng.resolve()),
                        checkpoint_sha256=record['checkpoint_sha256'],eval_rng_sha256=record['eval_rng_sha256'],
                        expected_srcc_x100=record['metrics']['srcc_x100']);break
            if match is None:missing.append('%s seed%d: %s' % (a,s,candidates[0]))
            else:records.append(match)
    if missing:
        raise FileNotFoundError('Missing or hash-mismatched original Ordered weights/RNG. Keep original v2 results; '
            'set TEACHER_ROOT to their result root. No training has started.\n'+'\n'.join(missing))
    core.immutable(root/'teacher_manifest.json',dict(records=records,teacher_root=str(teacher_root)))
    return records


def validate_cell(out,cell,smoke=False):
    import numpy as np
    from scipy.stats import spearmanr
    root=out.parents[4]
    expected=dict(argv=cell_args(root,cell,'0',smoke)[5:],protocol=cell[0])
    if core.read(out/'run_config.json')!=expected:raise ValueError('residual arguments changed')
    epochs=SMOKE_EPOCHS if smoke else core.BUDGET[cell[1]]
    train,test,check,qmd=[core.read(out/name) for name in ('train_result.json','test_result.json','teacher_check.json','qmd_training.json')]
    teacher=teacher_info(root,cell)
    source={k:teacher[k] for k in ('checkpoint_sha256','eval_rng_sha256','expected_srcc_x100','action','seed')}
    expected_meta=dict(action=cell[1],seed=cell[2],tag=cell[3],epochs=epochs,feats=2,
        protocol='july28_frozen_residual',weight=WEIGHTS[cell[3]],branch=dict(input_dim=768,hidden=64,bound=.1),
        anchor='July28 Ordered frozen checkpoint',eval_policy='fixed_ordered_eval_rng',mask_ratio=.15,
        warmup=0. if smoke else .1,rampup=0. if smoke else .1,target_ratio=.075,beta_cap=1.,
        gt='full corrected score',kd_target='frozen full-input Ordered score')
    for data in (train,test):
        if data['run']!=expected_meta or data['source']!=source:raise ValueError('residual model/recipe/source mismatch')
    if check['source']!=source or not check['zero_branch_exact']:raise ValueError('zero-branch anchor parity failed')
    if check['branch_config']!=expected_meta['branch'] or check['branch_parameters']!=50817:
        raise ValueError('unexpected residual architecture')
    baseline=check['baseline']['coef']
    if abs(baseline-source['expected_srcc_x100'])>1e-6:raise ValueError('original Ordered score not recovered')
    if test['evaluation']!=dict(policy='fixed_ordered_eval_rng',replayed_rng=True):raise ValueError('eval replay missing')
    if abs(test['baseline_srcc_x100']-baseline)>1e-6:raise ValueError('baseline changed during reload')
    for data in (test,check['baseline']):
        y,p=[np.asarray(data['predictions'][k]) for k in ('label','prediction')]
        n=100 if cell[1] in ('TES','PCS') else 50
        if y.shape!=(n,) or p.shape!=y.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError('missing/nonfinite predictions')
        coef=spearmanr(y,p)[0]*100
        if not math.isfinite(coef) or abs(coef-data['coef'])>1e-6 or abs(np.mean((y-p)**2)*100-data['loss_x100'])>1e-6:
            raise ValueError('prediction metrics disagree')
        if abs(np.std(p)-data['prediction_std'])>1e-8:raise ValueError('prediction spread mismatch')
    if test['predictions']['label']!=check['baseline']['predictions']['label']:
        raise ValueError('test labels/order changed from the frozen anchor')
    correction=np.asarray(test['predictions']['prediction'])-np.asarray(check['baseline']['predictions']['prediction'])
    if np.max(np.abs(correction))>.100001:raise ValueError('correction exceeded fixed bound')
    hist=[core.read(out/'model.pth.selection.json')]
    import json
    lines=[json.loads(v) for v in (out/'model.pth.epochs.jsonl').read_text().splitlines()]
    if [v['epoch'] for v in lines]!=list(range(1,epochs+1)):raise ValueError('incomplete/duplicate epochs')
    selected=hist[0]['epoch']; score=test['coef']
    if not 1<=selected<=epochs:raise ValueError('epoch zero is not an eligible checkpoint')
    if any(not math.isfinite(v) or abs(v-score)>1e-6 for v in (train['saved_model_srcc']*100,hist[0]['coef'],lines[selected-1]['coef'],
                                     max(v['coef'] for v in lines if math.isfinite(v['coef'])))):
        raise ValueError('checkpoint selection/reload mismatch')
    if (not qmd['anchor_unchanged'] or not qmd['anchor_gradients_none']
            or qmd['anchor_state_before']!=qmd['anchor_state_after']
            or qmd['anchor_state_before']!=check['initial_anchor_sha256']):
        raise ValueError('Ordered state or gradients changed')
    if not qmd['branch_updated'] or qmd['initial_branch_sha256']==qmd['final_branch_sha256']:
        raise ValueError('branch never updated')
    if qmd['initial_branch_sha256']!=check['initial_branch_sha256']:raise ValueError('branch initialization changed')
    if qmd['mode']!='residual_kd' or qmd['weight']!=WEIGHTS[cell[3]] or qmd['epoch_zero_eligible']:
        raise ValueError('wrong auxiliary or artificial baseline fallback')
    if [v['epoch'] for v in qmd['epochs']]!=list(range(1,epochs+1)) or qmd['extra_stage_epochs']!=epochs:
        raise ValueError('incomplete auxiliary history')
    if qmd['active_batches']!=sum(v['active_batches'] for v in qmd['epochs']) or qmd['active_batches']<=0:
        raise ValueError('auxiliary diagnostics never activated')
    if not math.isfinite(qmd['weighted_kd_sum']) or abs(qmd['weighted_kd_sum']-sum(v['weighted_kd_sum'] for v in qmd['epochs']))>1e-8:
        raise ValueError('KD loss summary mismatch')
    if cell[3]==GT and qmd['weighted_kd_sum']!=0:raise ValueError('GT-only control contains KD update')
    calibration=qmd['calibration']
    if not calibration or not math.isfinite(calibration['beta']) or not 0<=calibration['beta']<=1:
        raise ValueError('invalid branch calibration')
    warnings=[]
    if WEIGHTS[cell[3]]>0 and (qmd['weighted_kd_sum']<=0 or qmd['positive_kd_gradient_batches']<=0):
        warnings.append('no_effective_branch_kd')
    return dict(protocol=cell[0],action=cell[1],seed=cell[2],method=cell[3],srcc_x100=score,
        mse=test['loss_x100']/100,selected_epoch=selected,prediction_std=test['prediction_std'],
        nan_eval_epochs=sum(not math.isfinite(v['coef']) for v in lines),
        active_batches=qmd['active_batches'],beta=calibration['beta'],weight=WEIGHTS[cell[3]],
        branch_conflict_rate=qmd['conflicts']/max(1,qmd['cosine_batches']),
        anchor_unchanged=True,branch_parameters=check['branch_parameters'],
        selected_before_aux=selected<=math.ceil(.1*epochs) if not smoke else False,
        extra_stage_epochs=epochs,weighted_kd_sum=qmd['weighted_kd_sum'],warnings=warnings)


def run_cell(root,repos,cell,gpu,smoke):
    return core.run_cell(root,repos,cell,gpu,smoke,argv=cell_args(root.parent,cell,gpu,smoke),
        validator=validate_cell,extra_required=('teacher_check.json',))


def run_wave(root,repos,jobs,gpus,smoke=False):
    return core.run_wave(root,repos,jobs,gpus,smoke,runner=run_cell,
        ordering=lambda cells: sorted(cells,key=lambda c:(-core.BUDGET[c[1]],c[2],c[3])))


def summarize(root):
    rows=[validate_reference(core.cell_path(root/'reference',c),c) for c in reference_cells()]
    for c in matrix():
        out=core.cell_path(root/'screen',c);record=split.verify_record(out)
        row=validate_cell(out,c)
        if abs(row['srcc_x100']-record['metrics']['srcc_x100'])>1e-6:raise ValueError('recorded result changed')
        rows.append(dict(row,origin='new_v12_two_stage_branch'))
    check_prewarm(root/'screen')
    result=full.aggregate(rows,methods=METHODS,controls=((BASE,'base'),(ORDERED,'ordered'),(GT,'branch_gt'),(KD,'kd1'),(HALF,'kd05')))
    result['note']=('Six actions x three seeds for all methods. 54 new residual-branch trainings plus 72 frozen references. '
        'Fixed Ordered anchor and its original evaluation RNG; only branch trains, no inference-time test label use. '
        'Second-stage extra training budget and 50,817 inference parameters; not equal total budget to one-stage controls. '
        'Epoch zero is not eligible for selection. Test-best retained, no per-action or seed cherry-picking. '
        'Key KD ablation is versus same-architecture/budget branch_GT, not only versus historical Ordered.')
    return result


def check_prewarm(root):
    import json
    for a in core.ACTIONS:
        for s in SEEDS:
            paths=[core.cell_path(root,('july28',a,s,m)) for m in WEIGHTS]
            checks=[core.read(p/'teacher_check.json') for p in paths]
            if any(c['initial_branch_sha256']!=checks[0]['initial_branch_sha256'] or c['baseline']!=checks[0]['baseline'] for c in checks[1:]):
                raise ValueError('paired methods differ at initialization or Ordered evaluation')
            histories=[[json.loads(v) for v in (p/'model.pth.epochs.jsonl').read_text().splitlines()] for p in paths]
            warm=math.ceil(.1*core.BUDGET[a])
            for history in histories[1:]:
                for left,right in zip(histories[0][:warm],history[:warm]):
                    for k in ('coef','loss_x100','prediction_std'):
                        if not ((math.isnan(left[k]) and math.isnan(right[k])) or abs(left[k]-right[k])<1e-6):
                            raise ValueError('paired branch prewarm trajectories differ')


def write_report(root):
    report=summarize(root);core.save(root/'summary.json',report)
    strength.write_csv(root/'scores.csv',report['raw'])
    for s in SEEDS:strength.write_csv(root/('scores_seed%d.csv'%s),[r for r in report['raw'] if r['seed']==s])
    for key,column in (('per_seed','seed'),('per_action','action')):
        strength.write_csv(root/(key+'.csv'),[dict(method=g['method'],**{column:k},**v) for g in report['methods'] for k,v in g[key].items()])


def verify_training_contract(repo):
    scopes=['main.py','pipeline.py','datasets.py','models','configs','utils']
    if subprocess.check_output(['git','-C',str(repo),'diff',V11_COMMITS['july28'],'HEAD','--']+scopes):
        raise ValueError('v12 must preserve original Ordered model and old training paths')


def unit_check(repo,log,modules,gpu):
    # In particular, the CUDA RNG test must not create contexts on other users'
    # GPUs. env runs in the child; no process-global mutation during scheduling.
    core.command(['env','CUDA_VISIBLE_DEVICES='+str(gpu),os.sys.executable,'-u','-m','unittest']+modules,repo,log)


def execute(args,root):
    if args.stage=='package':
        if (root/'completion.json').exists():write_report(root)
        return
    core.immutable(root/'manifest.json',dict(study='full',jobs=matrix(),total_new_cells=54,reference_cells=72,total_rows=126,
        actions=core.ACTIONS,seeds=SEEDS,weights=WEIGHTS,epochs=core.BUDGET,smoke_epochs=2,smoke_cells=18,
        branch=dict(input_dim=768,hidden=64,bound=.1),selection='fixed Ordered eval draw; trained-epoch test-best; no epoch-zero fallback',
        budgets='new additional branch stage; inherited action learning rates; all candidate arms identical budgets'))
    gpus=args.gpus.split()
    if not gpus or len(gpus)!=len(set(gpus)) or any(not g.isdigit() for g in gpus):raise ValueError('provide distinct GPU indices')
    cal.check_resources(root,gpus)
    repos=dict(modern=REPO,july28=Path(args.july28_repo).resolve())
    print('PREFLIGHT: code/input provenance, 72 v11 reference rows, 18 original Ordered checkpoint/RNG pairs',flush=True)
    verify_training_contract(repos['july28'])
    source=core.provenance(repos);core.immutable(root/'source.json',source)
    checked_v11(Path(args.reference_root).resolve(),source,root)
    teachers=resolve_teachers(root,Path(args.teacher_root).resolve())
    print('TEACHERS READY: 18/18 hash-verified; Ordered references kept read-only',flush=True)
    gate=root/'smoke/gate.json'
    expected=dict(decision='QMD_RESIDUAL_SMOKE_PASS',cells=18,source=source,teachers=teachers)
    if args.stage in ('all','smoke') and not gate.exists():
        unit_check(REPO,root/'smoke/unit_tools.log',['tests.test_qmd_residual_tools','tests.test_qmd_strength_tools'],gpus[0])
        unit_check(repos['july28'],root/'smoke/unit_july28.log',['tests.test_qmd_residual','tests.test_qmd_full',
            'tests.test_qmd_split','tests.test_july28_qmd','tests.test_qmd_gradient'],gpus[0])
        run_wave(root/'smoke',repos,smoke_cells(matrix()),[gpus[0]],True)
        checks=[validate_cell(core.cell_path(root/'smoke',c),c,True) for c in smoke_cells(matrix())]
        core.save(root/'smoke/summary.json',dict(raw=checks,note='Functionality only, no score filtering'))
        if any(r['warnings'] for r in checks):raise RuntimeError('smoke found ineffective KD; inspect diagnostics')
        core.save(gate,expected)
    if not gate.exists() or core.read(gate)!=expected:raise RuntimeError('matching smoke PASS required')
    if args.stage=='smoke':core.package(root,'_smoke');return
    run_wave(root/'screen',repos,matrix(),gpus)
    write_report(root)
    core.save(root/'completion.json',dict(experiments_complete=True,new_cells=54,reference_cells=72,total_rows=126,
        source=source,completed_at=time.time()))


def status(root):
    cal.status(root,smoke_selector=smoke_cells,smoke_epochs=2)
    for m in WEIGHTS:
        done=sum((core.cell_path(root/'screen',c)/'complete.json').exists() for c in matrix() if c[3]==m)
        print('%s: %d/18' % (m,done))
    teacher=core.read(root/'teacher_manifest.json') if (root/'teacher_manifest.json').exists() else {}
    print('Hash-verified Ordered teachers: %d/18' % len(teacher.get('records',[])))
    print('54 new = 6 actions x 3 seeds x 3 arms; complete cells are kept on rerun.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('all','smoke','screen','status','package'),default='all')
    p.add_argument('--root',default='results/qmd_residual_v12');p.add_argument('--gpus',default='0 1 3 4')
    p.add_argument('--reference_root',default='../PAMFN-qmd-strength-v11/results/qmd_strength_v11')
    p.add_argument('--teacher_root',default='../PAMFN-qmd-dual-v2/results/qmd_dual_v2')
    p.add_argument('--july28_repo',default='../PAMFN-july28-qmd-residual-v12')
    args=p.parse_args();root=Path(args.root).resolve()
    if args.stage=='status':status(root);return
    full.check_result_roots(root,(args.reference_root,args.teacher_root))
    root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (root/'.runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for path in root.rglob('*.process.json'):
            if not {'reference','attempts'}.intersection(path.relative_to(root).parts) and alive(core.read(path).get('identity')):
                raise RuntimeError('live child remains: '+str(path))
        record=dict(pid=os.getpid(),identity=identity(os.getpid()),state='running',stage=args.stage,started_at=time.time())
        core.save(root/'runner.json',record)
        try:
            execute(args,root)
            record.update(state='finished',ended_at=time.time());core.save(root/'runner.json',record)
            if (root/'completion.json').exists():
                core.package(root,'');print('QMD_RESIDUAL_COMPLETE: 54/54 new; 126 rows; final ZIP ready.',flush=True)
            elif args.stage=='package':core.package(root,'_partial')
        except Exception as error:
            record.update(state='failed',ended_at=time.time(),error=repr(error));core.save(root/'runner.json',record);raise


if __name__=='__main__':main()
