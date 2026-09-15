"""Two-protocol, fixed gradient guard screen; original scoring/inference unchanged."""
import argparse
import csv
import math
from pathlib import Path
import shutil
import sys
import time

from tools import run_qmd_dual as core
from tools import run_qmd_light as light
from tools import run_qmd_cal as cal
from tools.qmd_gradient import recipe
from tools.qmd_process import alive, identity

REPO=Path(__file__).resolve().parents[1]
MODES=('observe','kd','all')
SMOKE_EPOCHS=2
BASE=dict(modern='score_qmd',july28='ordered_score_qmd')


def method(protocol,mode): return BASE[protocol]+'_guard_'+mode


def mode_for(cell):
    mode=cell[3].rsplit('_',1)[-1]
    if cell[0] not in BASE or mode not in MODES or cell[3]!=method(cell[0],mode):
        raise ValueError('unknown gradient cell')
    return mode


def matrix(study='screen'):
    if study not in ('screen','full'): raise ValueError('unknown study')
    actions=('Ball','Hoop','Ribbon') if study=='screen' else core.ACTIONS
    return [(p,a,s,method(p,m)) for p in BASE for a in actions
            for s in ((0,1) if study=='screen' else core.SEEDS[p]) for m in MODES]


def smoke_cells(jobs):
    return [(p,a,0,method(p,m)) for p in sorted({c[0] for c in jobs}) for a in ('Ball','TES') for m in MODES]


def settings(cell,smoke=False):
    return dict(mode='score_qmd',weight=1.,mask_ratio=.15,mask_gt_weight=.5,target_ratio=.075,
                beta_cap=1.,warmup=0. if smoke else .1,rampup=0. if smoke else .1,grad_mode=mode_for(cell))


def cell_args(cell,gpu,smoke=False):
    p,a,s,_=cell
    argv=core.cell_args(p,a,s,BASE[p],gpu,smoke)
    argv[argv.index('--run_tag')+1]=cell[3]
    if smoke: argv[argv.index('--epochs')+1]=str(SMOKE_EPOCHS)
    return argv+['--qmd_grad_mode',mode_for(cell)]


def validate_cell(out,cell,smoke=False):
    row=core.validate_cell(out,cell,smoke,SMOKE_EPOCHS if smoke else None)
    for name in ('train_result.json','test_result.json'):
        if core.read(out/name)['run']['qmd']!=settings(cell,smoke):
            raise ValueError('gradient QMD recipe mismatch')
    qmd=core.read(out/'qmd_training.json')
    from scipy.stats import spearmanr
    import numpy as np
    test=core.read(out/'test_result.json'); predictions=test['predictions']
    y=np.array(predictions['label']); pred=np.array(predictions['prediction'])
    expected_samples=100 if cell[1] in ('TES','PCS') else 50
    if (len(y)!=expected_samples or len(pred)!=len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all()
            or abs(spearmanr(y,pred)[0]*100-test['coef'])>1e-6
            or abs(float(np.mean((pred-y)**2))-test['loss_x100']/100)>1e-6):
        raise ValueError('saved prediction/label results disagree')
    guard=qmd.get('gradient_guard',{})
    rows=guard.get('rows',[])
    if guard.get('recipe')!=recipe(mode_for(cell)) or guard.get('batches')!=qmd['active_batches']:
        raise ValueError('missing or incomplete gradient guard')
    if not rows or sum(r['batches'] for r in rows)!=guard['batches'] or sum(r['projections'] for r in rows)!=guard['projections']:
        raise ValueError('gradient diagnostic counts disagree')
    if len({(r['epoch'],r['modality']) for r in rows})!=len(rows):
        raise ValueError('duplicate gradient diagnostics')
    expected_epochs={r['epoch']:r['active_batches'] for r in qmd['epochs'] if r['active_batches']>0}
    actual_epochs={e:sum(r['batches'] for r in rows if r['epoch']==e) for e in {r['epoch'] for r in rows}}
    if actual_epochs!=expected_epochs:
        raise ValueError('gradient diagnostics do not cover every active epoch')
    for r in rows:
        if (r['modality'] not in ('V','F','A') or r['batches']<=0 or
                any(not math.isfinite(v) for v in r.values() if isinstance(v,(int,float)))):
            raise ValueError('invalid gradient diagnostic values')
        if not 0<=r['projections']<=r['batches']:
            raise ValueError('invalid projection count')
        if any(not 0<=r[key]<=r['batches'] for key in ('gt_conflicts','kd_conflicts','total_conflicts')):
            raise ValueError('invalid conflict counts')
    if mode_for(cell)=='observe' and guard['projections']!=0:
        raise ValueError('observe control changed gradients')
    count=guard['batches']
    row.update(guard_mode=mode_for(cell),projection_fraction=guard['projections']/count,
               gt_conflict_fraction=sum(r['gt_conflicts'] for r in rows)/count,
               kd_conflict_fraction=sum(r['kd_conflicts'] for r in rows)/count,
               all_conflict_fraction=sum(r['total_conflicts'] for r in rows)/count,
               mse=core.read(out/'test_result.json')['loss_x100']/100)
    return row


def check_observe_reference(out,cell,reference):
    """Check entire native trajectory, not only a coincidentally equal best score."""
    ref=core.cell_path(reference,(cell[0],cell[1],cell[2],BASE[cell[0]]))
    import json
    def history(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    left,right=history(out/'model.pth.epochs.jsonl'),history(ref/'model.pth.epochs.jsonl')
    if len(left)!=len(right): raise ValueError('observe/reference epoch count differs')
    for a,b in zip(left,right):
        for key in ('epoch','coef','loss_x100','prediction_std'):
            x,y=a[key],b[key]
            if not ((math.isnan(x) and math.isnan(y)) or abs(x-y)<=1e-6):
                raise ValueError('observe baseline drift at epoch %s, %s' % (a['epoch'],key))


def run_cell(root,repos,cell,gpu,smoke):
    def checked(out,c,s):
        row=validate_cell(out,c,s)
        if not s and mode_for(c)=='observe': check_observe_reference(out,c,root.parent/'reference')
        return row
    return core.run_cell(root,repos,cell,gpu,smoke,argv=cell_args(cell,gpu,smoke),validator=checked)


def run_wave(root,repos,jobs,gpus,smoke=False):
    return core.run_wave(root,repos,jobs,gpus,smoke,runner=run_cell,
             ordering=lambda cells:sorted(cells,key=lambda c:(-core.BUDGET[c[1]],c[2],MODES.index(mode_for(c)),c[0])))


def reference_cells(jobs):
    cells={(p,a,s,BASE[p]) for p,a,s,m in jobs}
    cells.update((p,a,s,'ordered') for p,a,s,m in jobs if p=='july28')
    return sorted(cells)


def checked_reference(reference,source,jobs,root):
    original=core.read(reference/'source.json')
    complete=core.read(reference/'completion.json')
    if (original['commits']!=light.REFERENCE_COMMITS or complete.get('source')!=original
            or complete.get('cells')!=126 or not complete.get('experiments_complete')):
        raise ValueError('requires complete frozen qmd_dual_v2 reference')
    if any(original[k]!=source[k] for k in ('inputs','environment')):
        raise ValueError('reference data/environment changed')
    rows=[]
    for cell in reference_cells(jobs):
        origin=core.cell_path(reference/'screen',cell)
        record=core.read(origin/'complete.json')
        row=core.validate_cell(origin,cell)
        if abs(row['srcc_x100']-record['metrics']['srcc_x100'])>1e-6: raise ValueError('reference score changed')
        for name,digest in record['files'].items():
            if Path(name).name!=name or core.sha(origin/name)!=digest: raise ValueError('reference changed: '+name)
        q=core.read(origin/'test_result.json')['run']['qmd']
        expected=dict(mode='off' if cell[3]=='ordered' else 'score_qmd',weight=1.,mask_ratio=.15,
                      mask_gt_weight=.5,target_ratio=.075,beta_cap=1.,warmup=.1,rampup=.1)
        if q!=expected: raise ValueError('reference QMD recipe changed')
        dest=core.cell_path(root/'reference',cell); dest.mkdir(parents=True,exist_ok=True)
        for name in list(record['files'])+['complete.json']:
            if (dest/name).exists() and core.sha(dest/name)!=core.sha(origin/name): raise ValueError('copied reference changed')
            shutil.copyfile(origin/name,dest/name)
        rows.append(dict(row,origin='frozen_v2_control',source_path=str(origin)))
    core.immutable(root/'reference_manifest.json',dict(source=original,records=rows))
    return rows


def summarize(root,jobs,references):
    rows=[validate_cell(core.cell_path(root/'screen',c),c) for c in jobs]
    lookup={(r['protocol'],r['action'],r['seed'],r['method']):r for r in references}
    observed={(r['protocol'],r['action'],r['seed']):r for r in rows if r['guard_mode']=='observe'}
    for row in rows:
        key=(row['protocol'],row['action'],row['seed'])
        ref=lookup[key+(BASE[row['protocol']],)]
        row['original_combination_srcc']=ref['srcc_x100']
        row['delta_vs_original']=row['srcc_x100']-ref['srcc_x100']
        row['delta_vs_observe']=row['srcc_x100']-observed[key]['srcc_x100']
        control=lookup[key+('ordered',)] if key[0]=='july28' else ref
        row['control_srcc']=control['srcc_x100']
        row['delta_vs_control']=row['srcc_x100']-control['srcc_x100']
        if row['guard_mode']=='observe': check_observe_reference(core.cell_path(root/'screen',key+(row['method'],)),key+(row['method'],),root/'reference')
    groups=[]
    for p,m in sorted({(r['protocol'],r['method']) for r in rows}):
        group=[r for r in rows if (r['protocol'],r['method'])==(p,m)]
        actions=sorted({r['action'] for r in group}); seeds=sorted({r['seed'] for r in group})
        if len(group)!=len(actions)*len(seeds): raise ValueError('incomplete paired grid')
        mean=lambda key, subset=group: sum(r[key] for r in subset)/len(subset)
        groups.append(dict(protocol=p,method=m,actions=actions,seeds=seeds,cells=len(group),
             mean_srcc_x100=mean('srcc_x100'),delta_vs_original=mean('delta_vs_original'),delta_vs_control=mean('delta_vs_control'),
             per_action={a:mean('srcc_x100',[r for r in group if r['action']==a]) for a in actions},
             per_action_delta={a:mean('delta_vs_control',[r for r in group if r['action']==a]) for a in actions},
             per_seed={str(s):mean('srcc_x100',[r for r in group if r['seed']==s]) for s in seeds}))
    return dict(raw=rows,methods=groups,note='Fixed whole-run recipes; no per-action/seed selection. Control=modern Score-QMD or July28 Ordered. Original=within-protocol unprotected QMD combination. Projection is Euclidean, not an Adam or SRCC guarantee.')


def execute(args,root):
    if args.stage=='package':
        if (root/'completion.json').exists():
            summarize(root,[tuple(c) for c in core.read(root/'manifest.json')['jobs']],core.read(root/'reference_manifest.json')['records'])
        return
    jobs=matrix(args.study)
    core.immutable(root/'manifest.json',dict(study=args.study,jobs=jobs,total_new_cells=len(jobs),
                   recipes={m:recipe(m) for m in MODES},epochs=core.BUDGET,smoke_epochs=SMOKE_EPOCHS))
    gpus=args.gpus.split()
    if not gpus or len(gpus)!=len(set(gpus)) or any(not x.isdigit() for x in gpus): raise ValueError('distinct GPU indices required')
    cal.check_resources(root,gpus)
    repos=dict(modern=REPO,july28=Path(args.july28_repo).resolve())
    if core.sha(REPO/'tools/qmd_gradient.py')!=core.sha(repos['july28']/'tools/qmd_gradient.py'):
        raise ValueError('gradient guard implementation differs between protocols')
    print('PREFLIGHT: source, 37 input hashes, environment, frozen paired controls',flush=True)
    source=core.provenance(repos); core.immutable(root/'source.json',source)
    refs=checked_reference(Path(args.reference_root).resolve(),source,jobs,root)
    if getattr(args,'reuse_root',None):
        from tools.reuse_qmd_gradient_v8 import reuse_completed
        reuse_completed(args.reuse_root,root,repos,source,jobs)
    gate=root/'smoke/gate.json'; smoke=smoke_cells(jobs)
    if args.stage in ('all','smoke') and not gate.exists():
        for p in repos:
            core.command([sys.executable,'-u','-m','unittest','tests.test_qmd_gradient'],repos[p],root/'smoke'/('unit_'+p+'.log'))
            core.command([sys.executable,'-u',str(REPO/'tools/verify_qmd_gradient_runtime.py'),
                          '--protocol',p,'--gpu',gpus[0],'--output',str(root/'smoke'/('parity_'+p+'.json'))],
                         repos[p],root/'smoke'/('parity_'+p+'.log'))
        core.command([sys.executable,'-u','-m','unittest','tests.test_qmd_gradient_tools'],REPO,root/'smoke/unit_tools.log')
        run_wave(root/'smoke',repos,smoke,gpus,True)
        rows=[validate_cell(core.cell_path(root/'smoke',c),c,True) for c in smoke]
        if any(r['warnings'] for r in rows): raise RuntimeError('ineffective QMD smoke; inspect warnings')
        core.save(gate,dict(decision='QMD_GRAD_SMOKE_PASS',cells=len(smoke),source=source))
    if not gate.exists() or core.read(gate)!=dict(decision='QMD_GRAD_SMOKE_PASS',cells=len(smoke),source=source):
        raise RuntimeError('matching smoke PASS required')
    if args.stage=='smoke': core.package(root,'_smoke'); return
    pilot=[c for c in jobs if c[2]==0]
    run_wave(root/'screen',repos,pilot,gpus)
    core.save(root/'pilot_summary.json',summarize(root,pilot,refs)); core.package(root,'_pilot')
    run_wave(root/'screen',repos,[c for c in jobs if c not in pilot],gpus)
    report=summarize(root,jobs,refs); core.save(root/'summary.json',report)
    with (root/'scores.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(report['raw'][0])); writer.writeheader(); writer.writerows(report['raw'])
    core.save(root/'completion.json',dict(experiments_complete=True,new_cells=len(jobs),reference_cells=len(refs),source=source,completed_at=time.time()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('all','smoke','screen','status','package'),default='all')
    p.add_argument('--study',choices=('screen','full'),default='screen')
    p.add_argument('--root',default='results/qmd_grad_v8_fix1')
    p.add_argument('--gpus',default='0 1 3 4')
    p.add_argument('--reference_root',default='../PAMFN-qmd-dual-v2/results/qmd_dual_v2')
    p.add_argument('--july28_repo',default='../PAMFN-july28-qmd-grad-v8-fix1')
    p.add_argument('--reuse_root',help='Read-only import of completed original v8 cells into a separate fix1 root')
    args=p.parse_args(); root=Path(args.root).resolve()
    if args.stage=='status':
        cal.status(root,smoke_selector=smoke_cells,smoke_epochs=SMOKE_EPOCHS)
        if (root/'reuse_manifest.json').exists():
            reused={tuple(r['cell']) for r in core.read(root/'reuse_manifest.json')['cells']}
            jobs=[tuple(c) for c in core.read(root/'manifest.json')['jobs']]
            remaining=[c for c in jobs if c not in reused]
            done=sum((core.cell_path(root/'screen',c)/'complete.json').is_file() for c in remaining)
            print('Reused original v8: %d ; fix1 training completed: %d/%d' % (len(reused),done,len(remaining)))
        return
    import fcntl
    import os
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for path in root.rglob('*.process.json'):
            if not {'attempts','original_v8'}.intersection(path.relative_to(root).parts) and alive(core.read(path).get('identity')):
                raise RuntimeError('live child remains: '+str(path))
        record=dict(pid=os.getpid(),identity=identity(os.getpid()),state='running',stage=args.stage,started_at=time.time())
        core.save(root/'runner.json',record)
        try:
            execute(args,root)
            record.update(state='finished',ended_at=time.time()); core.save(root/'runner.json',record)
            if (root/'completion.json').exists():
                core.package(root,''); print('QMD_GRAD_COMPLETE: full results and ZIP ready.',flush=True)
            elif args.stage=='package': core.package(root,'_partial')
        except Exception as error:
            record.update(state='failed',ended_at=time.time(),error=repr(error)); core.save(root/'runner.json',record)
            raise


if __name__=='__main__': main()
