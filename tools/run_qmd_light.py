"""36-cell light-QMD screen with read-only v2 controls and optional Ball probe."""
import argparse
import csv
import math
from pathlib import Path
import shutil
import sys
import time

from tools import run_qmd_dual as core

REPO = Path(__file__).resolve().parents[1]
REFERENCE_COMMITS = dict(modern='db579d5cdc977c2193a3838628ffc10454b79543',
                         july28='ad13082a15b36041268f0eb9685978a0c87e346b')
CONTROLS = dict(modern='score_qmd', july28='ordered')
RECIPES = {
    ('modern','reliable_score_qmd_floor05'): dict(mode='reliable_score_qmd',weight=1.,mask_ratio=.15,score_gate_floor=.5),
    ('july28','ordered_score_qmd_w010'): dict(mode='score_qmd',weight=.1,mask_ratio=.15),
    ('july28','ordered_score_qmd_w025'): dict(mode='score_qmd',weight=.25,mask_ratio=.15),
    ('modern','score_qmd_mask05'): dict(mode='score_qmd',weight=1.,mask_ratio=.05),
}
SCREEN = tuple(RECIPES)[:3]


def matrix(study='screen', selected=()):
    if study == 'screen':
        recipes,actions = SCREEN,core.ACTIONS
    elif study == 'ball-mask':
        recipes,actions = (('modern','score_qmd_mask05'),),('Ball',)
    elif study == 'confirm':
        recipes,actions = tuple(selected),core.ACTIONS
        if not recipes or len(set(recipes))!=len(recipes) or any(r not in SCREEN for r in recipes):
            raise ValueError('confirmation requires distinct known protocol:method names')
    else:
        raise ValueError('unknown study')
    return [(p,a,s,m) for p,m in recipes for a in actions
            for s in ((0,1) if study=='screen' else (0,1,42) if study=='ball-mask'
                      else (42,) if p=='modern' else (2,))]


def smoke_cells(jobs):
    recipes = sorted(set((p,m) for p,a,s,m in jobs))
    actions = ('Ball',) if set(c[1] for c in jobs)=={'Ball'} else ('Ball','TES')
    return [(p,a,0,m) for p,m in recipes for a in actions]


def qmd_settings(cell, smoke=False):
    p,a,s,m=cell
    recipe=RECIPES[(p,m)]
    return dict(recipe,mask_gt_weight=.5,target_ratio=.075,beta_cap=1.,
                warmup=0. if smoke else .1,rampup=0. if smoke else .1)


def cell_args(cell, gpu, smoke=False):
    p,a,s,m=cell
    if (p,m) not in RECIPES or a not in core.ACTIONS or s not in core.SEEDS[p]:
        raise ValueError('cell outside supported recipes/actions/seeds')
    settings=qmd_settings(cell,smoke)
    base='reliable_score_qmd' if settings['mode']=='reliable_score_qmd' else ('ordered_score_qmd' if p=='july28' else 'score_qmd')
    argv=core.cell_args(p,a,s,base,gpu,smoke)
    for flag,value in [('--run_tag',m),('--qmd_weight',settings['weight']),('--qmd_mask_ratio',settings['mask_ratio'])]:
        argv[argv.index(flag)+1]=str(value)
    if 'score_gate_floor' in settings:
        argv += ['--qmd_score_gate_floor',str(settings['score_gate_floor'])]
    return argv


def validate_cell(out, cell, smoke=False):
    row=core.validate_cell(out,cell,smoke)
    expected=qmd_settings(cell,smoke)
    for name in ('train_result.json','test_result.json'):
        if core.read(out/name)['run']['qmd']!=expected:
            raise ValueError('QMD recipe differs from planned settings: '+str(out))
    qmd=core.read(out/'qmd_training.json')
    if expected['mode']=='reliable_score_qmd' and qmd['score_gate'].get('floor')!=expected['score_gate_floor']:
        raise ValueError('soft gate floor was not applied')
    views=qmd.get('view_diagnostics',{}).get('rows',[])
    if not views or not all(v['samples']>0 and v['modality'] in ('V','F','A') for v in views):
        raise ValueError('missing training view diagnostics')
    if expected['mode']=='reliable_score_qmd' and sum(v['samples'] for v in views)!=qmd['score_gate']['total_samples']:
        raise ValueError('view diagnostic denominator mismatch')
    row.update(qmd_weight=expected['weight'],mask_ratio=expected['mask_ratio'],
               score_gate_floor=expected.get('score_gate_floor',0.),origin='new_training')
    return row


def run_cell(root,repos,cell,gpu,smoke):
    return core.run_cell(root,repos,cell,gpu,smoke,argv=cell_args(cell,gpu,smoke),validator=validate_cell)


def dispatch_order(jobs):
    return sorted(jobs,key=lambda c:(-core.BUDGET[c[1]],c[2],list(RECIPES).index((c[0],c[3]))))


def run_wave(root,repos,jobs,gpus,smoke=False):
    return core.run_wave(root,repos,jobs,gpus,smoke,runner=run_cell,ordering=dispatch_order)


def checked_reference(reference,source,jobs,output):
    """Validate actual v2 records, never silently replace them by rounded scores."""
    ref_source=core.read(reference/'source.json')
    complete=core.read(reference/'completion.json')
    if ref_source['commits']!=REFERENCE_COMMITS or complete.get('source')!=ref_source:
        raise ValueError('reference must be the completed frozen qmd_dual_v2 study')
    if complete.get('cells')!=126 or not complete.get('experiments_complete'):
        raise ValueError('reference study is incomplete')
    for key in ('inputs','environment'):
        if ref_source[key]!=source[key]:
            raise ValueError('reference '+key+' differ; cannot reuse controls in this environment')
    rows=[]
    for cell in sorted(set((p,a,s,CONTROLS[p]) for p,a,s,m in jobs)):
        origin=core.cell_path(reference/'screen',cell)
        record=core.read(origin/'complete.json')
        for name,digest in record['files'].items():
            if Path(name).name!=name or core.sha(origin/name)!=digest:
                raise ValueError('reference file changed or unsafe: '+str(origin/name))
        row=core.validate_cell(origin,cell)
        if abs(row['srcc_x100']-record['metrics']['srcc_x100'])>1e-6:
            raise ValueError('reference completion metric mismatch')
        p,a,s,m=cell
        expected=dict(mode='score_qmd' if p=='modern' else 'off',weight=1.,mask_ratio=.15,
                      mask_gt_weight=.5,target_ratio=.075,beta_cap=1.,warmup=.1,rampup=.1)
        if core.read(origin/'test_result.json')['run']['qmd']!=expected:
            raise ValueError('reference recipe changed')
        dest=core.cell_path(output/'reference',cell)
        dest.mkdir(parents=True,exist_ok=True)
        names=list(record['files'])+['complete.json']
        for name in names:
            if (dest/name).exists() and core.sha(dest/name)!=core.sha(origin/name):
                raise ValueError('previously copied reference changed')
            shutil.copyfile(origin/name,dest/name)
        rows.append(dict(row,origin='reused_v2_control',source_path=str(origin),
                         completion_sha256=core.sha(origin/'complete.json')))
    result=dict(source=ref_source,records=rows)
    core.immutable(output/'reference_manifest.json',result)
    return rows


def summarize(root,jobs,references,validator=None):
    check = validate_cell if validator is None else validator
    rows=[check(core.cell_path(root/'screen',c),c) for c in jobs]
    lookup={(r['protocol'],r['action'],r['seed']):r for r in references}
    for row in rows:
        ref=lookup[(row['protocol'],row['action'],row['seed'])]
        row['control_srcc_x100']=ref['srcc_x100']
        row['delta_vs_control']=row['srcc_x100']-ref['srcc_x100']
    groups=[]
    for p,m in sorted(set((r['protocol'],r['method']) for r in rows)):
        group=[r for r in rows if (r['protocol'],r['method'])==(p,m)]
        seeds=sorted(set(r['seed'] for r in group))
        actions=sorted(set(r['action'] for r in group))
        if len(group)!=len(seeds)*len(actions):
            raise ValueError('incomplete candidate matrix')
        groups.append(dict(protocol=p,method=m,seeds=seeds,actions=actions,cells=len(group),
            control=CONTROLS[p],mean_srcc_x100=sum(r['srcc_x100'] for r in group)/len(group),
            delta_vs_control=sum(r['delta_vs_control'] for r in group)/len(group),
            per_action={a:sum(r['srcc_x100'] for r in group if r['action']==a)/len(seeds) for a in actions},
            per_action_delta={a:sum(r['delta_vs_control'] for r in group if r['action']==a)/len(seeds) for a in actions},
            per_seed_delta={str(s):sum(r['delta_vs_control'] for r in group if r['seed']==s)/len(actions) for s in seeds}))
    views=[]
    for c in jobs:
        data=core.read(core.cell_path(root/'screen',c)/'qmd_training.json')
        for lo,hi in ((0,.1),(.1,.2),(.2,.6),(.6,1.)):
            for modality in ('V','F','A'):
                values=[v for v in data['view_diagnostics']['rows'] if v['modality']==modality
                        and lo*core.BUDGET[c[1]]<v['epoch']<=hi*core.BUDGET[c[1]]]
                n=sum(v['samples'] for v in values)
                if not n:
                    continue
                full=sum(v['teacher_abs_error_sum'] for v in values)/n
                masked=sum(v['student_abs_error_sum'] for v in values)/n
                views.append(dict(protocol=c[0],action=c[1],seed=c[2],method=c[3],modality=modality,
                    epoch_fraction_start=lo,epoch_fraction_end=hi,samples=n,teacher_mae=full,
                    masked_mae=masked,mask_error_increase=masked-full,
                    teacher_better_fraction=sum(v['accepted_samples'] for v in values)/n))
    return dict(raw=rows,methods=groups,training_view_diagnostics=views,
                note='SRCC x100, equal action/seed weights, paired within protocol; controls reused from v2, not retrained; Ball-only probes are not all-action scores')


def status(root):
    if not (root/'manifest.json').exists():
        print('Manifest not created yet: inspect launcher/preflight log; training may not have started.')
        return
    manifest=core.read(root/'manifest.json')
    jobs=[tuple(c) for c in manifest['jobs']]
    done=sum((core.cell_path(root/'screen',c)/'complete.json').exists() for c in jobs)
    print('Study:',manifest['study'],'; new training completed: %d/%d'%(done,len(jobs)))
    gate=root/'smoke/gate.json'
    smoke=smoke_cells(jobs)
    print('Smoke: %d/%d; gate: %s'%(sum((core.cell_path(root/'smoke',c)/'complete.json').exists() for c in smoke),
          len(smoke),'PASS' if gate.exists() else 'not yet passed'))
    for c in jobs if gate.exists() else smoke:
        out=core.cell_path(root/('screen' if gate.exists() else 'smoke'),c)
        if (out/'complete.json').exists() or not (out/'train.log').exists():
            continue
        lines=[s for s in (out/'train.log').read_text(errors='replace').splitlines() if s.startswith('Epoch[')]
        print('UNFINISHED','/'.join(map(str,c)),lines[-1].split('\t')[0] if lines else 'loading / first epoch')
    print('Completion:', 'READY' if (root/'completion.json').exists() else 'not yet created')
    for suffix in ('','_pilot','_partial','_smoke'):
        path=root.parent/(root.name+suffix+'_return.zip')
        print('Archive:',path,'READY' if path.exists() else 'not yet packaged')
    print('These are recorded results. Check pgrep and nvidia-smi for live processes.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('all','smoke','screen','status','package'),default='all')
    p.add_argument('--study',choices=('screen','ball-mask','confirm'),default='screen')
    p.add_argument('--methods',nargs='*',default=[])
    p.add_argument('--root')
    p.add_argument('--reference_root',default='../PAMFN-qmd-dual-v2/results/qmd_dual_v2')
    p.add_argument('--july28_repo',default='../PAMFN-july28-qmd-light-v3')
    p.add_argument('--gpus',default='0 1 3 4')
    args=p.parse_args()
    root=Path(args.root or ('results/qmd_light_v3'+('' if args.study=='screen' else '_'+args.study))).resolve()
    if args.stage=='status':
        status(root)
        return
    root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (root/'.runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=='package':
            if (root/'completion.json').exists():
                jobs=[tuple(c) for c in core.read(root/'manifest.json')['jobs']]
                summarize(root,jobs,core.read(root/'reference_manifest.json')['records'])
                core.package(root,'')
            else:
                core.package(root,'_partial')
            return
        if args.methods and args.study!='confirm':
            p.error('--methods is only for confirmation')
        selected=[tuple(m.split(':')) for m in args.methods]
        jobs=matrix(args.study,selected)
        gpus=args.gpus.split()
        if not gpus or len(set(gpus))!=len(gpus) or any(not g.isdigit() for g in gpus):
            raise ValueError('use distinct physical GPU indices')
        repos=dict(modern=REPO,july28=Path(args.july28_repo).resolve())
        print('Checking code, 37 input hashes, environment and read-only v2 controls...',flush=True)
        source=core.provenance(repos)
        core.immutable(root/'source.json',source)
        core.immutable(root/'manifest.json',dict(study=args.study,jobs=jobs,recipes={p+':'+m:RECIPES[(p,m)] for p,a,s,m in jobs},
            epochs=core.BUDGET,selection='unchanged original test-best per protocol',total_new_cells=len(jobs)))
        references=checked_reference(Path(args.reference_root).resolve(),source,jobs,root)
        gate_path=root/'smoke/gate.json'
        smoke=smoke_cells(jobs)
        if args.stage in ('all','smoke') and not gate_path.exists():
            checks=[('modern',['-m','unittest','tests.test_qmd_light','tests.test_qmd_light_tools',
                              'tests.test_reliable_score','tests.test_score_qmd','tests.test_sr_qmd'], 'unit_modern'),
                    ('july28',['-m','unittest','tests.test_july28_qmd','tests.test_light_diagnostics'],'unit_july28'),
                    ('modern',['-m','tools.verify_sr_qmd_runtime','--gpu',gpus[0],
                               '--output',str(root/'smoke/score_parity.json')],'score_parity'),
                    ('july28',['-m','tools.verify_july28_qmd','--gpu',gpus[-1],
                               '--output',str(root/'smoke/july28_parity.json')],'july28_parity')]
            for protocol,argv,name in checks:
                print('CHECK '+name,flush=True)
                core.command([sys.executable,'-u']+argv,repos[protocol],root/'smoke'/(name+'.log'))
            run_wave(root/'smoke',repos,smoke,gpus,True)
            rows=[validate_cell(core.cell_path(root/'smoke',c),c,True) for c in smoke]
            if any(r['warnings'] for r in rows):
                raise RuntimeError('smoke found ineffective auxiliary: inspect its raw logs')
            core.save(gate_path,dict(decision='QMD_LIGHT_SMOKE_PASS',source=source,cells=len(smoke),results=rows))
        if not gate_path.exists() or core.read(gate_path)['source']!=source:
            raise RuntimeError('matching smoke gate required before full training')
        if args.stage=='smoke':
            core.package(root,'_smoke')
            return
        pilot=[c for c in jobs if c[2]==0]
        if pilot:
            run_wave(root/'screen',repos,pilot,gpus)
            core.save(root/'pilot_summary.json',summarize(root,pilot,references))
            core.package(root,'_pilot')
        run_wave(root/'screen',repos,[c for c in jobs if c not in pilot],gpus)
        report=summarize(root,jobs,references)
        core.save(root/'summary.json',report)
        for name,rows in [('scores.csv',report['raw']),('training_views.csv',report['training_view_diagnostics'])]:
            with (root/name).open('w',newline='',encoding='utf-8') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        core.save(root/'completion.json',dict(experiments_complete=True,study=args.study,new_cells=len(jobs),
            reference_cells=len(references),source=source,completed_at=time.time()))
        core.package(root,'')
        print('QMD_LIGHT_COMPLETE: %d/%d new trainings; return ZIP ready.'%(len(jobs),len(jobs)),flush=True)


if __name__=='__main__':
    main()
