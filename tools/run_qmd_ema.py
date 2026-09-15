"""Six independent training trajectories, four scoring views, automatic return ZIP."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

from tools import run_qmd_dual as core
from tools import run_qmd_light as light
from tools import run_qmd_cal as cal
from tools.qmd_process import alive, identity

REPO = Path(__file__).resolve().parents[1]
METHOD = 'score_qmd_ema'
VIEWS = ('online_bn', 'ema_short', 'ema_long')
SMOKE_EPOCHS = 2


def matrix(study='screen'):
    if study == 'screen':
        actions, seeds = ('Ball', 'Hoop', 'Ribbon'), (0, 1)
    elif study == 'all-actions':
        actions, seeds = core.ACTIONS, (0, 1)
    elif study == 'confirm':
        actions, seeds = core.ACTIONS, (42,)
    else:
        raise ValueError('unknown study')
    return [('modern', a, s, METHOD) for a in actions for s in seeds]


def smoke_cells(jobs):
    return [('modern', a, 0, METHOD) for a in ('Ball', 'TES')] if jobs else []


def settings(smoke=False):
    return dict(mode='score_qmd', weight=1., mask_ratio=.15, mask_gt_weight=.5,
                target_ratio=.075, beta_cap=1., warmup=0. if smoke else .1, rampup=0. if smoke else .1)


def cell_args(cell, gpu, smoke=False):
    p, a, s, m = cell
    if p != 'modern' or m != METHOD:
        raise ValueError('EMA only supports the fixed modern Score-QMD recipe')
    argv = core.cell_args(p, a, s, 'score_qmd', gpu, smoke)
    argv[argv.index('--run_tag') + 1] = METHOD
    if smoke:
        argv[argv.index('--epochs') + 1] = str(SMOKE_EPOCHS)
    return argv + ['--ema_screen']


def check_close(a, b):
    if not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > 1e-6:
        raise ValueError('EMA recorded reload/selection mismatch: %r versus %r' % (a,b))


def validate_cell(out, cell, smoke=False):
    from models.ema_shadow import recipe
    row = core.validate_cell(out, cell, smoke, SMOKE_EPOCHS if smoke else None)
    epochs = SMOKE_EPOCHS if smoke else core.BUDGET[cell[1]]
    for filename in ('train_result.json', 'test_result.json'):
        run = core.read(out/filename)['run']
        if (run['qmd'] != settings(smoke) or run.get('ema') != dict(recipe=recipe(), view='online')
                or run['eval_seed'] != 20260908 or 'score_head' in run):
            raise ValueError('original Score-QMD or EMA recipe changed')
    report = core.read(out/'ema_training.json')
    if report['recipe'] != recipe() or report['epochs'] != epochs or report['updates'] < epochs:
        raise ValueError('incomplete EMA training report')
    qmd = core.read(out/'qmd_training.json')
    if qmd['calibration']['scope'] != 'score_head_parameters_first_active_training_batch':
        raise ValueError('original first-batch calibration required')
    views = {}
    for view in VIEWS:
        test = core.read(out/(view+'.test_result.json'))
        selected = core.read(out/(view+'.selection.json'))
        hist = [json.loads(s) for s in (out/(view+'.epochs.jsonl')).read_text().splitlines() if s.strip()]
        if [h['epoch'] for h in hist] != list(range(1,epochs+1)):
            raise ValueError('incomplete EMA evaluation history')
        if test['run'] != dict(core.read(out/'test_result.json')['run'], ema=dict(recipe=recipe(),view=view)):
            raise ValueError('EMA reload used a different configuration')
        if selected != report['best'][view]:
            raise ValueError('EMA selection report changed')
        for value in (selected['coef'], hist[selected['epoch']-1]['coef'],
                      max(h['coef'] for h in hist if math.isfinite(h['coef']))):
            check_close(test['coef'], value)
        for key in ('loss_x100','prediction_std','prediction_std_ratio','mae','bias'):
            check_close(test[key], selected[key])
        for h in hist:
            if h['bn_calibration']['source'] != 'train' or h['bn_calibration']['samples'] < 2:
                raise ValueError('missing train-only BN coverage')
        if hist[-1]['updates'] != report['updates']:
            raise ValueError('EMA update counter differs from training')
        late = [h['coef'] for h in hist if h['epoch'] > .6*epochs and math.isfinite(h['coef'])]
        views[view] = dict(srcc_x100=test['coef'], selected_epoch=selected['epoch'],
            mse=test['loss_x100']/100, prediction_std_ratio=test['prediction_std_ratio'],
            nan_eval_epochs=sum(not math.isfinite(h['coef']) for h in hist),
            final_srcc_x100=hist[-1]['coef'], late_mean=sum(late)/len(late) if late else None,
            delta_vs_online=test['coef']-row['srcc_x100'])
    row.update(origin='one_training_trajectory_four_views', ema_views=views)
    return row


def reload_views(out, argv, repo):
    for view in VIEWS:
        core.command(argv + ['--test','--ema_view',view,'--ckpt_path',str(out/(view+'.pth')),
                            '--result_json',str(out/(view+'.test_result.json'))], repo, out/(view+'.test.log'))


def run_cell(root, repos, cell, gpu, smoke=False):
    return core.run_cell(root, repos, cell, gpu, smoke, argv=cell_args(cell,gpu,smoke),
        validator=validate_cell, post_test=reload_views, extra_required=('ema_training.json',),
        extra_checkpoints=tuple(v+'.pth' for v in VIEWS))


def run_wave(root, repos, jobs, gpus, smoke=False):
    return core.run_wave(root,repos,jobs,gpus,smoke,runner=run_cell,
        ordering=lambda cells: sorted(cells,key=lambda c:(0 if c[1]=='Ball' else 1 if c[1]=='Ribbon' else 2,c[2])))


def summarize(root, jobs, references):
    refs = {(r['action'],r['seed']):r for r in references}
    rows, reference_checks = [], []
    for c in jobs:
        out = core.cell_path(root/'screen',c)
        result = validate_cell(out,c)
        # Historical replay is diagnostic; candidate deltas ALWAYS use their own paired student.
        ref = refs[(c[1],c[2])]
        old_path = core.cell_path(root/'reference',(c[0],c[1],c[2],'score_qmd'))
        old = [json.loads(s) for s in (old_path/'model.pth.epochs.jsonl').read_text().splitlines() if s.strip()]
        new = [json.loads(s) for s in (out/'model.pth.epochs.jsonl').read_text().splitlines() if s.strip()]
        if len(old) != len(new):
            raise ValueError('frozen reference epoch budget differs')
        differences=[abs(a['coef']-b['coef']) for a,b in zip(old,new)
                     if math.isfinite(a['coef']) and math.isfinite(b['coef'])]
        nonfinite_mismatch=any(math.isfinite(a['coef']) != math.isfinite(b['coef']) for a,b in zip(old,new))
        matches=not nonfinite_mismatch and max(differences,default=0.)<=1e-6
        reference_checks.append(dict(action=c[1],seed=c[2],matches=matches,
            max_epoch_score_difference=max(differences,default=None),nonfinite_mismatch=nonfinite_mismatch,
            historical_score=ref['srcc_x100'],student_score=result['srcc_x100']))
        all_views = dict(online=dict(srcc_x100=result['srcc_x100'],selected_epoch=result['selected_epoch'],delta_vs_online=0.),
                         **result['ema_views'])
        bn_score = all_views['online_bn']['srcc_x100']
        for view,data in all_views.items():
            rows.append(dict(action=c[1],seed=c[2],view=view,control_srcc_x100=result['srcc_x100'],
                delta_vs_online_bn=data['srcc_x100']-bn_score,**data))
    actions, seeds = sorted({c[1] for c in jobs}),sorted({c[2] for c in jobs})
    groups=[]
    for view in ('online',)+VIEWS:
        selected=[r for r in rows if r['view']==view]
        if len(selected)!=len(actions)*len(seeds):
            raise ValueError('incomplete EMA comparison matrix')
        groups.append(dict(view=view,mean_srcc_x100=sum(r['srcc_x100'] for r in selected)/len(selected),
            delta_vs_online=sum(r['delta_vs_online'] for r in selected)/len(selected),
            delta_vs_online_bn=sum(r['delta_vs_online_bn'] for r in selected)/len(selected),
            per_action={a:sum(r['srcc_x100'] for r in selected if r['action']==a)/len(seeds) for a in actions},
            per_seed={str(s):sum(r['srcc_x100'] for r in selected if r['seed']==s)/len(actions) for s in seeds}))
    return dict(raw=rows,groups=groups,actions=actions,seeds=seeds,new_training_runs=len(jobs),
        scoring_views=len(rows),student_reference_trajectory_matches=all(x['matches'] for x in reference_checks),
        reference_checks=reference_checks,
        warnings=[] if all(x['matches'] for x in reference_checks) else ['student_differs_from_historical_reference; compare EMA to same-trajectory online student'],
        note='SRCC x100; same-trajectory paired comparisons; all planned seeds included. '
             'Three-action screen is not an all-action gain. EMA views include train-only BN refresh; '
             'online_bn isolates that refresh. No student feedback, best-seed substitution or winner mosaic.')


def status(root):
    cal.status(root,smoke_selector=smoke_cells,smoke_epochs=SMOKE_EPOCHS)
    manifest=cal.read_optional(root/'manifest.json')
    jobs=[tuple(c) for c in manifest.get('jobs',[])]
    count=sum((core.cell_path(root/'screen',c)/(v+'.test_result.json')).exists() for c in jobs for v in VIEWS)
    print('Shadow/BN reload results: %d/%d (three extra views per training, NOT extra training runs)' % (count,3*len(jobs)))
    phase='screen' if (root/'smoke/gate.json').exists() else 'smoke'
    for c in jobs if phase=='screen' else smoke_cells(jobs):
        out=core.cell_path(root/phase,c)
        if (out/'complete.json').exists() or not (out/'model.pth.epochs.jsonl').exists():
            continue
        progress={v:len((out/(v+'.epochs.jsonl')).read_text().splitlines())
                  if (out/(v+'.epochs.jsonl')).exists() else 0 for v in VIEWS}
        print('Shadow epoch progress:',c[1],'seed'+str(c[2]),progress)


def execute(args, root):
    if args.stage=='package':
        if (root/'completion.json').exists():
            jobs=[tuple(c) for c in core.read(root/'manifest.json')['jobs']]
            core.save(root/'summary.json',summarize(root,jobs,core.read(root/'reference_manifest.json')['records']))
        core.package(root,'' if (root/'completion.json').exists() else '_partial')
        return
    from models.ema_shadow import recipe
    jobs=matrix(args.study)
    core.immutable(root/'manifest.json',dict(study=args.study,jobs=jobs,total_new_cells=len(jobs),
        scoring_views=4*len(jobs),ema=recipe(),qmd=settings(),epochs=core.BUDGET,smoke_epochs=SMOKE_EPOCHS,
        selection='argmax test-best independently within each fixed view; all seeds retained'))
    gpus=args.gpus.split()
    if not gpus or len(set(gpus))!=len(gpus) or any(not g.isdigit() for g in gpus):
        raise ValueError('provide distinct physical GPU indices')
    cal.check_resources(root,gpus)
    print('Preflight: inputs, environment and original Score-QMD controls...',flush=True)
    repos=dict(modern=REPO)
    source=core.provenance(repos)
    core.immutable(root/'source.json',source)
    references=light.checked_reference(Path(args.reference_root).resolve(),source,jobs,root)
    gate_path=root/'smoke/gate.json'
    smoke=smoke_cells(jobs)
    if args.stage in ('all','smoke') and not gate_path.exists():
        for argv,name in [(['-m','unittest','tests.test_ema_shadow','tests.test_qmd_ema_tools',
                           'tests.test_score_qmd','tests.test_sr_qmd'],'unit'),
                          (['-m','tools.verify_ema_runtime','--gpu',gpus[0],
                            '--output',str(root/'smoke/ema_parity.json')],'ema_parity')]:
            core.command([sys.executable,'-u']+argv,REPO,root/'smoke'/(name+'.log'))
        run_wave(root/'smoke',repos,smoke,gpus,True)
        rows=[validate_cell(core.cell_path(root/'smoke',c),c,True) for c in smoke]
        if any(r['warnings'] for r in rows):
            raise RuntimeError('Score-QMD ineffective in smoke')
        core.save(gate_path,dict(decision='QMD_EMA_SMOKE_PASS',source=source,cells=len(smoke),results=rows))
    if not gate_path.exists() or core.read(gate_path)['source']!=source:
        raise RuntimeError('matching smoke PASS required')
    if args.stage=='smoke':
        core.package(root,'_smoke')
        return
    # Both seeds are dispatched together; no score-based stopping or selection.
    run_wave(root/'screen',repos,jobs,gpus)
    core.save(root/'summary.json',summarize(root,jobs,references))
    core.save(root/'completion.json',dict(experiments_complete=True,new_cells=len(jobs),scoring_views=4*len(jobs),
        reference_cells=len(references),source=source,study=args.study,completed_at=time.time()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('all','smoke','screen','status','package'),default='all')
    p.add_argument('--study',choices=('screen','all-actions','confirm'),default='screen')
    p.add_argument('--root',default='results/qmd_ema_v6')
    p.add_argument('--reference_root',default='../PAMFN-qmd-dual-v2/results/qmd_dual_v2')
    p.add_argument('--gpus',default='0 1 3 4')
    args=p.parse_args()
    root=Path(args.root).resolve()
    if args.stage=='status':
        status(root)
        return
    root.mkdir(parents=True,exist_ok=True)
    import fcntl
    import os
    with (root/'.runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path in root.rglob('*.process.json'):
            if 'attempts' not in path.relative_to(root).parts and alive(core.read(path).get('identity')):
                raise RuntimeError('live child still exists: '+str(path))
        record=dict(pid=os.getpid(),identity=identity(os.getpid()),state='running',stage=args.stage,started_at=time.time())
        core.save(root/'runner.json',record)
        try:
            execute(args,root)
            record.update(state='finished',ended_at=time.time())
            core.save(root/'runner.json',record)
            if (root/'completion.json').exists():
                core.package(root,'')
                n=core.read(root/'completion.json')['new_cells']
                print('QMD_EMA_COMPLETE: %d/%d training; %d/%d scoring views; final ZIP ready.'%(n,n,4*n,4*n),flush=True)
        except Exception as error:
            record.update(state='failed',ended_at=time.time(),error=repr(error))
            core.save(root/'runner.json',record)
            raise


if __name__=='__main__':
    main()
