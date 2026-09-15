"""Six-action, two-seed checkpoint-only HBN study; three frozen model families."""
import argparse
import csv
import math
import os
from pathlib import Path
import shutil
import sys
import time

from tools import run_qmd_dual as core
from tools.run_qmd_light import REFERENCE_COMMITS
from tools.qmd_process import alive, identity
from tools.hbn_stats import ALPHAS, recipe

REPO = Path(__file__).resolve().parents[1]
FAMILIES = (('modern', 'score_qmd'), ('july28', 'ordered'), ('july28', 'ordered_score_qmd'))


def matrix():
    # Ball first, but NEVER stop or drop another action according to test scores.
    return [(p, a, s, m) for a in ('Ball','Ribbon','Hoop','Clubs','TES','PCS')
            for p,m in FAMILIES for s in (0,1)]


def smoke_cells():
    return [(p,a,0,m) for p,m in FAMILIES for a in ('Ball','TES')]


def check_resources(root, gpus):
    if not gpus or len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
        raise ValueError('provide distinct physical GPU indices')
    # Read-only checkpoints plus small JSON/CSV/buffer outputs; no multi-GB copies.
    if shutil.disk_usage(root).free < 1024**3:
        raise RuntimeError('less than 1 GiB free; choose another output volume')
    output = core.subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                    '--format=csv,noheader,nounits'], universal_newlines=True)
    free = {a.strip(): int(b.strip()) for a,b in (line.split(',') for line in output.splitlines())}
    if any(free.get(g,0) < 4096 for g in gpus):
        raise RuntimeError('selected GPU has less than 4096 MiB free; change GPU_LIST')
    print('GPU free MiB:', {g:free[g] for g in gpus}, flush=True)


def preflight(root, repos, reference):
    print('PREFLIGHT: source and 37 input fingerprints (GPU may be idle here).', flush=True)
    core.save(root/'preflight_progress.json',dict(phase='37 input fingerprints',checked_checkpoints=0,total=36,updated_at=time.time()))
    source = core.provenance(repos)
    ref_source, complete = core.read(reference/'source.json'), core.read(reference/'completion.json')
    if (ref_source['commits'] != REFERENCE_COMMITS or complete.get('source') != ref_source
            or not complete.get('experiments_complete') or complete.get('cells') != 126):
        raise ValueError('requires the completed frozen qmd_dual_v2 reference')
    if source['commits']['july28'] != REFERENCE_COMMITS['july28']:
        raise ValueError('July28 evaluator must use the unchanged v2 historical repository')
    for key in ('inputs','environment'):
        if source[key] != ref_source[key]:
            raise ValueError('reference '+key+' differs; do not silently rebase historical scores')
    core.immutable(root/'source.json', source)
    records = []
    for index, cell in enumerate(matrix(), 1):
        p,a,s,m = cell
        origin = core.cell_path(reference/'screen', cell)
        # Check all required checkpoints BEFORE dispatching any evaluation.
        expected_files = ['model.pth'] + (['model.pth.eval_rng.pt'] if p=='july28' else [])
        missing = [str(origin/name) for name in expected_files if not (origin/name).is_file()]
        if missing:
            raise FileNotFoundError('missing original checkpoint/RNG; no automatic retraining: '+', '.join(missing))
        record = core.read(origin/'complete.json')
        for name, digest in record['files'].items():
            if Path(name).name != name or core.sha(origin/name) != digest:
                raise ValueError('reference file hash changed: '+str(origin/name))
        metrics = core.validate_cell(origin, cell)
        if abs(metrics['srcc_x100']-record['metrics']['srcc_x100']) > 1e-6:
            raise ValueError('reference completion score mismatch')
        if core.sha(origin/'model.pth') != record['checkpoint_sha256']:
            raise ValueError('source checkpoint differs from completion hash')
        expected = dict(mode='off' if m=='ordered' else 'score_qmd', weight=1., mask_ratio=.15,
                        mask_gt_weight=.5, target_ratio=.075, beta_cap=1., warmup=.1, rampup=.1)
        test = core.read(origin/'test_result.json')
        if test['run']['qmd'] != expected:
            raise ValueError('unexpected historical QMD settings')
        dest = core.cell_path(root/'reference', cell)
        dest.mkdir(parents=True, exist_ok=True)
        for name in list(record['files']) + ['complete.json']:
            if Path(name).suffix in ('.pth','.pt'):
                continue
            shutil.copyfile(origin/name, dest/name)
        spec = dict(cell=list(cell), repo=str(repos[p]), recipe=recipe(),
            checkpoint=str(origin/'model.pth'), checkpoint_sha256=record['checkpoint_sha256'],
            expected_test={k:test[k] for k in ('coef','loss_x100','prediction_std')},
            argv=core.cell_args(p,a,s,m,'-1')[3:],
            reference_completion_sha256=core.sha(origin/'complete.json'))
        if p=='july28':
            spec.update(eval_rng=str(origin/'model.pth.eval_rng.pt'),
                        eval_rng_sha256=core.sha(origin/'model.pth.eval_rng.pt'))
            if spec['eval_rng_sha256'] != record.get('eval_rng_sha256'):
                raise ValueError('July28 RNG is not authenticated by the reference completion')
        core.immutable(dest/'spec.json', spec)
        records.append(dict(cell=list(cell), score=test['coef'], checkpoint_sha256=record['checkpoint_sha256']))
        core.save(root/'preflight_progress.json',dict(phase='original checkpoints',checked_checkpoints=index,total=36,updated_at=time.time()))
        print('PREFLIGHT checkpoints %d/36 %s' % (index, '/'.join(map(str,cell))), flush=True)
    core.immutable(root/'reference_manifest.json', dict(source=ref_source, records=records))
    return source


def validate_result(out, cell, smoke=False):
    result = core.read(out/'result.json')
    if result['cell'] != list(cell) or result['recipe'] != recipe() or result['calibrated_layers'] != 1:
        raise ValueError('wrong cell or calibration recipe')
    expected_checks = ('original_replay','identity_exact','buffers_restored','parameters_unchanged','tiny_state_reload_exact')
    if not all(result['checks'].get(k) is True for k in expected_checks):
        raise ValueError('missing original-preservation/reload check')
    if result['checks'].get('test_labels_used_for_calibration') is not False:
        raise ValueError('test label calibration is forbidden')
    if smoke and not result['checks'].get('full_vs_cache_exact'):
        raise ValueError('smoke must verify full-network versus cached-head outputs')
    rows = result['results']
    if [r['alpha'] for r in rows] != list(ALPHAS) or rows[0]['delta'] != 0:
        raise ValueError('incomplete fixed views or identity mismatch')
    baseline = result['baseline']['coef']
    if not math.isfinite(baseline) or abs(rows[0]['coef']-baseline) > 1e-6:
        raise ValueError('invalid original baseline')
    for row in rows:
        expected = None if row['coef'] is None else row['coef']-baseline
        if row['delta'] != expected:
            raise ValueError('incorrect paired delta')
    return result


def validate_completed(out, cell, smoke=False):
    record = core.read(out/'complete.json')
    for name, digest in record['files'].items():
        if Path(name).name != name or core.sha(out/name) != digest:
            raise ValueError('completed result changed: '+str(out/name))
    result = validate_result(out, cell, smoke)
    if record['checkpoint_sha256'] != result['checkpoint_sha256']:
        raise ValueError('completion checkpoint identity mismatch')
    return result


def run_cell(root, repos, cell, gpu, smoke=False):
    out = core.cell_path(root, cell)
    spec = core.cell_path(root.parent/'reference',cell)/'spec.json'
    if (out/'complete.json').exists():
        result = validate_completed(out, cell, smoke)
        if result['checkpoint_sha256'] != core.read(spec)['checkpoint_sha256']:
            raise ValueError('cached result source checkpoint differs')
        return 'SKIP', dict(srcc_x100=result['baseline']['coef'])
    # Preserve failed/interrupted attempts; never mix their outputs into a completed cell.
    if out.exists() and any(out.iterdir()):
        archive = out/'attempts'/str(time.time_ns())
        archive.mkdir(parents=True)
        for path in list(out.iterdir()):
            if path.is_file():
                path.rename(archive/path.name)
    out.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable,'-u',str(REPO/'tools/eval_qmd_hbn.py'),
            '--spec',str(spec),'--output',str(out),'--gpu',gpu]
    if smoke:
        argv.append('--smoke')
    core.command(argv, repos[cell[0]], out/'eval.log')
    result = validate_result(out, cell, smoke)
    files = {path.name:core.sha(path) for path in out.iterdir()
             if path.is_file() and path.name != 'complete.json'}
    core.save(out/'complete.json', dict(cell=list(cell), files=files,
              checkpoint_sha256=result['checkpoint_sha256'], completed_at=time.time()))
    return 'FINISH', dict(srcc_x100=result['baseline']['coef'])


def mean_complete(values):
    # Do not drop failed/collapsed views to improve a mean.
    return sum(values)/len(values) if values and all(v is not None for v in values) else None


def summarize(root):
    rows = []
    for cell in matrix():
        result = validate_completed(core.cell_path(root/'screen',cell), cell)
        p,a,s,m = cell
        rows.extend(dict(protocol=p,action=a,seed=s,method=m,**r) for r in result['results'])
    groups = []
    for p,m in FAMILIES:
        for alpha in ALPHAS:
            selected = [r for r in rows if (r['protocol'],r['method'],r['alpha'])==(p,m,alpha)]
            if len(selected) != 12:
                raise ValueError('incomplete six-action/two-seed matrix')
            groups.append(dict(protocol=p,method=m,alpha=alpha,
                mean_srcc_x100=mean_complete([r['coef'] for r in selected]),
                delta=mean_complete([r['delta'] for r in selected]),
                nonfinite_cells=sum(r['coef'] is None for r in selected),
                per_action={a:mean_complete([r['coef'] for r in selected if r['action']==a]) for a in core.ACTIONS},
                ball_delta=mean_complete([r['delta'] for r in selected if r['action']=='Ball']),
                per_seed={str(s):mean_complete([r['coef'] for r in selected if r['seed']==s]) for s in (0,1)}))
    comparisons = []
    for alpha in ALPHAS:
        for action in core.ACTIONS:
            for seed in (0,1):
                paired = {r['method']:r['coef'] for r in rows if r['protocol']=='july28'
                          and r['alpha']==alpha and r['action']==action and r['seed']==seed}
                combined, ordered = paired['ordered_score_qmd'], paired['ordered']
                comparisons.append(dict(alpha=alpha,action=action,seed=seed,
                    combined_minus_ordered=None if combined is None or ordered is None else combined-ordered))
    report = dict(raw=rows,groups=groups,july28_combination_pairs=comparisons,
        note='Fixed original checkpoints, not new training or new epoch search. Same-protocol deltas only. '
             'All seeds/actions/alphas retained; no per-action winner mosaic. July28 has no QMD-only arm; '
             'combined-minus-Ordered is not a full factorial interaction estimate.')
    core.save(root/'summary.json',report)
    with (root/'scores.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return report


def optional(path):
    return core.read(path) if path.exists() else {}


def status(root):
    manifest, runner = optional(root/'manifest.json'), optional(root/'runner.json')
    jobs = [tuple(c) for c in manifest.get('jobs',matrix())]
    print('Runner:',runner.get('state','not started'), 'PID:',runner.get('pid','-'), 'alive:',alive(runner.get('identity')))
    preflight_state=optional(root/'preflight_progress.json')
    print('Preflight:',preflight_state.get('phase','not started'),
          '%s/36 checkpoints'%preflight_state.get('checked_checkpoints',0))
    smoke = sum((core.cell_path(root/'smoke',c)/'complete.json').exists() for c in smoke_cells())
    done = sum((core.cell_path(root/'screen',c)/'complete.json').exists() for c in jobs)
    print('Smoke: %d/6 ; gate: %s' % (smoke, optional(root/'smoke/gate.json').get('decision','pending')))
    print('Checkpoint evaluations completed: %d/36 ; scoring views: %d/108 ; NEW TRAINING: 0' % (done,3*done))
    phase = 'screen' if (root/'smoke/gate.json').exists() else 'smoke'
    for c in (jobs if phase=='screen' else smoke_cells()):
        out=core.cell_path(root/phase,c)
        if (out/'complete.json').exists():
            continue
        process, progress = optional(out/'eval.process.json'), optional(out/'progress.json')
        if not process and not progress:
            continue
        live = alive(process.get('identity'))
        state = 'RUNNING' if live else 'FAILED/INTERRUPTED' if process else 'PENDING'
        age = int(time.time()-progress['updated_at']) if progress.get('updated_at') else -1
        print(state,'/'.join(map(str,c)),progress.get('phase','starting'),
              '%s/%s'%(progress.get('done',0),progress.get('total',0)), 'age=%ss'%age)
    if runner.get('state')=='failed':
        print('LAUNCHER ERROR:',runner.get('error'))
    print('Completion:', 'READY' if (root/'completion.json').exists() else 'not yet created')
    archive=root.parent/(root.name+'_return.zip')
    print('Archive:',archive,'READY' if archive.exists() else 'not yet packaged')


def execute(args, root):
    if args.stage=='package':
        if (root/'completion.json').exists():
            summarize(root)
        core.package(root,'' if (root/'completion.json').exists() else '_partial')
        return
    core.immutable(root/'manifest.json',dict(jobs=matrix(),smoke_jobs=smoke_cells(),recipe=recipe(),
        checkpoint_cells=36,scoring_views=108,new_training=0,reference_root=str(Path(args.reference_root).resolve())))
    gpus=args.gpus.split()
    check_resources(root,gpus)
    repos=dict(modern=REPO,july28=Path(args.july28_repo).resolve())
    source=preflight(root,repos,Path(args.reference_root).resolve())
    gate=root/'smoke/gate.json'
    if args.stage in ('smoke','all') and not gate.exists():
        core.command([sys.executable,'-u','-m','unittest','tests.test_hbn_stats','tests.test_qmd_hbn_tools'],
                     REPO,root/'smoke/unit.log')
        core.run_wave(root/'smoke',repos,smoke_cells(),gpus,True,runner=run_cell,ordering=lambda x:x)
        for c in smoke_cells():
            validate_completed(core.cell_path(root/'smoke',c),c,True)
        core.save(gate,dict(decision='QMD_HBN_SMOKE_PASS',source=source,cells=6))
    gate_record=optional(gate)
    if (gate_record.get('source')!=source or gate_record.get('decision')!='QMD_HBN_SMOKE_PASS'
            or gate_record.get('cells')!=6):
        raise RuntimeError('matching smoke PASS required')
    for c in smoke_cells():
        validate_completed(core.cell_path(root/'smoke',c),c,True)
    if args.stage=='smoke':
        core.package(root,'_smoke')
        return
    core.run_wave(root/'screen',repos,matrix(),gpus,False,runner=run_cell,ordering=lambda x:x)
    summarize(root)
    core.save(root/'completion.json',dict(experiments_complete=True,checkpoint_cells=36,scoring_views=108,
                                         new_training=0,source=source,completed_at=time.time()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('all','smoke','screen','status','package'),default='all')
    p.add_argument('--root',default='results/qmd_hbn_v7')
    p.add_argument('--reference_root',default='../PAMFN-qmd-dual-v2/results/qmd_dual_v2')
    p.add_argument('--july28_repo',default='../PAMFN-july28-hbn-v7')
    p.add_argument('--gpus',default='0 1 4 5')
    args=p.parse_args(); root=Path(args.root).resolve()
    if args.stage=='status':
        status(root); return
    root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (root/'.runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path in root.rglob('*.process.json'):
            if 'attempts' not in path.relative_to(root).parts and alive(core.read(path).get('identity')):
                raise RuntimeError('live child remains; do not launch twice: '+str(path))
        record=dict(pid=os.getpid(),identity=identity(os.getpid()),state='running',stage=args.stage,started_at=time.time())
        core.save(root/'runner.json',record)
        try:
            execute(args,root)
            record.update(state='finished',ended_at=time.time()); core.save(root/'runner.json',record)
            if (root/'completion.json').exists():
                core.package(root,'')
                print('QMD_HBN_COMPLETE: 36/36 checkpoints; 108/108 views; final ZIP ready.',flush=True)
        except Exception as error:
            record.update(state='failed',error=repr(error),ended_at=time.time()); core.save(root/'runner.json',record)
            core.package(root,'_partial')
            raise


if __name__=='__main__':
    main()
