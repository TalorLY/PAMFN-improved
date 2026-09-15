"""Fixed 24-cell Score-QMD head screen with live status and automatic return ZIP."""
import argparse
import csv
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
RECIPES = dict(score_qmd_head_linear='linear', score_qmd_head_gn='groupnorm')
SMOKE_EPOCHS = 2


def matrix(study='screen', methods=()):
    if study == 'screen':
        if methods:
            raise ValueError('screen includes both fixed heads and seeds 0/1')
        methods, seeds = tuple(RECIPES), (0, 1)
    elif study == 'confirm':
        if not methods or len(set(methods)) != len(methods) or any(m not in RECIPES for m in methods):
            raise ValueError('confirm requires distinct known --methods')
        seeds = (42,)
    else:
        raise ValueError('unknown study')
    return [('modern', a, s, m) for m in methods for a in core.ACTIONS for s in seeds]


def smoke_cells(jobs):
    return [('modern', a, 0, m) for m in RECIPES if m in {c[3] for c in jobs} for a in ('Ball', 'TES')]


def settings(cell, smoke=False):
    return dict(mode='score_qmd', weight=1., mask_ratio=.15, mask_gt_weight=.5,
                target_ratio=.075, beta_cap=1., warmup=0. if smoke else .1, rampup=0. if smoke else .1)


def cell_args(cell, gpu, smoke=False):
    p, a, s, m = cell
    if p != 'modern' or m not in RECIPES:
        raise ValueError('unknown head cell')
    argv = core.cell_args(p, a, s, 'score_qmd', gpu, smoke)
    argv[argv.index('--run_tag') + 1] = m
    if smoke:
        argv[argv.index('--epochs') + 1] = str(SMOKE_EPOCHS)
    return argv + ['--score_head', RECIPES[m], '--score_head_diagnostics']


def validate_cell(out, cell, smoke=False):
    row = core.validate_cell(out, cell, smoke, SMOKE_EPOCHS if smoke else None)
    variant = RECIPES[cell[3]]
    config = settings(cell, smoke)
    for name in ('train_result.json', 'test_result.json'):
        run = core.read(out/name)['run']
        if (run['qmd'] != config or run['eval_seed'] != 20260908
                or run.get('score_head') != dict(variant=variant, diagnostics=True)):
            raise ValueError('head/QMD recipe differs from fixed settings')
    diagnostics = [json.loads(x) for x in (out/'model.pth.head.jsonl').read_text().splitlines() if x.strip()]
    epochs = SMOKE_EPOCHS if smoke else core.BUDGET[cell[1]]
    if [d['epoch'] for d in diagnostics] != list(range(1, epochs + 1)):
        raise ValueError('head diagnostic history is incomplete')
    test = core.read(out/'test_result.json')
    norm, output = ('GroupNorm', 'Sigmoid') if variant == 'groupnorm' else ('BatchNorm1d', 'Identity')
    for d in [test['score_head']] + [x[k] for x in diagnostics for k in ('train', 'test')]:
        if (d['variant'], d['normalization'], d['output']) != (variant, norm, output):
            raise ValueError('actual score head does not match requested variant')
        if d['pre_output']['count'] <= 0 or d['trainable_parameters'] <= 0:
            raise ValueError('missing head observations')
        if variant == 'linear' and d['saturation_fraction'] is not None:
            raise ValueError('linear output was mislabeled as sigmoid saturation')
    selected = diagnostics[row['selected_epoch'] - 1]['test']
    for group in ('pre_output', 'prediction', 'target'):
        for key in ('count', 'mean', 'std', 'min', 'max'):
            x, y = test['score_head'][group][key], selected[group][key]
            if not math.isfinite(x) or abs(x-y) > 1e-6:
                raise ValueError('reloaded head diagnostics disagree with selected checkpoint')
    qmd = core.read(out/'qmd_training.json')
    if qmd['calibration'].get('scope') != 'score_head_parameters_first_active_training_batch':
        raise ValueError('original single-batch Score-QMD calibration was not restored')
    if not qmd.get('view_diagnostics', {}).get('rows'):
        raise ValueError('missing full/masked training views')
    row.update(origin='new_training', score_head=variant, mse=test['loss_x100']/100,
               prediction_std_ratio=test['prediction_std_ratio'],
               pre_output_std=test['score_head']['pre_output']['std'],
               saturation_fraction=test['score_head']['saturation_fraction'])
    return row


def run_cell(root, repos, cell, gpu, smoke):
    return core.run_cell(root, repos, cell, gpu, smoke, argv=cell_args(cell, gpu, smoke), validator=validate_cell)


def run_wave(root, repos, jobs, gpus, smoke=False):
    return core.run_wave(root, repos, jobs, gpus, smoke, runner=run_cell,
        ordering=lambda cells: sorted(cells, key=lambda c: (
            0 if c[1] == 'Ball' else 1 if c[1] == 'Ribbon' else 2, -core.BUDGET[c[1]], c[2], c[3])))


def summarize(root, jobs, references):
    report = light.summarize(root, jobs, references, validator=validate_cell)
    for g in report['methods']:
        rows = [x for x in report['raw'] if x['method'] == g['method']]
        g['per_seed_score'] = {str(s): sum(x['srcc_x100'] for x in rows if x['seed'] == s)/len(g['actions'])
                               for s in g['seeds']}
    report['note'] = ('SRCC x100, six actions and all planned seeds equally weighted. '
                      'Original Score-QMD controls reused from frozen v2. '
                      'Only the final head changes; single-batch beta cap1 retained. '
                      'Per-seed scores are reported alongside the mean, not selected in its place.')
    return report


def status(root):
    return cal.status(root, smoke_selector=smoke_cells, smoke_epochs=SMOKE_EPOCHS)


def execute(args, root):
    if args.stage == 'package':
        if (root/'completion.json').exists():
            jobs = [tuple(c) for c in core.read(root/'manifest.json')['jobs']]
            summarize(root, jobs, core.read(root/'reference_manifest.json')['records'])
            core.package(root, '')
        else:
            core.package(root, '_partial')
        return
    jobs = matrix(args.study, args.methods)
    core.immutable(root/'manifest.json', dict(study=args.study, jobs=jobs, total_new_cells=len(jobs),
        heads=RECIPES, qmd=settings(jobs[0]), epochs=core.BUDGET, smoke_epochs=SMOKE_EPOCHS,
        selection='unchanged modern argmax test-best', reporting='all planned seeds, mean plus per-seed'))
    gpus = args.gpus.split()
    if not gpus or len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
        raise ValueError('provide distinct physical GPU indices')
    cal.check_resources(root, gpus)
    print('Preflight: code, input hashes, environment and original Score-QMD references...', flush=True)
    repos = dict(modern=REPO)
    source = core.provenance(repos)
    core.immutable(root/'source.json', source)
    references = light.checked_reference(Path(args.reference_root).resolve(), source, jobs, root)
    gate_path = root/'smoke/gate.json'
    smoke = smoke_cells(jobs)
    if args.stage in ('all', 'smoke') and not gate_path.exists():
        checks = [(['-m', 'unittest', 'tests.test_qmd_head', 'tests.test_qmd_head_tools',
                    'tests.test_score_qmd', 'tests.test_sr_qmd', 'tests.test_qmd_cal_tools'], 'unit'),
                  (['-m', 'tools.verify_sr_qmd_runtime', '--gpu', gpus[0],
                    '--output', str(root/'smoke/score_parity.json')], 'score_parity')]
        for argv, name in checks:
            print('CHECK ' + name, flush=True)
            core.command([sys.executable, '-u'] + argv, REPO, root/'smoke'/(name + '.log'))
        run_wave(root/'smoke', repos, smoke, gpus, True)
        rows = [validate_cell(core.cell_path(root/'smoke', c), c, True) for c in smoke]
        if any(r['warnings'] for r in rows):
            raise RuntimeError('ineffective Score-QMD in smoke; inspect reports')
        core.save(gate_path, dict(decision='QMD_HEAD_SMOKE_PASS', source=source, cells=len(smoke), results=rows))
    if not gate_path.exists() or core.read(gate_path)['source'] != source:
        raise RuntimeError('matching smoke PASS required before full training')
    if args.stage == 'smoke':
        core.package(root, '_smoke')
        return
    pilot = [c for c in jobs if c[2] == 0]
    if pilot:
        run_wave(root/'screen', repos, pilot, gpus)
        core.save(root/'pilot_summary.json', summarize(root, pilot, references))
        core.package(root, '_pilot')
    run_wave(root/'screen', repos, [c for c in jobs if c not in pilot], gpus)
    report = summarize(root, jobs, references)
    core.save(root/'summary.json', report)
    with (root/'scores.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report['raw'][0]))
        writer.writeheader()
        writer.writerows(report['raw'])
    core.save(root/'completion.json', dict(experiments_complete=True, new_cells=len(jobs),
        reference_cells=len(references), study=args.study, source=source, completed_at=time.time()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('all', 'smoke', 'screen', 'status', 'package'), default='all')
    p.add_argument('--study', choices=('screen', 'confirm'), default='screen')
    p.add_argument('--methods', nargs='*', default=[])
    p.add_argument('--root', default='results/qmd_head_v5')
    p.add_argument('--reference_root', default='../PAMFN-qmd-dual-v2/results/qmd_dual_v2')
    p.add_argument('--gpus', default='0 1 3 4')
    args = p.parse_args()
    root = Path(args.root).resolve()
    if args.stage == 'status':
        status(root)
        return
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    import os
    with (root/'.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path in root.rglob('*.process.json'):
            if 'attempts' not in path.relative_to(root).parts and alive(core.read(path).get('identity')):
                raise RuntimeError('live child still exists: ' + str(path))
        record = dict(pid=os.getpid(), identity=identity(os.getpid()), state='running',
                      stage=args.stage, started_at=time.time())
        core.save(root/'runner.json', record)
        try:
            execute(args, root)
            record.update(state='finished', ended_at=time.time())
            core.save(root/'runner.json', record)
            if (root/'completion.json').exists():
                core.package(root, '')
                count = core.read(root/'completion.json')['new_cells']
                print('QMD_HEAD_COMPLETE: %d/%d; final return ZIP ready.' % (count, count), flush=True)
        except Exception as error:
            record.update(state='failed', ended_at=time.time(), error=repr(error))
            core.save(root/'runner.json', record)
            raise


if __name__ == '__main__':
    main()
