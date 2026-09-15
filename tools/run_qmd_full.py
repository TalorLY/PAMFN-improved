"""Fixed v10: complete July28 KD-only and add a matched original-MSE Base.

30 new cells (18 Base + 12 KD), 6 frozen v9 KD cells, 36 frozen v2 references.
All final comparisons use six actions x seeds0/1/2; no per-action selection.
"""
import argparse
import csv
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from tools import run_qmd_dual as core
from tools import run_qmd_gradient as grad
from tools import run_qmd_cal as cal
from tools import run_qmd_split as split
from tools.qmd_process import alive, identity
from tools.reuse_qmd_gradient_v8 import copy_unchanged

REPO = Path(__file__).resolve().parents[1]
BASE, ORDERED, KD, COMBO = 'base', 'ordered', split.KD, 'ordered_score_qmd'
SEEDS = (0, 1, 2)
V9_COMMITS = dict(modern='00a5f6abc8b172b7522813f095101c3d62cfecf8',
                  july28='ee70832bc82862d432342a6ff0020e8b53f51600')
SMOKE_EPOCHS = 2


def reused_cells():
    return [('july28', a, s, KD) for a in split.ACTIONS for s in (0, 1)]


def matrix():
    reused = set(reused_cells())
    return [('july28', a, s, m) for a in core.ACTIONS for s in SEEDS
            for m in (BASE, KD) if ('july28', a, s, m) not in reused]


def smoke_cells(jobs):
    return [('july28', a, 0, m) for a in ('Ball', 'TES') for m in (BASE, KD)] if jobs else []


def cell_args(cell, gpu, smoke=False):
    if cell[0] != 'july28' or cell[3] not in (BASE, KD):
        raise ValueError('unknown full-study cell')
    if cell[3] == KD:
        return split.cell_args(cell, gpu, smoke)
    argv = core.cell_args(*cell[:3], 'ordered', gpu, smoke)
    argv[argv.index('--run_tag') + 1] = BASE
    argv[argv.index('--ordinal_prototype_weight') + 1] = '0'
    if smoke:
        argv[argv.index('--epochs') + 1] = str(SMOKE_EPOCHS)
    return argv + ['--save_test_predictions']


def validate_cell(out, cell, smoke=False):
    if cell[3] == KD:
        return split.validate_cell(out, cell, smoke)
    if cell[3] != BASE:
        raise ValueError('unknown full-study method')
    row = core.validate_cell(out, cell, smoke, SMOKE_EPOCHS if smoke else None)
    expected = dict(mode='off', weight=1., mask_ratio=.15, mask_gt_weight=.5,
                    target_ratio=.075, beta_cap=1., warmup=.1, rampup=.1)
    for name in ('train_result.json', 'test_result.json'):
        if core.read(out / name)['run']['qmd'] != expected:
            raise ValueError('Base auxiliary unexpectedly enabled')
    if (out / 'qmd_training.json').exists():
        raise ValueError('Base contains auxiliary-training output')
    import numpy as np
    from scipy.stats import spearmanr
    test = core.read(out / 'test_result.json')
    y, pred = [np.asarray(test['predictions'][k]) for k in ('label', 'prediction')]
    n = 100 if cell[1] in ('TES', 'PCS') else 50
    if y.shape != (n,) or pred.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Base predictions missing or invalid')
    coef = spearmanr(y, pred)[0] * 100
    if not math.isfinite(coef) or abs(coef - test['coef']) > 1e-6 or abs(np.mean((pred-y)**2) - test['loss_x100']/100) > 1e-6:
        raise ValueError('Base prediction metrics disagree')
    row['mse'] = test['loss_x100']/100
    return row


def check_args(out, cell, smoke=False):
    expected = dict(argv=cell_args(cell, '0', smoke)[5:], protocol=cell[0])
    if core.read(out / 'run_config.json') != expected:
        raise ValueError('fixed cell arguments changed: ' + str(out))


def run_cell(root, repos, cell, gpu, smoke):
    def checked(out, c, s):
        check_args(out, c, s)
        row = validate_cell(out, c, s)
        if c[3] == KD and not s:
            split.check_warmup(out, c, root.parent / 'reference')
        return row
    return core.run_cell(root, repos, cell, gpu, smoke, argv=cell_args(cell, gpu, smoke), validator=checked)


def run_wave(root, repos, jobs, gpus, smoke=False):
    return core.run_wave(root, repos, jobs, gpus, smoke, runner=run_cell,
                         ordering=lambda cells: sorted(cells, key=lambda c: (-core.BUDGET[c[1]], c[2], c[3])))


def checked_v9(reference, source, root):
    old, complete = core.read(reference / 'source.json'), core.read(reference / 'completion.json')
    if old['commits'] != V9_COMMITS or complete.get('source') != old or complete.get('new_cells') != 15 or not complete.get('experiments_complete'):
        raise ValueError('requires completed frozen v9 results')
    if any(old[k] != source[k] for k in ('inputs', 'environment')):
        raise ValueError('v9 data/environment mismatch')
    rows = []
    for c in reused_cells():
        origin = core.cell_path(reference / 'screen', c)
        record = split.verify_record(origin)
        check_args(origin, c)
        row = validate_cell(origin, c)
        split.check_warmup(origin, c, root / 'reference')
        if abs(row['srcc_x100'] - record['metrics']['srcc_x100']) > 1e-6:
            raise ValueError('v9 recorded score changed')
        dest = core.cell_path(root / 'reference_v9', c)
        for name in list(record['files']) + ['complete.json', 'train.log', 'test.log']:
            copy_unchanged(origin / name, dest / name)
        rows.append(dict(row, origin='frozen_v9_kd', source_path=str(origin)))
    for name in ('source.json', 'completion.json', 'manifest.json'):
        copy_unchanged(reference / name, root / 'reference_v9' / name)
    core.immutable(root / 'v9_reference_manifest.json', dict(source=old, records=rows))
    return rows


def verify_training_unchanged(repo):
    changed = subprocess.check_output(['git', '-C', str(repo), 'diff', V9_COMMITS['july28'], 'HEAD',
        '--', 'pipeline.py', 'datasets.py', 'models', 'configs'], text=True)
    if changed:
        raise ValueError('v10 must preserve the frozen v9 training recipe')


def check_result_roots(root, references):
    root = root.resolve()
    for reference in references:
        old = Path(reference).resolve()
        if root == old or root in old.parents or old in root.parents:
            raise ValueError('new result root must be separate from old reference roots')


def aggregate(rows, methods=(BASE, ORDERED, KD, COMBO),
              controls=((BASE, 'base'), (ORDERED, 'ordered'), (COMBO, 'combination'))):
    expected = {(a, s, m) for a in core.ACTIONS for s in SEEDS for m in methods}
    lookup = {(r['action'], r['seed'], r['method']): r for r in rows}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise ValueError('incomplete or duplicate six-action three-seed grid')
    for row in rows:
        key = row['action'], row['seed']
        for control, label in controls:
            row['delta_vs_' + label] = row['srcc_x100'] - lookup[key + (control,)]['srcc_x100']
    groups = []
    for method in methods:
        group = [r for r in rows if r['method'] == method]
        def stats(subset):
            keys = ('srcc_x100',) + tuple('delta_vs_' + label for _, label in controls)
            return {k: sum(r[k] for r in subset)/len(subset) for k in keys}
        seed_means = {str(s): stats([r for r in group if r['seed'] == s]) for s in SEEDS}
        groups.append(dict(method=method, cells=18, **stats(group),
            seed_mean_std=statistics.stdev(v['srcc_x100'] for v in seed_means.values()),
            per_seed=seed_means, per_action={a: stats([r for r in group if r['action'] == a]) for a in core.ACTIONS},
            wins_vs_ordered=sum(r['delta_vs_ordered'] > 1e-7 for r in group),
            ties_vs_ordered=sum(abs(r['delta_vs_ordered']) <= 1e-7 for r in group)))
    return dict(raw=rows, methods=groups,
        note='Six actions x seeds0/1/2 equally weighted. Base is the original-MSE PAMFN reproduction in the July28 protocol, not the paper table score. New: 18 Base + 12 KD. Frozen: 6 v9 KD + 18 v2 Ordered + 18 v2 original combination. Legacy global-RNG Ordered initialization and test-best/RNG replay are retained. Report total recipe gain, not an RNG-isolated causal module claim. Seed std is sample std of six-action seed means.')


def summarize(root):
    rows = []
    for c in matrix() + reused_cells():
        phase = 'reference_v9' if c in reused_cells() else 'screen'
        out = core.cell_path(root / phase, c)
        split.verify_record(out); check_args(out, c)
        row = validate_cell(out, c)
        if c[3] == KD:
            split.check_warmup(out, c, root / 'reference')
        rows.append(dict(row, origin='frozen_v9_kd' if phase == 'reference_v9' else 'new_v10_training'))
    for a in core.ACTIONS:
        for s in SEEDS:
            for m in (ORDERED, COMBO):
                c = ('july28', a, s, m); out = core.cell_path(root / 'reference', c)
                split.verify_record(out)
                rows.append(dict(core.validate_cell(out, c), origin='frozen_v2_control'))
    return aggregate(rows)


def write_report(root):
    report = summarize(root)
    core.save(root / 'summary.json', report)
    with (root / 'scores.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in report['raw'] for k in r}))
        writer.writeheader(); writer.writerows(report['raw'])


def execute(args, root):
    if args.stage == 'package':
        if (root / 'completion.json').exists():
            write_report(root)
        return
    jobs = matrix()
    core.immutable(root / 'manifest.json', dict(study='full', jobs=jobs, total_new_cells=30,
        actions=core.ACTIONS, seeds=SEEDS, reused_kd_cells=reused_cells(),
        primary_methods=[BASE, ORDERED, KD], original_combination_reference=COMBO,
        epochs=core.BUDGET, smoke_epochs=SMOKE_EPOCHS, selection='legacy test-best and evaluation RNG replay'))
    gpus = args.gpus.split()
    if not gpus or len(gpus) != len(set(gpus)) or any(not g.isdigit() for g in gpus):
        raise ValueError('provide distinct physical GPU indices')
    cal.check_resources(root, gpus)
    repos = dict(modern=REPO, july28=Path(args.july28_repo).resolve())
    print('PREFLIGHT: frozen recipe, 37 input hashes, v2 controls, v9 KD reuse', flush=True)
    verify_training_unchanged(repos['july28'])
    source = core.provenance(repos); core.immutable(root / 'source.json', source)
    grad.checked_reference(Path(args.reference_root).resolve(), source, jobs, root)
    checked_v9(Path(args.v9_root).resolve(), source, root)
    smoke = smoke_cells(jobs); gate = root / 'smoke/gate.json'
    expected_gate = dict(decision='QMD_FULL_SMOKE_PASS', cells=4, source=source)
    if args.stage in ('all', 'smoke') and not gate.exists():
        core.command([os.sys.executable, '-u', '-m', 'unittest', 'tests.test_qmd_full_tools',
                      'tests.test_qmd_split_tools'], REPO, root / 'smoke/unit_tools.log')
        core.command([os.sys.executable, '-u', '-m', 'unittest', 'tests.test_qmd_full', 'tests.test_qmd_split',
                      'tests.test_july28_qmd', 'tests.test_qmd_gradient'], repos['july28'], root / 'smoke/unit_july28.log')
        core.command([os.sys.executable, '-u', '-m', 'tools.verify_july28_qmd', '--base', '--max_batches', '3',
                      '--gpu', gpus[0], '--output', str(root / 'smoke/base_parity.json')],
                     repos['july28'], root / 'smoke/base_parity.log')
        core.command([os.sys.executable, '-u', str(REPO / 'tools/verify_qmd_split_runtime.py'),
                      '--gpu', gpus[0], '--output', str(root / 'smoke/kd_parity.json')],
                     repos['july28'], root / 'smoke/kd_parity.log')
        run_wave(root / 'smoke', repos, smoke, [gpus[0]], True)
        checks = [validate_cell(core.cell_path(root / 'smoke', c), c, True) for c in smoke]
        core.save(root / 'smoke/summary.json', dict(raw=checks))
        if any(r['warnings'] for r in checks):
            raise RuntimeError('ineffective auxiliary in smoke; inspect diagnostics')
        core.save(gate, expected_gate)
    if not gate.exists() or core.read(gate) != expected_gate:
        raise RuntimeError('matching smoke PASS required')
    if args.stage == 'smoke':
        core.package(root, '_smoke'); return
    run_wave(root / 'screen', repos, jobs, gpus)
    write_report(root)
    core.save(root / 'completion.json', dict(experiments_complete=True, new_cells=30, reused_kd_cells=6,
        primary_cells=54, reference_cells=36, source=source, completed_at=time.time()))


def status(root):
    cal.status(root, smoke_selector=smoke_cells, smoke_epochs=SMOKE_EPOCHS)
    for label, method, total in [('Base new', BASE, 18), ('KD new', KD, 12)]:
        count = sum((core.cell_path(root / 'screen', c) / 'complete.json').exists() for c in matrix() if c[3] == method)
        print('%s: %d/%d' % (label, count, total))
    reused = sum((core.cell_path(root / 'reference_v9', c) / 'complete.json').exists() for c in reused_cells())
    fresh = sum((core.cell_path(root / 'screen', c) / 'complete.json').exists() for c in matrix() if c[3] == KD)
    print('KD full coverage: %d/18 (new %d/12 + frozen %d/6)' % (fresh + reused, fresh, reused))
    ordered = sum((core.cell_path(root / 'reference', ('july28', a, s, ORDERED)) / 'complete.json').exists()
                  for a in core.ACTIONS for s in SEEDS)
    print('Frozen Ordered: %d/18; counts are recorded results, not fresh validation.' % ordered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('all', 'smoke', 'screen', 'status', 'package'), default='all')
    parser.add_argument('--root', default='results/qmd_full_v10')
    parser.add_argument('--gpus', default='0 1 3 4')
    parser.add_argument('--july28_repo', default='../PAMFN-july28-qmd-full-v10')
    parser.add_argument('--reference_root', default='../PAMFN-qmd-dual-v2/results/qmd_dual_v2')
    parser.add_argument('--v9_root', default='../PAMFN-qmd-split-v9/results/qmd_split_v9')
    args = parser.parse_args(); root = Path(args.root).resolve()
    if args.stage == 'status':
        status(root); return
    check_result_roots(root, (args.reference_root, args.v9_root))
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path in root.rglob('*.process.json'):
            if not {'attempts', 'reference', 'reference_v9'}.intersection(path.relative_to(root).parts) and alive(core.read(path).get('identity')):
                raise RuntimeError('live child remains: ' + str(path))
        record = dict(pid=os.getpid(), identity=identity(os.getpid()), state='running', stage=args.stage, started_at=time.time())
        core.save(root / 'runner.json', record)
        try:
            execute(args, root)
            record.update(state='finished', ended_at=time.time()); core.save(root / 'runner.json', record)
            if (root / 'completion.json').exists():
                core.package(root, '')
                print('QMD_FULL_COMPLETE: 30/30 new; KD 18/18; Base 18/18; Ordered 18/18; final ZIP ready.', flush=True)
            elif args.stage == 'package':
                core.package(root, '_partial')
        except Exception as error:
            record.update(state='failed', ended_at=time.time(), error=repr(error)); core.save(root / 'runner.json', record)
            raise


if __name__ == '__main__':
    main()
