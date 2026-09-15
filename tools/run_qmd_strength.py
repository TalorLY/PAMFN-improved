"""v11: six actions x three seeds x two pure-KD strengths, no pilot filtering.

The July28 v10 trainer is unchanged. Import all 72 v10 comparison rows read-only,
then run 36 new full-budget cells with weight .25/.5; no action/seed overrides.
"""
import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import time

from tools import run_qmd_dual as core
from tools import run_qmd_cal as cal
from tools import run_qmd_full as full
from tools import run_qmd_split as split
from tools.qmd_process import alive, identity
from tools.reuse_qmd_gradient_v8 import copy_unchanged

REPO = Path(__file__).resolve().parents[1]
BASE, ORDERED, KD, COMBO = full.BASE, full.ORDERED, full.KD, full.COMBO
WEIGHTS = {'ordered_score_qmd_split_kd_w025': .25, 'ordered_score_qmd_split_kd_w050': .5}
SEEDS = (0, 1, 2)
REFERENCES = (BASE, ORDERED, KD, COMBO)
METHODS = REFERENCES + tuple(WEIGHTS)
V10_COMMITS = dict(modern='cb97b373fca7f40b35e7c434e3653de73e5a1ce3',
                   july28='12b03b561a122ebe23c0a40b6103f7071bbde298')
SMOKE_EPOCHS = 2


def matrix():
    return [('july28', a, s, m) for a in core.ACTIONS for s in SEEDS for m in WEIGHTS]


def reference_cells():
    return [('july28', a, s, m) for a in core.ACTIONS for s in SEEDS for m in REFERENCES]


def smoke_cells(jobs):
    return [c for c in matrix() if c[2] == 0] if jobs else []


def settings(cell, smoke=False):
    if cell not in matrix():
        raise ValueError('cell outside fixed six-action three-seed strength grid')
    return dict(split.settings(cell[:3] + (KD,), smoke), weight=WEIGHTS[cell[3]])


def cell_args(cell, gpu, smoke=False):
    recipe = settings(cell, smoke)
    argv = full.cell_args(cell[:3] + (KD,), gpu, smoke)
    argv[argv.index('--run_tag') + 1] = cell[3]
    argv[argv.index('--qmd_weight') + 1] = str(recipe['weight'])
    return argv


def check_args(out, cell, smoke=False):
    expected = dict(argv=cell_args(cell, '0', smoke)[5:], protocol=cell[0])
    if core.read(out / 'run_config.json') != expected:
        raise ValueError('fixed strength arguments changed: ' + str(out))


def validate_cell(out, cell, smoke=False):
    check_args(out, cell, smoke)
    row = split.validate_cell(out, cell, smoke, recipe=settings(cell, smoke))
    row['qmd_weight'] = WEIGHTS[cell[3]]
    return row


def check_pre_aux(out, cell, reference):
    split.check_warmup(out, cell, reference)
    old = core.cell_path(reference, cell[:3] + (KD,))
    left = core.read(out / 'qmd_training.json')['calibration']
    right = dict(core.read(old / 'qmd_training.json')['calibration'])
    # Beta calibration is weight-independent. Only its achieved ratio scales.
    right['achieved_ratio_at_full_ramp'] *= WEIGHTS[cell[3]]
    if set(left) != set(right):
        raise ValueError('first-active calibration fields changed')
    for key, value in left.items():
        if isinstance(value, float):
            same = math.isfinite(value) and math.isclose(value, right[key], rel_tol=1e-6, abs_tol=1e-10)
        else:
            same = value == right[key]
        if not same:
            raise ValueError('first-active calibration changed beyond weight: ' + key)


def run_cell(root, repos, cell, gpu, smoke):
    def checked(out, c, s):
        row = validate_cell(out, c, s)
        if not s:
            check_pre_aux(out, c, root.parent / 'reference')
        return row
    return core.run_cell(root, repos, cell, gpu, smoke, argv=cell_args(cell, gpu, smoke), validator=checked)


def run_wave(root, repos, jobs, gpus, smoke=False):
    return core.run_wave(root, repos, jobs, gpus, smoke, runner=run_cell,
        ordering=lambda cells: sorted(cells, key=lambda c: (-core.BUDGET[c[1]], c[2], c[3])))


def validate_reference(out, cell, reference):
    record = split.verify_record(out)
    if cell[3] in (BASE, KD):
        full.check_args(out, cell)
        row = full.validate_cell(out, cell)
    else:
        expected = dict(argv=core.cell_args(*cell, '0')[5:], protocol=cell[0])
        if core.read(out / 'run_config.json') != expected:
            raise ValueError('frozen control arguments changed')
        row = core.validate_cell(out, cell)
    if cell[3] == KD:
        split.check_warmup(out, cell, reference)
    if abs(row['srcc_x100'] - record['metrics']['srcc_x100']) > 1e-6:
        raise ValueError('frozen v10 recorded score changed')
    return dict(row, origin='frozen_v10_reference')


def checked_v10(reference, source, root):
    old, complete = [core.read(reference / name) for name in ('source.json', 'completion.json')]
    if (old['commits'] != V10_COMMITS or complete.get('source') != old
            or not complete.get('experiments_complete') or complete.get('new_cells') != 30
            or complete.get('reference_cells') != 36 or complete.get('reused_kd_cells') != 6
            or complete.get('primary_cells') != 54):
        raise ValueError('requires completed frozen v10 results')
    if any(old[k] != source[k] for k in ('inputs', 'environment')):
        raise ValueError('v10 data/environment mismatch')
    # Validate the full source grid before copying, not merely its summary.
    full.summarize(reference)
    for c in reference_cells():
        phase = ('reference' if c[3] in (ORDERED, COMBO) else
                 'reference_v9' if c in full.reused_cells() else 'screen')
        origin = core.cell_path(reference / phase, c)
        record = split.verify_record(origin)
        dest = core.cell_path(root / 'reference', c)
        for name in list(record['files']) + ['complete.json']:
            copy_unchanged(origin / name, dest / name)
        # v10's frozen v2 rows retain hashed JSON/history but not text logs.
        for name in ('train.log', 'test.log'):
            if (origin / name).exists():
                copy_unchanged(origin / name, dest / name)
    rows = [validate_reference(core.cell_path(root / 'reference', c), c, root / 'reference')
            for c in reference_cells()]
    for name in ('source.json', 'completion.json', 'manifest.json'):
        copy_unchanged(reference / name, root / 'reference' / name)
    core.immutable(root / 'reference_manifest.json', dict(source=old, records=rows, source_path=str(reference)))
    return rows


def verify_training_unchanged(repo):
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != V10_COMMITS['july28']:
        raise ValueError('v11 must use the exact frozen v10 July28 trainer')


def aggregate(rows):
    if any(r.get('protocol') != 'july28' or not math.isfinite(r['srcc_x100']) for r in rows):
        raise ValueError('nonfinite score or wrong protocol')
    report = full.aggregate(rows, methods=METHODS,
        controls=((BASE, 'base'), (ORDERED, 'ordered'), (KD, 'kd1'), (COMBO, 'combination')))
    report['note'] = ('Six actions x seeds0/1/2 for EVERY method; no pilot filtering or per-action method selection. '
        '36 new pure-KD cells, weight .25/.5; 72 frozen v10 comparison rows. '
        'Base is the matched July28 PAMFN reproduction, not the paper table. '
        'Only qmd_weight and run_tag differ from v10 KD; native budgets, test-best/RNG replay retained. '
        'Sample seed std uses three six-action means; all low scores retained.')
    return report


def summarize(root):
    rows = [validate_reference(core.cell_path(root / 'reference', c), c, root / 'reference')
            for c in reference_cells()]
    for c in matrix():
        out = core.cell_path(root / 'screen', c)
        record = split.verify_record(out)
        row = validate_cell(out, c)
        check_pre_aux(out, c, root / 'reference')
        if abs(row['srcc_x100'] - record['metrics']['srcc_x100']) > 1e-6:
            raise ValueError('new recorded score changed')
        rows.append(dict(row, origin='new_v11_training'))
    return aggregate(rows)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader(); writer.writerows(rows)


def write_report(root):
    report = summarize(root)
    core.save(root / 'summary.json', report)
    write_csv(root / 'scores.csv', report['raw'])
    for seed in SEEDS:
        write_csv(root / ('scores_seed%d.csv' % seed), [r for r in report['raw'] if r['seed'] == seed])
    for key, column in (('per_seed', 'seed'), ('per_action', 'action')):
        write_csv(root / (key + '.csv'), [dict(method=g['method'], **{column: k}, **v)
            for g in report['methods'] for k, v in g[key].items()])


def execute(args, root):
    if args.stage == 'package':
        if (root / 'completion.json').exists():
            write_report(root)
        return
    jobs = matrix()
    core.immutable(root / 'manifest.json', dict(study='full', jobs=jobs, total_new_cells=36,
        actions=core.ACTIONS, seeds=SEEDS, weights=WEIGHTS, reference_cells=72, total_rows=108,
        primary_cells=90, epochs=core.BUDGET, smoke_epochs=SMOKE_EPOCHS, smoke_cells=12,
        selection='legacy test-best and evaluation RNG replay; all actions and seeds retained'))
    gpus = args.gpus.split()
    if not gpus or len(gpus) != len(set(gpus)) or any(not g.isdigit() for g in gpus):
        raise ValueError('provide distinct physical GPU indices')
    cal.check_resources(root, gpus)
    repos = dict(modern=REPO, july28=Path(args.july28_repo).resolve())
    print('PREFLIGHT: unchanged v10 trainer, input hashes, 72 frozen v10 reference rows', flush=True)
    verify_training_unchanged(repos['july28'])
    source = core.provenance(repos); core.immutable(root / 'source.json', source)
    checked_v10(Path(args.reference_root).resolve(), source, root)
    smoke = smoke_cells(jobs); gate = root / 'smoke/gate.json'
    expected_gate = dict(decision='QMD_STRENGTH_SMOKE_PASS', cells=12, source=source)
    if args.stage in ('all', 'smoke') and not gate.exists():
        core.command([os.sys.executable, '-u', '-m', 'unittest', 'tests.test_qmd_strength_tools',
                      'tests.test_qmd_full_tools', 'tests.test_qmd_split_tools'], REPO, root / 'smoke/unit_tools.log')
        core.command([os.sys.executable, '-u', '-m', 'unittest', 'tests.test_qmd_full', 'tests.test_qmd_split',
                      'tests.test_july28_qmd', 'tests.test_qmd_gradient'], repos['july28'], root / 'smoke/unit_july28.log')
        core.command([os.sys.executable, '-u', str(REPO / 'tools/verify_qmd_split_runtime.py'),
                      '--gpu', gpus[0], '--kd_weights', '.25', '.5', '--output', str(root / 'smoke/kd_parity.json')],
                     repos['july28'], root / 'smoke/kd_parity.log')
        run_wave(root / 'smoke', repos, smoke, [gpus[0]], True)
        checks = [validate_cell(core.cell_path(root / 'smoke', c), c, True) for c in smoke]
        core.save(root / 'smoke/summary.json', dict(raw=checks, note='Functionality only; no score-improvement filter.'))
        if any(r['warnings'] for r in checks):
            raise RuntimeError('ineffective auxiliary in smoke; inspect diagnostics')
        core.save(gate, expected_gate)
    if not gate.exists() or core.read(gate) != expected_gate:
        raise RuntimeError('matching smoke PASS required')
    if args.stage == 'smoke':
        core.package(root, '_smoke'); return
    run_wave(root / 'screen', repos, jobs, gpus)
    write_report(root)
    core.save(root / 'completion.json', dict(experiments_complete=True, new_cells=36,
        primary_cells=90, reference_cells=72, total_rows=108, source=source, completed_at=time.time()))


def status(root):
    cal.status(root, smoke_selector=smoke_cells, smoke_epochs=SMOKE_EPOCHS)
    for m, weight in WEIGHTS.items():
        count = sum((core.cell_path(root / 'screen', c) / 'complete.json').exists() for c in matrix() if c[3] == m)
        print('KD weight %.2f: %d/18 (six actions x three seeds)' % (weight, count))
    for m in REFERENCES:
        count = sum((core.cell_path(root / 'reference', c) / 'complete.json').exists() for c in reference_cells() if c[3] == m)
        print('Frozen %s: %d/18' % (m, count))
    print('No pilot filtering. Counts are recorded completions; final summary revalidates all 108 rows.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('all', 'smoke', 'screen', 'status', 'package'), default='all')
    parser.add_argument('--root', default='results/qmd_strength_v11')
    parser.add_argument('--gpus', default='0 1 3 4')
    parser.add_argument('--july28_repo', default='../PAMFN-july28-qmd-strength-v11')
    parser.add_argument('--reference_root', default='../PAMFN-qmd-full-v10/results/qmd_full_v10')
    args = parser.parse_args(); root = Path(args.root).resolve()
    if args.stage == 'status':
        status(root); return
    full.check_result_roots(root, (args.reference_root,))
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path in root.rglob('*.process.json'):
            if not {'attempts', 'reference'}.intersection(path.relative_to(root).parts) and alive(core.read(path).get('identity')):
                raise RuntimeError('live child remains: ' + str(path))
        record = dict(pid=os.getpid(), identity=identity(os.getpid()), state='running', stage=args.stage, started_at=time.time())
        core.save(root / 'runner.json', record)
        try:
            execute(args, root)
            record.update(state='finished', ended_at=time.time()); core.save(root / 'runner.json', record)
            if (root / 'completion.json').exists():
                core.package(root, '')
                print('QMD_STRENGTH_COMPLETE: 36/36 new; weights .25/.5 each 18/18; 108 rows; final ZIP ready.', flush=True)
            elif args.stage == 'package':
                core.package(root, '_partial')
        except Exception as error:
            record.update(state='failed', ended_at=time.time(), error=repr(error)); core.save(root / 'runner.json', record)
            raise


if __name__ == '__main__':
    main()
