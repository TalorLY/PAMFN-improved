"""July28 auxiliary-signal split: 12 new cells plus 3 KD-guard confirmations.

Old controls are verified/read-only, not counted as newly trained results.
The six imported guard cells plus seed2 form a separate three-seed comparison.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

from tools import run_qmd_dual as core
from tools import run_qmd_gradient as grad
from tools import run_qmd_cal as cal
from tools.qmd_process import alive, identity
from tools.reuse_qmd_gradient_v8 import copy_unchanged

REPO = Path(__file__).resolve().parents[1]
ACTIONS = ('Ball', 'Hoop', 'Ribbon')
GT = 'ordered_score_qmd_split_gt'
KD = 'ordered_score_qmd_split_kd'
BOTH = 'ordered_score_qmd_split_both'
GUARD = 'ordered_score_qmd_guard_kd'
TERMS = {GT: 'gt', KD: 'kd', BOTH: 'both', GUARD: 'both'}
V8_COMMITS = dict(modern='39a395cd07b57283701f6d220094311ab9988c3b',
                  july28='4d95c157f0efd745d534d604d9502895bb586c76')
SMOKE_EPOCHS = 2


def matrix():
    return [('july28', a, s, m) for a in ACTIONS for s in (0, 1) for m in (GT, KD)] + [
        ('july28', a, 2, GUARD) for a in ACTIONS]


def smoke_cells(jobs):
    return [('july28', a, 0, m) for a in ('Ball', 'TES') for m in (BOTH, GT, KD, GUARD)] if jobs else []


def settings(cell, smoke=False):
    if cell[0] != 'july28' or cell[3] not in TERMS:
        raise ValueError('unknown split cell')
    result = dict(mode='score_qmd', weight=1., mask_ratio=.15, mask_gt_weight=.5,
                  target_ratio=.075, beta_cap=1., warmup=0. if smoke else .1, rampup=0. if smoke else .1)
    if cell[3] in (GT, KD):
        result['terms'] = TERMS[cell[3]]
    if cell[3] == GUARD:
        result['grad_mode'] = 'kd'
    return result


def cell_args(cell, gpu, smoke=False):
    settings(cell, smoke)
    p, a, s, m = cell
    argv = core.cell_args(p, a, s, 'ordered_score_qmd', gpu, smoke)
    argv[argv.index('--run_tag') + 1] = m
    if smoke:
        argv[argv.index('--epochs') + 1] = str(SMOKE_EPOCHS)
    if m in (GT, KD):
        argv += ['--qmd_terms', TERMS[m]]
    if m == GUARD:
        argv += ['--qmd_grad_mode', 'kd']
    return argv


def validate_cell(out, cell, smoke=False, recipe=None):
    expected = settings(cell, smoke) if recipe is None else recipe
    terms = expected.get('terms', 'both')
    if cell[3] == GUARD:
        row = grad.validate_cell(out, cell, smoke)
    else:
        row = core.validate_cell(out, cell, smoke, SMOKE_EPOCHS if smoke else None)
    for name in ('train_result.json', 'test_result.json'):
        if core.read(out / name)['run']['qmd'] != expected:
            raise ValueError('split QMD configuration mismatch')
    test = core.read(out / 'test_result.json')
    import numpy as np
    from scipy.stats import spearmanr
    y = np.asarray(test['predictions']['label'])
    pred = np.asarray(test['predictions']['prediction'])
    n = 100 if cell[1] in ('TES', 'PCS') else 50
    if y.shape != (n,) or pred.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('saved prediction shape/values disagree')
    coef = spearmanr(y, pred)[0] * 100
    if (not math.isfinite(coef) or abs(coef - test['coef']) > 1e-6
            or abs(float(np.mean((pred - y) ** 2)) - test['loss_x100'] / 100) > 1e-6):
        raise ValueError('saved prediction results disagree')
    qmd = core.read(out / 'qmd_training.json')
    if sum(e['active_batches'] for e in qmd['epochs']) != qmd['active_batches']:
        raise ValueError('active batch counts disagree')
    if terms in ('gt', 'kd'):
        comp = qmd.get('components', {})
        if comp.get('terms') != terms or comp.get('batches') != qmd['active_batches']:
            raise ValueError('split component records missing/incomplete')
        disabled = 'weighted_kd_sum' if terms == 'gt' else 'weighted_gt_sum'
        enabled = 'weighted_gt_sum' if terms == 'gt' else 'weighted_kd_sum'
        if comp.get(disabled) != 0. or not math.isfinite(comp[enabled]) or comp[enabled] < 0:
            raise ValueError('disabled component contributed or invalid component loss')
        if qmd.get('gradient_guard') is not None:
            raise ValueError('split experiment unexpectedly projected gradients')
        if terms == 'gt':
            # Beta is still measured identically, but is not applied in GT-only.
            row['warnings'] = [w for w in row['warnings'] if w != 'degenerate_score_calibration']
        norm = comp.get('first_positive_head_gradient_norm')
        if comp.get('gradient_verified') and (norm is None or not math.isfinite(norm) or norm <= 0):
            raise ValueError('invalid split gradient evidence')
        if not comp.get('gradient_verified') or comp[enabled] == 0:
            row['warnings'].append('no_verified_split_gradient')
        row.update(weighted_gt_sum=comp['weighted_gt_sum'], weighted_kd_sum=comp['weighted_kd_sum'])
    row.update(terms=terms, mse=test['loss_x100'] / 100)
    return row


def check_warmup(out, cell, reference):
    ref = core.cell_path(reference, (cell[0], cell[1], cell[2], 'ordered_score_qmd'))
    histories = [[json.loads(line) for line in (p / 'model.pth.epochs.jsonl').read_text().splitlines() if line.strip()]
                 for p in (out, ref)]
    count = int(math.ceil(.1 * core.BUDGET[cell[1]]))
    if any(len(h) < count for h in histories):
        raise ValueError('missing pre-auxiliary history')
    for left, right in zip(histories[0][:count], histories[1][:count]):
        for key in ('epoch', 'coef', 'loss_x100', 'prediction_std'):
            x, y = left[key], right[key]
            if not ((math.isnan(x) and math.isnan(y)) or abs(x - y) <= 1e-6):
                raise ValueError('pre-auxiliary trajectory changed at epoch %s: %s' % (left['epoch'], key))


def run_cell(root, repos, cell, gpu, smoke):
    def checked(out, c, s):
        row = validate_cell(out, c, s)
        if not s:
            check_warmup(out, c, root.parent / 'reference')
        return row
    return core.run_cell(root, repos, cell, gpu, smoke, argv=cell_args(cell, gpu, smoke), validator=checked)


def run_wave(root, repos, jobs, gpus, smoke=False):
    return core.run_wave(root, repos, jobs, gpus, smoke, runner=run_cell,
                         ordering=lambda cells: sorted(cells, key=lambda c: (-core.BUDGET[c[1]], c[2], c[3])))


def verify_record(out):
    record = core.read(out / 'complete.json')
    for name, digest in record['files'].items():
        if Path(name).name != name or core.sha(out / name) != digest:
            raise ValueError('result file changed: ' + str(out / name))
    return record


def checked_v8(reference, source, root):
    old = core.read(reference / 'source.json')
    completion = core.read(reference / 'completion.json')
    if (old['commits'] != V8_COMMITS or completion.get('source') != old
            or completion.get('new_cells') != 36 or not completion.get('experiments_complete')):
        raise ValueError('requires completed frozen v8 fix1 results')
    if any(old[k] != source[k] for k in ('inputs', 'environment')):
        raise ValueError('v8 data/environment mismatch')
    rows = []
    for a in ACTIONS:
        for s in (0, 1):
            cell = ('july28', a, s, GUARD)
            origin = core.cell_path(reference / 'screen', cell)
            record = verify_record(origin)
            row = grad.validate_cell(origin, cell)
            if abs(record['metrics']['srcc_x100'] - row['srcc_x100']) > 1e-6:
                raise ValueError('v8 completion score changed')
            dest = core.cell_path(root / 'reference_v8', cell)
            for name in list(record['files']) + ['complete.json', 'train.log', 'test.log']:
                copy_unchanged(origin / name, dest / name)
            rows.append(dict(row, origin='frozen_v8_fix1_guard', source_path=str(origin)))
    for name in ('source.json', 'completion.json', 'manifest.json', 'reuse_manifest.json'):
        if (reference / name).exists():
            copy_unchanged(reference / name, root / 'reference_v8' / name)
    core.immutable(root / 'v8_reference_manifest.json', dict(source=old, records=rows))
    return rows


def summarize(root, jobs, refs, previous):
    rows = []
    for c in jobs:
        out = core.cell_path(root / 'screen', c)
        verify_record(out)
        row = validate_cell(out, c)
        check_warmup(out, c, root / 'reference')
        rows.append(dict(row, origin='new_v9_training'))
    for old in previous:
        cell = ('july28', old['action'], old['seed'], GUARD)
        out = core.cell_path(root / 'reference_v8', cell)
        verify_record(out)
        checked = grad.validate_cell(out, cell)
        if abs(old['srcc_x100'] - checked['srcc_x100']) > 1e-6:
            raise ValueError('imported v8 reference score differs')
        rows.append(dict(old))
    lookup = {(r['action'], r['seed'], r['method']): r['srcc_x100'] for r in refs}
    for row in rows:
        key = (row['action'], row['seed'])
        row['ordered_srcc'] = lookup[key + ('ordered',)]
        row['original_combination_srcc'] = lookup[key + ('ordered_score_qmd',)]
        row['delta_vs_ordered'] = row['srcc_x100'] - row['ordered_srcc']
        row['delta_vs_combination'] = row['srcc_x100'] - row['original_combination_srcc']
    groups = []
    for method in (GT, KD, GUARD):
        group = [r for r in rows if r['method'] == method]
        expected = 9 if method == GUARD else 6
        if len(group) != expected or len({(r['action'], r['seed']) for r in group}) != expected:
            raise ValueError('incomplete fixed method grid')
        mean = lambda key, subset=group: sum(r[key] for r in subset) / len(subset)
        groups.append(dict(method=method, cells=expected, seeds=[0, 1, 2] if method == GUARD else [0, 1],
            mean_srcc_x100=mean('srcc_x100'), delta_vs_ordered=mean('delta_vs_ordered'),
            delta_vs_combination=mean('delta_vs_combination'),
            per_action={a: dict(score=mean('srcc_x100', [r for r in group if r['action'] == a]),
                               delta_vs_ordered=mean('delta_vs_ordered', [r for r in group if r['action'] == a])) for a in ACTIONS},
            per_seed={str(s): dict(score=mean('srcc_x100', [r for r in group if r['seed'] == s]),
                                  delta_vs_ordered=mean('delta_vs_ordered', [r for r in group if r['seed'] == s]))
                      for s in sorted({r['seed'] for r in group})}))
    return dict(raw=rows, methods=groups, note='Split methods: 3 actions x seeds0/1. Guard confirmation: same 3 actions x seeds0/1/2, with six frozen v8 cells. Compare each group to its own matched Ordered and unprotected combination; do not compare different-seed means directly. Native July28 test-best and evaluation RNG replay unchanged.')


def execute(args, root):
    if args.stage == 'package':
        if (root / 'completion.json').exists():
            summarize(root, matrix(), core.read(root / 'reference_manifest.json')['records'],
                      core.read(root / 'v8_reference_manifest.json')['records'])
        return
    jobs = matrix()
    core.immutable(root / 'manifest.json', dict(study='screen', jobs=jobs, total_new_cells=15,
        recipes={m: settings(('july28', 'Ball', 0, m)) for m in (GT, KD, GUARD)},
        epochs=core.BUDGET, smoke_epochs=SMOKE_EPOCHS, frozen_guard_cells=6))
    gpus = args.gpus.split()
    if not gpus or len(gpus) != len(set(gpus)) or any(not g.isdigit() for g in gpus):
        raise ValueError('provide distinct physical GPU indices')
    cal.check_resources(root, gpus)
    repos = dict(modern=REPO, july28=Path(args.july28_repo).resolve())
    print('PREFLIGHT: code, 37 input hashes, v2 paired controls and completed v8 guard references', flush=True)
    source = core.provenance(repos)
    core.immutable(root / 'source.json', source)
    refs = grad.checked_reference(Path(args.reference_root).resolve(), source, jobs, root)
    previous = checked_v8(Path(args.v8_root).resolve(), source, root)
    smoke = smoke_cells(jobs)
    expected_gate = dict(decision='QMD_SPLIT_SMOKE_PASS', cells=len(smoke), source=source)
    gate = root / 'smoke/gate.json'
    if args.stage in ('all', 'smoke') and not gate.exists():
        core.command([os.sys.executable, '-u', '-m', 'unittest', 'tests.test_qmd_split_tools'], REPO, root / 'smoke/unit_tools.log')
        core.command([os.sys.executable, '-u', '-m', 'unittest', 'tests.test_qmd_split', 'tests.test_july28_qmd',
                      'tests.test_qmd_gradient'], repos['july28'], root / 'smoke/unit_july28.log')
        core.command([os.sys.executable, '-u', str(REPO / 'tools/verify_qmd_split_runtime.py'),
                      '--gpu', gpus[0], '--output', str(root / 'smoke/parity.json')], repos['july28'], root / 'smoke/parity.log')
        # All short smoke cells stay on a single selected GPU.
        run_wave(root / 'smoke', repos, smoke, [gpus[0]], True)
        checks = [validate_cell(core.cell_path(root / 'smoke', c), c, True) for c in smoke]
        core.save(root / 'smoke/summary.json', dict(raw=checks))
        if any(r['warnings'] for r in checks):
            raise RuntimeError('ineffective auxiliary in smoke; inspect diagnostics')
        core.save(gate, expected_gate)
    if not gate.exists() or core.read(gate) != expected_gate:
        raise RuntimeError('matching smoke PASS required')
    if args.stage == 'smoke':
        core.package(root, '_smoke')
        return
    run_wave(root / 'screen', repos, jobs, gpus)
    report = summarize(root, jobs, refs, previous)
    core.save(root / 'summary.json', report)
    with (root / 'scores.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in report['raw'] for k in r}))
        writer.writeheader(); writer.writerows(report['raw'])
    core.save(root / 'completion.json', dict(experiments_complete=True, new_cells=15, frozen_guard_cells=6,
        reference_cells=len(refs), source=source, completed_at=time.time()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('all', 'smoke', 'screen', 'status', 'package'), default='all')
    parser.add_argument('--root', default='results/qmd_split_v9')
    parser.add_argument('--gpus', default='4')
    parser.add_argument('--july28_repo', default='../PAMFN-july28-qmd-split-v9')
    parser.add_argument('--reference_root', default='../PAMFN-qmd-dual-v2/results/qmd_dual_v2')
    parser.add_argument('--v8_root', default='../PAMFN-qmd-grad-v8-fix1/results/qmd_grad_v8_fix1')
    args = parser.parse_args(); root = Path(args.root).resolve()
    if args.stage == 'status':
        cal.status(root, smoke_selector=smoke_cells, smoke_epochs=SMOKE_EPOCHS)
        for label, jobs in [('Split signals', [c for c in matrix() if c[3] != GUARD]),
                            ('Guard seed2 confirmation', [c for c in matrix() if c[3] == GUARD])]:
            count = sum((core.cell_path(root / 'screen', c) / 'complete.json').exists() for c in jobs)
            print('%s: %d/%d' % (label, count, len(jobs)))
        print('Frozen guard reference: %d/6 (not new training)' % (
            len(core.read(root / 'v8_reference_manifest.json')['records']) if (root / 'v8_reference_manifest.json').exists() else 0))
        return
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path in root.rglob('*.process.json'):
            if not {'attempts', 'reference', 'reference_v8'}.intersection(path.relative_to(root).parts) and alive(core.read(path).get('identity')):
                raise RuntimeError('live child remains: ' + str(path))
        record = dict(pid=os.getpid(), identity=identity(os.getpid()), state='running', stage=args.stage, started_at=time.time())
        core.save(root / 'runner.json', record)
        try:
            execute(args, root)
            record.update(state='finished', ended_at=time.time()); core.save(root / 'runner.json', record)
            if (root / 'completion.json').exists():
                core.package(root, '')
                print('QMD_SPLIT_COMPLETE: 15/15 new; full results and ZIP ready.', flush=True)
            elif args.stage == 'package':
                core.package(root, '_partial')
        except Exception as error:
            record.update(state='failed', ended_at=time.time(), error=repr(error)); core.save(root / 'runner.json', record)
            raise


if __name__ == '__main__':
    main()
