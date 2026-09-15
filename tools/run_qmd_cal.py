"""Modern Score-QMD: two fixed calibration candidates, six actions, seeds 0/1."""
import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import sys
import time

from models.qmd_calibration import WindowCalibration
from tools import run_qmd_dual as core
from tools import run_qmd_light as light
from tools.qmd_process import alive, identity

REPO = Path(__file__).resolve().parents[1]
RECIPES = dict(score_qmd_cal_shared='shared', score_qmd_cal_modality='modality')
CALIBRATION_BATCHES = 9
SMOKE_EPOCHS = 2


def matrix(study='screen', methods=()):
    if study == 'screen':
        if methods:
            raise ValueError('screen always includes both fixed candidates')
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
                target_ratio=.075, beta_cap=3., warmup=0. if smoke else .1,
                rampup=0. if smoke else .1, calibration_mode=RECIPES[cell[3]],
                calibration_batches=CALIBRATION_BATCHES)


def cell_args(cell, gpu, smoke=False):
    p, a, s, m = cell
    if p != 'modern' or m not in RECIPES:
        raise ValueError('unknown calibration cell')
    argv = core.cell_args(p, a, s, 'score_qmd', gpu, smoke)
    for flag, value in (('--run_tag', m), ('--qmd_beta_cap', '3')):
        argv[argv.index(flag) + 1] = value
    if smoke:
        argv[argv.index('--epochs') + 1] = str(SMOKE_EPOCHS)
    return argv + ['--qmd_calibration_mode', RECIPES[m], '--qmd_calibration_batches', str(CALIBRATION_BATCHES)]


def validate_calibration(qmd, config):
    report = qmd.get('calibration')
    if not report:
        raise ValueError('calibration window did not finish')
    recomputed = WindowCalibration(config['calibration_mode'], config['calibration_batches'],
                                  config['target_ratio'], config['beta_cap'], config['weight'])
    observations = report.get('observations', [])
    if len(observations) != CALIBRATION_BATCHES:
        raise ValueError('wrong number of calibration batches')
    for row in observations:
        recomputed.observe(row['modality'], row['main_gradient_norm'], row['kd_gradient_norm'],
                           row['samples'], row['epoch'] - 1)
    if recomputed.report != report:
        raise ValueError('calibration report disagrees with its raw training observations')
    progress = qmd.get('window_progress', {})
    if (progress.get('mode') != config['calibration_mode'] or not progress.get('frozen')
            or progress.get('collected_batches') != CALIBRATION_BATCHES
            or progress.get('requested_batches') != CALIBRATION_BATCHES):
        raise ValueError('incomplete calibration progress')
    counts = progress.get('kd_applied_batches', {})
    if set(counts) != {'V', 'F', 'A'} or any(not isinstance(n, int) or n < 0 for n in counts.values()):
        raise ValueError('invalid KD application counts')
    if sum(counts.values()) > qmd['active_batches'] - CALIBRATION_BATCHES:
        raise ValueError('KD was applied during collection')
    flags = []
    if report['degenerate']:
        flags.append('degenerate_score_calibration')
    if any(n == 0 for n in counts.values()):
        flags.append('no_positive_kd_for_some_modality')
    return dict(report['beta']), flags


def validate_cell(out, cell, smoke=False):
    config = settings(cell, smoke)
    result = core.validate_cell(out, cell, smoke, SMOKE_EPOCHS if smoke else None,
                               lambda q: validate_calibration(q, config))
    for name in ('train_result.json', 'test_result.json'):
        run = core.read(out/name)['run']
        if run['qmd'] != config or run['eval_seed'] != 20260908:
            raise ValueError('run differs from fixed calibration recipe')
    qmd = core.read(out/'qmd_training.json')
    rows = qmd.get('view_diagnostics', {}).get('rows', [])
    if not rows or {r['modality'] for r in rows} != {'V', 'F', 'A'}:
        raise ValueError('missing modality diagnostics')
    if qmd['active_batches'] != sum(r['active_batches'] for r in qmd['epochs']):
        raise ValueError('active batch history mismatch')
    result.update(origin='new_training', calibration_mode=config['calibration_mode'],
                  calibration_batches=CALIBRATION_BATCHES, beta_cap=config['beta_cap'])
    return result


def run_cell(root, repos, cell, gpu, smoke):
    return core.run_cell(root, repos, cell, gpu, smoke,
                         argv=cell_args(cell, gpu, smoke), validator=validate_cell)


def run_wave(root, repos, jobs, gpus, smoke=False):
    return core.run_wave(root, repos, jobs, gpus, smoke, runner=run_cell,
                         ordering=lambda cells: sorted(cells, key=lambda c: (-core.BUDGET[c[1]], c[2], c[3])))


def summarize(root, jobs, references):
    result = light.summarize(root, jobs, references, validator=validate_cell)
    result['note'] = ('SRCC x100; equal action/seed weights; paired with frozen original Score-QMD controls. '
                      'Both candidates change calibration window AND beta cap (1 to 3); '
                      'shared versus modality isolates grouping under the same window/cap.')
    return result


def check_resources(root, gpus):
    if shutil.disk_usage(root).free < 8 * 1024**3:
        raise RuntimeError('less than 8 GiB free; choose a volume with enough space (no automatic deletion)')
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                       '--format=csv,noheader,nounits'], universal_newlines=True)
    free = {a.strip(): int(b.strip()) for a, b in (line.split(',') for line in output.splitlines())}
    if any(g not in free or free[g] < 4096 for g in gpus):
        raise RuntimeError('selected GPU missing or less than 4096 MiB free; check nvidia-smi and change --gpus')
    print('Selected physical GPUs, free MiB:', {g: free[g] for g in gpus}, flush=True)


def read_optional(path):
    return core.read(path) if path.exists() else {}


def status(root, smoke_selector=smoke_cells, smoke_epochs=SMOKE_EPOCHS):
    manifest = read_optional(root/'manifest.json')
    runner = read_optional(root/'runner.json')
    print('Runner:', runner.get('state', 'not started'), '; PID:', runner.get('pid', '-'),
          '; alive:', alive(runner.get('identity')))
    # A runner can die while a child continues. Always inspect each child's PID.
    jobs = [tuple(c) for c in manifest.get('jobs', [])]
    smoke = smoke_selector(jobs)
    done = sum((core.cell_path(root/'screen', c)/'complete.json').exists() for c in jobs)
    print('New training completed: %d/%d ; study: %s' % (done, len(jobs), manifest.get('study', 'preflight')))
    gate = (root/'smoke/gate.json').exists()
    print('Smoke: %d/%d ; gate: %s' % (sum((core.cell_path(root/'smoke', c)/'complete.json').exists()
                                         for c in smoke), len(smoke), 'PASS' if gate else 'not yet passed'))
    running = failed = pending = 0
    for cell in jobs if gate else smoke:
        out = core.cell_path(root/('screen' if gate else 'smoke'), cell)
        if (out/'complete.json').exists():
            continue
        train = read_optional(out/'train.process.json')
        test = read_optional(out/'test.process.json')
        process = test if test and test.get('started_at', 0) >= train.get('started_at', 0) else train
        live = alive(process.get('identity'))
        failure = read_optional(out/'failure.json')
        label = 'RUNNING' if live else 'FAILED/INTERRUPTED' if process or failure else 'PENDING'
        running += int(label == 'RUNNING')
        failed += int(label == 'FAILED/INTERRUPTED')
        pending += int(label == 'PENDING')
        if label == 'PENDING':
            continue
        history = out/'model.pth.epochs.jsonl'
        epochs = len(history.read_text().splitlines()) if history.exists() else 0
        log = out/process.get('log', 'train.log')
        age = int(time.time() - log.stat().st_mtime) if log.exists() else -1
        print(label, '/'.join(map(str, cell)), 'completed epochs=%d/%d' % (
              epochs, core.BUDGET[cell[1]] if gate else smoke_epochs),
              'PID=%s log_age=%ss' % (process.get('pid', '-'), age))
        if failure and not failure.get('resolved_at'):
            print('  previous/current error:', failure.get('error'))
        if live and age > 1800:
            print('  WARNING: no log update for >30 min; inspect GPU and this cell log.')
    print('Running: %d ; failed/interrupted: %d ; pending: %d' % (running, failed, pending))
    for path in sorted((root/'smoke').glob('*.process.json')) if (root/'smoke').exists() else []:
        record = core.read(path)
        if alive(record.get('identity')):
            print('CHECK RUNNING:', path.stem, 'PID=', record['pid'])
    if runner.get('error') and runner.get('state') == 'failed':
        print('LAUNCHER ERROR:', runner['error'])
    print('Completion:', 'READY' if (root/'completion.json').exists() else 'not yet created')
    archive = root.parent/(root.name + '_return.zip')
    print('Final archive:', archive, 'READY' if archive.exists() else 'not yet packaged')
    if failed and not running:
        print('Not finished: inspect errors, then rerun SAME root/command on available GPUs. Completed cells are kept.')


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
        recipes={m: settings(('modern', 'Ball', 0, m)) for m in sorted({c[3] for c in jobs})},
        epochs=core.BUDGET, smoke_epochs=SMOKE_EPOCHS, selection='unchanged modern argmax test-best'))
    gpus = args.gpus.split()
    if not gpus or len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
        raise ValueError('provide distinct physical GPU indices')
    check_resources(root, gpus)
    print('Preflight: code, 37 input hashes, environment, frozen v2 Score-QMD controls...', flush=True)
    repos = dict(modern=REPO)
    source = core.provenance(repos)
    core.immutable(root/'source.json', source)
    references = light.checked_reference(Path(args.reference_root).resolve(), source, jobs, root)
    gate_path = root/'smoke/gate.json'
    smoke = smoke_cells(jobs)
    if args.stage in ('all', 'smoke') and not gate_path.exists():
        checks = [(['-m', 'unittest', 'tests.test_qmd_calibration', 'tests.test_qmd_cal_tools',
                    'tests.test_qmd_light', 'tests.test_qmd_light_tools', 'tests.test_score_qmd',
                    'tests.test_sr_qmd', 'tests.test_reliable_score'], 'unit'),
                  (['-m', 'tools.verify_sr_qmd_runtime', '--gpu', gpus[0],
                    '--output', str(root/'smoke/score_parity.json')], 'score_parity')]
        for argv, name in checks:
            print('CHECK ' + name, flush=True)
            core.command([sys.executable, '-u'] + argv, REPO, root/'smoke'/(name + '.log'))
        run_wave(root/'smoke', repos, smoke, gpus, True)
        rows = [validate_cell(core.cell_path(root/'smoke', c), c, True) for c in smoke]
        if any(r['warnings'] for r in rows):
            raise RuntimeError('ineffective calibration/KD in smoke: inspect reports')
        core.save(gate_path, dict(decision='QMD_CAL_SMOKE_PASS', source=source, cells=len(smoke), results=rows))
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
    p.add_argument('--root', default='results/qmd_cal_v4')
    p.add_argument('--reference_root', default='../PAMFN-qmd-dual-v2/results/qmd_dual_v2')
    p.add_argument('--gpus', default='3 4 5')
    args = p.parse_args()
    root = Path(args.root).resolve()
    if args.stage == 'status':
        status(root)
        return
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root/'.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Do not start a second trainer if an orphaned child outlived its launcher.
        for path in root.rglob('*.process.json'):
            if 'attempts' not in path.relative_to(root).parts and alive(core.read(path).get('identity')):
                raise RuntimeError('live child still exists: ' + str(path))
        import os
        record = dict(pid=os.getpid(), identity=identity(os.getpid()), state='running',
                      stage=args.stage, started_at=time.time())
        core.save(root/'runner.json', record)
        try:
            execute(args, root)
        except Exception as error:
            record.update(state='failed', ended_at=time.time(), error=repr(error))
            core.save(root/'runner.json', record)
            raise
        record.update(state='finished', ended_at=time.time())
        core.save(root/'runner.json', record)
        if (root/'completion.json').exists():
            core.package(root, '')
            count = core.read(root/'completion.json')['new_cells']
            print('QMD_CAL_COMPLETE: %d/%d; final return ZIP ready.' % (count, count), flush=True)


if __name__ == '__main__':
    main()
