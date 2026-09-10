"""Two-protocol QMD study: 42 seed-0 pilot cells, then 84 confirmation cells.

Every cell trains in its own repository/process; July28 never imports modern
PAMFN. No old test score is used for gating or silently reused as a new run.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
import zipfile

REPO = Path(__file__).resolve().parents[1]
HISTORICAL = '53716f61c117376154c9335b1a689bc667404ce2'
MODERN_BASE = '536046dfdcd64835c6a06d4f2fd74593ff5b061d'
ACTIONS = ('TES', 'PCS', 'Ball', 'Clubs', 'Hoop', 'Ribbon')
SEEDS = {'modern': (0, 1, 42), 'july28': (0, 1, 2)}
METHODS = {
    'modern': ('score_qmd', 'ordered_score_qmd', 'reliable_score_qmd', 'ordered_reliable_score_qmd'),
    'july28': ('ordered', 'ordered_score_qmd', 'ordered_sr_qmd'),
}
BUDGET = dict(TES=100, PCS=100, Ball=500, Clubs=100, Hoop=300, Ribbon=20)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def immutable(path, value):
    value = json.loads(json.dumps(value))
    if Path(path).exists() and read(path) != value:
        raise ValueError('configuration/source changed; use a new root: ' + str(path))
    save(path, value)


def command(argv, cwd, log):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open('a', encoding='utf-8') as handle:
        handle.write('\nCOMMAND ' + json.dumps(argv) + '\n')
        handle.flush()
        env = dict(os.environ, PYTHONPATH=str(cwd), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        subprocess.run(argv, cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def mode_for(method):
    if method == 'ordered':
        return 'off'
    if 'reliable' in method:
        return 'reliable_score_qmd'
    if method == 'ordered_sr_qmd':
        return 'score_rank_qmd'
    return 'score_qmd'


def cell_args(protocol, action, seed, method, gpu, smoke=False):
    if method not in METHODS[protocol] or action not in ACTIONS or seed not in SEEDS[protocol]:
        raise ValueError('cell is outside the fixed experiment matrix')
    argv = [sys.executable, '-u', 'main.py', '--gpu', str(gpu), '--feats', '2',
            '--action', action, '--seed', str(seed), '--multi_modality',
            '--sequence_sampling', 'legacy', '--run_tag', method, '--qmd_mode', mode_for(method)]
    if protocol == 'modern':
        argv += ['--eval_policy', 'argmax', '--eval_seed', '20260908']
    if method.startswith('ordered'):
        argv += ['--ordinal_prototype_weight', '.0001', '--ordinal_prototype_dim', '128',
                 '--ordinal_prototype_temperature', '.1', '--ordinal_prototype_order_margin', '.1',
                 '--ordinal_prototype_order_scale', '.1']
        if protocol == 'modern':
            argv += ['--ordinal_prototype_target_mode', 'continuous', '--quality_aux_rng_isolated']
    if mode_for(method) != 'off':
        argv += ['--qmd_mask_ratio', '.15', '--qmd_gt_weight', '.5', '--qmd_target_ratio', '.075',
                 '--qmd_beta_cap', '1', '--qmd_weight', '1', '--qmd_warmup', '0' if smoke else '.1',
                 '--qmd_rampup', '0' if smoke else '.1']
    if mode_for(method) == 'score_rank_qmd':
        argv += ['--qmd_rank_weight', '.5', '--qmd_rank_gap', '.05', '--qmd_rank_gamma', '.1']
    if smoke:
        argv += ['--epochs', '1']
    return argv


def cells(smoke=False, pilot=False):
    result = []
    for protocol in METHODS:
        for seed in ((0,) if smoke or pilot else SEEDS[protocol]):
            for action in (('Ball', 'TES') if smoke else ACTIONS):
                for method in METHODS[protocol]:
                    result.append((protocol, action, seed, method))
    return result


def cell_path(root, cell):
    protocol, action, seed, method = cell
    return root / protocol / action / ('seed' + str(seed)) / method


def validate_cell(out, cell, smoke=False):
    protocol, action, seed, method = cell
    train, test = read(out/'train_result.json'), read(out/'test_result.json')
    epochs = 1 if smoke else BUDGET[action]
    for data in (train, test):
        run = data['run']
        expected = dict(action=action, seed=seed, tag=method, epochs=epochs, feats=2,
                        eval_policy='argmax' if protocol == 'modern' else 'legacy_gumbel')
        for key, value in expected.items():
            if run.get(key) != value:
                raise ValueError('wrong run identity: ' + str(out) + ' ' + key)
        if run['qmd']['mode'] != mode_for(method):
            raise ValueError('wrong QMD mode')
        if run['ordinal_prototype_weight'] != (.0001 if method.startswith('ordered') else 0.):
            raise ValueError('wrong Ordered weight')
        if method.startswith('ordered'):
            if run['ordinal_prototype_target_mode'] != ('continuous' if protocol == 'modern' else 'discrete'):
                raise ValueError('wrong Ordered version')
            if run['quality_aux_rng_isolated'] != (protocol == 'modern'):
                raise ValueError('wrong Ordered initialization protocol')
    if protocol == 'july28' and not test['evaluation']['replayed_rng']:
        raise ValueError('July28 checkpoint was not replayed with its original test RNG')
    score = test['coef']
    if not all(math.isfinite(x) for x in (score, train['saved_model_srcc'], test['loss_x100'])):
        raise ValueError('nonfinite final metric: ' + str(out))
    if abs(score - 100*train['saved_model_srcc']) > 1e-6:
        raise ValueError('checkpoint reload does not match selection: ' + str(out))
    selection = read(out/'model.pth.selection.json')
    if abs(selection['coef'] - score) > 1e-6 or not 1 <= selection['epoch'] <= epochs:
        raise ValueError('bad checkpoint-selection metadata')
    lines = [json.loads(x) for x in (out/'model.pth.epochs.jsonl').read_text().splitlines() if x.strip()]
    if [x['epoch'] for x in lines] != list(range(1, epochs+1)):
        raise ValueError('epoch history is incomplete or duplicated')
    recorded = lines[selection['epoch']-1]['coef']
    if not math.isfinite(recorded) or abs(recorded - score) > 1e-6:
        raise ValueError('selected checkpoint disagrees with the epoch history')
    flags = []
    active = accepted = total = 0
    beta = None
    if mode_for(method) != 'off':
        qmd = read(out/'qmd_training.json')
        if qmd['mode'] != mode_for(method) or len(qmd['epochs']) != epochs or qmd['active_batches'] <= 0:
            raise ValueError('QMD did not complete the requested training')
        active = qmd['active_batches']
        calibration = qmd['calibration']
        beta = calibration['beta']
        if not math.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError('invalid beta')
        if calibration['degenerate'] or beta == 0:
            flags.append('degenerate_score_calibration')
        if mode_for(method) == 'reliable_score_qmd':
            gate = qmd['score_gate']
            accepted, total = gate['accepted_samples'], gate['total_samples']
            if not gate['gradient_verified']:
                flags.append('no_verified_gated_gradient')
        if mode_for(method) == 'score_rank_qmd' and not qmd['rank']['gradient_verified']:
            flags.append('no_verified_rank_gradient')
    # Weak/degenerate runs are retained and flagged, never removed to raise means.
    return dict(protocol=protocol, action=action, seed=seed, method=method, srcc_x100=score,
                selected_epoch=selection['epoch'], prediction_std=test['prediction_std'],
                selected_before_aux=(mode_for(method) != 'off' and not smoke
                                     and selection['epoch'] <= math.ceil(.1*epochs)),
                nan_eval_epochs=sum(not math.isfinite(x['coef']) for x in lines),
                active_batches=active, beta=beta, accepted_samples=accepted, total_samples=total,
                warnings=flags)


def run_cell(root, repositories, cell, gpu, smoke):
    out = cell_path(root, cell)
    out.mkdir(parents=True, exist_ok=True)
    protocol, action, seed, method = cell
    argv = cell_args(protocol, action, seed, method, gpu, smoke)
    immutable(out/'run_config.json', dict(argv=argv[5:], protocol=protocol))
    if (out/'complete.json').exists():
        metrics = validate_cell(out, cell, smoke)
        for name, digest in read(out/'complete.json')['files'].items():
            if sha(out/name) != digest:
                raise ValueError('completed raw result changed: ' + str(out/name))
        if not (out/'model.pth').is_file():
            raise ValueError('completed checkpoint missing')
        if sha(out/'model.pth') != read(out/'complete.json')['checkpoint_sha256']:
            raise ValueError('completed checkpoint changed')
        if protocol == 'july28' and sha(out/'model.pth.eval_rng.pt') != read(out/'complete.json')['eval_rng_sha256']:
            raise ValueError('July28 replay RNG changed')
        return 'SKIP', metrics
    required = ['model.pth', 'train_result.json', 'model.pth.selection.json', 'model.pth.epochs.jsonl']
    if mode_for(method) != 'off':
        required.append('qmd_training.json')
    if protocol == 'july28':
        required.append('model.pth.eval_rng.pt')
    if not all((out/name).is_file() for name in required):
        # An interrupted cell restarts its full seed, not a partial optimizer state.
        # Move all partial cell artifacts into a recoverable attempt directory.
        existing = [p for p in out.iterdir() if p.is_file() and p.name != 'run_config.json']
        if existing:
            attempt = out/'attempts'/str(time.time_ns())
            attempt.mkdir(parents=True)
            for path in existing:
                path.rename(attempt/path.name)
        command(argv + ['--save_final_model', str(out/'model.pth'), '--result_json', str(out/'train_result.json')],
                repositories[protocol], out/'train.log')
    replay = ['--replay_rng_path', str(out/'model.pth.eval_rng.pt')] if protocol == 'july28' else []
    command(argv + ['--test', '--ckpt_path', str(out/'model.pth'), '--result_json', str(out/'test_result.json')] + replay,
            repositories[protocol], out/'test.log')
    metrics = validate_cell(out, cell, smoke)
    if (out/'failure.json').exists():
        previous = read(out/'failure.json')
        previous['resolved_at'] = time.time()
        save(out/'failure.json', previous)
    files = {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.json', '.jsonl')
             and p.name != 'complete.json'}
    save(out/'complete.json', dict(metrics=metrics, files=files, finished_at=time.time(),
                                  checkpoint_sha256=sha(out/'model.pth'),
                                  eval_rng_sha256=sha(out/'model.pth.eval_rng.pt') if protocol == 'july28' else None))
    return 'FINISH', metrics


def dispatch_order(jobs):
    # Interleave protocols at each cost/method tier so July28 starts immediately
    # alongside modern jobs instead of waiting for four modern Ball trainings.
    return sorted(jobs, key=lambda c: (-BUDGET[c[1]], METHODS[c[0]].index(c[3]),
                                      tuple(METHODS).index(c[0]), c[2]))


def run_wave(root, repositories, jobs, gpus, smoke=False):
    pending = queue.Queue()
    # Both protocols participate in each wave. Greedy longest-first scheduling.
    for job in dispatch_order(jobs):
        pending.put(job)
    failures = []
    def worker(gpu):
        while True:
            try:
                cell = pending.get_nowait()
            except queue.Empty:
                return
            print('START GPU=' + gpu + ' ' + '/'.join(map(str, cell)), flush=True)
            try:
                state, metrics = run_cell(root, repositories, cell, gpu, smoke)
                print(state + ' GPU=' + gpu + ' ' + '/'.join(map(str, cell)) + ' SRCC=' + str(metrics['srcc_x100']), flush=True)
            except Exception as error:
                failure = dict(cell=list(cell), gpu=gpu, error=repr(error), time=time.time())
                failures.append(failure)
                save(cell_path(root, cell)/'failure.json', failure)
                print('FAILED ' + json.dumps(failure), flush=True)
            finally:
                pending.task_done()
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, gpus))
    if failures:
        save(root/'failure.json', dict(unresolved=failures, time=time.time()))
        raise RuntimeError(str(len(failures)) + ' cells failed; inspect logs and rerun the same command')
    if (root/'failure.json').exists():
        old = read(root/'failure.json')
        old['resolved_at'] = time.time()
        save(root/'failure.json', old)


def summarize(root, jobs, smoke=False):
    rows = [validate_cell(cell_path(root, cell), cell, smoke) for cell in jobs]
    report = dict(raw=rows, protocols={})
    for protocol, methods in METHODS.items():
        selected = [x for x in rows if x['protocol'] == protocol]
        seed_set = sorted(set(x['seed'] for x in selected))
        actions = sorted(set(x['action'] for x in selected))
        control = methods[0]
        groups = {}
        for method in methods:
            values = [r for r in selected if r['method'] == method]
            if len(values) != len(seed_set)*len(actions):
                raise ValueError('incomplete method matrix')
            groups[method] = dict(mean_srcc_x100=sum(r['srcc_x100'] for r in values)/len(values),
                per_action={a: sum(r['srcc_x100'] for r in values if r['action']==a)/len(seed_set) for a in actions},
                per_seed={str(s): sum(r['srcc_x100'] for r in values if r['seed']==s)/len(actions) for s in seed_set})
        for method in methods:
            groups[method]['delta_vs_control'] = groups[method]['mean_srcc_x100']-groups[control]['mean_srcc_x100']
        report['protocols'][protocol] = dict(control=control, seeds=seed_set, methods=groups)
    return report


def package(root, suffix):
    output = root.parent/(root.name + suffix + '_return.zip')
    temporary = output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix not in ('.pth', '.pt', '.ckpt', '.pyc') and '__pycache__' not in path.parts:
                archive.write(path, str(path.relative_to(root)))
    temporary.replace(output)
    print('ARCHIVE ' + str(output) + ' SHA256=' + sha(output), flush=True)
    return output


def provenance(repositories):
    import torch
    import numpy
    import scipy
    import timm
    result = dict(commits={}, inputs={}, environment=dict(python=sys.version, torch=torch.__version__,
                  cuda=torch.version.cuda, numpy=numpy.__version__, scipy=scipy.__version__, timm=timm.__version__))
    for protocol, repo in repositories.items():
        def git(*args):
            return subprocess.check_output(['git', '-C', str(repo)] + list(args), text=True).strip()
        if git('status', '--porcelain', '--untracked-files=no'):
            raise ValueError('commit tracked changes before server runs: ' + str(repo))
        result['commits'][protocol] = git('rev-parse', 'HEAD')
        ancestor = HISTORICAL if protocol == 'july28' else MODERN_BASE
        subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor', ancestor, 'HEAD'], check=True)
        if protocol == 'july28' and git('diff', HISTORICAL, 'HEAD', '--', 'models', ':!models/score_qmd.py', 'configs'):
            raise ValueError('July28 model or configuration changed')
    paths = [Path('data')/d/f for d in ('RG', 'FISV') for f in ('train.txt', 'test.txt')]
    paths += [Path('data/features')/(a+'_'+m+'_'+f+'.npy') for a in ('Ball','Clubs','Hoop','Ribbon','FISV')
              for m,f in (('rgb','UMT'),('flow','I3D'),('audio','MAST'))]
    paths += [Path('pretrained_models/feats2')/(a+'_'+m+'_'+f+'.pth') for a in ACTIONS
              for m,f in (('rgb','UMT'),('flow','I3D'),('audio','MAST'))]
    for relative in paths:
        left, right = repositories['modern']/relative, repositories['july28']/relative
        digest = sha(left)
        if left.resolve() != right.resolve() and sha(right) != digest:
            raise ValueError('protocols use different data/pretrained weights: ' + str(relative))
        result['inputs'][relative.as_posix()] = dict(bytes=left.stat().st_size, sha256=digest)
    return result


def status(root):
    total = cells()
    done = sum((cell_path(root/'screen', cell)/'complete.json').is_file() for cell in total)
    print('Screen completed: %d/%d' % (done, len(total)))
    for protocol in METHODS:
        subset = [c for c in total if c[0] == protocol]
        count = sum((cell_path(root/'screen', c)/'complete.json').is_file() for c in subset)
        print(protocol + ': %d/%d' % (count, len(subset)))
    print('Completion report:', 'READY' if (root/'completion.json').exists() else 'not yet created')
    gate = root/'smoke'/'gate.json'
    smoke_count = sum((cell_path(root/'smoke', c)/'complete.json').is_file() for c in cells(smoke=True))
    print('Smoke cells: %d/14; gate: %s' % (smoke_count, 'PASS' if gate.exists() else 'not yet passed'))
    progress_root = root/('screen' if gate.exists() else 'smoke')
    progress_cells = total if gate.exists() else cells(smoke=True)
    if not gate.exists():
        print('Current phase: preflight/smoke. Full training starts after smoke PASS.')
    for cell in progress_cells:
        out = cell_path(progress_root, cell)
        if (out/'complete.json').exists() or not (out/'train.log').exists():
            continue
        epochs = [line for line in (out/'train.log').read_text(errors='replace').splitlines() if line.startswith('Epoch[')]
        print('UNFINISHED', '/'.join(map(str, cell)), epochs[-1].split('\t')[0] if epochs else 'loading / first epoch')
    for suffix in ('_pilot', '', '_partial'):
        p = root.parent/(root.name + suffix + '_return.zip')
        print('Archive:', str(p), 'READY' if p.exists() else 'not yet packaged')
    print('Logs describe recorded progress, not live processes. Check pgrep and nvidia-smi as well.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('all','smoke','screen','status','package'), default='all')
    p.add_argument('--root', default='results/qmd_dual_v2')
    p.add_argument('--july28_repo', default='../PAMFN-july28-qmd-v2')
    p.add_argument('--gpus', default='0 1 3 4')
    args = p.parse_args()
    root = Path(args.root).resolve()
    if args.stage == 'status':
        status(root)
        return
    root.mkdir(parents=True, exist_ok=True)
    # Linux flock is released on interruption. Do not delete a live lock file.
    import fcntl
    with (root/'.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'package':
            package(root, '' if (root/'completion.json').exists() else '_partial')
            return
        gpus = args.gpus.split()
        if not gpus or len(gpus) != len(set(gpus)) or any(not x.isdigit() for x in gpus):
            raise ValueError('provide distinct physical GPU indices')
        repos = dict(modern=REPO, july28=Path(args.july28_repo).resolve())
        print('Checking source, environment, labels, features and pretrained hashes...', flush=True)
        source = provenance(repos)
        immutable(root/'source.json', source)
        immutable(root/'manifest.json', dict(methods=METHODS, seeds=SEEDS, epochs=BUDGET,
                  actions=ACTIONS, pilot_cells=42, total_cells=126, selection='original test-best per protocol',
                  reliable_gate='training abs(full-y) < abs(masked-y); ungated beta; no Rank'))
        gate_path = root/'smoke'/'gate.json'
        if args.stage in ('all','smoke') and not gate_path.exists():
            checks = [
                ('modern', ['-m','unittest','tests.test_reliable_score','tests.test_qmd_dual_tools','tests.test_sr_qmd',
                            'tests.test_score_qmd','tests.test_coqm_rng_isolation'], 'unit_modern'),
                ('july28', ['-m','unittest','tests.test_july28_qmd'], 'unit_july28'),
                ('modern', ['-m','tools.verify_sr_qmd_runtime','--gpu',gpus[0],
                            '--output',str(root/'smoke'/'score_parity.json')], 'score_parity'),
                ('july28', ['-m','tools.verify_july28_qmd','--gpu',gpus[-1],
                            '--output',str(root/'smoke'/'july28_parity.json')], 'july28_parity'),
            ]
            for protocol, argv, name in checks:
                print('CHECK ' + name, flush=True)
                command([sys.executable, '-u']+argv, repos[protocol], root/'smoke'/(name+'.log'))
            run_wave(root/'smoke', repos, cells(smoke=True), gpus, smoke=True)
            smoke_report = summarize(root/'smoke', cells(smoke=True), smoke=True)
            save(root/'smoke'/'summary.json', smoke_report)
            if any(row['warnings'] for row in smoke_report['raw']):
                raise RuntimeError('smoke found an inactive/degenerate auxiliary; inspect smoke summary')
            save(gate_path, dict(source=source, decision='PASS', cells=14))
        if not gate_path.exists() or read(gate_path)['source'] != source:
            raise RuntimeError('matching smoke gate is required before full experiments')
        if args.stage == 'smoke':
            package(root, '_smoke')
            return
        run_wave(root/'screen', repos, cells(pilot=True), gpus)
        pilot = summarize(root/'screen', cells(pilot=True))
        save(root/'pilot_summary.json', pilot)
        package(root, '_pilot')
        run_wave(root/'screen', repos, [c for c in cells() if c[2] != 0], gpus)
        report = summarize(root/'screen', cells())
        save(root/'summary.json', report)
        with (root/'scores.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report['raw'][0]))
            writer.writeheader()
            writer.writerows(report['raw'])
        save(root/'completion.json', dict(experiments_complete=True, cells=126, source=source,
                                        completed_at=time.time()))
        package(root, '')
        print('ALL DONE: 126/126; both protocols summarized separately; return ZIP ready.', flush=True)


if __name__ == '__main__':
    main()
