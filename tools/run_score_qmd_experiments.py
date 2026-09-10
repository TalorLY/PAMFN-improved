"""Bounded smoke/screen runner with checked result reuse and compact returns."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.summarize_score_qmd import (ACTIONS, METHODS, SEEDS, budget, read_json,
                                       summarize, validate_cell)
from tools.verify_qmd_runtime import BASE_COMMIT

REPO = Path(__file__).resolve().parents[1]


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8')
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def command(argv, log):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open('a', encoding='utf-8') as handle:
        handle.write('\nCOMMAND ' + json.dumps(argv) + '\n')
        handle.flush()
        subprocess.run(argv, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)


def import_baselines(source, root):
    if (source/'git_commit.txt').read_text().strip() != BASE_COMMIT:
        raise ValueError('Base results must come from the frozen all-action formal experiment')
    identities = []
    pending = []
    for a in ACTIONS:
        for s in SEEDS:
            path = source/a/f'seed{s}'/'base'
            metrics = validate_cell(path, a, s, 'base', budget(a))
            dest = root/'baseline_reference'/a/f'seed{s}'
            pending.append((dest/'metrics.json', metrics))
            for name in ('train_result.json', 'test_result.json'):
                # Distinct filenames keep progress count limited to the 36 new runs.
                pending.append((dest/('source_'+name), read_json(path/name)))
                identities.append(dict(action=a, seed=s, file=name, sha256=sha(path/name)))
    previous = root/'baseline_provenance.json'
    if previous.exists() and read_json(previous) != identities:
        raise ValueError('historical baseline files changed')
    for path, data in pending:
        save(path, data)
    return identities


def run_cell(root, action, seed, mode, gpu, smoke, resume):
    out = root/action/f'seed{seed}'/mode
    out.mkdir(parents=True, exist_ok=True)
    epochs = 1 if smoke else budget(action)
    argv = [sys.executable, '-u', 'main.py', '--gpu', gpu, '--feats', '2',
            '--action', action, '--seed', str(seed), '--multi_modality',
            '--sequence_sampling', 'legacy', '--eval_policy', 'argmax',
            '--run_tag', mode, '--qmd_mode', mode,
            '--qmd_mask_ratio', '0.15', '--qmd_gt_weight', '0.5',
            '--qmd_target_ratio', '0.075', '--qmd_beta_cap', '1.0']
    if smoke:
        argv += ['--epochs', '1', '--qmd_warmup', '0', '--qmd_rampup', '0']
    # Device is scheduling, not training configuration. It may change on resume.
    contract = dict(command=argv[5:], epochs=epochs)
    cfg_path = out/'run_config.json'
    if cfg_path.exists() and read_json(cfg_path) != contract:
        raise ValueError('run config differs; use a new output directory: ' + str(out))
    save(cfg_path, contract)
    if (out/'complete.json').exists():
        validate_cell(out, action, seed, mode, epochs)
        if not (out/'model.pth').is_file() or (out/'model.pth').stat().st_size == 0:
            raise ValueError('completed run has lost its checkpoint: ' + str(out))
        if not resume:
            raise ValueError('completed run exists; enable RESUME=1 to skip')
        return 'SKIP'
    trained = all((out/n).is_file() for n in ('model.pth', 'train_result.json', 'qmd_training.json'))
    if not trained:
        command(argv + ['--save_final_model', str(out/'model.pth'),
                        '--result_json', str(out/'train_result.json')], out/'train.log')
    command(argv + ['--test', '--ckpt_path', str(out/'model.pth'),
                    '--result_json', str(out/'test_result.json')], out/'test.log')
    metrics = validate_cell(out, action, seed, mode, epochs)
    save(out/'complete.json', dict(metrics=metrics, finished_at=time.time()))
    return 'FINISH'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('smoke', 'screen'), required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--smoke_root', default='results/score_qmd_v1_smoke')
    parser.add_argument('--base_root', default='../PAMFN-coqm-all-action-screen-v1/results/coqm_all_action_formal_v1')
    parser.add_argument('--resume', type=int, choices=(0,1), default=1)
    args = parser.parse_args()
    os.chdir(REPO)
    gpus = args.gpus.split()
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError('provide distinct currently available GPUs')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True)
    if dirty:
        raise ValueError('commit tracked changes before server experiments')
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    smoke = args.stage == 'smoke'
    smoke_root = Path(args.smoke_root).resolve()
    base_root = Path(args.base_root).resolve()
    manifest = dict(commit=commit, base_commit=BASE_COMMIT, stage=args.stage,
                    actions=['Ball','TES'] if smoke else list(ACTIONS),
                    seeds=[0] if smoke else list(SEEDS), methods=list(METHODS),
                    base_source=None if smoke else str(base_root), version=1,
                    mask_ratio=0.15, mask_gt_weight=0.5, target_ratio=0.075, beta_cap=1.0,
                    warmup=0.0 if smoke else 0.1, rampup=0.0 if smoke else 0.1,
                    evaluation='legacy-test-best checkpoint + argmax',
                    resume_semantics='skip complete cells; restart unfinished training cells')
    manifest_path = root/'manifest.json'
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError('output provenance differs; use a new root')
    save(manifest_path, manifest)
    (root/'git_commit.txt').write_text(commit+'\n')
    (root/'git_branch.txt').write_text(subprocess.check_output(
        ['git', 'branch', '--show-current'], text=True), encoding='utf-8')
    environment = subprocess.check_output([sys.executable, '-c',
        'import sys,json,torch,numpy,scipy; print(json.dumps(dict(python=sys.version,'
        'torch=torch.__version__,cuda=torch.version.cuda,numpy=numpy.__version__,'
        'scipy=scipy.__version__)))'], text=True)
    save(root/'environment.json', json.loads(environment))
    if smoke:
        print('SMOKE phase=unit_tests', flush=True)
        command([sys.executable, '-m', 'unittest', 'tests.test_score_qmd',
                 'tests.test_score_qmd_tools', 'tests.test_coqm_eval_b0_tools', '-v'], root/'unit_tests.log')
        print('SMOKE phase=real_data_frozen_base_first_epoch', flush=True)
        command([sys.executable, '-u', 'tools/verify_qmd_runtime.py', '--gpu', gpus[0],
                 '--output', str(root/'report/runtime_regression.json')], root/'runtime_regression.log')
    else:
        gate = read_json(smoke_root/'report/smoke_gate.json')
        if gate['decision'] != 'SCORE_QMD_SMOKE_PASS' or gate['commit'] != commit:
            raise ValueError('this commit needs a passed server smoke first')
        regression = read_json(smoke_root/'report/runtime_regression.json')
        if not regression['full_epoch'] or regression['decision'] != 'QMD_BASE_REGRESSION_PASS':
            raise ValueError('baseline reuse requires full first-epoch parity')
        identities = import_baselines(base_root, root)
        save(root/'baseline_provenance.json', identities)
        save(root/'report/smoke_gate.json', gate)
        save(root/'report/runtime_regression.json', regression)
    cells = [(a,s) for a in manifest['actions'] for s in manifest['seeds']]
    def worker(index, gpu):
        log = root/'worker_launchers'/('worker%d_gpu%s.log' % (index,gpu))
        log.parent.mkdir(exist_ok=True)
        with log.open('a', encoding='utf-8', buffering=1) as handle:
            for cell_index, (action,seed) in enumerate(cells):
                if cell_index % len(gpus) != index:
                    continue
                for mode in METHODS:
                    ident = 'action=%s seed=%s mode=%s gpu=%s' % (action,seed,mode,gpu)
                    print('START '+ident, file=handle)
                    try:
                        status = run_cell(root, action, seed, mode, gpu, smoke, args.resume)
                    except Exception as error:
                        print('FAILED '+ident+' '+repr(error), file=handle)
                        raise
                    print(status+' '+ident, file=handle)
    errors = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, index, gpu) for index,gpu in enumerate(gpus)]
        print('LAUNCHED stage=%s cells=%d gpus=%s' % (args.stage,len(cells)*2,gpus), flush=True)
        for future in futures:
            try:
                future.result()
            except Exception as error:
                errors.append(repr(error))
    if errors:
        save(root/'report/incomplete.json', {'errors': errors})
        raise RuntimeError('SCORE_QMD_INCOMPLETE; inspect worker logs: '+str(errors))
    if smoke:
        regression = read_json(root/'report/runtime_regression.json')
        if regression['decision'] != 'QMD_BASE_REGRESSION_PASS' or not regression['full_epoch']:
            raise ValueError('first-epoch Base parity failed')
        save(root/'report/smoke_gate.json', dict(decision='SCORE_QMD_SMOKE_PASS', commit=commit,
                                                expected_new_runs=4, full_base_regression=True))
    else:
        summarize(root)
    save(root/'report/completion.json', dict(stage=args.stage, runs=len(cells)*2,
                                            experiments_complete=True, commit=commit))
    print('EXPERIMENTS_COMPLETE; packaging', flush=True)
    subprocess.run([sys.executable, 'tools/package_coqm_results.py', '--root', str(root),
                    '--output', str(root)+'_return.zip'], check=True)
    print('SCORE_QMD_%s_COMPLETE archive=%s_return.zip' % (args.stage.upper(),root), flush=True)


if __name__ == '__main__':
    main()
