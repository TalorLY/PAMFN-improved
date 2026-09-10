"""Six-task SR-QMD + Ordered runner; seed-0 return first, then all 54 cells."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from tools.run_score_qmd_experiments import save, sha, command
from tools.summarize_score_qmd import validate_cell as validate_v1
from tools.summarize_sr_qmd import ACTIONS, SEEDS, METHODS, REFERENCES, budget, read_json, summarize, validate_cell
from tools.verify_qmd_runtime import BASE_COMMIT
from tools.verify_sr_qmd_runtime import SCORE_COMMIT

REPO = Path(__file__).resolve().parents[1]


@contextmanager
def experiment_lock(root):
    # The kernel releases this Linux lock even if the launcher is killed.
    import fcntl
    with (root/'.runner.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another launcher is already using this result root')
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def import_references(source, root):
    completion = read_json(source/'report/completion.json')
    if (source/'git_commit.txt').read_text().strip() != SCORE_COMMIT or completion != dict(
            commit=SCORE_COMMIT, experiments_complete=True, runs=36, stage='screen'):
        raise ValueError('reference must be the completed frozen Score-QMD v1 screen')
    if read_json(source/'manifest.json')['base_commit'] != BASE_COMMIT:
        raise ValueError('unexpected Base provenance')
    pending, identities = [], []
    for a in ACTIONS:
        for s in SEEDS:
            for m in REFERENCES:
                src = source/'baseline_reference'/a/f'seed{s}' if m=='base' else source/a/f'seed{s}'/m
                dest = root/'reference'/a/f'seed{s}'/m
                names = ['train_result.json','test_result.json'] + ([] if m=='base' else ['qmd_training.json'])
                for name in names:
                    original = src/(('source_' if m=='base' else '')+name)
                    pending.append((original,dest/name))
                    identities.append(dict(action=a,seed=s,method=m,file=name,sha256=sha(original)))
    previous = root/'reference_provenance.json'
    if previous.exists() and read_json(previous) != identities:
        raise ValueError('reference results changed; use a new output root')
    for src, dest in pending:
        dest.parent.mkdir(parents=True,exist_ok=True)
        content = src.read_bytes()
        if dest.exists() and dest.read_bytes() != content:
            raise ValueError('existing reference snapshot differs: '+str(dest))
        dest.write_bytes(content)
    for a in ACTIONS:
        for s in SEEDS:
            for m in REFERENCES:
                dest = root/'reference'/a/f'seed{s}'/m
                save(dest/'metrics.json',validate_v1(dest,a,s,m,budget(a)))
    save(previous, identities)


def cell_command(action, seed, method, gpu, smoke):
    ordered = method in ('ordered','ordered_sr_qmd')
    argv = [sys.executable,'-u','main.py','--gpu',str(gpu),'--feats','2','--action',action,
            '--seed',str(seed),'--multi_modality','--sequence_sampling','legacy','--eval_policy','argmax',
            '--eval_seed','20260908','--run_tag',method,
            '--qmd_mode','off' if method=='ordered' else 'score_rank_qmd',
            '--qmd_mask_ratio','0.15','--qmd_gt_weight','0.5','--qmd_target_ratio','0.075',
            '--qmd_beta_cap','1.0','--qmd_rank_weight','0.5','--qmd_rank_gap','0.05','--qmd_rank_gamma','0.1']
    if ordered:
        argv += ['--ordinal_prototype_weight','0.0001','--ordinal_prototype_target_mode','continuous',
                 '--quality_aux_rng_isolated','--ordinal_prototype_dim','128',
                 '--ordinal_prototype_temperature','0.1','--ordinal_prototype_order_margin','0.1',
                 '--ordinal_prototype_order_scale','0.1']
    if smoke:
        argv += ['--epochs','1','--qmd_warmup','0','--qmd_rampup','0']
    return argv


def run_cell(root, action, seed, method, gpu, smoke, resume):
    out = root/action/f'seed{seed}'/method
    out.mkdir(parents=True,exist_ok=True)
    argv = cell_command(action,seed,method,gpu,smoke)
    contract = dict(command=argv[5:],epochs=1 if smoke else budget(action))
    config = out/'run_config.json'
    if config.exists() and read_json(config) != contract:
        raise ValueError('run config differs: '+str(out))
    save(config,contract)
    if (out/'complete.json').exists():
        metrics = validate_cell(out,action,seed,method,smoke)
        if metrics != read_json(out/'complete.json')['metrics']:
            raise ValueError('completed result changed')
        if not (out/'model.pth').is_file() or not (out/'model.pth').stat().st_size:
            raise ValueError('completed run has lost its checkpoint')
        if not resume:
            raise ValueError('completed run exists; enable RESUME=1')
        return 'SKIP'
    artifacts = ['model.pth','train_result.json'] + ([] if method=='ordered' else ['qmd_training.json'])
    trained = all((out/n).is_file() and (out/n).stat().st_size for n in artifacts)
    if not trained:
        command(argv+['--save_final_model',str(out/'model.pth'),'--result_json',str(out/'train_result.json')],out/'train.log')
    command(argv+['--test','--ckpt_path',str(out/'model.pth'),'--result_json',str(out/'test_result.json')],out/'test.log')
    metrics = validate_cell(out,action,seed,method,smoke)
    save(out/'complete.json',dict(metrics=metrics,finished_at=time.time()))
    return 'FINISH'


def run_wave(root, actions, seeds, gpus, smoke, resume):
    # Longest tasks first; three methods for a task/seed stay on one GPU.
    cells = sorted([(a,s) for a in actions for s in seeds],key=lambda x: -budget(x[0]))
    assignments = [[] for _ in gpus]
    loads = [0]*len(gpus)
    for cell in cells:
        index = loads.index(min(loads))
        assignments[index].append(cell)
        loads[index] += budget(cell[0])
    def worker(index,gpu):
        log = root/'worker_launchers'/('worker%d_gpu%s.log'%(index,gpu))
        log.parent.mkdir(exist_ok=True)
        with log.open('a',encoding='utf-8',buffering=1) as handle:
            for a,s in assignments[index]:
                for m in METHODS:
                    ident = 'action=%s seed=%s method=%s gpu=%s'%(a,s,m,gpu)
                    print('START '+ident,file=handle)
                    try:
                        status = run_cell(root,a,s,m,gpu,smoke,resume)
                    except Exception as error:
                        print('FAILED '+ident+' '+repr(error),file=handle)
                        raise
                    print(status+' '+ident,file=handle)
                    print(status+' '+ident,flush=True)
    errors = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for future in as_completed([pool.submit(worker,i,g) for i,g in enumerate(gpus)]):
            try:
                future.result()
            except Exception as error:
                errors.append(repr(error))
                print('WORKER_FAILED '+repr(error),flush=True)
    if errors:
        save(root/'report/failure.json',dict(errors=errors,at=time.time(),seeds=list(seeds)))
        raise RuntimeError('SR_QMD_INCOMPLETE; inspect worker logs')


def package(root, suffix='_return.zip'):
    subprocess.run([sys.executable,'tools/package_coqm_results.py','--root',str(root),
                    '--output',str(root)+suffix],cwd=REPO,check=True)


def execute(args, root, gpus):
    commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True):
        raise ValueError('commit tracked source changes before experiments')
    smoke = args.stage=='smoke'
    source, smoke_root = Path(args.reference_root).resolve(), Path(args.smoke_root).resolve()
    manifest = dict(version=1,commit=commit,stage=args.stage,base_commit=BASE_COMMIT,score_commit=SCORE_COMMIT,
                    actions=['Ball','TES'] if smoke else list(ACTIONS),seeds=[0] if smoke else list(SEEDS),
                    methods=list(METHODS),reference_source=None if smoke else str(source),
                    ordered='continuous; weight=0.0001; RNG-isolated; full-view only',
                    rank='weight=0.5; gap=0.05; gamma=0.1; denominator=eligible_pair_count',
                    evaluation='legacy-test-best checkpoint + argmax',
                    resume='skip validated complete cells; reuse completed training for reload; restart unfinished training')
    if (root/'manifest.json').exists() and read_json(root/'manifest.json') != manifest:
        raise ValueError('output provenance differs; use a new root')
    save(root/'manifest.json',manifest)
    (root/'git_commit.txt').write_text(commit+'\n',encoding='utf-8')
    (root/'git_branch.txt').write_text(subprocess.check_output(['git','branch','--show-current'],text=True),encoding='utf-8')
    environment = subprocess.check_output([sys.executable,'-c',
        'import sys,json,torch,numpy,scipy; print(json.dumps(dict(python=sys.version,torch=torch.__version__,'
        'cuda=torch.version.cuda,numpy=numpy.__version__,scipy=scipy.__version__)))'],text=True)
    current_env = json.loads(environment)
    if any(g != '-1' for g in gpus):
        subprocess.run([sys.executable,'-c',
            'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'],check=True)
    if (root/'environment.json').exists() and read_json(root/'environment.json') != current_env:
        raise ValueError('runtime changed during resume; use a new root')
    save(root/'environment.json',current_env)
    if smoke:
        print('SMOKE unit tests',flush=True)
        command([sys.executable,'-m','unittest','tests.test_sr_qmd','tests.test_sr_qmd_tools',
                 'tests.test_score_qmd','tests.test_score_qmd_tools','tests.test_coqm_rng_isolation','-v'],root/'unit_tests.log')
        for script in ('verify_qmd_runtime','verify_sr_qmd_runtime'):
            print('SMOKE '+script,flush=True)
            command([sys.executable,'-u','tools/'+script+'.py','--gpu',gpus[0],
                     '--output',str(root/'report'/(script+'.json'))],root/(script+'.log'))
        run_wave(root,manifest['actions'],[0],gpus,True,args.resume)
        gate = dict(decision='SR_QMD_SMOKE_PASS',commit=commit,new_runs=6,full_epoch_regressions=True)
        save(root/'report/smoke_gate.json',gate)
    else:
        gate = read_json(smoke_root/'report/smoke_gate.json')
        if gate != dict(decision='SR_QMD_SMOKE_PASS',commit=commit,new_runs=6,full_epoch_regressions=True):
            raise ValueError('this commit needs a passed server smoke first')
        for script, decision in [('verify_qmd_runtime','QMD_BASE_REGRESSION_PASS'),
                                 ('verify_sr_qmd_runtime','SR_QMD_V1_PARITY_PASS')]:
            data = read_json(smoke_root/'report'/(script+'.json'))
            if data['decision'] != decision or not data['full_epoch']:
                raise ValueError('full-epoch reuse regression missing')
            save(root/'report'/(script+'.json'),data)
        if read_json(source/'environment.json') != current_env:
            raise ValueError('reference runtime differs; paired references must be rerun or reviewed')
        import_references(source,root)
        save(root/'report/smoke_gate.json',gate)
        run_wave(root,ACTIONS,[0],gpus,False,args.resume)
        summarize(root,(0,),'pilot_report')
        save(root/'pilot_report/completion.json',dict(pilot_complete=True,new_runs=18,final_complete=False,commit=commit))
        if not Path(str(root)+'_pilot_return.zip').exists():
            package(root,'_pilot_return.zip')
        print('SR_QMD_PILOT_COMPLETE new_runs=18; continuing seeds 1/42',flush=True)
        run_wave(root,ACTIONS,[1,42],gpus,False,args.resume)
        summarize(root)
    save(root/'report/completion.json',dict(experiments_complete=True,new_runs=6 if smoke else 54,commit=commit,stage=args.stage))
    if (root/'report/failure.json').exists():
        history=read_json(root/'report/failure.json')
        history['resolved_at']=time.time()
        save(root/'report/failure.json',history)
    package(root)
    print('SR_QMD_%s_COMPLETE archive=%s_return.zip'%(args.stage.upper(),root),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage',choices=('smoke','screen'),required=True)
    p.add_argument('--root',required=True)
    p.add_argument('--gpus',required=True)
    p.add_argument('--reference_root',default='../PAMFN-score-qmd-v1/results/score_qmd_v1_screen')
    p.add_argument('--smoke_root',default='results/sr_qmd_ordered_v1_smoke')
    p.add_argument('--resume',type=int,choices=(0,1),default=1)
    args = p.parse_args()
    os.chdir(REPO)
    gpus = args.gpus.split()
    if not gpus or len(gpus)!=len(set(gpus)) or any(not re.fullmatch(r'\d+|-1',g) for g in gpus):
        raise ValueError('provide distinct available GPU indices')
    root = Path(args.root).resolve()
    if root==REPO or REPO not in root.parents:
        raise ValueError('result root must be a child directory of this worktree')
    root.mkdir(parents=True,exist_ok=True)
    with experiment_lock(root):
        execute(args,root,gpus)


if __name__ == '__main__':
    main()
