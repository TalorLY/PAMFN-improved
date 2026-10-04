"""Orchestration only: invoke byte-preserved v17 entry points in new processes.

Never imports/reimplements the training loop, modifies the trainer checkout,
uses historical predictions as new results, or deletes checkpoints.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

COMMIT = '1437dd4b9ce3fc220bdf530946a83d4bac88c4f7'
HISTORY_SHA = 'd460aab030de69585896b1e8a1081d81f57ae60eed9c6ffe11e9c23dcb072e52'
ACTIONS = ('Ball', 'Clubs', 'Hoop', 'Ribbon', 'TES', 'PCS')
METHODS = ('base', 'ordered', 'ordered_modal_fused_gt', 'e_modal_scaled')
TRAINER_TESTS = ('tests.test_qmd_supp', 'tests.test_qmd_modal',
                 'tests.test_qmd_residual', 'tests.test_july28_qmd')
BUDGET = dict(Ball=500, Clubs=100, Hoop=300, Ribbon=20, TES=100, PCS=100)
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def immutable(path, value):
    if Path(path).exists() and read(path) != value:
        raise ValueError('run contract changed; use a new root: ' + str(path))
    save(path, value)


def history(path):
    if sha(path) != HISTORY_SHA: raise ValueError('not the verified original v17 return ZIP')
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None: raise ValueError('history ZIP damaged')
        def get(name): return json.loads(z.read(name))
        templates, old = {}, []
        for action in ACTIONS:
            for seed in range(3):
                for method in METHODS:
                    prefix = 'reference' if method in METHODS[:2] else ('e_reference' if method == METHODS[2] else 'screen')
                    base = '%s/july28/%s/seed%d/%s/' % (prefix, action, seed, method)
                    record = get(base + 'complete.json')
                    cfg = get(base + 'run_config.json')
                    if sha_bytes(z.read(base+'run_config.json')) != record['files']['run_config.json']:
                        raise ValueError('historical command checksum mismatch')
                    key = '%s/%d/%s' % (action, seed, method)
                    templates[key] = cfg['argv']
                    old.append(dict(action=action, seed=seed, method=method,
                                    srcc_x100=record['metrics']['srcc_x100']))
        return dict(source=get('source.json'), templates=templates, old=old)


def sha_bytes(value): return hashlib.sha256(value).hexdigest()


def cell(root, stage, action, seed, method):
    return Path(root) / stage / 'july28' / action / ('seed%d' % seed) / method


def active(directory, verify=True):
    marker = directory / 'active.json'
    if not marker.exists(): return None
    rel = read(marker)['attempt']
    if not re.fullmatch(r'attempts/\d{4}', rel): raise ValueError('unsafe active attempt')
    path = directory / rel
    if not (path/'complete.json').exists(): return None
    if verify:
        rec = read(path/'complete.json')
        for name, digest in rec['files'].items():
            if Path(name).name != name or sha(path/name) != digest:
                raise ValueError('completed asset changed: ' + str(path/name))
        if sha(path/'model.pth') != rec['checkpoint_sha256'] or sha(path/'model.pth.eval_rng.pt') != rec['eval_rng_sha256']:
            raise ValueError('completed checkpoint/RNG changed: '+str(path))
    return path


def arguments(hist, root, stage, action, seed, method, gpu):
    # Use EACH recorded historical argv, changing only dependency paths/device/output.
    argv = list(hist['templates']['%s/%d/%s' % (action, seed, method)])
    replacements = {}
    if method in METHODS[2:]:
        ordered = active(cell(root, stage, action, seed, 'ordered'))
        if ordered is None: raise ValueError('new Ordered is not complete')
        replacements.update(teacher_model=str(ordered/'model.pth'),
                            teacher_rng=str(ordered/'model.pth.eval_rng.pt'),
                            teacher_record=str(ordered/'complete.json'))
    if method == METHODS[3]:
        e = active(cell(root, stage, action, seed, METHODS[2]))
        if e is None: raise ValueError('new E is not complete')
        replacements.update(e_model=str(e/'model.pth'), e_record=str(e/'complete.json'))
    for key, value in replacements.items(): argv[argv.index('--'+key)+1] = value
    if stage == 'smoke':
        argv += ['--epochs', '2']
        if method in METHODS[2:]: argv += ['--smoke']
    entry = 'main.py' if method in METHODS[:2] else ('tools/train_qmd_modal.py' if method == METHODS[2] else 'tools/train_qmd_supp.py')
    return [sys.executable, '-u', entry, '--gpu', str(gpu)] + argv


def proc_identity(pid=None):
    pid = os.getpid() if pid is None else pid
    try:
        fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z': return dict(pid=pid,start=None,boot=None)
        return dict(pid=pid, start=fields[19], boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except OSError: return dict(pid=pid, start=None, boot=None)


def alive(record):
    ident = record.get('identity', record)
    if ident.get('start') is None: return False
    current = proc_identity(ident['pid'])
    return current == ident


def environment(repo):
    env = dict(os.environ, PYTHONPATH=str(repo), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('CUDA_VISIBLE_DEVICES', None)  # entry point selects the physical GPU itself
    return env


def trainer_unit_argv():
    return [sys.executable, '-u', '-m', 'unittest'] + list(TRAINER_TESTS)


def command(argv, repo, directory, name, visible_gpu=None):
    with (directory/(name+'.log')).open('w', encoding='utf-8') as log:
        log.write('COMMAND '+json.dumps(argv)+'\n'); log.flush()
        env=environment(repo)
        if visible_gpu is not None: env['CUDA_VISIBLE_DEVICES']=str(visible_gpu)
        p = subprocess.Popen(argv, cwd=str(repo), env=env, stdout=log, stderr=subprocess.STDOUT)
        rec = dict(identity=proc_identity(p.pid), state='running', argv=argv, started=time.time())
        save(directory/(name+'.process.json'), rec)
        try: code = p.wait()
        except BaseException:
            p.terminate(); p.wait(); raise
        rec.update(state='finished' if code == 0 else 'failed', returncode=code, ended=time.time())
        save(directory/(name+'.process.json'), rec)
        if code != 0: raise RuntimeError('%s exited %d: %s' % (name, code, directory/(name+'.log')))


def validate(directory, action, seed, method, stage):
    import numpy as np
    from scipy.stats import spearmanr
    train, test, reload = [read(directory/n) for n in ('train_result.json', 'test_result.json', 'reload_result.json')]
    # Original PAMFN includes wall-clock timing in test output, not in predictions.
    if {k:v for k,v in test.items() if k!='time'} != {k:v for k,v in reload.items() if k!='time'}:
        raise ValueError('two saved-RNG evaluations differ')
    epochs = 2 if stage == 'smoke' else BUDGET[action]
    for obj in (train, test):
        run = obj['run']
        for k, v in dict(action=action, seed=seed, tag=method, feats=2, epochs=epochs).items():
            if run[k] != v: raise ValueError('wrong run metadata: '+k)
    score = float(test['coef'])
    if not math.isfinite(score) or abs(score-train['saved_model_srcc']*100) > 1e-6:
        raise ValueError('checkpoint does not replay the selected score')
    p = test['predictions']; y, pred = np.asarray(p['label']), np.asarray(p['prediction'])
    n = 100 if action in ('TES', 'PCS') else 50
    if y.shape != (n,) or pred.shape != (n,) or not np.isfinite(pred).all(): raise ValueError('invalid predictions')
    if abs(float(spearmanr(y, pred)[0])*100-score) > 1e-6: raise ValueError('prediction SRCC mismatch')
    rows = [json.loads(s) for s in (directory/'model.pth.epochs.jsonl').read_text().splitlines()]
    if [r['epoch'] for r in rows] != list(range(1, epochs+1)): raise ValueError('incomplete/duplicated epochs')
    selection = read(directory/'model.pth.selection.json')
    winner = max(rows, key=lambda r:r['coef'])  # first maximum; original strict > update
    if selection['epoch'] != winner['epoch'] or abs(winner['coef']-score) > 1e-6:
        raise ValueError('not historical trained-epoch test-best selection')
    if method in METHODS[:2]:
        run = test['run']
        if run['eval_policy'] != 'legacy_gumbel' or run['qmd']['mode'] != 'off': raise ValueError('main scorer recipe changed')
        if run['ordinal_prototype_weight'] != (0. if method == 'base' else .0001): raise ValueError('Ordered strength changed')
    else:
        frozen = read(directory/'qmd_training.json')
        if method == METHODS[2]:
            if not frozen['anchor_unchanged'] or frozen['anchor_state_before'] != frozen['anchor_state_after']:
                raise ValueError('Ordered changed during E training')
        elif not frozen['frozen_unchanged'] or frozen['frozen_before'] != frozen['frozen_after']:
            raise ValueError('Ordered/E changed during supplement training')
    return dict(protocol='july28', action=action, seed=seed, method=method,
                srcc_x100=score, selected_epoch=selection['epoch'], epochs=epochs)


def run_cell(hist, root, repo, stage, action, seed, method, gpu):
    directory = cell(root, stage, action, seed, method)
    if active(directory):
        print('KEEP %s/%s/seed%d/%s' % (stage, action, seed, method), flush=True); return
    directory.mkdir(parents=True, exist_ok=True)
    prior = sorted((directory/'attempts').glob('[0-9][0-9][0-9][0-9]')) if (directory/'attempts').exists() else []
    for attempt in prior:
        for record in attempt.glob('*.process.json'):
            if alive(read(record)): raise RuntimeError('child still running; do not duplicate: '+str(record))
    index = max([int(p.name) for p in prior]+[0])+1
    out = directory/'attempts'/('%04d'%index); out.mkdir(parents=True)
    save(directory/'active.json', dict(attempt='attempts/%04d'%index))
    argv = arguments(hist, root, stage, action, seed, method, gpu)
    save(out/'run_config.json', dict(argv=argv[5:], protocol='july28', stage=stage))
    print('START GPU=%s %s/%s/seed%d/%s attempt=%d' % (gpu,stage,action,seed,method,index), flush=True)
    command(argv+['--save_final_model',str(out/'model.pth'),'--result_json',str(out/'train_result.json')], repo, out, 'train')
    eval_args = argv+['--test','--ckpt_path',str(out/'model.pth'),'--replay_rng_path',str(out/'model.pth.eval_rng.pt')]
    if method in METHODS[:2] and '--save_test_predictions' not in eval_args: eval_args += ['--save_test_predictions']
    for name in ('test_result', 'reload_result'):
        command(eval_args+['--result_json',str(out/(name+'.json'))], repo, out, name)
    metrics = validate(out, action, seed, method, stage)
    if method in METHODS[2:]:
        check = read(out/'teacher_check.json')
        previous = 'ordered' if method == METHODS[2] else METHODS[2]
        expected = read(active(cell(root,stage,action,seed,previous))/'test_result.json')
        actual = check['baseline']
        if actual['predictions'] != expected['predictions']: raise ValueError('frozen teacher predictions changed')
    files = {p.name:sha(p) for p in out.iterdir() if p.suffix in ('.json','.jsonl') and not p.name.endswith('.process.json')}
    record = dict(metrics=metrics, files=files, checkpoint_sha256=sha(out/'model.pth'),
                  eval_rng_sha256=sha(out/'model.pth.eval_rng.pt'), completed=time.time())
    save(out/'complete.json', record)
    print('FINISH %s/%s/seed%d/%s SRCC=%.6f' % (stage,action,seed,method,metrics['srcc_x100']), flush=True)


def wave(hist, root, repo, stage, gpus):
    jobs = queue.Queue()
    pairs = [('Ball',0),('TES',0)] if stage == 'smoke' else [(a,s) for a in ACTIONS for s in range(3)]
    for pair in pairs: jobs.put(pair)
    errors = queue.Queue()
    def worker(gpu):
        while True:
            try: action, seed = jobs.get_nowait()
            except queue.Empty: return
            try:
                for method in METHODS: run_cell(hist,root,repo,stage,action,seed,method,gpu)
            except Exception as exc:
                errors.put(dict(action=action,seed=seed,error=repr(exc)))
                print('FAILED %s/seed%d: %r'%(action,seed,exc),flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, gpus))
    failures = []
    while not errors.empty(): failures.append(errors.get())
    if failures:
        save(root/(stage+'_failures.json'), failures)
        raise RuntimeError('%d chains failed; completed cells retained; inspect %s_failures.json'%(len(failures),stage))


def git(repo, *argv):
    return subprocess.check_output(['git','-C',str(repo)]+list(argv), text=True).strip()


def preflight(hist, root, repo, gpus):
    if git(repo,'rev-parse','HEAD') != COMMIT: raise ValueError('trainer must be original v17 commit')
    if git(repo,'status','--porcelain','--untracked-files=no'): raise ValueError('tracked v17 source changed')
    # Detect injected Python modules without treating data/weights as source files.
    extras = git(repo,'ls-files','--others','--exclude-standard').splitlines()
    if any(p.endswith('.py') for p in extras): raise ValueError('untracked Python code in trainer checkout')
    probe = "import json,sys,torch,numpy,scipy,timm;print(json.dumps(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,numpy=numpy.__version__,scipy=scipy.__version__,timm=timm.__version__)))"
    actual = json.loads(subprocess.check_output([sys.executable,'-c',probe],cwd=str(repo),env=environment(repo),text=True).splitlines()[-1])
    if actual != hist['source']['environment']: raise ValueError('environment differs from original v17: '+json.dumps(actual))
    for i,(name,expected) in enumerate(hist['source']['inputs'].items(),1):
        path = repo/name
        if not path.is_file() or path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise ValueError('historical input missing/changed: '+str(path))
        print('INPUT %d/%d PASS %s'%(i,len(hist['source']['inputs']),name),flush=True)
    free = shutil.disk_usage(root).free
    if free < 10*1024**3: raise RuntimeError('need >=10 GiB free for models plus an automatic asset backup; no automatic deletion')
    current = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,name','--format=csv,noheader,nounits'],text=True)
    inventory = {r[0].strip():dict(free_mib=int(r[1]),name=r[2].strip()) for r in csv.reader(io.StringIO(current))}
    if gpus == ['auto']: gpus = [g for g,v in inventory.items() if g in ('0','1','2','3','4','5') and v['free_mib']>=12000]
    if not gpus or len(set(gpus)) != len(gpus) or any(g not in inventory or inventory[g]['free_mib']<12000 for g in gpus):
        raise RuntimeError('selected GPUs need >=12000 MiB free: '+json.dumps(inventory))
    save(root/'preflight.json',dict(commit=COMMIT,environment=actual,inputs=hist['source']['inputs'],
                                   gpu_inventory=inventory,gpus=gpus,free_bytes=free,checked_at=time.time()))
    return gpus


def summarize(hist, root):
    raw = []
    for action in ACTIONS:
        for seed in range(3):
            for method in METHODS:
                out = active(cell(root,'screen',action,seed,method))
                if out is None: raise ValueError('cannot summarize unfinished experiment')
                raw.append(read(out/'complete.json')['metrics'])
    def means(rows):
        values = {m:sum(r['srcc_x100'] for r in rows if r['method']==m)/len([r for r in rows if r['method']==m]) for m in METHODS}
        return [dict(method=m,srcc_x100=v,delta_base=v-values['base'],delta_ordered=v-values['ordered']) for m,v in values.items()]
    summary = dict(raw=raw,overall=means(raw),per_seed={str(s):means([r for r in raw if r['seed']==s]) for s in range(3)},
                   per_action={a:means([r for r in raw if r['action']==a]) for a in ACTIONS},historical_reference=means(hist['old']),
                   scope='six-task arithmetic mean; all three seeds; new training, not historical checkpoint recovery',
                   selection='original trained-epoch test-best, NOT validation-selected')
    save(root/'summary.json',summary)
    fields = ['action','seed','method','srcc_x100','selected_epoch','epochs']
    with (root/'scores.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(raw)
    return summary


def write_return(root):
    return_path = root.parent/(root.name+'_return.zip')
    with zipfile.ZipFile(str(return_path)+'.tmp','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.csv') and p.name!='archive.json':
                z.write(p,str(p.relative_to(root)).replace('\\','/'))
        for name in ('run.py','test_runner.py','setup.sh','start.sh','README.md','qmd_supp_v17_full.bundle','v17_history.zip'):
            if (HERE/name).is_file(): z.write(HERE/name,'rebuild_code/'+name)
    with zipfile.ZipFile(str(return_path)+'.tmp') as z:
        if z.testzip() is not None: raise ValueError('return ZIP failed CRC')
    Path(str(return_path)+'.tmp').replace(return_path)
    return dict(path=str(return_path),sha256=sha(return_path))


def package(root):
    # Two separate artifacts: small analysis return, plus COMPLETE active model assets.
    assets_path = root.parent/(root.name+'_assets.tar.gz')
    selected = []
    for name in ('contract.json','preflight.json','summary.json','completion.json','smoke_gate.json'):
        if (root/name).is_file():selected.append((root/name,sha(root/name)))
    for stage,pairs in [('smoke',[('Ball',0),('TES',0)]),('screen',[(a,s) for a in ACTIONS for s in range(3)])]:
        for action,seed in pairs:
            for method in METHODS:
                out=active(cell(root,stage,action,seed,method))
                if out is None: raise ValueError('cannot back up unfinished cells')
                selected.extend((p,sha(p)) for p in out.iterdir() if p.is_file() and not p.name.endswith('.tmp'))
                marker=out.parent.parent/'active.json';selected.append((marker,sha(marker)))
    receipt=dict(files={str(p.relative_to(root)).replace('\\','/'):h for p,h in selected},
                 note='all 80 active smoke/formal model and RNG pairs; no historical dependencies')
    save(root/'asset_manifest.json',receipt)
    with tarfile.open(str(assets_path)+'.tmp','w:gz',compresslevel=1) as tar:
        for p,_ in selected: tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
        tar.add(root/'asset_manifest.json',arcname='asset_manifest.json')
    with tarfile.open(str(assets_path)+'.tmp','r:gz') as tar:
        for member in tar:
            if member.name=='asset_manifest.json': continue
            h=hashlib.sha256();f=tar.extractfile(member)
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            if h.hexdigest()!=receipt['files'][member.name]:raise ValueError('asset backup verification failed')
    Path(str(assets_path)+'.tmp').replace(assets_path)
    save(root/'archive.json',dict(ready=True,return_zip=write_return(root),
                                 assets=dict(path=str(assets_path),sha256=sha(assets_path))))
    print('BACKUP VERIFIED '+str(assets_path),flush=True)


def status(root):
    runner=read(root/'runner.json') if (root/'runner.json').exists() else {}
    print('Runner: %s ; alive: %s ; phase: %s'%(runner.get('state','not started'),alive(runner),runner.get('phase','-')))
    for stage,pairs in [('smoke',[('Ball',0),('TES',0)]),('screen',[(a,s) for a in ACTIONS for s in range(3)])]:
        counts={m:0 for m in METHODS};pending=[]
        for a,s in pairs:
            for m in METHODS:
                d=cell(root,stage,a,s,m)
                if active(d,verify=False):counts[m]+=1
                elif (d/'active.json').exists():pending.append((a,s,m,d/read(d/'active.json')['attempt']))
        print('%s completed: %d/%d ; %s'%(stage,sum(counts.values()),len(pairs)*4,counts))
        for a,s,m,d in pending:
            for record in d.glob('*.process.json'):
                r=read(record)
                if r['state']=='finished':continue
                log=d/(record.name.replace('.process.json','.log'))
                with log.open('rb') as f:
                    f.seek(max(0,log.stat().st_size-32768));text=f.read().decode(errors='replace')
                epochs=re.findall(r'Epoch\[(\d+)/(\d+)\]',text)
                print('%s %s/%s/seed%d/%s %s epoch=%s log_age=%ds'%(
                    'RUNNING' if alive(r) else 'FAILED/INTERRUPTED',stage,a,s,m,record.stem,
                    '/'.join(epochs[-1]) if epochs else '-',time.time()-log.stat().st_mtime))
    if runner.get('error'):print('ERROR:',runner['error'])
    print('Experiments complete:',(root/'completion.json').exists(),'; Archives verified:',(root/'archive.json').exists())
    print('Resume SAME command/root. No completed cells deleted. Incomplete cells restart in new attempts.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('all','check','smoke','status','package'),default='all')
    p.add_argument('--root',type=Path,default=HERE/'results/v17_rebuild_v29')
    p.add_argument('--repo',type=Path,default=HERE/'trainer')
    p.add_argument('--history',type=Path,default=HERE/'v17_history.zip')
    p.add_argument('--gpus',default='auto')
    args=p.parse_args();root=args.root.resolve();repo=args.repo.resolve()
    if args.stage=='status':status(root);return
    if root==repo or repo in root.parents or root in repo.parents:raise ValueError('results and original trainer must be separate')
    root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (root/'.runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for f in root.rglob('*.process.json'):
            if alive(read(f)):raise RuntimeError('existing live child: '+str(f))
        runner=dict(identity=proc_identity(),state='running',phase='preflight',started=time.time());save(root/'runner.json',runner)
        try:
            hist=history(args.history)
            contract=dict(code_commit=COMMIT,source=hist['source'],templates=hist['templates'],runner_sha256=sha(Path(__file__)),
                          methods=METHODS,actions=ACTIONS,seeds=[0,1,2],budget=BUDGET,formal_cells=72,
                          selection='trained-epoch test-best, saved legacy stochastic RNG; NO validation split',
                          base='new training',ordered='new training',e='new training on new Ordered',supplement='new training on new E')
            immutable(root/'contract.json',json.loads(json.dumps(contract)))
            if args.stage=='package':
                summarize(hist,root);package(root)
            else:
                gpus=preflight(hist,root,repo,args.gpus.split())
                runner.update(phase='unit_checks');save(root/'runner.json',runner)
                checks=root/'checks';checks.mkdir(exist_ok=True)
                if not (checks/'unit_pass.json').exists():
                    command([sys.executable,'-u','-m','unittest','discover','-s',str(HERE),'-p','test_runner.py'],
                            repo,checks,'runner_unit',visible_gpu=gpus[0])
                    command(trainer_unit_argv(),repo,checks,'unit',visible_gpu=gpus[0])
                    save(checks/'unit_pass.json',dict(passed=True,commit=COMMIT,log_sha256=sha(checks/'unit.log')))
                if args.stage!='check':
                    runner.update(phase='smoke');save(root/'runner.json',runner)
                    wave(hist,root,repo,'smoke',gpus[:2]);save(root/'smoke_gate.json',dict(passed=True,cells=8))
                if args.stage=='all':
                    runner.update(phase='screen');save(root/'runner.json',runner)
                    wave(hist,root,repo,'screen',gpus);summarize(hist,root)
                    save(root/'completion.json',dict(experiments_complete=True,formal_cells=72,completed=time.time()))
                    runner.update(phase='packaging');save(root/'runner.json',runner);package(root)
            runner.update(state='finished',phase='checked' if args.stage=='check' else 'done',ended=time.time());save(root/'runner.json',runner)
            if (root/'archive.json').exists():
                receipt=read(root/'archive.json');receipt['return_zip']=write_return(root);save(root/'archive.json',receipt)
            print('V17_REBUILD_DONE '+args.stage,flush=True)
        except Exception as exc:
            runner.update(state='failed',error=repr(exc),ended=time.time());save(root/'runner.json',runner);raise


if __name__=='__main__':main()
