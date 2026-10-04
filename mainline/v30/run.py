"""Read-only v17 references -> exact replay -> retraining gate -> matched evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import os
from pathlib import Path
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import zipfile
import contract as c
import json
import math
import tarfile


def identity(pid):
    try:
        values=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,host=socket.gethostname(),start_ticks=values[19],
                    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),state=values[0])
    except (OSError,IndexError): return None


def alive(record):
    if not record: return False
    current=identity(record['pid'])
    return bool(current and all(current[k]==record[k] for k in ('pid','host','start_ticks','boot')) and current['state'] not in ('Z','X'))


def active_dir(directory):
    path=Path(directory)/'active.json'
    if not path.is_file(): return None
    target=Path(c.read(path)['directory']).resolve()
    target.relative_to(Path(directory).resolve())
    return target


def valid_done(directory):
    out=active_dir(directory)
    if out is None or not (out/'complete.json').is_file(): return False
    record=c.read(out/'complete.json')
    for name,digest in record['files'].items():
        if c.sha(c.child(out,name))!=digest: raise ValueError('completed evidence changed: '+str(out/name))
    if record.get('checkpoint_sha256') and c.sha(out/'model.pth')!=record['checkpoint_sha256']:
        raise ValueError('completed checkpoint changed: '+str(out))
    if record.get('eval_rng_sha256') and c.sha(out/'model.pth.eval_rng.pt')!=record['eval_rng_sha256']:
        raise ValueError('completed RNG changed: '+str(out))
    return True


def check_space(root):
    free=shutil.disk_usage(root).free
    if free < 5*1024**3: raise OSError('less than 5 GiB free; no files were deleted')
    if hasattr(os,'statvfs'):
        stat=os.statvfs(root)
        if stat.f_files and stat.f_favail < 1000: raise OSError('less than 1000 free inodes; no files were deleted')


def run_cell(root,phase,cell,gpu):
    directory=c.cell_dir(root,phase,cell)
    if valid_done(directory):
        print('KEEP',phase,cell,flush=True); return
    previous=active_dir(directory)
    if previous and (previous/'process.json').exists() and alive(c.read(previous/'process.json').get('identity')):
        raise RuntimeError('existing worker is still alive: '+str(previous))
    check_space(root)
    # Never append a restarted run to old history or overwrite its checkpoint.
    attempts=directory/'attempts'; attempts.mkdir(parents=True,exist_ok=True)
    index=1
    while (attempts/('%03d'%index)).exists(): index+=1
    out=attempts/('%03d'%index); out.mkdir()
    c.save(directory/'active.json',dict(directory=str(out.resolve())))
    a,s,m=cell
    argv=[sys.executable,'-u',str(c.HERE/'worker.py'),'--gpu',gpu,'--manifest',str(root/'manifest.json'),
          '--action',a,'--seed',str(s),'--method',m,'--output',str(out.resolve())]
    if phase=='smoke': argv+=['--smoke']
    env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(c.HERE),str(c.REPO))),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    env.pop('CUDA_VISIBLE_DEVICES',None)
    print('START',phase,cell,'GPU='+gpu,flush=True)
    with (out/'train.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(argv,cwd=str(c.REPO),env=env,stdout=log,stderr=subprocess.STDOUT)
        record=dict(identity=identity(process.pid),pid=process.pid,gpu=gpu,argv=argv,state='running',started_at=time.time())
        c.save(out/'process.json',record)
        code=process.wait(); record.update(state='finished' if code==0 else 'failed',returncode=code,ended_at=time.time())
        c.save(out/'process.json',record)
    if code: raise RuntimeError('worker failed, see '+str(out/'train.log'))
    if m!='replay':
        with (out/'reload.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(argv+['--evaluate'],cwd=str(c.REPO),env=env,stdout=log,stderr=subprocess.STDOUT)
            record=dict(identity=identity(process.pid),pid=process.pid,gpu=gpu,argv=argv+['--evaluate'],state='running',started_at=time.time())
            c.save(out/'reload.process.json',record)
            code=process.wait();record.update(state='finished' if code==0 else 'failed',returncode=code,ended_at=time.time())
            c.save(out/'reload.process.json',record)
        if code:raise RuntimeError('reload failed: '+str(out/'reload.log'))
    result=c.read(out/'test_result.json')
    required=['test_result.json']
    if m=='replay': required+=['diagnostics.json','inference_draws.json','cost.json']
    else:
        required+=['train_result.json','teacher_check.json','training_statistics.json','qmd_training.json','reload_result.json',
                   'last_epoch_result.json','model.pth.selection.json','model.pth.epochs.jsonl']
        expected=2 if phase=='smoke' else c.BUDGET[a]
        history=[__import__('json').loads(x) for x in (out/'model.pth.epochs.jsonl').read_text().splitlines()]
        if [r['epoch'] for r in history]!=list(range(1,expected+1)): raise ValueError('incomplete epoch history')
        selected=c.read(out/'model.pth.selection.json')
        finite=[r for r in history if math.isfinite(r['coef'])]
        winner=max(finite,key=lambda r:r['coef'])
        if selected['epoch']!=winner['epoch'] or abs(selected['coef']-result['coef'])>1e-6:
            raise ValueError('not the first trained-epoch maximum checkpoint')
        training=c.read(out/'qmd_training.json')
        if not training['statistics_unchanged'] or not training.get('anchor_unchanged',training.get('frozen_unchanged',False)):
            raise ValueError('frozen main model/statistics changed')
        if training['epoch_zero_eligible'] or training['weighted_kd_sum']!=0:
            raise ValueError('unexpected fallback or KD in the fixed no-KD study')
        c.exact_predictions(c.read(out/'reload_result.json'),result,'saved checkpoint reload')
        from scipy.stats import spearmanr
        pred=result['predictions']
        coef=float(spearmanr(pred['label'],pred['prediction'])[0])*100
        if not math.isfinite(coef) or abs(coef-result['coef'])>1e-6:raise ValueError('predictions/SRCC mismatch')
        if m=='rgb_e':required+=['rgb_baseline.json']
        if phase!='smoke' and m in ('repeat_e','repeat_s'): required+=['parity.json']
    done=dict(cell=list(cell),phase=phase,srcc_x100=result['coef'],files={n:c.sha(out/n) for n in required})
    if m!='replay': done.update(checkpoint_sha256=c.sha(out/'model.pth'),eval_rng_sha256=c.sha(out/'model.pth.eval_rng.pt'))
    c.save(out/'complete.json',done)
    print('FINISH',phase,cell,'SRCC=%.6f'%result['coef'],flush=True)


def wave(root,phase,jobs,gpus):
    pending=queue.Queue(); stop=threading.Event()
    for cell in jobs: pending.put(cell)
    def worker(gpu):
        while not stop.is_set():
            try: cell=pending.get_nowait()
            except queue.Empty: return
            try: run_cell(root,phase,cell,gpu)
            except Exception: stop.set(); raise
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(worker,g) for g in gpus]
        for f in futures: f.result()


def package(root,partial=False):
    path=root.parent/(root.name+('_partial' if partial else '')+'_return.zip')
    temp=path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(root.rglob('*')):
            if f.is_file() and f.suffix in ('.json','.jsonl','.log','.md','.csv') and f.name!='archive.json':
                archive.write(f,f.relative_to(root).as_posix())
        for f in c.HERE.iterdir():
            if f.is_file() and f.suffix in ('.py','.sh','.md','.json','.bundle'):
                archive.write(f,'evidence_code/'+f.name)
    with zipfile.ZipFile(temp) as archive:
        if archive.testzip() is not None:raise ValueError('return ZIP CRC failed')
    temp.replace(path)
    print('ARCHIVE',path,'SHA256='+c.sha(path),flush=True)
    return path


def backup(root):
    files=[]
    for phase,methods in [('smoke',c.SMOKE_METHODS),('formal',c.TRAIN_METHODS)]:
        for cell in c.cells(methods,phase=='smoke'):
            directory=c.cell_dir(root,phase,cell)
            if not valid_done(directory):raise ValueError('backup requires completed cell: '+str(cell))
            out=active_dir(directory)
            files.append(directory/'active.json')
            files.extend(p for p in out.iterdir() if p.is_file())
    files.extend(root/n for n in ('manifest.json','completion.json','summary.json'))
    manifest={p.relative_to(root).as_posix():c.sha(p) for p in files}
    c.save(root/'asset_manifest.json',dict(files=manifest,
        scope='new v30 heads and records only; retained v29 full models/RNG remain required',
        v29_origin=c.read(root/'manifest.json')['origin']))
    path=root.parent/(root.name+'_assets.tar.gz');tmp=path.with_suffix('.gz.tmp')
    with tarfile.open(tmp,'w:gz',compresslevel=1) as tar:
        for file in files+[root/'asset_manifest.json']:
            tar.add(file,arcname=file.relative_to(root).as_posix(),recursive=False)
    import hashlib
    with tarfile.open(tmp,'r:gz') as tar:
        for member in tar:
            if member.name=='asset_manifest.json':continue
            digest=hashlib.sha256();stream=tar.extractfile(member)
            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
            if digest.hexdigest()!=manifest[member.name]:raise ValueError('backup hash mismatch')
    tmp.replace(path)
    c.save(root/'archive.json',dict(assets_verified=True,assets_path=str(path),assets_sha256=c.sha(path)))
    print('BACKUP VERIFIED',path,flush=True)


def report(root):
    manifest=c.read(root/'manifest.json'); rows=[]
    for ref in manifest['references']:
        for method in ('base','ordered','e','s'):
            rows.append(dict(action=ref['action'],seed=ref['seed'],method='v29_'+method,
                             srcc_x100=ref[method]['test']['coef'],origin='v29 matched retraining reference; not a new v30 training'))
    for cell in c.cells(tuple(manifest.get('methods',c.TRAIN_METHODS))):
        directory=c.cell_dir(root,'formal',cell)
        if valid_done(directory):
            out=active_dir(directory); record=c.read(out/'complete.json')
            rows.append(dict(action=cell[0],seed=cell[1],method=cell[2],srcc_x100=record['srcc_x100'],origin='new matched training'))
            if cell[2]=='rgb_e':
                rows.append(dict(action=cell[0],seed=cell[1],method='rgb_reference',
                    srcc_x100=c.read(out/'rgb_baseline.json')['coef'],origin='fixed public RGB scorer; not independently retrained per seed'))
    data=dict(policy=c.POLICY,rows=rows,summary=c.aggregate(rows),
              warning='post-hoc modality removal, scale curves and inference draws are diagnostic, not extra trained methods',
              missing=list(manifest['source']['remaining_evidence']),study=manifest.get('study','all'),
              deferred_methods=manifest.get('deferred_methods',[]))
    if data['deferred_methods']:
        data['missing'].append('Base-transfer extension deferred: base_e/base_s, 36 cells; historical Base scores retained, no Base-transfer claim')
    # Paired per-action/seed changes, not a new mean selected from winning seeds.
    baseline={r['method']:{(v['action'],v['seed']):v['srcc_x100'] for v in rows if v['method']==r['method']}
              for r in data['summary']}
    data['paired_comparisons']=[]
    for method in baseline:
        for reference in (('rgb_reference',) if method.startswith('rgb') else ('v29_base','v29_ordered','v29_e','v29_s')):
            if method==reference: continue
            changes=[baseline[method][key]-value for key,value in baseline[reference].items()]
            data['paired_comparisons'].append(dict(method=method,reference=reference,mean_delta=sum(changes)/18,
                wins=sum(x>1e-8 for x in changes),ties=sum(abs(x)<=1e-8 for x in changes),losses=sum(x < -1e-8 for x in changes)))
    data['factorial_interaction']=[]
    if all(m in baseline for m in ('base_e','base_s','v29_base','v29_ordered','v29_e','v29_s')):
        for action in c.ACTIONS:
            for seed in c.SEEDS:
                key=(action,seed)
                delta_e_ordered=baseline['v29_e'][key]-baseline['v29_ordered'][key]
                delta_e_base=baseline['base_e'][key]-baseline['v29_base'][key]
                delta_full_ordered=baseline['v29_s'][key]-baseline['v29_ordered'][key]
                delta_full_base=baseline['base_s'][key]-baseline['v29_base'][key]
                data['factorial_interaction'].append(dict(action=action,seed=seed,
                    e_interaction=delta_e_ordered-delta_e_base,full_interaction=delta_full_ordered-delta_full_base,
                    note='positive interaction alone does not establish that combination beats both single modules'))
    data['literature_reference']=c.read(c.HERE/'literature_reference.json')
    data['comparison_scope']='v29/v30 matched ablations are primary causal comparisons; paper numbers are separately labeled, not matched reruns'
    costs=[]; draws=[]; training_costs=[]
    for ref in manifest['references']:
        for method in ('e','s'):
            if 'record' not in ref[method]:continue
            path=Path(ref[method]['record']).parent/'qmd_training.json'
            if path.exists():
                training_costs.append(dict(action=ref['action'],seed=ref['seed'],method='v29_'+method,
                    training_seconds=c.read(path).get('training_seconds'),
                    scope='recorded correction-stage wall time; shared-server historical measurement, not isolated speed comparison'))
    for cell in c.cells(('replay',)):
        directory=c.cell_dir(root,'replay',cell)
        if valid_done(directory):
            out=active_dir(directory)
            costs.append(dict(action=cell[0],seed=cell[1],**c.read(out/'cost.json')))
            draws.append(dict(action=cell[0],seed=cell[1],**c.read(out/'inference_draws.json')))
    data['cost_records']=costs; data['inference_draw_records']=draws;data['training_cost_records']=training_costs
    c.save(root/'summary.json',data)
    text=io.StringIO(); writer=csv.DictWriter(text,fieldnames=('method','action','seed','srcc_x100','origin')); writer.writeheader(); writer.writerows(rows)
    (root/'all_action_seed_results.csv').write_text(text.getvalue(),encoding='utf-8-sig')
    md=['# v17 paper evidence','',c.POLICY,'','| Method | six-task mean | seed SD | RG Fisher | FisV Fisher |','|---|---:|---:|---:|---:|']
    for r in data['summary']: md.append('| {method} | {six_macro:.6f} | {sample_sd_seed_means:.6f} | {RG_fisher:.6f} | {FisV_fisher:.6f} |'.format(**r))
    md+=['','## Paired changes (18 action/seed cells)','','| Method | Reference | delta | wins/ties/losses |','|---|---|---:|---|']
    for r in data['paired_comparisons']:
        md.append('| {method} | {reference} | {mean_delta:+.6f} | {wins}/{ties}/{losses} |'.format(**r))
    for seed in c.SEEDS:
        md+=['','## Seed %d'%seed,'','| Method | Ball | Clubs | Hoop | Ribbon | TES | PCS | six-task mean |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in data['summary']:
            sr=next(v for v in r['per_seed'] if v['seed']==seed)
            md.append('| '+r['method']+' | '+' | '.join('%.6f'%sr['actions'][a] for a in c.ACTIONS)+' | %.6f |'%sr['six_macro'])
    md+=['','Not yet established:']+['- '+x for x in data['missing']]
    (root/'tables.md').write_text('\n'.join(md)+'\n',encoding='utf-8')


def status(root):
    runner=c.read(root/'runner.json') if (root/'runner.json').exists() else {}
    print('Runner:',runner.get('state','not started'),'alive:',alive(runner.get('identity')),'phase:',runner.get('phase','-'))
    if runner.get('error'):print('ERROR:',runner['error'])
    for phase,jobs in [('replay',c.cells(('replay',))),('smoke',c.cells(c.SMOKE_METHODS,True)),('formal',c.cells())]:
        count=0; per={m:0 for m in set(v[2] for v in jobs)}
        for cell in jobs:
            out=active_dir(c.cell_dir(root,phase,cell))
            if not out:continue
            if (out/'complete.json').exists():count+=1;per[cell[2]]+=1;continue
            proc='reload.process.json' if (out/'reload.process.json').exists() else 'process.json'
            process=c.read(out/proc) if (out/proc).exists() else {}
            log=out/('reload.log' if proc.startswith('reload') else 'train.log')
            tail=''
            if log.exists():
                with log.open('rb') as f:f.seek(max(0,log.stat().st_size-3000));tail=f.read().decode(errors='replace')
            epochs=re.findall(r'Epoch\[\d+/\d+\]',tail)
            print('RUNNING' if alive(process.get('identity')) else 'FAILED/INTERRUPTED',phase,cell,
                  epochs[-1] if epochs else 'loading/evaluation','GPU='+str(process.get('gpu')),str(log))
            if tail and not alive(process.get('identity')):print('  last:',tail.splitlines()[-1][:240])
        print('%s completed: %d/%d'%(phase,count,len(jobs)),dict(sorted(per.items())))
    for name in ('replay_gate.json','smoke_gate.json','completion.json','archive.json'):
        print(name,'READY' if (root/name).exists() else 'not yet created')
    if (root/'archive.json').exists():print('Assets verified:',c.read(root/'archive.json').get('assets_verified',False))
    print('Formal: 126 NEW branch trainings; v29 72 results are references, not counted as new.')
    print('Final return:',root.parent/(root.name+'_return.zip'))
    print('Same root resume keeps verified completions; no files deleted; interrupted cell restarts in a new attempt.')


def verify_units(root,gpu):
    env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(c.HERE),str(c.REPO))),
             CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    commands=[
        ('unit_v30.log',[sys.executable,'-m','unittest','discover','-s',str(c.HERE),'-p','test_*.py','-v']),
        ('unit_original.log',[sys.executable,'-m','unittest','tests.test_qmd_supp','tests.test_qmd_modal',
                              'tests.test_qmd_residual','tests.test_july28_qmd'])]
    for name,argv in commands:
        with (root/name).open('w',encoding='utf-8') as log:
            subprocess.run(argv,cwd=str(c.REPO),env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    c.save(root/'unit_pass.json',dict(passed=True,files={n:c.sha(root/n) for n,_ in commands}))


def select_gpus(requested):
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    available={r[0].strip():int(r[1].strip()) for r in csv.reader(io.StringIO(text))}
    selected=[g for g,n in available.items() if g in ('0','1','2','3','4','5') and n>=8192] if requested=='auto' else requested.split()
    if not selected or len(set(selected))!=len(selected) or any(g not in available or available[g]<8192 for g in selected):
        raise RuntimeError('selected GPUs need >=8192 MiB free; available: '+str(available))
    print('Selected physical GPUs:',selected,'free MiB:',available,flush=True)
    return selected


def execute(args,root,record):
    def phase(name):
        record['phase']=name;c.save(root/'runner.json',record)
    phase('preflight');check_space(root)
    gpus=select_gpus(args.gpus)
    old,refs=c.build_references(args.origin)
    source=c.provenance(old)
    manifest=dict(source=source,references=refs,origin=str(Path(args.origin).resolve()),
                  actions=list(c.ACTIONS),seeds=list(c.SEEDS),methods=list(c.TRAIN_METHODS),
                  smoke_methods=list(c.SMOKE_METHODS),study='v29_matched_evidence')
    c.immutable(root/'manifest.json',manifest)
    phase('unit_checks');verify_units(root,gpus[0])
    if args.stage=='check':phase('checked');return
    phase('replay_profile')
    wave(root,'replay',c.cells(('replay',)),[gpus[0]])
    c.save(root/'replay_gate.json',dict(exact_replays=18,passed=True))
    phase('smoke')
    for method in c.SMOKE_METHODS:wave(root,'smoke',c.cells((method,),True),gpus)
    c.save(root/'smoke_gate.json',dict(cells=18,passed=True))
    if args.stage=='smoke':
        report(root);package(root,partial=True);return
    phase('formal')
    # base_s depends on the new base_e for the same action/seed. Never mix old dependencies.
    for method in c.TRAIN_METHODS:wave(root,'formal',c.cells((method,)),gpus)
    phase('report');report(root)
    c.save(root/'completion.json',dict(experiments_complete=True,new_formal_cells=126,smoke_cells=18,replay_cells=18,
        v29_reference_cells=72,paper_ready=False,remaining=list(c.REMAINING),manifest_sha256=c.sha(root/'manifest.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('all','check','smoke','status','pack'),default='all')
    p.add_argument('--root',type=Path,default=c.HERE/'results/v17_evidence_v30')
    p.add_argument('--origin',default='/szztdata/ly/exp0_source/PAMFN-v17-rebuild-v29/results/v17_rebuild_v29')
    p.add_argument('--gpus',default='0 1 3 4')
    args=p.parse_args();root=args.root.resolve()
    if args.stage=='status':status(root);return
    expected=(c.HERE/'results').resolve()
    if root==expected or expected not in root.parents:raise ValueError('root must be a named child of v30/results')
    old=Path(args.origin).resolve()
    if root==old or old in root.parents or root in old.parents:raise ValueError('output overlaps read-only v29')
    root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (root/'runner.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('another launcher owns this result root')
        # Parent may have stopped while a worker remained alive. Refuse duplicate launches/packaging.
        for name in ('process.json','reload.process.json'):
            for path in root.glob('*/**/'+name):
                if alive(c.read(path).get('identity')):raise RuntimeError('worker still alive: '+str(path))
        if args.stage=='pack':
            report(root);package(root,partial=not (root/'completion.json').exists());return
        record=dict(pid=os.getpid(),identity=identity(os.getpid()),state='running',started_at=time.time())
        c.save(root/'runner.json',record)
        try:
            execute(args,root,record)
            if args.stage=='all':
                record.update(phase='packaging');c.save(root/'runner.json',record)
                backup(root)
            record.update(state='finished',phase='checked' if args.stage=='check' else ('smoke_done' if args.stage=='smoke' else 'done'),ended_at=time.time())
            c.save(root/'runner.json',record)
            if args.stage=='all':
                final=package(root);receipt=c.read(root/'archive.json')
                receipt['return_zip']=dict(path=str(final),sha256=c.sha(final));c.save(root/'archive.json',receipt)
            print('V17_EVIDENCE_DONE '+args.stage,flush=True)
        except Exception as exc:
            record.update(state='failed',error=repr(exc),ended_at=time.time());c.save(root/'runner.json',record)
            try:package(root,partial=True)
            except Exception as error:print('Partial package failed:',repr(error),flush=True)
            raise


if __name__=='__main__':main()
