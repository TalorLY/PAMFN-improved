"""Resume verified cells, dependency-aware GPU queue, automatic smoke/formal/report/pack."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import zipfile
from spec import (HERE,POLICY,TASKS,SEEDS,NEW_METHODS,cells,epochs,read,save,sha,
                  cell_path,completed,dependencies,reference_path,artifact)
from storage import storage_report

def alive(pid):
    if not pid:return False
    try:os.kill(int(pid),0);return True
    except (OSError,TypeError,ValueError):return False

def environment(gpu=None):
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    return env

def runtime_python():return os.environ.get('V38_PYTHON',sys.executable)

def gpu_inventory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {r.split(',')[0].strip():int(r.split(',')[1]) for r in raw.strip().splitlines()}

def selected_gpus(value):
    ids=value.replace(',',' ').split();free=gpu_inventory()
    if not ids or len(ids)!=len(set(ids)) or any(free.get(g,0)<8192 for g in ids):
        raise RuntimeError('Choose distinct GPUs with >=8192 MiB free. Current free MiB: '+str(free))
    print('Selected physical GPUs:',ids,'free MiB:',free,flush=True);return ids

def code_check():
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        if sha(HERE/name)!=digest:raise ValueError('Delivery file changed: '+name)

def disk_check(root,workers=1):
    budget=read(HERE/'storage_budget.json')['checkpoint_bytes']
    remaining={m:0 for m in NEW_METHODS}
    for phase in ('smoke','formal'):
        for t,s,m in cells(phase=='smoke'):
            if completed(cell_path(root,phase,t,s,m)) is None:remaining[m]+=1
    value=storage_report(root,budget,remaining,concurrent_workers=workers)
    save(Path(root)/'storage_check.json',value)
    print('STORAGE free=%.2f GiB required=%.2f GiB pass=%s (no deletion)'%
          (value['free_bytes']/2**30,value['required_bytes']/2**30,value['pass']),flush=True)
    if not value['pass']:raise RuntimeError('Insufficient disk headroom; inspect storage_check.json. No files deleted.')
    return value

def source_records(source):
    source=Path(source).resolve()
    manifest=read(source/'manifest.json')
    policy=manifest['policy']
    for key,value in [('dataset','FS1000'),('train',1000),('evaluation',247),('train_crop',95),('test_crop','center95')]:
        if policy.get(key)!=value:raise ValueError('v37 protocol mismatch: '+key)
    for key in ('batch_size','loader_workers','test_batch_size','rgb_pool','input_dimensions'):
        if policy.get(key)!=POLICY[key]:raise ValueError('v37 matched data/loader mismatch: '+key)
    if policy['selection']!=POLICY['selection'] or policy['mcmoe_epochs']!={t:epochs('mcmoe',t) for t in TASKS}:
        raise ValueError('v37 MCMoE budget/selection mismatch')
    if manifest['input_manifest_sha256']!=sha(HERE/'FS1000_INPUT_SHA256SUMS'):raise ValueError('Input contract mismatch')
    if read(source/'completion.json')['status']!='EXPERIMENTS_COMPLETE':raise ValueError('v37 is not complete')
    records=[]
    required=('model.pth','model.pth.eval_rng.pt','test_result.json')
    for phase in ('smoke','formal'):
        for t in (('TES','PCS') if phase=='smoke' else TASKS):
            for s in ((0,) if phase=='smoke' else SEEDS):
                cell=source/phase/t/('seed%d'%s)/'mcmoe';out=completed(cell)
                if out is None:raise RuntimeError('Missing v37 MCMoE: '+str(cell))
                out.resolve().relative_to(source)
                done=read(cell/'complete.json');digests={}
                for name in required:
                    file=out/name
                    if file.is_symlink() or sha(file)!=done['files'][name]:raise ValueError('v37 reference hash failed: '+str(file))
                    digests[name]=done['files'][name]
                result=read(out/'test_result.json')
                if (result['task'],result['seed'],result['method'],result['phase'])!=(t,s,'mcmoe',phase):raise ValueError('Wrong reference cell')
                if not result['reloaded_exact'] or result['epochs']!=epochs('mcmoe',t,phase=='smoke'):raise ValueError('Invalid reference budget/reload')
                n=32 if phase=='smoke' else 247
                if len(result['ids'])!=n or len(set(result['ids']))!=n:raise ValueError('Reference sample identity invalid')
                records.append(dict(task=t,seed=s,phase=phase,directory=str(out.resolve()),
                    complete_sha256=sha(cell/'complete.json'),required_files=digests))
    return manifest,records

def prepare_references(args,root,current):
    old,records=source_records(args.source)
    expected=old['environments']['external']
    for k in ('torch','numpy','scipy','cuda'):
        if current[k]!=expected[k]:raise ValueError('External environment differs from v37: '+k)
    manifest=dict(policy=POLICY,source_root=str(Path(args.source).resolve()),data_root=str(Path(args.data).resolve()),
        environment=current,source_manifest_sha256=sha(Path(args.source)/'manifest.json'),
        code_manifest_sha256=sha(HERE/'SHA256SUMS'),input_manifest_sha256=sha(HERE/'FS1000_INPUT_SHA256SUMS'),references=records)
    if (root/'manifest.json').exists() and read(root/'manifest.json')!=manifest:
        raise ValueError('Run contract changed; refusing to mix results')
    save(root/'manifest.json',manifest)
    for record in records:
        dest=reference_path(root,record['phase'],record['task'],record['seed'])
        for name,value in [('record.json',record),('test_result.json',read(Path(record['directory'])/'test_result.json'))]:
            if (dest/name).exists() and read(dest/name)!=value:raise ValueError('Reference snapshot changed')
            save(dest/name,value)
    print('REFERENCES_PASS: 21 formal + 2 smoke MCMoE weights reused read-only',flush=True)

def check(args,root,workers):
    code_check();disk_check(root,workers)
    snippet="import sys,json,torch,numpy,scipy;print(json.dumps(dict(python=sys.version,torch=str(torch.__version__),numpy=numpy.__version__,scipy=scipy.__version__,cuda=torch.version.cuda,available=torch.cuda.is_available())))"
    current=json.loads(subprocess.check_output([runtime_python(),'-c',snippet],env=environment(),text=True))
    if (current['torch'],current['numpy'],current['scipy'])!=('1.12.1+cu113','1.21.6','1.10.1') or not current['available']:
        raise RuntimeError('Use existing v33 external environment; do not reinstall pamfn_ly: '+str(current))
    prepare_references(args,root,current)
    with (root/'check.log').open('w',encoding='utf-8') as log:
        commands=[['-m','unittest','test_modules','test_storage','test_runner','test_adapter','test_report'],
                  ['check_data.py','--data',str(Path(args.data).resolve())]]
        for command in commands:
            # These are CPU checks: never initialize CUDA contexts on unselected busy GPUs.
            subprocess.run([runtime_python(),'-u']+command,cwd=str(HERE),env=environment(''),stdout=log,stderr=subprocess.STDOUT,check=True)
    print('PREFLIGHT_PASS: code, environment, reference weights, input hashes and units',flush=True)

def validate_cell(out,phase,t,s,m,deps):
    result=read(out/'test_result.json')
    if (result['task'],result['seed'],result['method'],result['phase'])!=(t,s,m,phase):raise ValueError('Wrong returned cell')
    if result['epochs']!=epochs(m,t,phase=='smoke') or not result['reloaded_exact'] or result.get('synthetic_cpu_test_only'):
        raise ValueError('Wrong training budget, reload, or CPU-only result')
    if result['dependencies']!=deps:raise ValueError('Parent checkpoint mismatch')
    n=32 if phase=='smoke' else 247
    if len(result['ids'])!=n or len(set(result['ids']))!=n:raise ValueError('Wrong sample identities')
    history=[json.loads(v) for v in (out/'epochs.jsonl').read_text().splitlines() if v.strip()]
    if [r['epoch'] for r in history]!=list(range(1,result['epochs']+1)):raise ValueError('Incomplete history')
    valid=[r for r in history if r['coef'] is not None]
    if not valid:raise ValueError('No finite trained epoch')
    best=max(valid,key=lambda r:r['coef'])
    if best['epoch']!=result['selected_epoch'] or abs(best['coef']-result['coef'])>1e-8:raise ValueError('Selection mismatch')
    if m!='mc_ordered':
        checks=read(out/'qmd_training.json')
        if not all(checks[k] for k in ('anchor_unchanged','statistics_unchanged','branch_updated')):
            raise ValueError('Frozen-parent or branch-update check failed')
    return result

def run_cell(args,root,phase,t,s,m,gpu,workers):
    cell=cell_path(root,phase,t,s,m)
    if completed(cell,True):return
    disk_check(root,workers)
    free=gpu_inventory()
    if free.get(gpu,0)<8192:raise RuntimeError('GPU '+gpu+' now has <8192 MiB free; no worker launched')
    deps={d:sha(artifact(root,phase,t,s,d)/'model.pth') for d in dependencies(m)}
    attempt='attempt_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out=cell/attempt;out.mkdir(parents=True)
    cmd=[runtime_python(),'-u',str(HERE/'worker.py'),'--root',str(root),'--phase',phase,
         '--task',t,'--seed',str(s),'--method',m,'--out',str(out),'--data',str(Path(args.data).resolve())]
    record=dict(state='starting',attempt=attempt,started=time.time(),gpu=gpu,command=cmd,pid=None)
    save(cell/'active.json',record)
    try:
        with (out/'train.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(cmd,cwd=str(HERE),env=environment(gpu),stdout=log,stderr=subprocess.STDOUT)
            record.update(state='running',pid=proc.pid);save(cell/'active.json',record)
            print('START GPU=%s %s/%s/seed%d/%s'%(gpu,phase,t,s,m),flush=True)
            code=proc.wait()
        record.update(exit_code=code,ended=time.time())
        if code:raise RuntimeError('Worker exited %s: %s'%(code,out/'train.log'))
        result=validate_cell(out,phase,t,s,m,deps)
        files={p.name:sha(p) for p in out.iterdir() if p.is_file() and not p.name.endswith('.tmp')}
        save(cell/'complete.json',dict(attempt=attempt,files=files,coef=result['coef'],dependencies=deps))
        record['state']='finished';save(cell/'active.json',record)
        print('FINISH %s/%s/seed%d/%s SRCC=%.6f'%(phase,t,s,m,result['coef']),flush=True)
    except BaseException as error:
        record.update(state='failed',error=repr(error),ended=time.time());save(cell/'active.json',record);raise

def wave(args,root,phase,gpus,runner=run_cell):
    matrix=cells(phase=='smoke')
    done={c for c in matrix if completed(cell_path(root,phase,*c),True) is not None}
    for t,s,m in done:
        out=completed(cell_path(root,phase,t,s,m),True)
        current={d:sha(artifact(root,phase,t,s,d)/'model.pth') for d in dependencies(m)}
        result=read(out/'test_result.json')
        if result['dependencies']!=current:
            raise ValueError('Completed child has a missing/replaced parent; refusing mixed results: '+str(out))
    pending=[c for c in matrix if c not in done];failed={};inflight={};available=list(gpus)
    def ready(c):
        return all(d=='mcmoe' or (c[0],c[1],d) in done for d in dependencies(c[2]))
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        while pending or inflight:
            ready_cells=[c for c in pending if ready(c)]
            while available and ready_cells:
                c=ready_cells.pop(0);pending.remove(c);gpu=available.pop(0)
                job=pool.submit(runner,args,root,phase,*c,gpu,len(gpus));inflight[job]=(c,gpu)
            if not inflight:break
            finished,_=wait(tuple(inflight),return_when=FIRST_COMPLETED)
            for job in finished:
                c,gpu=inflight.pop(job);available.append(gpu)
                try:job.result();done.add(c)
                except Exception as error:
                    failed[c]=repr(error);print('FAILED',c,repr(error),flush=True)
    value=dict(phase=phase,completed=len(done),total=len(matrix),
        failures=[dict(task=c[0],seed=c[1],method=c[2],error=e) for c,e in failed.items()],
        blocked=[dict(task=c[0],seed=c[1],method=c[2]) for c in pending])
    save(root/(phase+'_wave.json'),value)
    if failed or pending:raise RuntimeError('%s: %d failed, %d dependency-blocked; completed cells retained'%(phase,len(failed),len(pending)))

def pack(root,partial=False):
    root=Path(root);dest=root.with_name(root.name+('_partial_return.zip' if partial else '_return.zip'))
    temp=dest.with_name(dest.name+'.tmp');records=[]
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.is_symlink() or p.suffix not in ('.json','.jsonl','.log','.csv','.md','.svg'):continue
            data=p.read_bytes();name=p.relative_to(root).as_posix();z.writestr(root.name+'/'+name,data)
            records.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest()))
        z.writestr('RETURN_CONTENTS.json',json.dumps(records,indent=2))
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:raise ValueError('Return archive CRC failed')
        for r in records:
            if hashlib.sha256(z.read(root.name+'/'+r['path'])).hexdigest()!=r['sha256']:raise ValueError('Archive hash failed')
    os.replace(str(temp),str(dest));print('ARCHIVE',dest,'SHA256='+sha(dest),flush=True);return dest

def status(root):
    rec=read(root/'runner.json') if (root/'runner.json').exists() else {}
    print('Runner:',rec.get('state','not started'),'alive:',alive(rec.get('pid')),'phase:',rec.get('phase'))
    for phase in ('smoke','formal'):
        matrix=cells(phase=='smoke');done=0;live=0;failed=0;started=set()
        for t,s,m in matrix:
            cell=cell_path(root,phase,t,s,m)
            if completed(cell):done+=1;continue
            if not (cell/'active.json').exists():continue
            a=read(cell/'active.json');started.add((t,s,m));is_live=a.get('state')=='running' and alive(a.get('pid'))
            live+=int(is_live);failed+=int(not is_live)
            log=cell/a['attempt']/'train.log';last='-'
            if log.exists():
                with log.open('rb') as f:f.seek(max(0,log.stat().st_size-12000));tail=f.read().decode('utf-8',errors='replace')
                matches=re.findall(r'Epoch\[\d+/\d+\]',tail)
                if matches:last=matches[-1]
            print('RUNNING' if is_live else 'FAILED/INTERRUPTED',phase,t,s,m,last,'GPU='+a['gpu'],'log='+str(log))
        refs=sum((reference_path(root,phase,t,s)/'test_result.json').exists()
                 for t in (('TES','PCS') if phase=='smoke' else TASKS) for s in ((0,) if phase=='smoke' else SEEDS))
        print('%s new training completed: %d/%d; reused MCMoE: %d; running: %d; failed/interrupted: %d; pending: %d'%
              (phase,done,len(matrix),refs,live,failed,len(matrix)-done-len(started)))
    completion=(root/'completion.json').exists();archive=root.with_name(root.name+'_return.zip')
    print('Smoke gate:',(root/'smoke_gate.json').exists(),'Completion:',completion)
    print('Return ZIP:',archive,'READY' if archive.exists() and completion else 'not complete')
    if rec.get('error'):print('LAUNCHER ERROR:',rec['error'])
    print('Resume SAME root/command; completed weights verified and kept; unfinished stage restarts, not epoch-resumes.')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('all','check','status','storage','report','pack'),default='all')
    p.add_argument('--root',default='results/v17_transfer_v38')
    p.add_argument('--source',default=os.environ.get('V38_SOURCE_ROOT','/szztdata/ly/exp0_source/PAMFN-v17-fs1000-v37/results/v17_fs1000_v37'))
    p.add_argument('--data',default=os.environ.get('V38_DATA','/szztdata/ly/exp0_source/shared_data/FS1000'))
    p.add_argument('--gpus',default=os.environ.get('GPU_LIST','4'))
    args=p.parse_args();root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    if args.stage=='status':return status(root)
    if args.stage=='storage':return disk_check(root,max(1,len(args.gpus.replace(',',' ').split())))
    if args.stage=='report':
        from report import build_report
        return build_report(root)
    if args.stage=='pack':return pack(root,not (root/'completion.json').exists())
    import fcntl
    with (root/'runner.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A launcher already holds this root')
        for file in root.glob('*/*/seed*/*/active.json'):
            a=read(file)
            if a['state']=='running' and alive(a.get('pid')):raise RuntimeError('Existing worker is still active: '+str(file))
        record=dict(pid=os.getpid(),state='running',phase='preflight',started=time.time());save(root/'runner.json',record)
        try:
            gpus=selected_gpus(args.gpus);check(args,root,len(gpus))
            if args.stage=='all':
                for phase in ('smoke','formal'):
                    record['phase']=phase;save(root/'runner.json',record);wave(args,root,phase,gpus)
                    if phase=='smoke':save(root/'smoke_gate.json',dict(status='PASS',new_cells=10,reused=2))
                record['phase']='report';save(root/'runner.json',record)
                from report import build_report
                build_report(root)
                save(root/'completion.json',dict(status='EXPERIMENTS_COMPLETE',new_formal=105,new_smoke=10,
                    reused_formal=21,note='Completion does not imply positive scientific outcome',ended=time.time()))
                record['phase']='pack';save(root/'runner.json',record);pack(root)
                record['phase']='complete'
            record.update(state='finished',ended=time.time());save(root/'runner.json',record)
            # Include final runner state in the verified final ZIP.
            if args.stage=='all':pack(root)
        except BaseException as error:
            record.update(state='failed',error=repr(error),ended=time.time());save(root/'runner.json',record)
            traceback.print_exc()
            try:pack(root,True)
            except Exception:traceback.print_exc()
            raise

if __name__=='__main__':main()
