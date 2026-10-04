"""Small independent v39 queue. No old weights are copied, trained or deleted."""
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
import bootstrap
from contract import HERE,POLICY,TASKS,SEEDS,METHODS,cells,budget,cell,reference,prepare,read,save,sha,completed
from storage import storage_report

def alive(pid):
    try:os.kill(int(pid),0);return True
    except (OSError,TypeError,ValueError):return False

def env(gpu=None):
    value=os.environ.copy();value.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if gpu is not None:value['CUDA_VISIBLE_DEVICES']=str(gpu)
    return value

def python():return os.environ.get('V39_PYTHON',sys.executable)

def gpu_inventory():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {line.split(',')[0].strip():int(line.split(',')[1]) for line in text.strip().splitlines()}

def select_gpus(value):
    ids=value.replace(',',' ').split();free=gpu_inventory()
    if not ids or len(set(ids))!=len(ids) or any(free.get(g,0)<8192 for g in ids):
        raise RuntimeError('Choose distinct physical GPUs with >=8192 MiB free: '+str(free))
    print('Selected GPUs',ids,'current free MiB',free,flush=True);return ids

def code_check():
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        if sha(HERE/name)!=digest:raise ValueError('Source file changed: '+name)
    legacy=read(HERE/'legacy_source.json')
    for name,digest in legacy['files'].items():
        if sha(bootstrap.LEGACY/name)!=digest:raise ValueError('Preserved v38 file changed: '+name)

def disk_check(root,workers=1):
    counts={m:sum(completed(cell(root,p,t,s,m)) is None for p in ('smoke','formal')
                 for t,s,k in cells(p=='smoke') if k==m) for m in METHODS}
    # No new backbone weights. Extra metadata includes serial benchmark and paper outputs.
    result=storage_report(root,{m:262144 for m in METHODS},counts,workers,reserve_gib=8)
    result['required_bytes']+=512*1024**2
    result['pass']=result['free_bytes']>=result['required_bytes']
    save(Path(root)/'storage_check.json',result)
    print('STORAGE free=%.2f GiB required=%.2f GiB pass=%s (no deletion)'%
          (result['free_bytes']/2**30,result['required_bytes']/2**30,result['pass']),flush=True)
    if not result['pass']:raise RuntimeError('Insufficient disk headroom; no files deleted. Inspect storage_check.json')
    return result

def check(args,root,workers):
    code_check();disk_check(root,workers)
    snippet="import sys,json,torch,numpy,scipy;print(json.dumps(dict(python=sys.version,torch=str(torch.__version__),numpy=numpy.__version__,scipy=scipy.__version__,cuda=torch.version.cuda,available=torch.cuda.is_available())))"
    current=json.loads(subprocess.check_output([python(),'-c',snippet],env=env(),text=True))
    if (current['torch'],current['numpy'],current['scipy'])!=('1.12.1+cu113','1.21.6','1.10.1') or not current['available']:
        raise RuntimeError('Reuse the working v38 external environment; no environment installation needed: '+str(current))
    prepare(args.source,root,args.data,current)
    print('CHECKING CPU regression tests and FS1000 input hashes; details in check.log',flush=True)
    with (root/'check.log').open('w',encoding='utf-8') as log:
        subprocess.run([python(),'-u','-m','unittest','test_v39'],cwd=str(HERE),env=env(''),stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([python(),'-u',str(bootstrap.LEGACY/'check_data.py'),'--data',str(Path(args.data).resolve())],
            cwd=str(bootstrap.LEGACY),env=env(''),stdout=log,stderr=subprocess.STDOUT,check=True)
    print('PREFLIGHT_PASS: v38 references read-only, unchanged data/recipe, v39 units',flush=True)

def deps(root,phase,t,s,m):
    keys=['mcmoe']+(['mc_e'] if m=='mc_fused_u' else [])
    return {k:sha(reference(root,phase,t,s,k)/'model.pth') for k in keys}

def validate_control(out,phase,t,s,m,expected):
    r=read(out/'test_result.json');n=32 if phase=='smoke' else 247
    if (r['phase'],r['task'],r['seed'],r['method'])!=(phase,t,s,m):raise ValueError('Wrong control identity')
    if r['synthetic_cpu_test_only'] or not r['reloaded_exact'] or r['epochs']!=budget(phase=='smoke'):
        raise ValueError('Incomplete or synthetic control')
    if r['dependencies']!=expected or len(r['ids'])!=n or len(set(r['ids']))!=n:raise ValueError('Control source/sample mismatch')
    if r.get('study_version')!=39 or r['parameters']!={'mc_plain_e':4128,'mc_fused_u':12384}[m]:
        raise ValueError('Control version/capacity mismatch')
    declared=read(out/'control_contract.json')
    if declared['policy']!=POLICY or declared['method']!=m:raise ValueError('Control design changed')
    from scipy.stats import spearmanr
    import numpy as np
    p=np.asarray(r['predictions']['prediction']);y=np.asarray(r['predictions']['label'])
    if p.shape!=(n,) or y.shape!=(n,) or not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('Invalid predictions')
    if not np.isfinite(r['coef']) or abs(spearmanr(p,y)[0]*100-r['coef'])>1e-8:raise ValueError('SRCC mismatch')
    history=[json.loads(x) for x in (out/'epochs.jsonl').read_text().splitlines() if x.strip()]
    if [x['epoch'] for x in history]!=list(range(1,r['epochs']+1)):raise ValueError('Incomplete control epochs')
    best=max((x for x in history if x['coef'] is not None),key=lambda x:x['coef'])
    if best['epoch']!=r['selected_epoch'] or abs(best['coef']-r['coef'])>1e-8:raise ValueError('Selection mismatch')
    training=read(out/'qmd_training.json')
    if not all(training[k] for k in ('anchor_unchanged','statistics_unchanged','branch_updated')):raise ValueError('Freeze/update gate failed')
    if training['positive_gradient_batches']<=0:raise ValueError('No gradients')
    return r

def process(c,cmd,gpu,validator):
    attempt='attempt_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out=c/attempt;out.mkdir(parents=True);cmd=cmd+['--out',str(out)]
    rec=dict(state='starting',attempt=attempt,started=time.time(),gpu=gpu,command=cmd,pid=None);save(c/'active.json',rec)
    try:
        with (out/'train.log').open('w',encoding='utf-8') as log:
            p=subprocess.Popen(cmd,cwd=str(HERE),env=env(gpu),stdout=log,stderr=subprocess.STDOUT)
            rec.update(state='running',pid=p.pid);save(c/'active.json',rec)
            print('START GPU='+gpu,str(c),flush=True);rc=p.wait()
        rec.update(exit_code=rc,ended=time.time())
        if rc:raise RuntimeError('Worker exited %d: %s'%(rc,out/'train.log'))
        validator(out)
        files={p.name:sha(p) for p in out.iterdir() if p.is_file() and not p.name.endswith('.tmp')}
        save(c/'complete.json',dict(attempt=attempt,files=files))
        rec['state']='finished';save(c/'active.json',rec);print('FINISH',str(c),flush=True)
    except BaseException as error:
        rec.update(state='failed',error=repr(error),ended=time.time());save(c/'active.json',rec);raise

def run_control(args,root,phase,t,s,m,gpu,workers):
    c=cell(root,phase,t,s,m);expected=deps(root,phase,t,s,m)
    done=completed(c,True)
    if done:validate_control(done,phase,t,s,m,expected);return
    disk_check(root,workers)
    if gpu_inventory().get(gpu,0)<8192:raise RuntimeError('Selected GPU now has <8192 MiB free: '+gpu)
    cmd=[python(),'-u',str(HERE/'design_worker.py'),'--root',str(root),'--phase',phase,'--task',t,
         '--seed',str(s),'--method',m,'--data',str(Path(args.data).resolve())]
    process(c,cmd,gpu,lambda out:validate_control(out,phase,t,s,m,expected))

def wave(args,root,phase,gpus):
    queue=list(cells(phase=='smoke'));running={};available=list(gpus);failures=[]
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        while queue or running:
            while queue and available:
                item=queue.pop(0);gpu=available.pop(0)
                future=pool.submit(run_control,args,root,phase,*item,gpu,len(gpus));running[future]=(item,gpu)
            finished,_=wait(tuple(running),return_when=FIRST_COMPLETED)
            for future in finished:
                item,gpu=running.pop(future);available.append(gpu)
                try:future.result()
                except Exception as error:
                    failures.append(dict(cell=item,error=repr(error)));print('FAILED',item,repr(error),flush=True)
    save(root/(phase+'_wave.json'),dict(total=len(cells(phase=='smoke')),failures=failures))
    if failures:raise RuntimeError('%d control cells failed; completed cells preserved'%len(failures))

def bench_cells(smoke=False):
    return [(t,s,p) for t in (('TES','PCS') if smoke else TASKS) for s in ((0,) if smoke else SEEDS) for p in ('mcmoe','mc_ordered')]

def bench_root(root,phase,exclusive=False):
    return Path(root)/('benchmark_exclusive' if exclusive else 'benchmark')/phase

def validate_benchmark(out,root,phase,t,s,parent,exclusive):
    r=read(out/'benchmark.json')
    names=('mcmoe','mc_e','mc_u') if parent=='mcmoe' else ('mc_ordered','mc_ordered_e','mc_ordered_u')
    expected={k:sha(reference(root,phase,t,s,m)/'model.pth') for k,m in zip(('anchor','e','u'),names)}
    if (r['phase'],r['task'],r['seed'],r['parent'])!=(phase,t,s,parent) or r['dependencies']!=expected:
        raise ValueError('Benchmark source/identity mismatch')
    if r['synthetic_cpu_test_only'] or not r['state_unchanged'] or r['parity']['status']!='PASS':raise ValueError('Benchmark parity missing')
    if r['parity']['samples']!=(32 if phase=='smoke' else 247) or r['cost']['status']!='measured':raise ValueError('Benchmark incomplete')
    if exclusive and not r['cost'].get('observed_exclusive'):raise ValueError('Exclusive measurement unavailable')
    return r

def benchmarks(args,root,phase,gpus):
    from benchmark_worker import gpu_processes
    gpu=args.bench_gpu
    if gpu=='auto':
        free=gpu_inventory();idle=[g for g in gpus if gpu_processes(g).get('other_pids')==[]]
        gpu=max(idle or gpus,key=lambda g:free.get(g,0))
    if gpu not in gpus:raise ValueError('BENCH_GPU must be auto or one of GPU_LIST')
    for t,s,parent in bench_cells(phase=='smoke'):
        c=bench_root(root,phase,args.require_exclusive)/t/('seed%d'%s)/parent
        done=completed(c,True)
        if done:validate_benchmark(done,root,phase,t,s,parent,args.require_exclusive);continue
        disk_check(root,1)
        if gpu_inventory().get(gpu,0)<8192:raise RuntimeError('Benchmark GPU has <8192 MiB free')
        cmd=[python(),'-u',str(HERE/'benchmark_worker.py'),'--root',str(root),'--phase',phase,'--task',t,
             '--seed',str(s),'--parent',parent,'--data',str(Path(args.data).resolve()),'--physical-gpu',gpu]
        if args.require_exclusive:cmd+=['--require-exclusive']
        process(c,cmd,gpu,lambda out:validate_benchmark(out,root,phase,t,s,parent,args.require_exclusive))

def pack(root,partial=False):
    target=root.with_name(root.name+('_partial_return.zip' if partial else '_return.zip'));temp=target.with_name(target.name+'.tmp');entries=[]
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.is_symlink() or p.suffix not in ('.json','.jsonl','.log','.csv','.md','.tex','.bib','.svg','.html'):continue
            data=p.read_bytes();name=p.relative_to(root).as_posix();z.writestr(root.name+'/'+name,data)
            entries.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest()))
        z.writestr('RETURN_CONTENTS.json',json.dumps(entries,indent=2))
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        for r in entries:
            if hashlib.sha256(z.read(root.name+'/'+r['path'])).hexdigest()!=r['sha256']:raise ValueError('Return hash mismatch')
    os.replace(str(temp),str(target));print('ARCHIVE',str(target),'SHA256='+sha(target),flush=True)
    return target

def status(root):
    r=read(root/'runner.json') if (root/'runner.json').exists() else {}
    print('Runner:',r.get('state','not started'),'alive:',alive(r.get('pid')),'phase:',r.get('phase'))
    for phase in ('smoke','formal'):
        jobs=cells(phase=='smoke');n=sum(completed(cell(root,phase,*c)) is not None for c in jobs)
        print('%s controls completed: %d/%d'%(phase,n,len(jobs)))
        for kind in ('benchmark','benchmark_exclusive'):
            js=bench_cells(phase=='smoke')
            count=sum(completed(root/kind/phase/t/('seed%d'%s)/p) is not None for t,s,p in js)
            if kind=='benchmark' or count:print('%s %s replay/timing: %d/%d'%(phase,kind,count,len(js)))
    for file in root.rglob('active.json'):
        if (file.parent/'complete.json').exists():continue
        a=read(file);live=a.get('state')=='running' and alive(a.get('pid'));out=file.parent/a['attempt'];log=out/'train.log';last='-'
        if log.exists():
            with log.open('rb') as stream:stream.seek(max(0,log.stat().st_size-12000));tail=stream.read().decode('utf-8',errors='replace')
            found=re.findall(r'Epoch\[\d+/\d+\]',tail)
            if found:last=found[-1]
        print('RUNNING' if live else 'FAILED/INTERRUPTED',str(file.parent.relative_to(root)),last,'GPU='+a['gpu'],'log='+str(log))
    done=(root/'completion.json').exists();target=root.with_name(root.name+'_return.zip')
    print('Smoke gate:',(root/'smoke_gate.json').exists(),'Completion:',done)
    print('Return ZIP:',target,'READY' if done and target.exists() else 'not complete')
    if r.get('error'):print('ERROR:',r['error'])
    if (root/'paper/gaps.json').exists():print('Paper gaps:',read(root/'paper/gaps.json')['open_items'])
    print('No deletion. SAME root/command resumes verified completed cells; interrupted training restarts only that cell.')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('all','check','storage','status','benchmark','paper','pack'),default='all')
    p.add_argument('--root',default='results/v17_finish_v39')
    p.add_argument('--source',default=os.environ.get('V39_SOURCE_ROOT','/szztdata/ly/exp0_source/PAMFN-v17-transfer-v38/results/v17_transfer_v38'))
    p.add_argument('--data',default=os.environ.get('V39_DATA','/szztdata/ly/exp0_source/shared_data/FS1000'))
    p.add_argument('--gpus',default=os.environ.get('GPU_LIST','4'))
    p.add_argument('--bench-gpu',default=os.environ.get('BENCH_GPU','auto'))
    p.add_argument('--require-exclusive',action='store_true',default=os.environ.get('BENCH_REQUIRE_EXCLUSIVE','0')=='1')
    args=p.parse_args();root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    if args.stage=='status':return status(root)
    if args.stage=='storage':return disk_check(root,len(args.gpus.replace(',',' ').split()))
    if args.stage=='pack':return pack(root,not (root/'completion.json').exists())
    import fcntl
    with (root/'runner.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A launcher is already active')
        for file in root.rglob('active.json'):
            a=read(file)
            if a.get('state')=='running' and alive(a.get('pid')):raise RuntimeError('Existing child still running: '+str(file))
        rec=dict(pid=os.getpid(),state='running',phase='preflight',started=time.time());save(root/'runner.json',rec)
        try:
            if args.stage=='paper':
                code_check()
                from paper_export import build
                build(root)
            else:
                gpus=select_gpus(args.gpus);check(args,root,len(gpus))
                if args.stage=='all':
                    rec['phase']='smoke';save(root/'runner.json',rec);wave(args,root,'smoke',gpus)
                    benchmarks(args,root,'smoke',gpus)
                    save(root/'smoke_gate.json',dict(status='V39_SMOKE_PASS',controls=4,benchmark_replays=4))
                    rec['phase']='formal';save(root/'runner.json',rec);wave(args,root,'formal',gpus)
                if args.stage in ('all','benchmark'):
                    rec['phase']='benchmark';save(root/'runner.json',rec);benchmarks(args,root,'formal',gpus)
                    rec['phase']='paper';save(root/'runner.json',rec)
                    from paper_export import build
                    build(root)
                    save(root/'completion.json',dict(status='EXPERIMENTS_COMPLETE',formal_controls=42,formal_benchmark_replays=42,
                        warning='Engineering completion is not scientific success or submission readiness'))
            rec.update(state='finished',phase='complete',ended=time.time());save(root/'runner.json',rec)
            if args.stage!='check':pack(root,not (root/'completion.json').exists())
        except BaseException as error:
            rec.update(state='failed',error=repr(error),ended=time.time());save(root/'runner.json',rec);traceback.print_exc()
            try:pack(root,True)
            except Exception:traceback.print_exc()
            raise
if __name__=='__main__':main()
