"""v39 predeclared controls and read-only v38 provenance."""
from pathlib import Path
import bootstrap
from spec import TASKS, SEEDS, SCORED, POLICY as V38_POLICY, read, save, sha, completed

HERE = bootstrap.HERE
METHODS = ('mc_plain_e', 'mc_fused_u')
POLICY = dict(version=39, parent_version=38, dataset='FS1000', tasks=list(TASKS), seeds=list(SEEDS),
    new_methods=list(METHODS), formal=42, smoke=4, epochs=500, smoke_epochs=2,
    inherited=V38_POLICY,
    plain_e='Same E initialization, 256->16->1 MLP, train-only bias/standardization/clipping and multiplier0.03; remove tanh and use ordinary float32 score MSE, no centering/mean penalty. Composite conventional residual comparator, not a one-factor ablation.',
    fused_u='Same U architecture, initialization, optimizer, loss, scale and 500 epochs; replace V/F/A by three copies of frozen fused256 feature. Same12384 parameters; isolates information source.',
    comparison_pairs={'mc_plain_e':'mc_e', 'mc_fused_u':'mc_u'},
    selection=V38_POLICY['selection'], test_split_role=V38_POLICY['test_split_role'],
    inference='No training changes. Persistent hooks and once-per-session metadata/checks; no altered native forward or mixed precision. Require bitwise same-shape replay and reference-batch replay.',
    cost='Benchmark both parents and both legacy/deployment paths after training; serial jobs, raw CUDA-event and synchronized wall times. Do not claim exclusive when other compute processes are observed.',
    no_new_modules=True, old_roots_read_only=True, no_disk_feature_cache=True,
    historical_base='81.277 matched RG/FisV Base and all old training/selection rules untouched')

def cells(smoke=False):
    return [(t,s,m) for t in (('TES','PCS') if smoke else TASKS)
            for s in ((0,) if smoke else SEEDS) for m in METHODS]

def budget(smoke=False): return 2 if smoke else 500

def cell(root, phase, task, seed, method):
    if phase not in ('smoke','formal') or task not in TASKS or seed not in SEEDS or method not in METHODS:
        raise ValueError('Unknown v39 cell')
    return Path(root)/phase/task/('seed%d'%seed)/method

def ref_file(root,phase,task,seed,method):
    if phase not in ('smoke','formal') or task not in TASKS or seed not in SEEDS or method not in SCORED:
        raise ValueError('Unknown reference')
    return Path(root)/'references'/phase/task/('seed%d'%seed)/method/'record.json'

def reference(root,phase,task,seed,method,verify=True):
    record=read(ref_file(root,phase,task,seed,method)); folder=Path(record['directory'])
    if verify:
        for name,digest in record['files'].items():
            p=folder/name
            if p.is_symlink() or not p.is_file() or sha(p)!=digest:
                raise ValueError('Read-only reference changed: '+str(p))
    return folder

def prepare(source,root,data,environment):
    source=Path(source).resolve(); root=Path(root).resolve()
    if source==root or source in root.parents or root in source.parents:
        raise ValueError('Results must not overlap the read-only v38 source')
    old=read(source/'manifest.json')
    if old['policy']!=V38_POLICY or read(source/'completion.json').get('new_formal')!=105:
        raise ValueError('Expected complete, unchanged v38 contract')
    for k in ('torch','numpy','scipy','cuda'):
        if environment[k]!=old['environment'][k]:raise ValueError('v38 environment mismatch: '+k)
    if sha(bootstrap.LEGACY/'FS1000_INPUT_SHA256SUMS')!=old['input_manifest_sha256']:
        raise ValueError('FS1000 input manifest mismatch')
    records=[]
    for phase in ('smoke','formal'):
        for t in (('TES','PCS') if phase=='smoke' else TASKS):
            for s in ((0,) if phase=='smoke' else SEEDS):
                for m in SCORED:
                    if m=='mcmoe':
                        r=read(source/'references'/phase/t/('seed%d'%s)/m/'record.json')
                        folder=Path(r['directory']); expected=r['required_files']
                    else:
                        c=source/phase/t/('seed%d'%s)/m
                        folder=completed(c,True)
                        if folder is None:raise ValueError('Missing v38 cell '+str(c))
                        expected=read(c/'complete.json')['files']
                    names=['model.pth','test_result.json']
                    if m in ('mcmoe','mc_ordered'):names.append('model.pth.eval_rng.pt')
                    files={}
                    for name in names:
                        p=folder/name
                        if p.is_symlink() or sha(p)!=expected[name]:raise ValueError('Source hash changed '+str(p))
                        files[name]=expected[name]
                    result=read(folder/'test_result.json')
                    if (result['phase'],result['task'],result['seed'],result['method'])!=(phase,t,s,m):
                        raise ValueError('Reference identity mismatch')
                    records.append(dict(phase=phase,task=t,seed=s,method=m,directory=str(folder.resolve()),files=files))
                print('READ_ONLY_REFERENCES %d/138: %s/%s/seed%d'%(len(records),phase,t,s),flush=True)
    manifest=dict(policy=POLICY,environment=environment,source_root=str(source),data_root=str(Path(data).resolve()),
        source_manifest_sha256=sha(source/'manifest.json'),code_sha256=sha(HERE/'SHA256SUMS'),references=records)
    if (root/'manifest.json').exists() and read(root/'manifest.json')!=manifest:
        raise ValueError('Immutable v39 run contract changed; refuse mixed results')
    save(root/'manifest.json',manifest)
    for r in records:
        dst=ref_file(root,r['phase'],r['task'],r['seed'],r['method'])
        save(dst,r)
        save(dst.parent/'test_result.json',read(Path(r['directory'])/'test_result.json'))
    return manifest
