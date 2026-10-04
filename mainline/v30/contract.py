"""v30 evidence on verified v29 checkpoints; original v17 trainer is read-only."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ORIGIN = '1437dd4b9ce3fc220bdf530946a83d4bac88c4f7'
ACTIONS = ('Ball', 'Clubs', 'Hoop', 'Ribbon', 'TES', 'PCS')
SEEDS = (0, 1, 2)
BUDGET = dict(Ball=500, Clubs=100, Hoop=300, Ribbon=20, TES=100, PCS=100)
TRAIN_METHODS = ('base_e', 'base_s', 'raw_e', 'fused_s', 'unbounded_s', 'no_e_s', 'rgb_e')
SMOKE_METHODS = ('repeat_e', 'repeat_s') + TRAIN_METHODS
POLICY = 'v17 historical: online crops; trained-epoch test-best; saved stochastic RNG; no new validation split'
HERE = Path(__file__).resolve().parent
REPO = HERE / 'trainer'
REMAINING = ['recent strong methods under matched inputs/protocol',
             'independent held-out model-selection evidence, separately from historical test-best',
             'external backbone/new feature or new dataset transfer beyond the RGB scorer probe',
             'isolated GPU timing if measured devices are shared']


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
        for data in iter(lambda: f.read(1024*1024), b''): h.update(data)
    return h.hexdigest()


def immutable(path, value):
    if Path(path).exists() and read(path) != value:
        raise ValueError('frozen experiment contract changed; use a new root: ' + str(path))
    save(path, value)


def child(root, name):
    root = Path(root).resolve(); target = (root / name).resolve()
    target.relative_to(root)
    if target == root: raise ValueError('expected child file')
    return target


def exact_predictions(actual, expected, label):
    if actual['predictions'] != expected['predictions'] or abs(actual['coef']-expected['coef']) > 1e-6:
        raise ValueError(label + ' predictions do not match')


def verify_record(directory):
    directory=Path(directory);record=read(directory/'complete.json')
    for name,digest in record['files'].items():
        if sha(child(directory,name))!=digest:raise ValueError('record file changed: '+name)
    return record


def source_from(record):
    row = record['metrics']
    return dict(action=row['action'], seed=row['seed'], checkpoint_sha256=record['checkpoint_sha256'],
                eval_rng_sha256=record['eval_rng_sha256'], expected_srcc_x100=row['srcc_x100'])


def build_references(origin):
    """Resolve active v29 attempts; do not recover/search/substitute historical weights."""
    origin = Path(origin).resolve()
    done, contract = read(origin/'completion.json'), read(origin/'contract.json')
    if not done.get('experiments_complete') or done.get('formal_cells') != 72:
        raise ValueError('v29 formal 72/72 required')
    if contract['code_commit'] != ORIGIN or contract['seeds'] != list(SEEDS):
        raise ValueError('not the preserved v17 rebuild')
    old = contract['source']; rows = []
    for action in ACTIONS:
        for seed in SEEDS:
            row = dict(action=action, seed=seed)
            for name, method in [('base','base'), ('ordered','ordered'),
                                 ('e','ordered_modal_fused_gt'), ('s','e_modal_scaled')]:
                cell = origin/'screen'/'july28'/action/('seed%d'%seed)/method
                directory = child(cell, read(cell/'active.json')['attempt'])
                rec = read(directory/'complete.json')
                metric = rec['metrics']
                if (metric['action'],metric['seed'],metric['method'],metric['epochs']) != (action,seed,method,BUDGET[action]):
                    raise ValueError('wrong v29 cell: '+str(directory))
                for n,digest in rec['files'].items():
                    if sha(child(directory,n)) != digest: raise ValueError('v29 record changed: '+str(directory/n))
                for n,key in [('model.pth','checkpoint_sha256'),('model.pth.eval_rng.pt','eval_rng_sha256')]:
                    if sha(directory/n) != rec[key]: raise ValueError('missing/changed v29 asset: '+str(directory/n))
                row[name] = dict(record=str(directory/'complete.json'),record_sha256=sha(directory/'complete.json'),
                                 source=source_from(rec),test=read(directory/'test_result.json'),
                                 model=str(directory/'model.pth'),rng=str(directory/'model.pth.eval_rng.pt'))
            rows.append(row)
            print('V29 REFERENCE VERIFIED %d/18: %s seed%d'%(len(rows),action,seed),flush=True)
    return old, rows


def provenance(old):
    import numpy, scipy, torch, timm
    env=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
             numpy=numpy.__version__,scipy=scipy.__version__,timm=timm.__version__)
    if env != old['environment']: raise ValueError('environment differs from v17/v29')
    def git(*args): return subprocess.check_output(['git']+list(args),cwd=str(REPO),text=True).strip()
    if git('rev-parse','HEAD') != ORIGIN: raise ValueError('wrong original trainer commit')
    if git('status','--porcelain','--untracked-files=no'): raise ValueError('original tracked trainer files changed')
    if any(p.endswith('.py') for p in git('ls-files','--others','--exclude-standard').splitlines()):
        raise ValueError('untracked Python injected into original trainer')
    for relative,expected in old['inputs'].items():
        # Deliberate shared data symlinks resolve outside the trainer; paths come from the verified contract.
        path=REPO/relative
        if dict(bytes=path.stat().st_size,sha256=sha(path)) != expected:
            raise ValueError('v17 input mismatch: '+relative)
        print('INPUT VERIFIED '+relative,flush=True)
    code={p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.sh')}
    return dict(origin_commit=ORIGIN,environment=env,inputs=old['inputs'],evidence_code=code,
                selection_policy=POLICY,remaining_evidence=list(REMAINING))


def cells(methods=TRAIN_METHODS, smoke=False):
    actions,seeds=(('Ball','TES'),(0,)) if smoke else (ACTIONS,SEEDS)
    return [(a,s,m) for m in methods for a in actions for s in seeds]


def cell_dir(root, phase, cell):
    a,s,m=cell
    return Path(root)/phase/a/('seed%d'%s)/m


def aggregate(rows):
    result=[]
    for method in sorted(set(r['method'] for r in rows)):
        subset=[r for r in rows if r['method']==method]
        if len(subset)!=18 or {(r['action'],r['seed']) for r in subset}!={(a,s) for a in ACTIONS for s in SEEDS}:
            continue
        seeds=[]
        for seed in SEEDS:
            table={r['action']:float(r['srcc_x100']) for r in subset if r['seed']==seed}
            if not all(math.isfinite(x) and -100<=x<=100 for x in table.values()): raise ValueError('invalid SRCC')
            def fisher(keys):
                return 100*math.tanh(sum(math.atanh(max(-1+1e-7,min(1-1e-7,table[k]/100))) for k in keys)/len(keys))
            seeds.append(dict(seed=seed,six_macro=sum(table.values())/6,RG_fisher=fisher(ACTIONS[:4]),
                              FisV_fisher=fisher(ACTIONS[4:]),actions=table))
        mean=sum(v['six_macro'] for v in seeds)/3
        result.append(dict(method=method,six_macro=mean,sample_sd_seed_means=(sum((v['six_macro']-mean)**2 for v in seeds)/2)**.5,
                           RG_fisher=sum(v['RG_fisher'] for v in seeds)/3,FisV_fisher=sum(v['FisV_fisher'] for v in seeds)/3,
                           per_seed=seeds))
    return result
