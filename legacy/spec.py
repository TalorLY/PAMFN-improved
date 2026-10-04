"""Predeclared FS1000 transfer study. Existing PAMFN and v37 artifacts are read-only."""
import hashlib
import json
import os
from pathlib import Path
import uuid

HERE = Path(__file__).resolve().parent
TASKS = ('TES', 'PCS', 'SS', 'TR', 'PE', 'CO', 'IN')
SEEDS = (0, 1, 2)
NEW_METHODS = ('mc_ordered', 'mc_e', 'mc_u', 'mc_ordered_e', 'mc_ordered_u')
METHODS = NEW_METHODS
SCORED = ('mcmoe',) + NEW_METHODS
FEATURES = ('output_feature_fs1000_new', 'ast_feature_fs1000_new', 'i3d_avg_clip8_5s_fs1000')
RANGES = dict(zip(TASKS, (130., 60., 10., 10., 10., 10., 10.)))
MC_EPOCHS = dict(zip(TASKS, (360, 460, 360, 210, 520, 520, 390)))
SOURCE_COMMIT = '302460fc8a18229c67a1689301d346e0fa3ec34d'
POLICY = dict(version=38, dataset='FS1000', train=1000, evaluation=247,
    tasks=list(TASKS), seeds=list(SEEDS), methods=list(SCORED), new_formal_cells=105,
    source='v37 MCMoE checkpoints, predictions and RNG read-only; no weight/feature copies',
    input_dimensions=[768,768,1024], rgb_pool='numpy float32 mean over 15 spatial tokens',
    train_crop=95, test_crop='center95', alignment='same v37 fs_data.py byte-for-byte',
    batch_size=32, loader_workers=2, test_batch_size=1,
    selection='trained-epoch evaluation-best SRCC; first tie; no epoch-zero fallback',
    test_split_role='official val used for epoch selection and reporting; not blind test',
    optimizer='MCMoE original Adam lr1e-4 wd1e-4, cosine eta_min1e-6',
    main_epochs=MC_EPOCHS, branch_epochs=500, new_hyperparameter_search=False,
    ordered='train-only quartiles; 256->128 projector, four prototypes, learned axis; loss weight1e-4, temperature/margin/order_scale all0.1',
    ordered_input='native decoder query embeddings mean over queries; gradients reach backbone',
    ordered_initialization='private CPU RNG; does not change native backbone initialization or loader RNG',
    original_mcmoe_losses='unchanged MSE*10 + original triplet + reconstruction; Ordered added, not substituted',
    e='frozen native eval decoder mean query (256D); 16-hidden tanh residual bounded0.03; train-only bias and feature standardization',
    u='frozen pre-fusion per-modality encoder mean (V,F,A; 3x256D); three 16-hidden heads; v30 unbounded output; fixed mean',
    branch_objective='centered score MSE + 0.1 residual mean squared; E float32, U float64; no KD',
    branch_optimizer='AdamW lr1e-4 wd0.01; 500 epochs; random train crops, no persistent train-feature cache',
    u_scale='min(0.003,0.1*training E prediction std); not a hard output bound',
    no_learned_gate=True, frozen_parents=True, labels_used_for_calibration='train only',
    protocol_scope='matched within each parent/child pair; E/U extra training explicitly counted, not equal total compute',
    historical_rg_fisv='unchanged; 81.277 and retained +1.549 not overwritten',
    undefined_srcc='finite constant prediction => diagnostic null and ineligible epoch; max10 consecutive or20 total then fail; nonfinite prediction/loss/gradient fatal',
    aggregation='SRCC x100 per score/seed; seven-score arithmetic macro auxiliary; TES/PCS separately',
    interpretation='cross-model adaptation, not exact PAMFN features; no guaranteed gain or missing-modality claim')

def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): value.update(block)
    return value.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(str(temp), str(path))

def cells(smoke=False):
    return [(t,s,m) for t in (('TES','PCS') if smoke else TASKS)
            for s in ((0,) if smoke else SEEDS) for m in NEW_METHODS]

def epochs(method, task, smoke=False):
    if method not in SCORED or task not in TASKS: raise ValueError('Unknown method/task')
    return 2 if smoke else MC_EPOCHS[task] if method in ('mcmoe','mc_ordered') else 500

def cell_path(root, phase, task, seed, method):
    if phase not in ('formal','smoke') or task not in TASKS or seed not in SEEDS or method not in SCORED:
        raise ValueError('Unknown cell')
    return Path(root)/phase/task/('seed%d'%seed)/method

def completed(path, verify=False):
    path=Path(path)
    if not (path/'complete.json').is_file(): return None
    value=read(path/'complete.json'); attempt=value['attempt']
    if Path(attempt).name!=attempt or '\\' in attempt or attempt in ('.','..'): raise ValueError('Unsafe attempt')
    out=path/attempt
    out.resolve().relative_to(path.resolve())
    if verify:
        for name,digest in value['files'].items():
            if Path(name).name!=name or '\\' in name: raise ValueError('Unsafe artifact path')
            file=out/name
            if file.is_symlink() or not file.is_file() or sha(file)!=digest:
                raise ValueError('Completed artifact changed: '+str(file))
    return out

def dependencies(method):
    return {'mc_ordered':(), 'mc_e':('mcmoe',), 'mc_u':('mcmoe','mc_e'),
            'mc_ordered_e':('mc_ordered',), 'mc_ordered_u':('mc_ordered','mc_ordered_e')}[method]

def anchor_method(method):
    return 'mc_ordered' if method.startswith('mc_ordered') else 'mcmoe'

def reference_path(root,phase,task,seed):
    return Path(root)/'references'/phase/task/('seed%d'%seed)/'mcmoe'

def artifact(root,phase,task,seed,method):
    if method=='mcmoe':
        record=read(reference_path(root,phase,task,seed)/'record.json')
        out=Path(record['directory'])
        for name,digest in record['required_files'].items():
            if not (out/name).is_file() or sha(out/name)!=digest: raise ValueError('v37 reference changed: '+str(out/name))
        return out
    out=completed(cell_path(root,phase,task,seed,method),True)
    if out is None: raise RuntimeError('Dependency is unfinished: '+method)
    return out
