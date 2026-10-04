"""v17 evidence worker. Original train_branch functions are called without edits."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from models.score_qmd import rng_state
from contract import (read, save, sha, exact_predictions, source_from, TRAIN_METHODS, SMOKE_METHODS, POLICY, REPO)
from tools.train_qmd_residual import configurations, load, metrics
from tools.qmd_residual import freeze_anchor, preserved_rng, state_digest
from tools.qmd_modal import ModalBranch, make_cache
from tools.qmd_supp import SupplementBranch, e_score
from tools.qmd_supp_config import E_TAG, SCALED
from tools.train_qmd_supp import load_e
from tools import train_qmd_supp as original_s
from tools import train_qmd_modal as original_e
from model import ControlledSupplement, ZeroE, predict_views, diagnose, RGBBranch, control_context


def profile(anchor, e, supplement, test_loader, device, gpu, base_anchor):
    """Wall clock synchronized GPU latency; frozen checkpoint, three stated videos."""
    import pipeline
    measurements = []
    def occupancy():
        try:
            text = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
            pids = [int(x.strip()) for x in text.splitlines() if x.strip().isdigit()]
            return dict(pids=pids, shared=any(p != os.getpid() for p in pids))
        except (OSError, subprocess.SubprocessError): return dict(pids=[], shared=None)
    before = occupancy()
    chosen = {0, len(test_loader) // 2, len(test_loader) - 1}
    with preserved_rng():
        for index, batch in enumerate(test_loader):
            if index not in chosen: continue
            inputs, _ = pipeline._prepare_batch(batch, device)
            for name in ('base', 'ordered', 'e', 'v17'):
                # Exclude unloaded correction heads from each view's GPU allocation.
                e.to('cpu'); supplement.to('cpu');base_anchor.to('cpu');anchor.to('cpu')
                if name=='base':base_anchor.to(device)
                else:anchor.to(device)
                if name in ('e','v17'): e.to(device)
                if name == 'v17': supplement.to(device)
                def forward():
                    if name == 'base': return base_anchor(inputs)[0]
                    if name == 'ordered': return anchor(inputs)[0]
                    from tools.qmd_modal import modal_view
                    base, packed = modal_view(anchor, inputs)
                    pred = e_score(e, base, packed)
                    return pred if name == 'e' else pred + supplement(packed).double()
                with torch.no_grad():
                    for _ in range(5): forward()
                    if device.type == 'cuda':
                        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                    times = []
                    for _ in range(20):
                        if device.type == 'cuda': torch.cuda.synchronize()
                        start = time.perf_counter(); forward()
                        if device.type == 'cuda': torch.cuda.synchronize()
                        times.append((time.perf_counter() - start) * 1000)
                measurements.append(dict(method=name, test_loader_index=index, input_shapes={k:list(v.shape) for k,v in inputs.items()},
                    warmup=5, repeats=20, latency_median_ms=float(np.median(times)), latency_p90_ms=float(np.percentile(times, 90)),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None))
    base_anchor.to('cpu');anchor.to(device);e.to(device); supplement.to(device)
    return dict(parameters={k:sum(p.numel() for p in m.parameters()) for k,m in [('base',base_anchor),('ordered',anchor),('e_head',e),('supplement',supplement)]},
                measurements=measurements, occupancy_before=before, occupancy_after=occupancy(),
                device=str(device), gpu_name=torch.cuda.get_device_name() if device.type=='cuda' else 'CPU',
                scope='inference only, batch1, three recorded videos, excludes feature extraction and data IO; shared GPU is not isolated timing')


def run_impl(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.gpu == '-1' else args.gpu
    if args.gpu != '-1' and not torch.cuda.is_available(): raise RuntimeError('requested GPU unavailable')
    import pipeline
    manifest = read(args.manifest)
    if args.method!='replay' and args.method not in SMOKE_METHODS:
        raise ValueError('worker method is not in the frozen experiment plan: '+args.method)
    ref = next(r for r in manifest['references'] if (r['action'],r['seed'])==(args.action,args.seed))
    base_mode = args.method in ('base_e', 'base_s', 'rgb_e')
    teacher = ref['base' if base_mode else 'ordered']; source = teacher['source']
    for key, expected in [('model', source['checkpoint_sha256']), ('rng', source['eval_rng_sha256'])]:
        if sha(teacher[key]) != expected: raise ValueError('teacher changed: ' + teacher[key])
    args.epochs = 2 if args.smoke else None
    pipeline.setup_seed(args.seed)
    _, epochs, dataset_cfg, model_cfg, _, _ = configurations(args)
    if base_mode:
        model_cfg['model_conf']['ordinal_prototype_weight'] = 0.
        model_cfg['model_conf']['ordinal_quality_thresholds'] = ()
    loader, test_loader = pipeline.build_dataset(**dataset_cfg)
    if args.method=='rgb_e':
        rgb_relative='pretrained_models/feats2/%s_rgb_UMT.pth'%args.action
        rgb_path=REPO/rgb_relative
        rgb_sha=manifest['source']['inputs'][rgb_relative]['sha256']
        if sha(rgb_path)!=rgb_sha:raise ValueError('RGB pretrained scorer changed')
        anchor=load(rgb_path)
        if not isinstance(anchor,torch.nn.Module) or anchor.modality!='V':raise ValueError('expected pretrained RGB scorer')
        source=dict(source,checkpoint_sha256=rgb_sha,architecture='RGB_only_BaseModel')
        teacher=dict(teacher,model=str(rgb_path),source=source)
    else:
        anchor = pipeline.build_model(**model_cfg)
        anchor.load_state_dict(pipeline._load_state_dict(teacher['model']), strict=True)
    device = torch.device('cpu' if args.gpu == '-1' else 'cuda')
    freeze_anchor(anchor); anchor.to(device)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    training_e = args.method in ('repeat_e','base_e','raw_e','rgb_e')
    if training_e:
        branch = (RGBBranch(seed=args.seed) if args.method=='rgb_e' else ModalBranch(E_TAG, seed=args.seed)).to(device)
    else:
        if base_mode:
            prior = output.parents[2] / 'base_e' / 'active.json'
            prior_dir = Path(read(prior)['directory'])
            record = read(prior_dir/'complete.json')
            if sha(prior_dir/'model.pth') != record['checkpoint_sha256']: raise ValueError('base E dependency changed')
            saved = load(prior_dir/'model.pth')
            if saved['source'] != source or saved['tag'] != E_TAG: raise ValueError('wrong Base E dependency')
            e = ModalBranch(E_TAG, seed=args.seed).to(device); e.load_state_dict(saved['branch'], strict=True); freeze_anchor(e)
            e_reference = read(prior_dir/'test_result.json')
            e_source = dict(checkpoint_sha256=record['checkpoint_sha256'], expected_srcc_x100=e_reference['coef'],
                            eval_rng_sha256=source['eval_rng_sha256'], action=args.action, seed=args.seed)
        else:
            args.e_model=ref['e']['model']; args.e_record=ref['e']['record']
            e,e_source,e_reference = load_e(args,source,device)
        if args.method == 'no_e_s':
            e = freeze_anchor(ZeroE()).to(device); e_reference = teacher['test']; e_source = dict(removed=True)
        branch = (ControlledSupplement(args.method, seed=args.seed) if args.method in ('fused_s','unbounded_s')
                  else SupplementBranch(SCALED,seed=args.seed)).to(device)
    if args.evaluate:
        if sha(output/'model.pth.eval_rng.pt')!=source['eval_rng_sha256']:
            raise ValueError('saved branch evaluation RNG changed')
        replay=load(output/'model.pth.eval_rng.pt')
    else:
        replay = load(teacher['rng'])
    if args.method == 'replay':
        if sha(ref['s']['model']) != ref['s']['source']['checkpoint_sha256']:
            raise ValueError('retained v17 supplement checkpoint changed')
        saved = load(ref['s']['model'])
        if saved['source'] != dict(ordered=source,e=e_source) or saved['config'] != branch.config or saved['tag'] != SCALED:
            raise ValueError('not the retained v17 Scaled checkpoint')
        branch.load_state_dict(saved['branch'],strict=True)
        if original_e.statistics_digest(branch) != saved['statistics_sha256']: raise ValueError('v17 statistics changed')
    elif args.evaluate:
        saved=load(output/'model.pth')
        if saved['config']!=branch.config:raise ValueError('reload branch configuration changed')
        branch.load_state_dict(saved['branch'],strict=True)
        if original_e.statistics_digest(branch)!=saved['statistics_sha256']:raise ValueError('reload statistics changed')
    else:
        # Same order as historic entry points; preservation restores loader/model RNG.
        train_cache = make_cache(anchor,loader,device,(rng_state(),np.random.get_state()))
        statistics = branch.fit_training_statistics(train_cache) if training_e else branch.fit_training_statistics(e_score(e,train_cache[0],train_cache[1]),train_cache[1])
        del train_cache
    cache = make_cache(anchor,test_loader,device,replay)
    if args.method=='rgb_e':
        teacher['test']=metrics(cache[0],cache[2]);source['expected_srcc_x100']=teacher['test']['coef']
        save(output/'rgb_baseline.json',dict(teacher['test'],source=source,
             note='one fixed public RGB scorer per task; only E is trained with seed0/1/2; not three retrained RGB anchors'))
    exact_predictions(metrics(cache[0],cache[2]),teacher['test'],'frozen main scorer')
    if not training_e: exact_predictions(metrics(e_score(e,cache[0],cache[1]),cache[2]),e_reference,'frozen E')
    if args.method == 'replay':
        result = original_s.evaluate(e,branch,cache)
        exact_predictions(result,ref['s']['test'],'v17')
        save(output/'diagnostics.json',diagnose(e,branch,cache))
        # Stochastic inference draws, NOT extra training seeds and NOT selectable checkpoints.
        draws = []
        for draw in range(10):
            with preserved_rng():
                pipeline.setup_seed(170000 + 100 * args.seed + draw)
                draw_cache = make_cache(anchor,test_loader,device,(rng_state(),np.random.get_state()))
            draws.append(dict(draw=draw,ordered=metrics(draw_cache[0],draw_cache[2])['coef'],
                              e=metrics(e_score(e,draw_cache[0],draw_cache[1]),draw_cache[2])['coef'],
                              v17=original_s.evaluate(e,branch,draw_cache)['coef']))
            del draw_cache
        import math
        for draw in draws:
            for key in ('ordered','e','v17'):
                if not math.isfinite(draw[key]):draw[key]=None
        save(output/'inference_draws.json',dict(kind='same selected checkpoint, 10 fixed inference draws, not training seeds',
             undefined='null means undefined SRCC, not zero; do not silently drop undefined draws',rows=draws))
        del cache
        if device.type=='cuda': torch.cuda.empty_cache()
        import copy
        base_cfg=copy.deepcopy(model_cfg)
        base_cfg['model_conf'].update(ordinal_prototype_weight=0.,ordinal_quality_thresholds=())
        base_anchor=pipeline.build_model(**base_cfg)
        base_anchor.load_state_dict(pipeline._load_state_dict(ref['base']['model']),strict=True)
        freeze_anchor(base_anchor)
        save(output/'cost.json',profile(anchor,e,branch,test_loader,device,args.gpu,base_anchor))
        save(output/'test_result.json',result)
        return
    if args.evaluate:
        if saved['source'] != (source if training_e else dict(ordered=source,e=e_source)):
            raise ValueError('reloaded dependency source changed')
        result=original_e.evaluate(branch,cache) if training_e else original_s.evaluate(e,branch,cache)
        exact_predictions(result,read(output/'test_result.json'),'fresh-process reload')
        save(output/'reload_result.json',result)
        return
    if training_e:
        result = original_e.train_branch(anchor,branch,loader,cache,epochs,E_TAG,args.seed,args.smoke,
                                         output,source,teacher['rng'],statistics)
    else:
        result = original_s.train_branch(anchor,e,branch,loader,cache,epochs,SCALED,args.seed,args.smoke,
                                         output,dict(ordered=source,e=e_source),teacher['rng'],statistics)
    save(output/'train_result.json',dict(result,paper_method=args.method,policy=POLICY,
         ablation='raw MSE replaces centered MSE; original mean penalty unchanged' if args.method=='raw_e' else None))
    saved = load(output/'model.pth'); branch.load_state_dict(saved['branch'],strict=True)
    test = original_e.evaluate(branch,cache) if training_e else original_s.evaluate(e,branch,cache)
    if abs(test['coef']-result['saved_model_srcc']*100)>1e-6: raise ValueError('saved checkpoint score mismatch')
    save(output/'test_result.json',test)
    if not args.smoke and args.method in ('repeat_e','repeat_s'):
        original = ref['e' if training_e else 's']
        exact_predictions(test,original['test'],'original-recipe retraining')
        old_state = load(original['model'])['branch']
        if state_digest(branch) != state_digest_from_dict(branch,old_state): raise ValueError('retrained weights differ despite matching scores')
        old_dir=Path(original['record']).parent
        old_selection=read(old_dir/'model.pth.selection.json')
        if result['selected_epoch']!=old_selection['epoch']: raise ValueError('retrained selected epoch differs')
        save(output/'parity.json',dict(predictions_exact=True,branch_state_exact=True,policy=POLICY))


def state_digest_from_dict(branch,state):
    import copy
    clone=copy.deepcopy(branch); clone.load_state_dict(state,strict=True)
    return state_digest(clone)


def run(args):
    with control_context(args.method):run_impl(args)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',required=True); p.add_argument('--manifest',required=True)
    p.add_argument('--action',required=True); p.add_argument('--seed',type=int,required=True)
    p.add_argument('--method',choices=('replay',)+SMOKE_METHODS,required=True)
    p.add_argument('--output',required=True); p.add_argument('--smoke',action='store_true')
    p.add_argument('--evaluate',action='store_true')
    args=p.parse_args(); run(args)


if __name__=='__main__': main()
