"""Read one frozen checkpoint in its OWN repository and evaluate three fixed views."""
import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def metrics(predictions, labels, logits):
    import numpy as np
    from scipy.stats import spearmanr
    pred, target = np.asarray(predictions), np.asarray(labels)
    if pred.shape != target.shape or not np.isfinite(pred).all():
        raise ValueError('nonfinite prediction or label alignment failure')
    def rank(x):
        if np.unique(x).size < 2 or np.unique(target).size < 2:
            return None
        value = float(spearmanr(x, target).correlation * 100)
        return value if math.isfinite(value) else None
    return dict(coef=rank(pred), mse=float(np.mean((pred-target)**2)),
                prediction_std=float(pred.std()), unique_predictions=int(np.unique(pred).size),
                saturation_fraction=float(np.mean((pred < .001) | (pred > .999))),
                pre_sigmoid_coef=rank(logits), samples=len(pred),
                prediction_std_ratio=float(pred.std()/target.std()) if target.std() else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding='utf-8'))
    out, repo = Path(args.output).resolve(), Path(spec['repo']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.gpu == '-1' else args.gpu
    # Only model/data/pipeline imports come from the specified historical repo.
    sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    import numpy as np
    import torch
    from hbn_stats import (ALPHAS, recipe, isolated_rng, head_bn, collect_head_stats,
                           blended_stats, use_head_stats, score_cached_head)
    entry, pipeline = importlib.import_module('main'), importlib.import_module('pipeline')
    if Path(pipeline.__file__).resolve().parent != repo or Path(entry.__file__).resolve().parent != repo:
        raise RuntimeError('historical repository import isolation failed')
    if spec['recipe'] != recipe():
        raise ValueError('worker recipe differs from manifest')
    torch.set_num_threads(1)
    if args.gpu != '-1' and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA GPU unavailable')
    def progress(phase, done=0, total=0):
        save(out/'progress.json', dict(phase=phase, done=done, total=total, updated_at=time.time()))
        if not total or done == 1 or done == total or done % 20 == 0:
            print('%s %d/%d' % (phase, done, total), flush=True)
    progress('load_checkpoint')
    checkpoint = Path(spec['checkpoint'])
    if digest(checkpoint) != spec['checkpoint_sha256']:
        raise ValueError('source checkpoint hash changed')
    protocol, action, seed, method = spec['cell']
    sys.argv = ['main.py'] + spec['argv'] + ['--test']
    cfg = entry.parser_args()
    current, epochs, data_config, model_config, _, _ = entry.build_configs(cfg, entry.load_config(action, 2))
    pipeline.setup_seed(seed)
    train, test = pipeline.build_dataset(**data_config)
    model = pipeline.build_model(**model_config)
    model.load_state_dict(pipeline._load_state_dict(str(checkpoint)), strict=True)
    model.to(torch.device('cpu' if args.gpu == '-1' else 'cuda')).eval()
    bn = head_bn(model)
    original_buffers = {n: b.clone() for n, b in model.named_buffers()}
    original_versions = {n: p._version for n, p in model.named_parameters()}
    rng = None
    if protocol == 'july28':
        if digest(spec['eval_rng']) != spec['eval_rng_sha256']:
            raise ValueError('July28 original evaluation RNG changed')
        rng = pipeline._load_state_dict(spec['eval_rng'])
    def capture_evaluation(phase):
        cache, preds, logits = [], [], []
        def cache_input(module, inputs):
            cache.append(inputs[0].detach().cpu().clone())
            progress(phase, len(cache), len(test))
        hooks = [bn.register_forward_pre_hook(cache_input),
                 model.fc[-1].register_forward_pre_hook(
                     lambda module, inputs: logits.extend(inputs[0].detach().cpu().reshape(-1).tolist())),
                 model.fc.register_forward_hook(
                     lambda module, inputs, output: preds.extend(output.detach().cpu().reshape(-1).tolist()))]
        try:
            with isolated_rng():
                if protocol == 'july28':
                    from models.score_qmd import set_rng_state
                    set_rng_state(rng[0])
                    np.random.set_state(rng[1])
                    result = pipeline.test_epoch(model, test)
                else:
                    result = pipeline.test_epoch(model, test, eval_policy='argmax', eval_seed=20260908)
        finally:
            for hook in hooks:
                hook.remove()
        return result, cache, preds, logits
    baseline, cache, baseline_pred, baseline_logits = capture_evaluation('baseline_full_test')
    # Match native _prepare_batch's label.float(), including possible float32 ties.
    labels = torch.tensor([label for _, label in test.dataset.labels], dtype=torch.float32).tolist()
    names = [name for name, _ in test.dataset.labels]
    if len(baseline_pred) != len(labels):
        raise ValueError('test sample coverage mismatch')
    computed = metrics(baseline_pred, labels, baseline_logits)
    expected = spec['expected_test']
    for key in ('coef', 'loss_x100', 'prediction_std'):
        if not math.isfinite(baseline[key]) or abs(baseline[key]-expected[key]) > 1e-6:
            raise RuntimeError('original checkpoint replay mismatch: '+key)
    if computed['coef'] is None or abs(computed['coef']-baseline['coef']) > 1e-6:
        raise RuntimeError('cached sample ordering differs from native evaluation')
    if abs(computed['mse']-baseline['loss_x100']/100) > 1e-6:
        raise RuntimeError('cached label precision/scale differs from native evaluation')
    zero_pred, _ = score_cached_head(model, cache)
    if zero_pred != baseline_pred:
        raise RuntimeError('identity cached-head evaluation differs from native output')
    save(out/'baseline.json', computed)
    stats = collect_head_stats(model, train.dataset, pipeline._prepare_batch, progress)
    calibration = {k: v.tolist() if torch.is_tensor(v) else v for k, v in stats.items()}
    calibration.update(original_running_mean=bn.running_mean.detach().cpu().tolist(),
                       original_running_var=bn.running_var.detach().cpu().tolist(), mixed={})
    save(out/'calibration.json', calibration)
    rows, records = [], []
    for alpha in ALPHAS:
        progress('head_alpha_%s' % alpha)
        values = blended_stats(bn, stats, alpha)
        calibration['mixed'][str(alpha)] = {k:v.detach().cpu().tolist() for k,v in values.items()}
        path = out/('alpha_%s.bn.pt' % alpha)
        torch.save({k: v.detach().cpu() for k, v in values.items()}, str(path))
        with use_head_stats(model, values):
            pred, logits = score_cached_head(model, cache)
            result = metrics(pred, labels, logits)
            if args.smoke and alpha > 0:
                _, full_cache, full_pred, _ = capture_evaluation('smoke_full_alpha_%s' % alpha)
                if len(full_cache) != len(cache) or any(not torch.equal(a,b) for a,b in zip(cache,full_cache)):
                    raise RuntimeError('head calibration changed upstream activations or RNG replay')
                if pred != full_pred:
                    raise RuntimeError('cached-head candidate differs from full-network forward')
        # Verify the tiny saved buffer artifact, never save/copy another full model.
        reloaded = pipeline._load_state_dict(str(path))
        with use_head_stats(model, reloaded):
            again, _ = score_cached_head(model, cache)
        if again != pred:
            raise RuntimeError('head statistics save/reload mismatch')
        row = dict(alpha=alpha, delta=None if result['coef'] is None else result['coef']-computed['coef'], **result)
        rows.append(row)
        save(out/('alpha_%s.json' % alpha), row)
        for name, label, score, pre in zip(names, labels, pred, logits):
            records.append(dict(sample=name, label=label, alpha=alpha, prediction=score, pre_sigmoid=pre))
    restored, _ = score_cached_head(model, cache)
    if restored != baseline_pred or rows[0]['delta'] != 0:
        raise RuntimeError('original model not exactly preserved')
    if any(not torch.equal(b, original_buffers[n]) for n,b in model.named_buffers()):
        raise RuntimeError('non-restored model buffers')
    if any(p._version != original_versions[n] for n,p in model.named_parameters()):
        raise RuntimeError('model parameters changed')
    with (out/'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    save(out/'calibration.json', calibration)
    checks = dict(original_replay=True, identity_exact=True, buffers_restored=True,
                  parameters_unchanged=True, tiny_state_reload_exact=True,
                  full_vs_cache_exact=bool(args.smoke), test_labels_used_for_calibration=False)
    save(out/'result.json', dict(cell=spec['cell'], recipe=recipe(), results=rows, checks=checks,
        baseline=computed, total_bn_layers=sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()),
        calibrated_layers=1, checkpoint_sha256=spec['checkpoint_sha256'],
        model_source=str(Path(sys.modules['models.pamfn'].__file__).resolve())))
    progress('finished', 3, 3)
    print('QMD_HBN_CELL_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
