"""Strict paired screen summary; incomplete cells never produce a pass."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

ACTIONS = ('TES', 'PCS', 'Ball', 'Clubs', 'Hoop', 'Ribbon')
SEEDS = (0, 1, 42)
METHODS = ('mask_gt', 'score_qmd')


def budget(action):
    from configs import RG_feats2, FISV_feats2
    config = FISV_feats2 if action in ('TES', 'PCS') else RG_feats2
    return config.multi_modality['epoch'][action]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_cell(path, action, seed, mode, epochs=None):
    path = Path(path)
    train, test = read_json(path / 'train_result.json'), read_json(path / 'test_result.json')
    for record in (train, test):
        run = record['run']
        if (run['action'], run['seed'], run['tag']) != (action, seed, mode):
            raise ValueError('run identity mismatch: ' + str(path))
        if run['eval_policy'] != 'argmax' or run['eval_seed'] != 20260908:
            raise ValueError('evaluation protocol mismatch')
        if run['feats'] != 2 or run['coqm_weight'] != 0 or run['ordinal_prototype_weight'] != 0:
            raise ValueError('unexpected baseline or auxiliary method')
        if epochs is not None and run['epochs'] != epochs:
            raise ValueError('epoch budget mismatch')
    evaluation = test['evaluation']
    if (evaluation['policy'] != 'argmax' or evaluation['seed'] != 20260908
            or evaluation['passes'] != 1 or not evaluation['rng_isolated']):
        raise ValueError('evaluation RNG mismatch')
    if train['run'] != test['run']:
        raise ValueError('train/test config mismatch')
    if not math.isclose(train['saved_model_srcc'] * 100, test['coef'], abs_tol=1e-8):
        raise ValueError('checkpoint reload mismatch')
    metrics = dict(srcc=test['coef'], mse=test['loss_x100']/100, mae=test['mae'],
                   bias=test['bias'], std_ratio=test['prediction_std_ratio'])
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError('non-finite result metrics')
    if mode in METHODS:
        short_smoke = test['run']['epochs'] == 1
        expected = dict(mode=mode, weight=1.0, mask_ratio=0.15, mask_gt_weight=0.5,
                        target_ratio=0.075, beta_cap=1.0,
                        warmup=0.0 if short_smoke else 0.1,
                        rampup=0.0 if short_smoke else 0.1)
        if test['run']['qmd'] != expected:
            raise ValueError('QMD configuration mismatch')
        qmd = read_json(path / 'qmd_training.json')
        if qmd['active_batches'] <= 0:
            raise ValueError('QMD never activated')
        if mode == 'score_qmd':
            if not qmd['calibration'] or qmd['calibration']['degenerate']:
                raise ValueError('score distillation calibration was degenerate')
            if qmd['calibration']['beta'] <= 0:
                raise ValueError('score distillation was not trained')
    return dict(action=action, seed=seed, method=mode, **metrics)


def aggregate(rows):
    expected = {(a,s,m) for a in ACTIONS for s in SEEDS for m in ('base',)+METHODS}
    index = {(r['action'], r['seed'], r['method']):r for r in rows}
    if len(rows) != len(expected) or set(index) != expected:
        raise ValueError('expected exactly 54 unique paired rows (18 Base + 36 new)')
    macro = {m: statistics.mean(r['srcc'] for r in rows if r['method']==m)
             for m in ('base',)+METHODS}
    actions = []
    for a in ACTIONS:
        item = dict(action=a)
        for m in ('base',)+METHODS:
            values = [index[a,s,m]['srcc'] for s in SEEDS]
            item[m] = statistics.mean(values)
            item[m+'_sd'] = statistics.stdev(values)
        for m in METHODS:
            item[m+'_delta_pp'] = item[m]-item['base']
        item['kd_minus_mask_gt_pp'] = item['score_qmd']-item['mask_gt']
        actions.append(item)
    checks = {m: dict(delta_pp=macro[m]-macro['base'],
                     positive_actions=sum(r[m+'_delta_pp'] > 0 for r in actions),
                     paired_wins=sum(index[a,s,m]['srcc'] > index[a,s,'base']['srcc']
                                     for a in ACTIONS for s in SEEDS),
                     ball_delta_pp=next(r[m+'_delta_pp'] for r in actions if r['action']=='Ball'))
              for m in METHODS}
    viable = [m for m in METHODS if checks[m]['delta_pp'] >= 0.4
              and checks[m]['positive_actions'] >= 4 and checks[m]['ball_delta_pp'] > -1.0]
    selected = max(viable, key=lambda m:macro[m]) if viable else None
    decision = ('SCORE_QMD_SIGNAL' if selected == 'score_qmd' and macro['score_qmd'] > macro['mask_gt']
                else 'MASK_GT_SIGNAL' if selected == 'mask_gt' else 'QMD_NO_CLEAR_GAIN')
    return dict(decision=decision, macro_srcc=macro, checks=checks, selected_method=selected,
                score_qmd_minus_mask_gt_pp=macro['score_qmd']-macro['mask_gt'],
                screening_only=True, actions=actions)


def summarize(root):
    root = Path(root)
    rows = []
    for a in ACTIONS:
        for s in SEEDS:
            base = read_json(root/'baseline_reference'/a/f'seed{s}'/'metrics.json')
            rows.append(base)
            for m in METHODS:
                rows.append(validate_cell(root/a/f'seed{s}'/m, a, s, m, budget(a)))
    result = aggregate(rows)
    report = root/'report'
    report.mkdir(exist_ok=True)
    for name, data in [('runs', rows), ('actions', result['actions'])]:
        with (report/(name+'.csv')).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (report/'decision.json').write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    text = '# Score-QMD 六动作三种子筛选\n\n' + result['decision'] + '\n\n'
    text += '| 动作 | Base | Mask-GT | Score-QMD | QMD−Base pp | QMD−Mask-GT pp |\n'
    text += '|---|---:|---:|---:|---:|---:|\n'
    for r in result['actions']:
        text += ('| {action} | {base:.3f} | {mask_gt:.3f} | {score_qmd:.3f} | '
                 '{score_qmd_delta_pp:+.3f} | {kd_minus_mask_gt_pp:+.3f} |\n').format(**r)
    text += '\nMacro SRCC ×100: ' + json.dumps(result['macro_srcc'])
    text += '\n\nScreen only; legacy test-best checkpoint selection + deterministic Argmax.\n'
    (report/'summary.md').write_text(text, encoding='utf-8')
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    summarize(parser.parse_args().root)
