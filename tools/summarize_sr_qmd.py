"""Paired SR-QMD/Ordered comparisons. Pilot is never a final success claim."""
import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

from tools.summarize_score_qmd import ACTIONS, SEEDS, budget, read_json

METHODS = ('sr_qmd', 'ordered', 'ordered_sr_qmd')
REFERENCES = ('base', 'mask_gt', 'score_qmd')


def qmd_contract(method, smoke=False):
    config = dict(mode='off' if method == 'ordered' else 'score_rank_qmd', weight=1.,
                  mask_ratio=.15, mask_gt_weight=.5, target_ratio=.075, beta_cap=1.,
                  warmup=0. if smoke else .1, rampup=0. if smoke else .1)
    if method != 'ordered':
        config.update(rank_weight=.5, rank_gap=.05, rank_gamma=.1)
    return config


def validate_cell(path, action, seed, method, smoke=False):
    path = Path(path)
    train, test = read_json(path/'train_result.json'), read_json(path/'test_result.json')
    run = test['run']
    ordered = method in ('ordered', 'ordered_sr_qmd')
    if train['run'] != run or (run['action'], run['seed'], run['tag']) != (action, seed, method):
        raise ValueError('run identity mismatch: ' + str(path))
    expected = dict(epochs=1 if smoke else budget(action), feats=2, coqm_weight=0.,
                    eval_policy='argmax', eval_seed=20260908,
                    ordinal_prototype_weight=.0001 if ordered else 0.,
                    ordinal_prototype_target_mode='continuous' if ordered else 'discrete',
                    quality_aux_rng_isolated=ordered, qmd=qmd_contract(method, smoke),
                    ordered_config=dict(dim=128, temperature=.1, margin=.1, order_scale=.1)
                                   if ordered else None)
    if any(run.get(k) != v for k, v in expected.items()):
        raise ValueError('experiment config mismatch: ' + str(path))
    if test['evaluation'] != dict(policy='argmax', seed=20260908, passes=1, rng_isolated=True):
        raise ValueError('evaluation protocol mismatch')
    if not math.isclose(train['saved_model_srcc'] * 100, test['coef'], abs_tol=1e-8):
        raise ValueError('checkpoint reload mismatch')
    if method != 'ordered':
        qmd = read_json(path/'qmd_training.json')
        if qmd['mode'] != 'score_rank_qmd' or qmd['active_batches'] <= 0:
            raise ValueError('QMD never activated')
        if len(qmd['epochs']) != run['epochs'] or sum(e['active_batches'] for e in qmd['epochs']) != qmd['active_batches']:
            raise ValueError('incomplete training diagnostics')
        if not qmd['calibration'] or qmd['calibration']['degenerate'] or qmd['calibration']['beta'] <= 0:
            raise ValueError('invalid Score-KD calibration')
        rank = qmd['rank']
        if (rank['weight'], rank['gap'], rank['gamma'], rank['normalization']) != (.5, .05, .1, 'eligible_pair_count'):
            raise ValueError('rank configuration mismatch')
        if not rank['gradient_verified'] or rank['eligible_pairs'] <= 0:
            raise ValueError('Rank-KD never contributed a nonzero gradient')
    metrics = dict(srcc=test['coef'], mse=test['loss_x100']/100, mae=test['mae'],
                   bias=test['bias'], std_ratio=test['prediction_std_ratio'])
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError('non-finite metrics')
    return dict(action=action, seed=seed, method=method, **metrics)


def aggregate(rows, seeds=SEEDS):
    methods = REFERENCES + METHODS
    expected = {(a,s,m) for a in ACTIONS for s in seeds for m in methods}
    index = {(r['action'], r['seed'], r['method']): r for r in rows}
    if len(rows) != len(expected) or set(index) != expected:
        raise ValueError('missing or duplicate paired experiment rows')
    macro = {m: st.mean(index[a,s,m]['srcc'] for a in ACTIONS for s in seeds) for m in methods}
    comparisons = [('rank_vs_score', 'sr_qmd', 'score_qmd'),
                   ('ordered_vs_base', 'ordered', 'base'),
                   ('combo_vs_ordered', 'ordered_sr_qmd', 'ordered'),
                   ('combo_vs_sr', 'ordered_sr_qmd', 'sr_qmd'),
                   ('combo_vs_score', 'ordered_sr_qmd', 'score_qmd')]
    actions, seed_rows = [], []
    for a in ACTIONS:
        row = dict(action=a)
        for m in methods:
            x = [index[a,s,m]['srcc'] for s in seeds]
            row[m], row[m+'_sd'] = st.mean(x), st.stdev(x) if len(x)>1 else None
        for name, left, right in comparisons:
            row[name+'_pp'] = row[left]-row[right]
        actions.append(row)
    for s in seeds:
        row = dict(seed=s, **{m: st.mean(index[a,s,m]['srcc'] for a in ACTIONS) for m in methods})
        seed_rows.append(row)
    checks = {}
    for name, left, right in comparisons:
        deltas = [index[a,s,left]['srcc']-index[a,s,right]['srcc'] for a in ACTIONS for s in seeds]
        checks[name] = dict(delta_pp=macro[left]-macro[right], wins=sum(d>1e-10 for d in deltas),
                            ties=sum(abs(d)<=1e-10 for d in deltas), losses=sum(d < -1e-10 for d in deltas),
                            positive_actions=sum(r[left]>r[right] for r in actions))
    # Descriptive signal, not significance or a guarantee on untested seeds.
    rank_signal = checks['rank_vs_score']['delta_pp'] > 0
    combo_signal = all(checks[k]['delta_pp'] > 0 for k in ('combo_vs_ordered','combo_vs_sr','combo_vs_score'))
    final = tuple(seeds) == tuple(SEEDS)
    winner = max(('score_qmd', 'sr_qmd', 'ordered_sr_qmd'), key=lambda m: macro[m])
    decision = ('PILOT_ONLY' if not final else 'RANK_AND_COMBO_SIGNAL' if rank_signal and combo_signal
                else 'COMBO_SIGNAL' if combo_signal else 'RANK_SIGNAL' if rank_signal else 'KEEP_SCORE_QMD_V1')
    return dict(decision=decision, screening_only=True, seeds=list(seeds), macro_srcc=macro,
                comparisons=checks, candidate=winner, actions=actions, seed_means=seed_rows)


def summarize(root, seeds=SEEDS, name='report'):
    root = Path(root)
    rows = []
    for a in ACTIONS:
        for s in seeds:
            for m in REFERENCES:
                rows.append(read_json(root/'reference'/a/f'seed{s}'/m/'metrics.json'))
            for m in METHODS:
                rows.append(validate_cell(root/a/f'seed{s}'/m, a, s, m))
    result = aggregate(rows, seeds)
    report = root/name
    report.mkdir(exist_ok=True)
    for label, data in [('runs',rows), ('actions',result['actions']), ('seeds',result['seed_means'])]:
        with (report/(label+'.csv')).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (report/'decision.json').write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    lines = ['# SR-QMD / Ordered', '', result['decision'], '',
             'SRCC x100; equal task/seed weights; legacy test-best checkpoint + argmax.', '',
             '|Task|Base|Mask-GT|Score-QMD|SR-QMD|Ordered|Ordered+SR-QMD|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in result['actions'] + [dict(action='Macro', **result['macro_srcc'])]:
        lines.append('|'+r['action']+'|'+'|'.join('%.3f'%r[m] for m in REFERENCES+METHODS)+'|')
    lines += ['', 'Comparisons (pp):', '```json', json.dumps(result['comparisons'],indent=2), '```',
              '', 'Pilot is a progress readout only. No historical July28 score is added to these results.']
    (report/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('actions','seed_means')},indent=2), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    summarize(args.root, (0,) if args.pilot else SEEDS, 'pilot_report' if args.pilot else 'report')
