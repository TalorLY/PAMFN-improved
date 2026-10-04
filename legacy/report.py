"""Verified descriptive tables for the fixed v38 experiment; never marks completion."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from spec import TASKS, SEEDS, SCORED, POLICY, anchor_method, artifact, cell_path, completed, dependencies, epochs, read, save, sha


NAMES = dict(mcmoe='MCMoE', mc_ordered='MCMoE + Ordered', mc_e='MCMoE + E',
             mc_u='MCMoE + E + U', mc_ordered_e='MCMoE + Ordered + E',
             mc_ordered_u='MCMoE + Ordered + E + U')
PAIRS = (('O-M', 'mc_ordered', 'mcmoe'), ('E-M', 'mc_e', 'mcmoe'),
         ('U-E', 'mc_u', 'mc_e'), ('OE-O', 'mc_ordered_e', 'mc_ordered'),
         ('OU-OE', 'mc_ordered_u', 'mc_ordered_e'), ('OU-M', 'mc_ordered_u', 'mcmoe'),
         ('OU-U', 'mc_ordered_u', 'mc_u'))
INTERACTION = '(OU-O)-(U-M)'


def write_csv(path, rows):
    if not rows:
        raise ValueError('No report rows: ' + str(path))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def predictions(result, expected_n):
    values = result['predictions']
    pred = np.asarray(values['prediction'], dtype=np.float64)
    label = np.asarray(values['label'], dtype=np.float64)
    if pred.shape != (expected_n,) or label.shape != (expected_n,):
        raise ValueError('Prediction/label count or shape mismatch')
    if not np.isfinite(pred).all() or not np.isfinite(label).all():
        raise ValueError('Nonfinite predictions or labels')
    coef = float(spearmanr(pred, label)[0] * 100)
    saved = float(result['coef'])
    if not np.isfinite(coef) or not np.isfinite(saved) or abs(coef-saved) > 1e-8:
        raise ValueError('Undefined or inconsistent SRCC')
    return pred, label, coef


def validate_result(result, task, seed, method, expected_n):
    if (result['task'], result['seed'], result['method'], result['phase']) != (task, seed, method, 'formal'):
        raise ValueError('Cell metadata mismatch: %s/%s/%s' % (task, seed, method))
    ids = result['ids']
    if not isinstance(ids, list) or len(ids) != expected_n or len(set(ids)) != expected_n:
        raise ValueError('Expected unique evaluation IDs')
    if not all(isinstance(value, str) and value for value in ids):
        raise ValueError('Invalid evaluation ID')
    if result.get('reloaded_exact') is not True:
        raise ValueError('Worker did not record exact checkpoint reload')
    if result['epochs'] != epochs(method, task) or not 1 <= result['selected_epoch'] <= result['epochs']:
        raise ValueError('Training budget/selection epoch mismatch')
    return predictions(result, expected_n)


def validate_history(folder, result, expected_n):
    records = [json.loads(line) for line in (folder/'epochs.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(records) != result['epochs'] or [r['epoch'] for r in records] != list(range(1, result['epochs']+1)):
        raise ValueError('History does not cover the full training budget')
    valid = []
    for record in records:
        value = record['coef']
        if value is not None:
            if not np.isfinite(value):
                raise ValueError('Nonfinite history SRCC must not be used as an epoch score')
            valid.append(record)
    if not valid:
        raise ValueError('No eligible epoch')
    selected = max(valid, key=lambda r: r['coef'])
    if selected['epoch'] != result['selected_epoch'] or abs(selected['coef']-result['coef']) > 1e-8:
        raise ValueError('Selected checkpoint does not match first best eligible epoch')
    last = read(folder/'last_epoch_result.json')
    if last['predictions']['label'] != result['predictions']['label']:
        raise ValueError('Last epoch used different labels')
    if last.get('coef') is None:
        raise ValueError('Final epoch still has undefined SRCC; recovery not confirmed')
    _, _, last_coef = predictions(last, expected_n)
    if records[-1]['coef'] is None or abs(last_coef-records[-1]['coef']) > 1e-8:
        raise ValueError('Last epoch and history disagree')
    return dict(undefined_epochs=sum(r['coef'] is None for r in records), last_srcc_x100=last_coef)


def optional_cost(result, key):
    value = result.get(key)
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0):
        raise ValueError('Invalid recorded cost: ' + key)
    return value


def validate_branch_evidence(folder, result, group, expected_n):
    """Verify server-produced frozen-parent evidence against the matched group."""
    method = result['method']
    if method in ('mcmoe', 'mc_ordered'):
        return None
    parent = anchor_method(method)
    teacher = read(folder/'teacher_check.json')
    training = read(folder/'qmd_training.json')
    statistics = read(folder/'training_statistics.json')
    if teacher['parent'] != parent or teacher.get('epoch_zero_eligible') is not False:
        raise ValueError('Branch teacher/selection contract changed')
    if teacher.get('zero_branch_exact') is not True or teacher.get('baseline_predicted_exact') is not True:
        raise ValueError('Branch parent replay did not pass')
    predictions(teacher['native'], expected_n)
    if teacher['native']['predictions'] != group[parent]['predictions'] or abs(teacher['native']['coef']-group[parent]['coef']) > 1e-8:
        raise ValueError('Branch teacher does not reproduce matched anchor')
    if statistics.get('source_split') != 'train' or statistics.get('passes') != 1:
        raise ValueError('Branch calibration is not declared single-pass train-only')
    if not all(training.get(k) is True for k in ('anchor_unchanged', 'statistics_unchanged', 'branch_updated')):
        raise ValueError('Branch frozen-parent/update checks failed')
    if training['frozen_before'] != training['frozen_after'] or training['frozen_before'] != teacher['initial_frozen_sha256']:
        raise ValueError('Frozen-parent digests changed')
    if training['epochs'] != result['epochs'] or training['positive_gradient_batches'] <= 0 or training['kd_weight'] != 0.:
        raise ValueError('Branch training evidence disagrees with the fixed experiment')
    if set(result['dependencies']) != set(dependencies(method)):
        raise ValueError('Branch dependency names differ from the experiment contract')
    if method.endswith('_u'):
        e_parent = 'mc_ordered_e' if parent == 'mc_ordered' else 'mc_e'
        if teacher['initial']['predictions'] != group[e_parent]['predictions']:
            raise ValueError('Zero U does not preserve matched E predictions')
    else:
        initial = np.asarray(teacher['initial']['predictions']['prediction'], dtype=np.float64)
        expected = np.asarray(group[parent]['predictions']['prediction'], dtype=np.float64)+statistics['global_bias']
        if teacher['initial']['predictions']['label'] != result['predictions']['label'] or not np.array_equal(initial, expected):
            raise ValueError('Zero E does not equal matched anchor plus train-only global bias')
    return dict(task=result['task'], seed=result['seed'], method=method, anchor=parent,
                anchor_prediction_exact=True, frozen_parent_exact=True, statistics_source='train',
                branch_updated=True, positive_gradient_batches=training['positive_gradient_batches'],
                kd_weight=training['kd_weight'], global_bias=statistics.get('global_bias'),
                u_scale=statistics.get('bound'), training_samples=statistics.get('samples'))


def optional_diagnostics(folder, result, expected_n):
    """Summarize measured artifacts when present, without requiring old snapshots."""
    mechanism_rows, inference_rows = [], []
    common = dict(task=result['task'], seed=result['seed'], method=result['method'])
    if (folder/'mechanism.json').is_file():
        evidence = read(folder/'mechanism.json')
        for name, value in evidence['variants'].items():
            if value['predictions']['label'] != result['predictions']['label']:
                raise ValueError('Mechanism intervention labels changed')
            if value['coef'] is None:
                vector = np.asarray(value['predictions']['prediction'], dtype=np.float64)
                if vector.shape != (expected_n,) or not np.isfinite(vector).all() or np.unique(vector).size != 1:
                    raise ValueError('Invalid undefined intervention SRCC')
                coef = None
            else:
                _, _, coef = predictions(value, expected_n)
            if name == 'full' and value['predictions']['prediction'] != result['predictions']['prediction']:
                raise ValueError('Full intervention does not reproduce selected predictions')
            counts = {key: value[key] for key in ('reliable_pairs', 'repaired_pairs', 'broken_pairs')}
            if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in counts.values()):
                raise ValueError('Invalid mechanism pair count')
            if counts['repaired_pairs']+counts['broken_pairs'] > counts['reliable_pairs']:
                raise ValueError('Mechanism pair counts exceed valid pairs')
            mechanism_rows.append(dict(common, intervention=name, srcc_x100=coef, **counts,
                                       label_gap=evidence['label_gap'], scope=evidence['kind']))
    if (folder/'cost.json').is_file():
        cost = read(folder/'cost.json')
        if cost.get('status') == 'measured':
            for mode, value in cost['variants'].items():
                samples = np.asarray(value['repetitions_ms'], dtype=np.float64)
                if samples.shape != (cost['repetitions'],) or not len(samples) or not np.isfinite(samples).all() or (samples <= 0).any():
                    raise ValueError('Invalid inference timing samples')
                if abs(float(np.median(samples))-value['median_ms']) > 1e-8 or abs(float(samples.mean())-value['mean_ms']) > 1e-8:
                    raise ValueError('Inference timing summary disagrees with repetitions')
                inference_rows.append(dict(common, variant=mode, median_ms=value['median_ms'], mean_ms=value['mean_ms'],
                                           repetitions=cost['repetitions'], warmup=cost['warmup'],
                                           gpu=cost.get('gpu'), exclusive_gpu=cost.get('exclusive_gpu'), scope=cost.get('scope')))
        elif cost.get('status') != 'not_measured':
            raise ValueError('Unknown inference measurement status')
    return mechanism_rows, inference_rows


def _build(root, expected_n):
    table, results, rows, costs, cases = {}, {}, [], [], []
    mechanisms, inference, branch_evidence = [], [], []
    task_identity = {}
    for task in TASKS:
        for seed in SEEDS:
            group, model_hashes = {}, {}
            for method in SCORED:
                if method == 'mcmoe':
                    folder = root/'references'/'formal'/task/('seed%d' % seed)/'mcmoe'
                else:
                    folder = completed(cell_path(root, 'formal', task, seed, method), verify=True)
                    if folder is None:
                        raise ValueError('Incomplete formal cell: %s/%s/%s' % (task, seed, method))
                result = read(folder/'test_result.json')
                if method == 'mcmoe':
                    # This is a read-only reference, not a copied model. Check the live
                    # model/RNG/prediction digests before trusting its small snapshot.
                    source_folder = artifact(root, 'formal', task, seed, method)
                    if read(source_folder/'test_result.json') != result:
                        raise ValueError('Reference prediction snapshot differs from source')
                    model_hashes[method] = read(folder/'record.json')['required_files']['model.pth']
                else:
                    model_hashes[method] = sha(folder/'model.pth')
                    expected_dependencies = {key: model_hashes[key] for key in dependencies(method)}
                    if result.get('dependencies') != expected_dependencies:
                        raise ValueError('Child checkpoint dependencies differ from current parent weights')
                _, _, coef = validate_result(result, task, seed, method, expected_n)
                history = validate_history(folder, result, expected_n) if method != 'mcmoe' else dict(undefined_epochs=None, last_srcc_x100=None)
                group[method] = result
                branch_record = validate_branch_evidence(folder, result, group, expected_n)
                if branch_record is not None:
                    branch_evidence.append(branch_record)
                results[task, seed, method] = result
                table[task, seed, method] = coef
                rows.append(dict(task=task, seed=seed, method=method, srcc_x100=coef,
                                 delta_mcmoe=coef-group['mcmoe']['coef'], selected_epoch=result['selected_epoch'],
                                 epochs=result['epochs'], **history))
                costs.append(dict(task=task, seed=seed, method=method,
                                  parameters=optional_cost(result, 'parameters'),
                                  total_parameters=optional_cost(result, 'total_parameters'),
                                  training_only_parameters=optional_cost(result, 'training_only_parameters'),
                                  training_seconds=optional_cost(result, 'training_seconds'),
                                  wall_seconds=optional_cost(result, 'wall_seconds'),
                                  scope='recorded stage only; no inferred runtime or peak memory'))
                extra_mechanism, extra_inference = optional_diagnostics(folder, result, expected_n)
                mechanisms.extend(extra_mechanism)
                inference.extend(extra_inference)
            identity = (group['mcmoe']['ids'], group['mcmoe']['predictions']['label'])
            if task in task_identity and identity != task_identity[task]:
                raise ValueError('Seed evaluation identities/labels disagree: ' + task)
            task_identity[task] = identity
            for method in SCORED:
                if (group[method]['ids'], group[method]['predictions']['label']) != identity:
                    raise ValueError('Method evaluation identities/labels disagree: %s/%s/%s' % (task, seed, method))
            truth_rank = rankdata(identity[1], method='average')
            predicted_ranks = {m: rankdata(group[m]['predictions']['prediction'], method='average') for m in SCORED}
            for index, video in enumerate(identity[0]):
                case = dict(task=task, seed=seed, video_id=video, label=identity[1][index], target_rank=float(truth_rank[index]))
                for method in SCORED:
                    case[method+'_prediction'] = group[method]['predictions']['prediction'][index]
                    case[method+'_rank_error'] = float(abs(predicted_ranks[method][index]-truth_rank[index]))
                for label, child, parent in PAIRS:
                    case[label+'_rank_error_reduction'] = case[parent+'_rank_error']-case[child+'_rank_error']
                # Keep task/seed/official ID order. Positive is improved, negative is harmed.
                cases.append(case)

    seed_rows, summaries, pair_rows, pair_summary = [], [], [], []
    macro = {}
    for seed in SEEDS:
        for method in SCORED:
            macro[seed, method] = float(np.mean([table[t, seed, method] for t in TASKS]))
            seed_rows.append(dict(seed=seed, method=method, seven_score_mean=macro[seed, method],
                                  tes=table['TES', seed, method], pcs=table['PCS', seed, method],
                                  tes_pcs_mean=float(np.mean([table[t, seed, method] for t in ('TES', 'PCS')]))))
    for method in SCORED:
        values = np.array([macro[s, method] for s in SEEDS])
        summaries.append(dict(method=method, seven_score_mean=float(values.mean()),
                              seed_sample_std=float(values.std(ddof=1)),
                              delta_mcmoe=float(np.mean([macro[s, method]-macro[s, 'mcmoe'] for s in SEEDS])),
                              tes=float(np.mean([table['TES', s, method] for s in SEEDS])),
                              pcs=float(np.mean([table['PCS', s, method] for s in SEEDS]))))
    for task in TASKS:
        for seed in SEEDS:
            for name, child, parent in PAIRS:
                pair_rows.append(dict(task=task, seed=seed, contrast=name, delta_srcc_x100=table[task, seed, child]-table[task, seed, parent]))
            interaction = table[task, seed, 'mc_ordered_u']-table[task, seed, 'mc_ordered']-table[task, seed, 'mc_u']+table[task, seed, 'mcmoe']
            pair_rows.append(dict(task=task, seed=seed, contrast=INTERACTION, delta_srcc_x100=interaction))
    for name in [p[0] for p in PAIRS]+[INTERACTION]:
        selected = [r for r in pair_rows if r['contrast'] == name]
        values = np.array([r['delta_srcc_x100'] for r in selected])
        row = dict(contrast=name, mean_delta=float(values.mean()), wins=int((values > 1e-10).sum()),
                   ties=int((np.abs(values) <= 1e-10).sum()), losses=int((values < -1e-10).sum()))
        for seed in SEEDS:
            row['seed%d_mean' % seed] = float(np.mean([r['delta_srcc_x100'] for r in selected if r['seed'] == seed]))
        pair_summary.append(row)

    # Only publish report tables after the complete matrix has passed validation.
    out = root/'paper'
    out.mkdir(exist_ok=True)
    for filename, values in (('metrics.csv', rows), ('method_summary.csv', summaries), ('seed_summary.csv', seed_rows),
                             ('paired_increments.csv', pair_rows), ('increment_summary.csv', pair_summary),
                             ('per_video_cases.csv', cases), ('recorded_costs.csv', costs),
                             ('branch_evidence.csv', branch_evidence)):
        write_csv(out/filename, values)
    if mechanisms:
        write_csv(out/'mechanism_interventions.csv', mechanisms)
    if inference:
        write_csv(out/'inference_measurements.csv', inference)
    text = ['# FS1000 跨模型验证 v38', '',
            'M=MCMoE，O=M+Ordered，E=M+E，U=M+E+U，OE=M+Ordered+E，OU=M+Ordered+E+U。', '',
            '单位为 SRCC×100；全部 seed 保留。七个评分项来自相同视频、彼此相关，不是七个独立动作。',
            'TES、PCS 单列；七项算术均值只是辅助汇总。标准差来自三个 seed 的七项均分。',
            '官方 val 用于逐轮选模及报告，不称为盲测。E/U 增加训练阶段，不能称总训练预算相等。',
            'v37 MCMoE 为只读复用；RG/FisV 历史主线与 81.277 基线不变。', '',
            '## 总体结果', '', '| 方法 | TES | PCS | 七项平均 ± seed标准差 | 相比M |',
            '|---|---:|---:|---:|---:|']
    for row in summaries:
        text.append('| %s | %.3f | %.3f | %.3f ± %.3f | %+.3f |' % (NAMES[row['method']], row['tes'], row['pcs'], row['seven_score_mean'], row['seed_sample_std'], row['delta_mcmoe']))
    for seed in SEEDS:
        text += ['', '## Seed %d' % seed, '', '| 评分项 | M | O | E | U | OE | OU | OU−M |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for task in TASKS:
            vals = [table[task, seed, m] for m in SCORED]
            text.append('| %s | %s | %+.3f |' % (task, ' | '.join('%.3f' % v for v in vals), vals[-1]-vals[0]))
        vals = [macro[seed, m] for m in SCORED]
        text.append('| 七项平均 | %s | %+.3f |' % (' | '.join('%.3f' % v for v in vals), vals[-1]-vals[0]))
    text += ['', '## 成对增量与描述性交互', '', '| 对比 | seed0 | seed1 | seed2 | 平均 | 胜/平/负 |', '|---|---:|---:|---:|---:|---:|']
    for row in pair_summary:
        text.append('| %s | %+.3f | %+.3f | %+.3f | %+.3f | %d/%d/%d |' % (row['contrast'], row['seed0_mean'], row['seed1_mean'], row['seed2_mean'], row['mean_delta'], row['wins'], row['ties'], row['losses']))
    text += ['', '交互定义：(OU−O)−(U−M)，即 E/U 在有 Ordered 与无 Ordered 时的增量之差。',
             '这是固定已选检查点的描述性数值；正值不能自动证明统计显著或“强互补”。胜负计数的21组也非独立样本。', '',
             '## 案例与成本', '',
             '`per_video_cases.csv` 保持评分项、seed与官方ID顺序；rank_error_reduction>0为修复，<0为损害，平均秩处理并列。',
             '每个样本同时保留全部模型预测与排名误差；不能凭数值自动推断视频动作或音频语义。',
             '`recorded_costs.csv` 仅保存实际存在的阶段参数/耗时字段，缺失值留空。',
             'parameters 为该阶段记录的参数量；total_parameters 仅在 worker 实际提供时表示完整推理参数。',
             'Ordered 的 training_only_parameters 为训练期辅助质量头；它计入该阶段训练参数，但不计入推理参数。',
             '共享GPU训练耗时不是独占GPU性能；报告没有补造推理延迟、显存或等计算量结论。', '',
             '若 worker 提供干预或测速文件，另输出 mechanism_interventions.csv / inference_measurements.csv。',
             '干预是固定检查点诊断，不是重新训练或新的选模候选；去除U某模态头不等于去除主模型该模态。',
             '推理计时只在实际 measured 时输出，均值/中位数从重复样本核验；共享GPU记录不是独占端到端成本。', '',
             '## 核验范围', '',
             '已验证126个结果、105个新训练日志、ID/标签对齐、原始预测SRCC以及新实验完成清单中的文件哈希。',
             '84组校正分支还核验父模型预测、冻结校验记录、训练内校准声明、零输出一致性及非零训练更新。',
             'reloaded_exact 是服务器 worker 的记录；本报告不重新加载权重，也不声称在报告端完成权重重载。',
             '任何缺项或不一致都会导致 report_check.json 为失败；本报告不创建实验完成标志。', '']
    (out/'TRANSFER_REPORT.md').write_text('\n'.join(text), encoding='utf-8')
    policy = read(root/'manifest.json').get('policy', POLICY) if (root/'manifest.json').is_file() else POLICY
    save(out/'report_policy.json', policy)
    return dict(passed=True, status='PASS', formal_result_count=len(rows), new_history_count=len(rows)-len(TASKS)*len(SEEDS),
                evaluation_count_per_cell=expected_n, per_video_case_rows=len(cases),
                mechanism_rows=len(mechanisms), measured_inference_rows=len(inference),
                verified_branch_evidence=len(branch_evidence),
                checked=['metadata', 'prediction_srcc', 'within_task_ids_labels', 'new_budget_history_first_best', 'new_completion_file_hashes', 'branch_frozen_parent_evidence', 'parent_checkpoint_dependency_closure'],
                weight_reload='server worker record checked; reporter does not load model weights',
                claims='descriptive matched comparisons; no automatic statistical or competitiveness conclusion',
                paper_directory=str(out))


def build_report(root, expected_n=247):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        if not isinstance(expected_n, int) or expected_n < 3:
            raise ValueError('Invalid evaluation count')
        result = _build(root, expected_n)
    except Exception as error:
        save(root/'report_check.json', dict(passed=False, status='FAIL', error=repr(error),
                                          note='Existing paper files, if any, are stale until this check passes.'))
        raise
    save(root/'report_check.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args.root), ensure_ascii=False))


if __name__ == '__main__':
    main()
