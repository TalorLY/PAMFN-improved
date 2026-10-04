"""Synthetic report contract tests; no model weights, CUDA, or training needed."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.stats import rankdata, spearmanr

import report
from spec import TASKS, SEEDS, SCORED, anchor_method, cell_path, dependencies, epochs, read, save, sha


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.n = 12
        cls.labels = np.array([0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        cls.coefs = {}
        for task in TASKS:
            for seed in SEEDS:
                group, model_hashes = {}, {}
                for index, method in enumerate(SCORED):
                    pred = cls.labels.copy()
                    # Distinct, nonmonotonic improvements exercise paired contrasts.
                    left = (index+seed) % 8
                    pred[[left, left+3]] = pred[[left+3, left]]
                    coef = float(spearmanr(pred, cls.labels)[0]*100)
                    cls.coefs[task, seed, method] = coef
                    budget = epochs(method, task)
                    value = dict(task=task, seed=seed, method=method, phase='formal', coef=coef,
                                 ids=['video_%02d' % i for i in range(cls.n)],
                                 predictions=dict(prediction=pred.tolist(), label=cls.labels.tolist()),
                                 epochs=budget, selected_epoch=1, parameters=100, total_parameters=200,
                                 training_seconds=2.5, reloaded_exact=True, dependencies={})
                    if method == 'mcmoe':
                        # v37 exact schema has no total_parameters or wall_seconds.
                        value.pop('total_parameters')
                    elif method == 'mc_ordered':
                        value.update(parameters=300, total_parameters=200, training_only_parameters=100)
                    else:
                        value['dependencies'] = {key: model_hashes[key] for key in dependencies(method)}
                    if method == 'mcmoe':
                        folder = cls.root/'references'/'formal'/task/('seed%d' % seed)/'mcmoe'
                        save(folder/'test_result.json', value)
                        source = cls.root/'fixture_external_readonly'/task/('seed%d' % seed)
                        save(source/'test_result.json', value)
                        save(source/'model.pth', dict(synthetic_checkpoint=task, seed=seed, method=method))
                        save(source/'model.pth.eval_rng.pt', dict(synthetic_rng=seed))
                        required = {key: sha(source/key) for key in ('model.pth', 'model.pth.eval_rng.pt', 'test_result.json')}
                        save(folder/'record.json', dict(directory=str(source), required_files=required))
                        model_hashes[method] = required['model.pth']
                    else:
                        folder = cell_path(cls.root, 'formal', task, seed, method)/'attempt_test'
                        save(folder/'test_result.json', value)
                        save(folder/'model.pth', dict(synthetic_checkpoint=task, seed=seed, method=method))
                        model_hashes[method] = sha(folder/'model.pth')
                        save(folder/'last_epoch_result.json', value)
                        history = '\n'.join(json.dumps(dict(epoch=i, coef=coef)) for i in range(1, budget+1))
                        (folder/'epochs.jsonl').write_text(history+'\n', encoding='utf-8')
                        if method != 'mc_ordered':
                            parent = anchor_method(method)
                            initial = group['mc_ordered_e' if parent == 'mc_ordered' else 'mc_e'] if method.endswith('_u') else group[parent]
                            save(folder/'teacher_check.json', dict(parent=parent, native=group[parent], initial=initial,
                                 zero_branch_exact=True, baseline_predicted_exact=True, epoch_zero_eligible=False,
                                 initial_frozen_sha256='frozen-digest'))
                            save(folder/'qmd_training.json', dict(anchor_unchanged=True, statistics_unchanged=True,
                                 branch_updated=True, frozen_before='frozen-digest', frozen_after='frozen-digest',
                                 epochs=budget, positive_gradient_batches=10, kd_weight=0.))
                            save(folder/'training_statistics.json', dict(source_split='train', samples=1000,
                                 passes=1, global_bias=0., bound=.003 if method.endswith('_u') else None))
                        cls.rehash(folder)
                    group[method] = value

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def rehash(folder):
        save(folder.parent/'complete.json', dict(attempt=folder.name,
             files={path.name: sha(path) for path in folder.iterdir() if path.is_file()}))

    def test_full_report_recomputes_every_cell_and_case(self):
        check = report.build_report(self.root, expected_n=self.n)
        self.assertEqual(check['formal_result_count'], 126)
        self.assertEqual(check['new_history_count'], 105)
        self.assertEqual(check['per_video_case_rows'], 21*self.n)
        self.assertEqual(check['verified_branch_evidence'], 84)
        self.assertFalse((self.root/'completion.json').exists())
        with (self.root/'paper'/'method_summary.csv').open(encoding='utf-8-sig', newline='') as stream:
            summary = list(csv.DictReader(stream))
        expected = np.mean([self.coefs[t, s, 'mc_ordered_u'] for t in TASKS for s in SEEDS])
        self.assertAlmostEqual(float(summary[-1]['seven_score_mean']), expected)
        with (self.root/'paper'/'increment_summary.csv').open(encoding='utf-8-sig', newline='') as stream:
            increments = list(csv.DictReader(stream))
        contrast = increments[-1]
        expected = np.mean([self.coefs[t, s, 'mc_ordered_u']-self.coefs[t, s, 'mc_ordered']-
                            self.coefs[t, s, 'mc_u']+self.coefs[t, s, 'mcmoe'] for t in TASKS for s in SEEDS])
        self.assertEqual(contrast['contrast'], report.INTERACTION)
        self.assertAlmostEqual(float(contrast['mean_delta']), expected)
        with (self.root/'paper'/'per_video_cases.csv').open(encoding='utf-8-sig', newline='') as stream:
            cases = list(csv.DictReader(stream))
        self.assertEqual(cases[0]['video_id'], 'video_00')
        self.assertEqual(cases[1]['target_rank'], str(float(rankdata(self.labels)[1])))
        self.assertTrue(read(self.root/'report_check.json')['passed'])
        with (self.root/'paper'/'recorded_costs.csv').open(encoding='utf-8-sig', newline='') as stream:
            costs = list(csv.DictReader(stream))
        self.assertEqual(costs[0]['total_parameters'], '')
        self.assertEqual(costs[0]['wall_seconds'], '')
        self.assertEqual(costs[1]['training_only_parameters'], '100')
        self.assertEqual(costs[1]['total_parameters'], '200')
        # Historical PAMFN score is explanatory text only, never a numerical baseline.
        self.assertAlmostEqual(float(summary[0]['delta_mcmoe']), 0.)

    def test_missing_cell_fails_and_marks_stale_report(self):
        path = cell_path(self.root, 'formal', 'TES', 0, 'mc_ordered')/'complete.json'
        original = path.read_text(encoding='utf-8')
        path.unlink()
        try:
            with self.assertRaisesRegex(ValueError, 'Incomplete formal cell'):
                report.build_report(self.root, expected_n=self.n)
            self.assertFalse(read(self.root/'report_check.json')['passed'])
        finally:
            path.write_text(original, encoding='utf-8')

    def test_changed_payload_is_rejected(self):
        folder = cell_path(self.root, 'formal', 'TES', 0, 'mc_ordered')/'attempt_test'
        path = folder/'test_result.json'
        original = path.read_text(encoding='utf-8')
        path.write_text(original+' ', encoding='utf-8')
        try:
            with self.assertRaisesRegex(ValueError, 'Completed artifact changed'):
                report.build_report(self.root, expected_n=self.n)
        finally:
            path.write_text(original, encoding='utf-8')

    def mutate_result(self, transform, pattern):
        folder = cell_path(self.root, 'formal', 'TES', 0, 'mc_ordered')/'attempt_test'
        original = read(folder/'test_result.json')
        value = json.loads(json.dumps(original))
        transform(value)
        save(folder/'test_result.json', value)
        self.rehash(folder)
        try:
            with self.assertRaisesRegex(ValueError, pattern):
                report.build_report(self.root, expected_n=self.n)
        finally:
            save(folder/'test_result.json', original)
            self.rehash(folder)

    def test_saved_score_mismatch_is_rejected(self):
        self.mutate_result(lambda value: value.update(coef=value['coef']+1), 'inconsistent SRCC')

    def test_method_id_misalignment_is_rejected(self):
        self.mutate_result(lambda value: value['ids'].reverse(), 'identities/labels disagree')

    def test_duplicate_id_is_rejected(self):
        self.mutate_result(lambda value: value['ids'].__setitem__(1, value['ids'][0]), 'unique evaluation IDs')

    def test_first_best_epoch_is_enforced(self):
        self.mutate_result(lambda value: value.update(selected_epoch=2), 'first best eligible epoch')

    def test_replaced_parent_does_not_validate_existing_child(self):
        folder = cell_path(self.root, 'formal', 'TES', 0, 'mc_ordered')/'attempt_test'
        original = read(folder/'model.pth')
        save(folder/'model.pth', dict(replaced=True))
        self.rehash(folder)
        try:
            with self.assertRaisesRegex(ValueError, 'dependencies differ from current parent weights'):
                report.build_report(self.root, expected_n=self.n)
        finally:
            save(folder/'model.pth', original)
            self.rehash(folder)

    def test_missing_cost_is_not_fabricated(self):
        self.assertIsNone(report.optional_cost({}, 'wall_seconds'))
        with self.assertRaises(ValueError):
            report.optional_cost(dict(training_seconds=-1), 'training_seconds')

    def test_measured_cost_is_recomputed_and_preserves_scope(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            value = dict(task='TES', seed=0, method='mc_u')
            cost = dict(status='measured', gpu='synthetic', exclusive_gpu=False,
                        scope='shared GPU cached-feature-only', warmup=10, repetitions=3,
                        variants=dict(u=dict(repetitions_ms=[1., 2., 3.], median_ms=2., mean_ms=2.)))
            save(folder/'cost.json', cost)
            _, rows = report.optional_diagnostics(folder, value, self.n)
            self.assertEqual(rows[0]['median_ms'], 2.)
            self.assertFalse(rows[0]['exclusive_gpu'])
            cost['variants']['u']['median_ms'] = 9.
            save(folder/'cost.json', cost)
            with self.assertRaisesRegex(ValueError, 'timing summary'):
                report.optional_diagnostics(folder, value, self.n)

    def test_actual_worker_e_u_evidence_is_compatible(self):
        # Call the actual worker's output constructors, not a parallel invented schema.
        import torch
        from modules import EBranch, UBranch
        from worker import benchmark, branch_evaluate, mechanism, metrics
        with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory() as name:
            torch.manual_seed(77)
            folder = Path(name)
            base = torch.linspace(.1, .9, self.n)
            fused, modal = torch.randn(self.n, 256), torch.randn(self.n, 768)
            labels = base+.01*torch.randn(self.n)
            cache = (base, fused, modal, labels)
            anchor = metrics(base.tolist(), labels.tolist())
            e = EBranch(256, 0)
            e_stats = e.fit(base, fused, labels)
            e_initial = branch_evaluate(cache, e)
            with torch.no_grad():
                e.net[-1].weight.fill_(.2)
            e_result = branch_evaluate(cache, e)
            u = UBranch(768, 0)
            u_stats = u.fit(e.evaluation_score(base, fused), modal)
            u_initial = branch_evaluate(cache, e, u)
            with torch.no_grad():
                for head in u.heads:
                    head[-1].weight.fill_(.3)
            u_result = branch_evaluate(cache, e, u)
            group = dict(mcmoe=anchor, mc_e=e_result, mc_u=u_result)
            for method, result, initial, stats, branch, count in (
                    ('mc_e', e_result, e_initial, e_stats, e, 4),
                    ('mc_u', u_result, u_initial, u_stats, u, 7)):
                result.update(task='TES', seed=0, method=method, epochs=500,
                              dependencies={key: 'digest' for key in dependencies(method)},
                              parameters=sum(p.numel() for p in branch.parameters()),
                              total_parameters=9122183+sum(p.numel() for p in e.parameters())+
                              (sum(p.numel() for p in u.parameters()) if method == 'mc_u' else 0))
                save(folder/'teacher_check.json', dict(parent='mcmoe', native=anchor, initial=initial,
                     zero_branch_exact=True, baseline_predicted_exact=True, epoch_zero_eligible=False,
                     initial_frozen_sha256='digest'))
                save(folder/'qmd_training.json', dict(frozen_before='digest', frozen_after='digest',
                     anchor_unchanged=True, statistics_unchanged=True, branch_updated=True,
                     positive_gradient_batches=2, epochs=500, kd_weight=0.))
                save(folder/'training_statistics.json', stats)
                record = report.validate_branch_evidence(folder, result, group, self.n)
                self.assertTrue(record['anchor_prediction_exact'])
                save(folder/'mechanism.json', mechanism(cache, e, u if method == 'mc_u' else None))
                save(folder/'cost.json', benchmark(None, e, u, None, torch.device('cpu')))
                interventions, timing = report.optional_diagnostics(folder, result, self.n)
                self.assertEqual(len(interventions), count)
                self.assertEqual(timing, [])
                self.assertEqual(result['parameters'], 4128 if method == 'mc_e' else 12384)
            # A teacher copied from a different anchor must not pass merely because flags say True.
            teacher = read(folder/'teacher_check.json')
            teacher['native'] = metrics(list(reversed(base.tolist())), labels.tolist())
            save(folder/'teacher_check.json', teacher)
            with self.assertRaisesRegex(ValueError, 'does not reproduce matched anchor'):
                report.validate_branch_evidence(folder, u_result, group, self.n)


if __name__ == '__main__':
    unittest.main()
