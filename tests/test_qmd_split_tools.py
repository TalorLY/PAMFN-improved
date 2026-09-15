from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from tools import run_qmd_split as run
from tools import run_qmd_dual as core


def fake_row(cell):
    p, a, s, m = cell
    # One very weak seed is intentional: summaries must retain it.
    return dict(protocol=p, action=a, seed=s, method=m, srcc_x100=70. if s == 1 else 81., warnings=[])


def fixture_references():
    return [dict(protocol='july28', action=a, seed=s, method=m, srcc_x100=80. if m == 'ordered' else 79.)
            for a in run.ACTIONS for s in (0, 1, 2) for m in ('ordered', 'ordered_score_qmd')]


def fixture_v8():
    return [dict(fake_row(('july28', a, s, run.GUARD)), origin='frozen_v8_fix1_guard')
            for a in run.ACTIONS for s in (0, 1)]


class SplitRunnerTest(unittest.TestCase):
    def test_fixed_matrix_and_original_july28_settings(self):
        jobs = run.matrix()
        self.assertEqual(len(jobs), 15)
        self.assertEqual(len(set(jobs)), 15)
        self.assertEqual(sum(c[3] == run.GUARD for c in jobs), 3)
        self.assertEqual(len(run.smoke_cells(jobs)), 8)
        self.assertEqual(run.smoke_cells([]), [])
        self.assertEqual(len(run.grad.reference_cells(jobs)), 18)
        for c in jobs:
            argv = run.cell_args(c, '4')
            self.assertNotIn('--eval_policy', argv)
            self.assertNotIn('--quality_aux_rng_isolated', argv)
            self.assertNotIn('--score_head', argv)
            self.assertEqual(argv[argv.index('--qmd_gt_weight') + 1], '.5')
            self.assertEqual(argv[argv.index('--ordinal_prototype_weight') + 1], '.0001')
            if c[3] == run.GUARD:
                self.assertEqual(argv, run.grad.cell_args(c, '4'))
            else:
                self.assertEqual(argv[argv.index('--qmd_terms') + 1], run.TERMS[c[3]])
                self.assertNotIn('--qmd_grad_mode', argv)
        with self.assertRaises(ValueError):
            run.settings(('modern', 'Ball', 0, run.GT))

    def test_component_and_prediction_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            c = ('july28', 'Ball', 0, run.GT)
            y = np.linspace(.1, .9, 50).tolist()
            result = dict(run=dict(qmd=run.settings(c)), coef=100., loss_x100=0., predictions=dict(label=y, prediction=y))
            core.save(out / 'train_result.json', result)
            core.save(out / 'test_result.json', result)
            qmd = dict(active_batches=2, epochs=[dict(active_batches=2)], components=dict(
                terms='gt', batches=2, weighted_gt_sum=.3, weighted_kd_sum=0., gradient_verified=True,
                first_positive_head_gradient_norm=.01))
            core.save(out / 'qmd_training.json', qmd)
            with patch.object(core, 'validate_cell', side_effect=lambda *a: fake_row(c)):
                self.assertEqual(run.validate_cell(out, c)['weighted_kd_sum'], 0)
                qmd['components']['weighted_kd_sum'] = .01
                core.save(out / 'qmd_training.json', qmd)
                with self.assertRaisesRegex(ValueError, 'disabled component'):
                    run.validate_cell(out, c)
                qmd['components']['weighted_kd_sum'] = 0
                core.save(out / 'qmd_training.json', qmd)
                result['predictions']['prediction'] = y[::-1]
                core.save(out / 'test_result.json', result)
                with self.assertRaisesRegex(ValueError, 'prediction'):
                    run.validate_cell(out, c)

    def test_warmup_compares_each_epoch_but_not_auxiliary_epochs(self):
        c = ('july28', 'Ribbon', 0, run.KD)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); out = root / 'new'; out.mkdir()
            ref = core.cell_path(root / 'ref', (c[0], c[1], c[2], 'ordered_score_qmd')); ref.mkdir(parents=True)
            history = [dict(epoch=i, coef=80., loss_x100=.1, prediction_std=.03) for i in (1, 2, 3)]
            for p in (out, ref):
                (p / 'model.pth.epochs.jsonl').write_text('\n'.join(map(json.dumps, history)))
            history[2]['coef'] = 70.
            (out / 'model.pth.epochs.jsonl').write_text('\n'.join(map(json.dumps, history)))
            run.check_warmup(out, c, root / 'ref')
            history[0]['coef'] = 70.
            (out / 'model.pth.epochs.jsonl').write_text('\n'.join(map(json.dumps, history)))
            with self.assertRaisesRegex(ValueError, 'trajectory changed'):
                run.check_warmup(out, c, root / 'ref')

    def test_copied_references_are_idempotent_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'source'; dest = root / 'new' / 'dest'
            source.write_bytes(b'original')
            run.copy_unchanged(source, dest)
            run.copy_unchanged(source, dest)
            self.assertEqual(source.read_bytes(), dest.read_bytes())
            dest.write_bytes(b'user data')
            with self.assertRaisesRegex(ValueError, 'already differs'):
                run.copy_unchanged(source, dest)
            self.assertEqual(dest.read_bytes(), b'user data')

    def test_v8_source_or_environment_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = dict(commits=run.V8_COMMITS, inputs={}, environment={})
            core.save(root / 'source.json', old)
            core.save(root / 'completion.json', dict(source=old, new_cells=36, experiments_complete=True))
            with self.assertRaisesRegex(ValueError, 'environment'):
                run.checked_v8(root, dict(old, environment={'changed': True}), root / 'new')
            old['commits'] = {}
            core.save(root / 'source.json', old)
            with self.assertRaisesRegex(ValueError, 'completed frozen'):
                run.checked_v8(root, old, root / 'new')

    def test_changed_completed_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); core.save(root / 'train_result.json', dict(value=1))
            core.save(root / 'complete.json', dict(files={'train_result.json': core.sha(root / 'train_result.json')}))
            run.verify_record(root)
            core.save(root / 'train_result.json', dict(value=2))
            with self.assertRaisesRegex(ValueError, 'changed'):
                run.verify_record(root)

    def test_summary_keeps_weak_seed_and_separates_confirmation_scope(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(run, 'verify_record'), \
             patch.object(run, 'validate_cell', side_effect=lambda out, c: fake_row(c)), \
             patch.object(run.grad, 'validate_cell', side_effect=lambda out, c: fake_row(c)), patch.object(run, 'check_warmup'):
            result = run.summarize(Path(temp), run.matrix(), fixture_references(), fixture_v8())
            self.assertEqual(len(result['raw']), 21)
            self.assertEqual([r['cells'] for r in result['methods']], [6, 6, 9])
            self.assertEqual(result['methods'][0]['mean_srcc_x100'], 75.5)
            self.assertEqual(result['methods'][0]['delta_vs_ordered'], -4.5)
            self.assertEqual(result['methods'][2]['seeds'], [0, 1, 2])
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                run.summarize(Path(temp), run.matrix()[:-1], fixture_references(), fixture_v8())

    def test_smoke_failure_blocks_new_training(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            args = types.SimpleNamespace(stage='all', gpus='4', july28_repo=temp, reference_root=temp, v8_root=temp)
            with patch.object(run.cal, 'check_resources'), patch.object(core, 'provenance', return_value={}), \
                 patch.object(run.grad, 'checked_reference', return_value=[]), patch.object(run, 'checked_v8', return_value=[]), \
                 patch.object(core, 'command', side_effect=RuntimeError('unit failed')), patch.object(run, 'run_wave') as wave:
                with self.assertRaisesRegex(RuntimeError, 'unit failed'):
                    run.execute(args, Path(temp))
                wave.assert_not_called()
                self.assertFalse((Path(temp) / 'completion.json').exists())

    def test_execute_smoke_on_one_gpu_then_fixed15_and_package_without_weights(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            args = types.SimpleNamespace(stage='all', gpus='0 1 3 4', july28_repo=temp, reference_root=temp, v8_root=temp)
            def wave(path, repos, jobs, gpus, smoke=False):
                for c in jobs:
                    out = core.cell_path(path, c); out.mkdir(parents=True, exist_ok=True)
                    core.save(out / 'complete.json', dict(fixture=True))
                    (out / 'model.pth').write_bytes(b'fixture not a model')
            with patch.object(run.cal, 'check_resources'), patch.object(core, 'provenance', return_value={}), \
                 patch.object(run.grad, 'checked_reference', return_value=fixture_references()), \
                 patch.object(run, 'checked_v8', return_value=fixture_v8()), patch.object(core, 'command'), \
                 patch.object(run, 'run_wave', side_effect=wave) as waves, patch.object(run, 'verify_record'), \
                 patch.object(run, 'validate_cell', side_effect=lambda out, c, smoke=False: fake_row(c)), \
                 patch.object(run.grad, 'validate_cell', side_effect=lambda out, c: fake_row(c)), patch.object(run, 'check_warmup'):
                run.execute(args, root)
                self.assertEqual(waves.call_args_list[0].args[3], ['0'])
                self.assertEqual(len(waves.call_args_list[0].args[2]), 8)
                self.assertEqual(waves.call_args_list[1].args[3], ['0', '1', '3', '4'])
                self.assertEqual(len(waves.call_args_list[1].args[2]), 15)
                self.assertEqual(core.read(root / 'completion.json')['new_cells'], 15)
                archive = core.package(root, '')
                with zipfile.ZipFile(archive) as z:
                    self.assertIn('summary.json', z.namelist())
                    self.assertFalse(any(n.endswith(('.pth', '.pt')) for n in z.namelist()))

    def test_status_shows_new_and_reference_counts(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()) as output:
            with patch('sys.argv', ['run', '--stage', 'status', '--root', temp]):
                run.main()
            self.assertIn('Split signals: 0/12', output.getvalue())
            self.assertIn('Guard seed2 confirmation: 0/3', output.getvalue())
            self.assertIn('Frozen guard reference: 0/6', output.getvalue())


if __name__ == '__main__':
    unittest.main()
