import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools.run_qmd_dual import (cells, cell_args, mode_for, immutable, package,
                                METHODS, SEEDS, BUDGET, validate_cell, run_cell, cell_path, save, dispatch_order)


class DualToolsTest(unittest.TestCase):
    def test_matrix_counts_and_no_duplicate_cells(self):
        self.assertEqual(len(cells()), 126)
        self.assertEqual(len(set(cells())), 126)
        self.assertEqual(len(cells(pilot=True)), 42)
        self.assertEqual(len(cells(smoke=True)), 14)
        self.assertEqual(SEEDS['july28'], (0, 1, 2))
        self.assertEqual(SEEDS['modern'], (0, 1, 42))

    def test_modern_matrix_has_no_rank(self):
        for method in METHODS['modern']:
            self.assertNotEqual(mode_for(method), 'score_rank_qmd')
            argv = cell_args('modern', 'Ball', 42, method, '4')
            self.assertIn('argmax', argv)
            self.assertNotIn('--qmd_rank_weight', argv)
            if method.startswith('ordered'):
                self.assertIn('continuous', argv)
                self.assertIn('--quality_aux_rng_isolated', argv)

    def test_first_four_jobs_include_both_protocols(self):
        jobs = dispatch_order(cells(pilot=True))
        self.assertEqual([c[0] for c in jobs[:4]], ['modern', 'july28', 'modern', 'july28'])
        self.assertTrue(all(c[1] == 'Ball' for c in jobs[:4]))
        self.assertEqual(set(jobs), set(cells(pilot=True)))

    def test_july28_retains_original_protocol_and_only_requested_rank(self):
        for method in METHODS['july28']:
            argv = cell_args('july28', 'TES', 2, method, '3')
            self.assertNotIn('--eval_policy', argv)
            self.assertNotIn('--quality_aux_rng_isolated', argv)
            self.assertNotIn('--ordinal_prototype_target_mode', argv)
            self.assertEqual('--qmd_rank_weight' in argv, method == 'ordered_sr_qmd')

    def test_smoke_changes_schedule_and_epoch_only(self):
        argv = cell_args('modern', 'Ball', 0, 'ordered_score_qmd', '0', True)
        self.assertEqual(argv[argv.index('--epochs')+1], '1')
        self.assertEqual(argv[argv.index('--qmd_warmup')+1], '0')

    def test_device_change_does_not_change_contract(self):
        a = cell_args('july28','Ball',0,'ordered','0')
        b = cell_args('july28','Ball',0,'ordered','4')
        self.assertEqual(a[5:], b[5:])

    def test_invalid_cells_rejected(self):
        with self.assertRaises(ValueError):
            cell_args('july28','Ball',42,'ordered','0')
        with self.assertRaises(ValueError):
            cell_args('modern','Ball',0,'ordered_sr_qmd','0')

    def test_manifest_roundtrip_and_changed_config_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'config.json'
            value = dict(methods=METHODS, seeds=SEEDS, epochs=BUDGET)
            immutable(path, value)
            immutable(path, value)
            with self.assertRaises(ValueError):
                immutable(path, dict(value, epochs=1))

    def test_package_excludes_weights_but_keeps_rng_metadata_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'results'
            root.mkdir()
            for name in ('model.pth','model.pth.eval_rng.pt','model.pth.selection.json','complete.json','model.pth.epochs.jsonl'):
                (root/name).write_text('{}')
            output = package(root, '_partial')
            with zipfile.ZipFile(output) as z:
                self.assertEqual(set(z.namelist()), {'model.pth.selection.json','complete.json','model.pth.epochs.jsonl'})

    def test_validation_rejects_wrong_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = dict(action='Ball',seed=0,tag='ordered',epochs=500,feats=2,eval_policy='argmax')
            for name in ('train_result.json','test_result.json'):
                (root/name).write_text(json.dumps(dict(run=run)))
            with self.assertRaisesRegex(ValueError, 'wrong run identity'):
                validate_cell(root, ('july28','Ball',0,'ordered'))

    def test_run_resume_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            cell=('modern','Ball',0,'score_qmd')
            out=cell_path(root,cell)
            calls=[]
            def fake_command(argv,cwd,log):
                calls.append(argv)
                run=dict(action='Ball',seed=0,tag='score_qmd',epochs=1,feats=2,eval_policy='argmax',
                         ordinal_prototype_weight=0.,qmd=dict(mode='score_qmd'))
                if '--test' not in argv:
                    (out/'model.pth').write_bytes(b'fixture')
                    save(out/'train_result.json',dict(run=run,saved_model_srcc=.8))
                    save(out/'model.pth.selection.json',dict(epoch=1,coef=80.))
                    (out/'model.pth.epochs.jsonl').write_text(json.dumps(dict(epoch=1,coef=80.))+'\n')
                    save(out/'qmd_training.json',dict(mode='score_qmd',epochs=[{}],active_batches=1,
                                                    calibration=dict(beta=.5,degenerate=False)))
                else:
                    save(out/'test_result.json',dict(run=run,coef=80.,loss_x100=1.,prediction_std=.1))
            with patch('tools.run_qmd_dual.command',side_effect=fake_command):
                self.assertEqual(run_cell(root,dict(modern=root),cell,'0',True)[0],'FINISH')
                self.assertEqual(run_cell(root,dict(modern=root),cell,'4',True)[0],'SKIP')
                self.assertEqual(len(calls),2)
                (out/'model.pth').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'checkpoint changed'):
                    run_cell(root,dict(modern=root),cell,'4',True)


if __name__ == '__main__':
    unittest.main()
