import subprocess
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

import main as entry
import pipeline
from tests.test_july28_qmd import TinyModel, TinyDataset

V9 = 'ee70832bc82862d432342a6ff0020e8b53f51600'


class FullStudyTest(unittest.TestCase):
    def test_prediction_flag_preserves_base_and_kd_configs(self):
        old = types.ModuleType('v9_main')
        exec(compile(subprocess.check_output(['git', 'show', V9 + ':main.py']).decode(), 'v9/main.py', 'exec'), old.__dict__)
        for action in entry.ACTIONS:
            for auxiliary in ([], ['--ordinal_prototype_weight', '.0001', '--qmd_mode', 'score_qmd', '--qmd_terms', 'kd']):
                argv = ['main.py', '--gpu', '-1', '--feats', '2', '--action', action, '--seed', '2', '--multi_modality'] + auxiliary
                with patch('sys.argv', argv):
                    before = old.parser_args()
                with patch('sys.argv', argv + ['--save_test_predictions']):
                    after = entry.parser_args()
                config = entry.load_config(action, 2)
                with patch.object(old, 'get_train_quality_thresholds', return_value=(.25, .5, .75)), \
                     patch.object(entry, 'get_train_quality_thresholds', return_value=(.25, .5, .75)):
                    self.assertEqual(old.build_configs(before, config), entry.build_configs(after, config))
                self.assertEqual(old.run_metadata(before, 20), entry.run_metadata(after, 20))
                if not auxiliary:
                    self.assertEqual(after.ordinal_prototype_weight, 0)
                    self.assertEqual(after.shared_rank_weight, 0)
                    self.assertEqual(after.shared_list_weight, 0)
                    self.assertEqual(after.qmd_mode, 'off')

    def test_prediction_collection_preserves_test_metrics_and_rng(self):
        rows = []
        for enabled in (False, True):
            pipeline.setup_seed(17)
            model = TinyModel()
            result = pipeline.test_epoch(model, DataLoader(TinyDataset(), batch_size=4), return_predictions=enabled)
            result.pop('time')
            self.assertEqual('predictions' in result, enabled)
            if enabled:
                self.assertEqual(len(result.pop('predictions')['label']), 12)
            rows.append((result, torch.get_rng_state(), np.random.get_state()[1]))
        self.assertEqual(rows[0][0], rows[1][0])
        self.assertTrue(torch.equal(rows[0][1], rows[1][1]))
        self.assertTrue(np.array_equal(rows[0][2], rows[1][2]))

    def test_base_entry_forwards_prediction_flag_only_to_test(self):
        argv = ['main.py', '--gpu', '-1', '--feats', '2', '--action', 'Ball', '--multi_modality', '--save_test_predictions']
        with patch('sys.argv', argv + ['--test']), patch.object(entry, 'test_model', return_value={}) as test, patch.object(entry, 'log_configuration'):
            entry.main()
            self.assertTrue(test.call_args.kwargs['return_predictions'])
            self.assertEqual(test.call_args.kwargs['model_kwargs']['model_conf']['ordinal_prototype_weight'], 0.)
        with patch('sys.argv', argv), patch.object(entry, 'go', return_value=(1, 1, 1, 1, 1)) as go, patch.object(entry, 'log_configuration'):
            entry.main()
            self.assertIsNone(go.call_args.kwargs['qmd_config'])
            self.assertNotIn('return_predictions', go.call_args.kwargs)


if __name__ == '__main__':
    unittest.main()
