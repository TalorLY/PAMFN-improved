import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.run_sr_qmd_experiments import run_cell, cell_command, import_references
from tools.summarize_sr_qmd import ACTIONS, SEEDS, METHODS, REFERENCES, aggregate, qmd_contract, validate_cell


class SRToolsTest(unittest.TestCase):
    def fixture(self,path,method='sr_qmd'):
        ordered=method!='sr_qmd'
        run=dict(action='Ball',seed=0,tag=method,epochs=1,feats=2,coqm_weight=0.,
                 eval_policy='argmax',eval_seed=20260908,ordinal_prototype_weight=.0001 if ordered else 0.,
                 ordinal_prototype_target_mode='continuous' if ordered else 'discrete',
                 quality_aux_rng_isolated=ordered,qmd=qmd_contract(method,True),
                 ordered_config=dict(dim=128,temperature=.1,margin=.1,order_scale=.1) if ordered else None)
        values={'train_result.json':dict(run=run,saved_model_srcc=.8),
                'test_result.json':dict(run=run,coef=80.,loss_x100=1.,mae=.05,bias=0.,prediction_std_ratio=.8,
                    evaluation=dict(policy='argmax',seed=20260908,passes=1,rng_isolated=True)),
                'qmd_training.json':dict(mode='score_rank_qmd',active_batches=2,epochs=[dict(active_batches=2)],
                    calibration=dict(degenerate=False,beta=.7),
                    rank=dict(weight=.5,gap=.05,gamma=.1,normalization='eligible_pair_count',
                              gradient_verified=True,eligible_pairs=8))}
        path.mkdir(parents=True,exist_ok=True)
        for name,value in values.items():
            if method=='ordered' and name=='qmd_training.json':
                continue
            (path/name).write_text(json.dumps(value),encoding='utf-8')
        (path/'model.pth').write_bytes(b'fixture-not-a-real-checkpoint')
        return values

    def test_all_three_methods_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            for method in METHODS:
                path=Path(directory)/method
                self.fixture(path,method)
                self.assertEqual(validate_cell(path,'Ball',0,method,True)['srcc'],80.)

    def test_empty_rank_and_reload_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            values=self.fixture(path)
            values['qmd_training.json']['rank']['gradient_verified']=False
            (path/'qmd_training.json').write_text(json.dumps(values['qmd_training.json']))
            with self.assertRaisesRegex(ValueError,'nonzero gradient'):
                validate_cell(path,'Ball',0,'sr_qmd',True)
            self.fixture(path)
            values['test_result.json']['coef']=79.
            (path/'test_result.json').write_text(json.dumps(values['test_result.json']))
            with self.assertRaisesRegex(ValueError,'reload mismatch'):
                validate_cell(path,'Ball',0,'sr_qmd',True)

    def test_resume_reloads_trained_then_skips_and_allows_gpu_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'Ball/seed0/ordered_sr_qmd'
            self.fixture(path,'ordered_sr_qmd')
            with patch('tools.run_sr_qmd_experiments.command') as command:
                self.assertEqual(run_cell(root,'Ball',0,'ordered_sr_qmd','1',True,1),'FINISH')
                self.assertEqual(command.call_count,1)
                self.assertIn('--test',command.call_args[0][0])
                command.reset_mock()
                self.assertEqual(run_cell(root,'Ball',0,'ordered_sr_qmd','3',True,1),'SKIP')
                command.assert_not_called()

    def test_partial_training_restarts_instead_of_being_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'Ball/seed0/sr_qmd'
            values=self.fixture(path)
            (path/'train_result.json').unlink()
            def command(argv,log):
                if '--test' not in argv:
                    (path/'train_result.json').write_text(json.dumps(values['train_result.json']))
            with patch('tools.run_sr_qmd_experiments.command',side_effect=command) as called:
                self.assertEqual(run_cell(root,'Ball',0,'sr_qmd','2',True,1),'FINISH')
                self.assertEqual(called.call_count,2)

    def test_cli_command_contract(self):
        from main import parser_args, qmd_config
        for method in METHODS:
            argv=cell_command('Ball',0,method,'1',True)
            with patch('sys.argv',argv[2:]):
                args=parser_args()
            self.assertEqual(qmd_config(args),qmd_contract(method,True))

    def rows(self,seeds=SEEDS):
        scores=dict(base=80.,mask_gt=80.3,score_qmd=80.7,sr_qmd=80.9,ordered=80.2,ordered_sr_qmd=81.1)
        return [dict(action=a,seed=s,method=m,srcc=scores[m]) for a in ACTIONS for s in seeds for m in REFERENCES+METHODS]

    def test_three_seed_and_pilot_decisions(self):
        result=aggregate(self.rows())
        self.assertEqual(result['decision'],'RANK_AND_COMBO_SIGNAL')
        self.assertAlmostEqual(result['comparisons']['combo_vs_sr']['delta_pp'],.2)
        self.assertEqual(aggregate(self.rows((0,)),(0,))['decision'],'PILOT_ONLY')

    def test_missing_duplicate_and_combo_not_better_than_components(self):
        rows=self.rows()
        with self.assertRaises(ValueError):
            aggregate(rows[:-1])
        with self.assertRaises(ValueError):
            aggregate(rows[:-1]+[rows[0]])
        for row in rows:
            if row['method']=='ordered_sr_qmd':
                row['srcc']=80.8
        self.assertEqual(aggregate(rows)['decision'],'RANK_SIGNAL')

    def test_source_commit_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'report').mkdir()
            (root/'report/completion.json').write_text('{}')
            (root/'git_commit.txt').write_text('wrong')
            with self.assertRaisesRegex(ValueError,'frozen Score-QMD'):
                import_references(root,root/'dest')


if __name__=='__main__':
    unittest.main()
