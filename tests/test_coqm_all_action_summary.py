import unittest

from tools.summarize_coqm_all_action import (
    ACTIONS,
    FORMAL_SEEDS,
    summarize_formal,
    summarize_screen,
)


class COQMAllActionSummaryTest(unittest.TestCase):
    def test_screen_selects_higher_macro_anchor_mode(self):
        rows = []
        for index, action in enumerate(ACTIONS):
            rows.extend([
                {
                    "action": action,
                    "seed": 0,
                    "method": "coqm_uniform",
                    "test_srcc": 0.70 + index * 0.01,
                },
                {
                    "action": action,
                    "seed": 0,
                    "method": "coqm_train_center",
                    "test_srcc": 0.71 + index * 0.01,
                },
            ])
        report = summarize_screen(rows)
        self.assertEqual(
            report["decision"], "COQM_TRAIN_CENTER_SELECTED"
        )
        self.assertAlmostEqual(
            report["train_center_minus_uniform_pp"], 1.0
        )
        self.assertEqual(report["train_center_action_wins"], 6)

    def test_formal_reports_all_thirty_paired_cells(self):
        rows = []
        for action in ACTIONS:
            for seed in FORMAL_SEEDS:
                rows.extend([
                    {
                        "action": action,
                        "seed": seed,
                        "method": "base",
                        "test_srcc": 0.80,
                    },
                    {
                        "action": action,
                        "seed": seed,
                        "method": "coqm_geom",
                        "test_srcc": 0.805,
                    },
                ])
        report = summarize_formal(rows, "train_centers")
        self.assertEqual(
            report["decision"], "COQM_ALL_ACTION_FORMAL_POSITIVE"
        )
        self.assertAlmostEqual(report["coqm_minus_base_pp"], 0.5)
        self.assertEqual(report["paired_wins"], 30)
        self.assertEqual(report["paired_total"], 30)


if __name__ == "__main__":
    unittest.main()
