import json
import os
import tempfile
import unittest

from tools.summarize_coqm_stage_a import (
    ACTIONS,
    METHODS,
    SEEDS,
    aggregate,
    load_runs,
)


class COQMExperimentToolsTest(unittest.TestCase):
    def test_complete_synthetic_protocol_passes_gate(self):
        scores = {
            "base": 0.700,
            "ordered_discrete": 0.701,
            "ordered_continuous": 0.702,
            "coqm_no_coord": 0.703,
            "coqm": 0.706,
        }
        with tempfile.TemporaryDirectory() as root:
            for action in ACTIONS:
                for seed in SEEDS:
                    for method in METHODS:
                        run_root = os.path.join(
                            root, action, f"seed{seed}", method
                        )
                        os.makedirs(run_root)
                        metadata = {
                            "tag": method,
                            "action": action,
                            "seed": seed,
                            "quality_aux_rng_isolated": method != "base",
                            "ordinal_prototype_target_mode": (
                                "continuous"
                                if method == "ordered_continuous"
                                else "discrete"
                            ),
                            "ordinal_prototype_weight": (
                                1e-4 if method.startswith("ordered") else 0.0
                            ),
                            "coqm_weight": (
                                1e-4 if method.startswith("coqm") else 0.0
                            ),
                            "coqm_coord_scale": (
                                0.0 if method == "coqm_no_coord" else 1.0
                            ),
                        }
                        train = {
                            "best_srcc": scores[method],
                            "best_avg_srcc": scores[method],
                            "saved_model_srcc": scores[method],
                            "run": metadata,
                        }
                        test = {
                            "coef": scores[method] * 100,
                            "loss_x100": (
                                1.02 if method == "coqm" else 1.0
                            ),
                            "prediction_std_ratio": 0.8,
                            "prediction_std": 0.08,
                            "target_std": 0.1,
                            "mae": 0.05,
                            "bias": 0.0,
                            "run": metadata,
                        }
                        if method.startswith("coqm"):
                            test["coqm"] = {
                                "curve_cosine": 0.7,
                                "coordinate_mae": 0.1,
                                "quality_std_ratio": 0.7,
                                "orthogonal_error": 0.0,
                                "score_q_spearman": 0.8,
                                "score_q_pearson": 0.8,
                                "soft_occupancy": [0.25] * 4,
                                "hard_occupancy": [0.25] * 4,
                                "adjacent_cosine_distance": [0.2] * 3,
                            }
                        for name, value in (
                                ("train_result.json", train),
                                ("test_result.json", test)):
                            with open(
                                    os.path.join(run_root, name),
                                    "w", encoding="utf-8") as handle:
                                json.dump(value, handle)
            gate = aggregate(load_runs(root))
        self.assertEqual(gate["decision"], "COQM_STAGE_A_PASS")
        self.assertAlmostEqual(gate["coqm_minus_base_pp"], 0.6)


if __name__ == "__main__":
    unittest.main()
