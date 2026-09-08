import json
import os
import tempfile
import unittest

from tools.summarize_coqm_eval_b0 import (
    ACTIONS,
    METHODS,
    POLICIES,
    SEEDS,
    aggregate,
    load_runs,
)


class COQMEvalB0ToolsTest(unittest.TestCase):
    @staticmethod
    def _result(action, seed, method, policy):
        result = {
            "coef": 70.0,
            "loss_x100": 1.0,
            "prediction_std_ratio": 0.8,
            "prediction_std": 0.1,
            "target_std": 0.125,
            "mae": 0.05,
            "bias": 0.0,
            "run": {
                "tag": method,
                "action": action,
                "seed": seed,
            },
        }
        if policy != "legacy_gumbel":
            passes = 20 if policy == "mc_gumbel" else 1
            result["run"].update({
                "eval_policy": policy,
                "eval_seed": 20260908,
            })
            result["evaluation"] = {
                "policy": policy,
                "passes": passes,
                "seed": 20260908,
                "rng_isolated": True,
            }
        return result

    def test_loader_requires_all_legacy_and_new_results(self):
        with tempfile.TemporaryDirectory() as checkpoint_root:
            with tempfile.TemporaryDirectory() as root:
                for action in ACTIONS:
                    for seed in SEEDS:
                        for method in METHODS:
                            for policy in POLICIES:
                                if policy == "legacy_gumbel":
                                    path = os.path.join(
                                        checkpoint_root,
                                        action,
                                        f"seed{seed}",
                                        method,
                                        "test_result.json",
                                    )
                                else:
                                    path = os.path.join(
                                        root,
                                        action,
                                        f"seed{seed}",
                                        method,
                                        policy,
                                        "test_result.json",
                                    )
                                os.makedirs(os.path.dirname(path))
                                with open(
                                        path, "w", encoding="utf-8"
                                ) as handle:
                                    json.dump(self._result(
                                        action, seed, method, policy
                                    ), handle)
                rows = load_runs(checkpoint_root, root)
        self.assertEqual(len(rows), 135)

    def test_signal_must_pass_under_argmax_and_mc(self):
        rows = []
        scores = {
            "base": 0.700,
            "ordered_discrete": 0.701,
            "ordered_continuous": 0.702,
            "coqm_no_coord": 0.708,
            "coqm": 0.703,
        }
        for policy in POLICIES:
            for action in ACTIONS:
                for seed in SEEDS:
                    for method in METHODS:
                        rows.append({
                            "policy": policy,
                            "action": action,
                            "seed": seed,
                            "method": method,
                            "test_srcc": scores[method],
                            "test_mse": 0.01,
                            "prediction_std_ratio": 0.8,
                        })
        gate = aggregate(rows)
        self.assertEqual(
            gate["decision"], "COQM_EVAL_B0_SIGNAL_PERSISTS"
        )
        self.assertTrue(gate["diagnostic_only"])
        for policy in ("argmax", "mc_gumbel"):
            self.assertTrue(gate["policies"][policy]["signal_pass"])


if __name__ == "__main__":
    unittest.main()
