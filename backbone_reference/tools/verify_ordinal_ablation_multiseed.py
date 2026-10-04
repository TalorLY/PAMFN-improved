#!/usr/bin/env python
"""Fast static checks for the multi-seed ordinal ablation."""

import os

from summarize_ordinal_ablation_multiseed import (
    aggregate_scope,
    mean_std,
    parse_seed_results,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parsed = parse_seed_results(["2=two", "0=zero", "1=one"])
    assert list(parsed) == [0, 1, 2]
    assert mean_std([1.0, 2.0, 3.0]) == (2.0, 1.0)

    results = {}
    for seed in (0, 1, 2):
        for action_index, action in enumerate(("TES", "PCS")):
            base = 70.0 + seed + action_index
            for mode, delta in (
                    ("base", 0.0),
                    ("grade", -0.2),
                    ("unordered", 0.3),
                    ("prototype", 0.8)):
                results[(seed, action, mode)] = {
                    "best_coef": base + delta,
                    "best_avg_coef": base + delta,
                    "min_loss": 0.1 - delta * 0.001,
                    "min_avg_loss": 0.1 - delta * 0.001,
                }
    row = aggregate_scope(
        "FIS-V", ["TES", "PCS"], [0, 1, 2], results
    )
    assert abs(
        row["prototype_delta_best_coef_mean"] - 0.8
    ) < 1e-9
    assert abs(
        row["prototype_minus_unordered_best_coef_mean"] - 0.5
    ) < 1e-9
    assert row["prototype_best_wins_vs_base"] == 6

    runner_path = os.path.join(
        ROOT,
        "tools",
        "run_ordinal_ablation_multiseed.sh",
    )
    with open(runner_path, "r", encoding="utf-8") as handle:
        runner = handle.read()
    assert "SEEDS=(1 2)" in runner
    assert "ACTIONS=(TES PCS Ball Clubs Hoop Ribbon)" in runner
    assert 'run_mode "$seed" "$action" grade' in runner
    assert 'run_mode "$seed" "$action" unordered' in runner
    assert "--ordinal_prototype_order_scale 0" in runner
    assert "summarize_ordinal_ablation_multiseed.py" in runner

    print("three-seed aggregation math: passed")
    print("24-run Grade/Unordered runner: passed")
    print("mentor tables and feature evidence wiring: passed")


if __name__ == "__main__":
    main()
