#!/usr/bin/env python
"""Verify that the recovered files match the July 28 Ordered snapshot."""

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASHES = {
    "main.py": "c904d8c45556649a80b0446479000ec6a1f52d86a08bd2fbfcc3aebdfbb63f05",
    "pipeline.py": "a99717f710bebc6bd07ca2e92795b3ebcdfc7732c73bf5b28fc5bba18c0938d0",
    "datasets.py": "c7a979e03e32cd1c8010b13f54b6dc4d32da4563b3e86af8510fd6b5de7995fb",
    "models/pamfn.py": "6e47f8259a92b15bcb3cdb6ae3bbe68a7d7e7fa934786db1ec14a4b126d4bad0",
}
EXPECTED_FINAL = {
    "TES": {"best_coef": 0.7654, "best_avg_coef": 0.7543},
    "PCS": {"best_coef": 0.8580, "best_avg_coef": 0.8547},
    "Ball": {"best_coef": 0.8243, "best_avg_coef": 0.7802},
    "Clubs": {"best_coef": 0.8491, "best_avg_coef": 0.8269},
    "Hoop": {"best_coef": 0.8372, "best_avg_coef": 0.8115},
    "Ribbon": {"best_coef": 0.8344, "best_avg_coef": 0.7823},
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_summary(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"best avg coef:\s*([-+0-9.eE]+)\s+"
        r"best coef:\s*([-+0-9.eE]+)",
        text,
    )
    if len(matches) != 1:
        raise RuntimeError(f"SUMMARY_NOT_UNIQUE: {path}")
    return {
        "best_avg_coef": float(matches[0][0]),
        "best_coef": float(matches[0][1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-root", default="")
    parser.add_argument(
        "--output",
        default="reports/july28_recovery/source_manifest.json",
    )
    args = parser.parse_args()

    observed = {
        name: sha256(ROOT / name) for name in EXPECTED_HASHES
    }
    mismatches = {
        name: {"expected": EXPECTED_HASHES[name], "observed": value}
        for name, value in observed.items()
        if value != EXPECTED_HASHES[name]
    }

    main_source = (ROOT / "main.py").read_text(encoding="utf-8")
    pipeline_source = (ROOT / "pipeline.py").read_text(encoding="utf-8")
    model_source = (ROOT / "models/pamfn.py").read_text(encoding="utf-8")
    structural_gates = {
        "fixed_four_grades": "ordinal_num_grades" not in main_source,
        "legacy_scalar_training_loss": (
            "call_training_loss_components" not in pipeline_source
            and "call_training_loss_components" not in model_source
        ),
        "no_prediction_std_diagnostic": "prediction_std" not in pipeline_source,
        "exact_soft_neighbor_target": (
            "torch.full_like(" in model_source
            and "distance, 0.1" in model_source
        ),
        "legacy_loader_rng": (
            "generator=" not in pipeline_source
            and "worker_init_fn=" not in pipeline_source
        ),
        "test_every_epoch": (
            "test_result = test_epoch(model, test_dataloader)"
            in pipeline_source
        ),
    }

    historical = {}
    historical_ok = True
    if args.historical_root:
        root = Path(args.historical_root)
        for action, expected in EXPECTED_FINAL.items():
            log = root / action / f"{action}_prototype_seed0.log"
            checkpoint = root / action / f"{action}_prototype_seed0.pth"
            if not log.is_file() or not checkpoint.is_file():
                historical[action] = {
                    "status": "missing",
                    "log": str(log),
                    "checkpoint": str(checkpoint),
                }
                historical_ok = False
                continue
            metrics = parse_summary(log)
            matches = all(
                abs(metrics[key] - expected[key]) <= 5e-5
                for key in expected
            )
            historical[action] = {
                "status": "complete" if matches else "metric_mismatch",
                "metrics": metrics,
                "checkpoint_bytes": checkpoint.stat().st_size,
            }
            historical_ok &= matches

    passed = (
        not mismatches
        and all(structural_gates.values())
        and historical_ok
    )
    report = {
        "decision": (
            "JULY28_SOURCE_RECOVERY_PASS"
            if passed else "JULY28_SOURCE_RECOVERY_FAIL"
        ),
        "passed": passed,
        "source": "Codex session snapshot immediately before the July 29 diagnostic refactor",
        "expected_hashes": EXPECTED_HASHES,
        "observed_hashes": observed,
        "hash_mismatches": mismatches,
        "structural_gates": structural_gates,
        "historical_artifacts": historical,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(report["decision"])
    print(output)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
