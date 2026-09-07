#!/usr/bin/env python
"""Compare a reproduced Ordered run with its July 28 training trajectory."""

import argparse
import csv
import json
import re
from pathlib import Path


EPOCH_RE = re.compile(
    r"^Epoch\[(?P<epoch>\d+)/(?P<total>\d+)\].*?"
    r"Loss\s+(?P<train_loss>[-+0-9.eE]+)/(?P<test_loss>[-+0-9.eE]+).*?"
    r"Objective\s+(?P<objective>[-+0-9.eE]+).*?"
    r"Coef\s+(?P<train_coef>[-+0-9.eE]+)/(?P<test_coef>[-+0-9.eE]+)"
)


def read_epochs(path):
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EPOCH_RE.search(line)
        if not match:
            continue
        row = match.groupdict()
        epoch = int(row.pop("epoch"))
        row["total"] = int(row["total"])
        for key in ("train_loss", "test_loss", "objective", "train_coef", "test_coef"):
            row[key] = float(row[key])
        rows[epoch] = row
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-log", required=True)
    parser.add_argument("--reproduced-log", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--loss-tolerance", type=float, default=0.0002)
    parser.add_argument("--coef-tolerance", type=float, default=0.03)
    args = parser.parse_args()

    historical = read_epochs(Path(args.historical_log))
    reproduced = read_epochs(Path(args.reproduced_log))
    rows = []
    passed = True
    fields = ("train_loss", "test_loss", "objective", "train_coef", "test_coef")
    for epoch in range(args.epochs):
        if epoch not in historical or epoch not in reproduced:
            rows.append({"epoch": epoch, "status": "missing"})
            passed = False
            continue
        row = {"epoch": epoch, "status": "complete"}
        for field in fields:
            old = historical[epoch][field]
            new = reproduced[epoch][field]
            delta = new - old
            tolerance = (
                args.coef_tolerance if "coef" in field
                else args.loss_tolerance
            )
            ok = abs(delta) <= tolerance
            row.update({
                f"historical_{field}": old,
                f"reproduced_{field}": new,
                f"delta_{field}": delta,
                f"pass_{field}": ok,
            })
            passed &= ok
        rows.append(row)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output / "trajectory_comparison.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    decision = {
        "decision": (
            "JULY28_TRAJECTORY_PASS"
            if passed else "JULY28_TRAJECTORY_MISMATCH"
        ),
        "passed": passed,
        "epochs_compared": args.epochs,
        "loss_tolerance": args.loss_tolerance,
        "coef_tolerance": args.coef_tolerance,
    }
    (output / "trajectory_gate.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(decision["decision"])
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
