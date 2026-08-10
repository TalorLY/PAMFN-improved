"""Create one write-once, label-free calibration split for a model screen."""

import argparse
import hashlib
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=("TES", "Ball"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--calibration_ratio", type=float, default=0.2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit_output", required=True)
    args = parser.parse_args()
    if not 0 < args.calibration_ratio < 1:
        parser.error("--calibration_ratio must be in (0, 1)")
    if os.path.abspath(args.output) == os.path.abspath(args.audit_output):
        parser.error("manifest and audit paths must differ")
    return args


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path, action, skip_header=False):
    sample_ids = []
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if skip_header and index == 0:
                continue
            fields = line.strip().split()
            if not fields:
                continue
            if action == "Ball" and not fields[0].startswith("Ball"):
                continue
            sample_ids.append(fields[0])
    if not sample_ids:
        raise ValueError(f"{path} contains no {action} samples")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{path} contains duplicate sample IDs")
    return sample_ids


def stable_key(seed, sample_id):
    return hashlib.sha256(
        f"{seed}:{sample_id}".encode("utf-8")
    ).hexdigest()


def write_once(path, payload):
    content = (
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if os.path.exists(path):
        with open(path, "rb") as handle:
            existing = handle.read()
        if existing != content:
            raise FileExistsError(
                f"refusing to overwrite a different locked artifact: {path}"
            )
        return
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(content)


def main():
    args = parse_args()
    if args.action == "TES":
        family = "FISV"
        train_file = "data/FISV/train.txt"
        test_file = "data/FISV/test.txt"
        train_ids = read_ids(train_file, args.action)
        test_ids = read_ids(test_file, args.action)
    else:
        family = "RG"
        train_file = "data/RG/train.txt"
        test_file = "data/RG/test.txt"
        train_ids = read_ids(train_file, args.action, skip_header=True)
        test_ids = read_ids(test_file, args.action, skip_header=True)
    if set(train_ids) & set(test_ids):
        raise ValueError("parent train and official test IDs overlap")

    calibration_size = int(round(
        len(train_ids) * args.calibration_ratio
    ))
    calibration_size = min(
        max(calibration_size, 1), len(train_ids) - 1
    )
    selected = set(sorted(
        train_ids, key=lambda sample_id: stable_key(args.seed, sample_id)
    )[:calibration_size])
    inner_train = [
        sample_id for sample_id in train_ids if sample_id not in selected
    ]
    calibration = [
        sample_id for sample_id in train_ids if sample_id in selected
    ]
    manifest = {
        "dataset_family": family,
        "seed": args.seed,
        "calibration_ratio": args.calibration_ratio,
        "train_ids": inner_train,
        "calibration_ids": calibration,
        "selection": {
            "method": "sha256(seed:sample_id)",
            "model_outputs_used": False,
            "label_values_used": False,
            "sequence_sampling": "normalized_full",
        },
    }
    write_once(args.output, manifest)
    audit = {
        "schema_version": 1,
        "manifest_path": os.path.abspath(args.output),
        "manifest_sha256": file_sha256(args.output),
        "source_files": {
            train_file: file_sha256(train_file),
            test_file: file_sha256(test_file),
        },
        "counts": {
            "parent_train": len(train_ids),
            "inner_train": len(inner_train),
            "calibration": len(calibration),
            "official_test": len(test_ids),
        },
        "checks": {
            "train_calibration_disjoint": not bool(
                set(inner_train) & set(calibration)
            ),
            "train_union_preserved": (
                set(inner_train) | set(calibration) == set(train_ids)
            ),
            "official_test_untouched": not bool(
                (set(inner_train) | set(calibration)) & set(test_ids)
            ),
        },
    }
    if not all(audit["checks"].values()):
        raise AssertionError("split audit failed")
    write_once(args.audit_output, audit)
    print(
        f"{family}: {len(inner_train)} inner-train / "
        f"{len(calibration)} calibration; seed={args.seed}"
    )
    print(f"manifest={os.path.abspath(args.output)}")
    print(f"audit={os.path.abspath(args.audit_output)}")


if __name__ == "__main__":
    main()
