import argparse
import hashlib
import json
import math
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Retire an inspected calibration set and create one fresh, "
            "label-stratified lockbox from the parent training IDs"
        )
    )
    parser.add_argument("--action", required=True, choices=("TES", "PCS"))
    parser.add_argument("--label_file", required=True)
    parser.add_argument("--parent_split", required=True)
    parser.add_argument("--selection_seed", type=int, required=True)
    parser.add_argument("--strata", type=int, default=8)
    parser.add_argument("--samples_per_stratum", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit_output", required=True)
    args = parser.parse_args()
    if args.strata < 1 or args.samples_per_stratum < 1:
        parser.error("strata and samples_per_stratum must be positive")
    if os.path.abspath(args.output) == os.path.abspath(args.audit_output):
        parser.error("output and audit_output must be different files")
    return args


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fisv_labels(path, action):
    label_index = 1 if action == "TES" else 2
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) <= label_index:
                raise ValueError(
                    f"{path}:{line_number} has too few columns"
                )
            sample_id = fields[0]
            try:
                label = float(fields[label_index]) / 45.0
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number} has a non-numeric label"
                ) from error
            if not math.isfinite(label):
                raise ValueError(
                    f"{path}:{line_number} has a non-finite label"
                )
            rows.append((sample_id, label))
    sample_ids = [sample_id for sample_id, _ in rows]
    if not rows or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("label file must contain unique, non-empty samples")
    return rows


def stable_key(seed, stratum, sample_id):
    payload = f"{seed}:{stratum}:{sample_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def summary(labels):
    mean = sum(labels) / len(labels)
    variance = sum((value - mean) ** 2 for value in labels) / len(labels)
    return {
        "count": len(labels),
        "mean": mean,
        "std": math.sqrt(variance),
        "minimum": min(labels),
        "maximum": max(labels),
    }


def serialize(payload):
    return (
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_once(path, payload):
    content = serialize(payload)
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
    label_rows = read_fisv_labels(args.label_file, args.action)
    label_order = [sample_id for sample_id, _ in label_rows]
    labels = dict(label_rows)

    with open(args.parent_split, "r", encoding="utf-8") as handle:
        parent = json.load(handle)
    if parent.get("dataset_family") != "FISV" or int(
            parent.get("seed", -1)) != 0 or not math.isclose(
                float(parent.get("calibration_ratio", -1)), 0.2):
        raise ValueError(
            "parent split must be the FISV seed-0 80/20 development split"
        )
    parent_train = [str(value) for value in parent["train_ids"]]
    parent_calibration = [
        str(value) for value in parent["calibration_ids"]
    ]
    if len(parent_train) != len(set(parent_train)) or len(
            parent_calibration) != len(set(parent_calibration)):
        raise ValueError("parent split contains duplicate IDs")
    parent_train_set = set(parent_train)
    parent_calibration_set = set(parent_calibration)
    if parent_train_set & parent_calibration_set:
        raise ValueError("parent train and calibration IDs overlap")
    if parent_train_set | parent_calibration_set != set(label_order):
        raise ValueError(
            "parent split IDs do not exactly match the label file"
        )
    if len(label_order) != 400 or len(parent_train) != 320 or len(
            parent_calibration) != 80:
        raise ValueError(
            "locked TES/PCS refresh protocol expects parent counts 320/80"
        )
    expected_parent_calibration = set(sorted(
        label_order,
        key=lambda sample_id: hashlib.sha256(
            f"0:{sample_id}".encode("utf-8")
        ).hexdigest(),
    )[:80])
    if parent_calibration_set != expected_parent_calibration:
        raise ValueError(
            "parent split does not match deterministic seed-0 selection"
        )
    if len(parent_train) % args.strata:
        raise ValueError(
            "parent training size must be divisible by the number of strata"
        )
    stratum_size = len(parent_train) // args.strata
    if args.samples_per_stratum >= stratum_size:
        raise ValueError(
            "samples_per_stratum must leave training samples in every stratum"
        )

    ranked = sorted(parent_train, key=lambda value: (labels[value], value))
    selected = []
    strata_audit = []
    for stratum in range(args.strata):
        start = stratum * stratum_size
        candidates = ranked[start:start + stratum_size]
        chosen = sorted(
            candidates,
            key=lambda value: stable_key(
                args.selection_seed, stratum, value
            ),
        )[:args.samples_per_stratum]
        selected.extend(chosen)
        strata_audit.append({
            "index": stratum,
            "candidate": summary([labels[value] for value in candidates]),
            "selected": summary([labels[value] for value in chosen]),
        })

    calibration_set = set(selected)
    train_ids = [
        sample_id for sample_id in label_order
        if sample_id not in calibration_set
    ]
    calibration_ids = [
        sample_id for sample_id in label_order
        if sample_id in calibration_set
    ]
    if len(calibration_ids) != args.strata * args.samples_per_stratum:
        raise AssertionError("fresh lockbox size is incorrect")
    if not parent_calibration_set <= set(train_ids):
        raise AssertionError("the retired parent calibration set was not merged")
    if not calibration_set <= parent_train_set:
        raise AssertionError("fresh lockbox escaped the parent training pool")

    parent_hash = file_sha256(args.parent_split)
    label_hash = file_sha256(args.label_file)
    manifest = {
        "dataset_family": "FISV",
        "seed": args.selection_seed,
        "calibration_ratio": len(calibration_ids) / len(label_order),
        "train_ids": train_ids,
        "calibration_ids": calibration_ids,
        "selection": {
            "action": args.action,
            "candidate_pool": "parent.train_ids",
            "forced_train_pool": "parent.calibration_ids",
            "method": (
                f"{args.strata}_equal_rank_label_strata_x_"
                f"{args.samples_per_stratum}"
            ),
            "selection_seed": args.selection_seed,
            "parent_split_sha256": parent_hash,
            "label_file_sha256": label_hash,
            "model_outputs_used": False,
        },
    }
    write_once(args.output, manifest)
    manifest_hash = file_sha256(args.output)

    audit = {
        "schema_version": 1,
        "manifest_path": os.path.abspath(args.output),
        "manifest_sha256": manifest_hash,
        "parent_split_path": os.path.abspath(args.parent_split),
        "parent_split_sha256": parent_hash,
        "label_file_path": os.path.abspath(args.label_file),
        "label_file_sha256": label_hash,
        "counts": {
            "all": len(label_order),
            "parent_train_candidates": len(parent_train),
            "retired_parent_calibration_merged_into_train": len(
                parent_calibration
            ),
            "new_train": len(train_ids),
            "new_lockbox": len(calibration_ids),
        },
        "labels": {
            "all": summary([labels[value] for value in label_order]),
            "new_train": summary([labels[value] for value in train_ids]),
            "new_lockbox": summary([
                labels[value] for value in calibration_ids
            ]),
            "strata": strata_audit,
        },
        "checks": {
            "train_lockbox_disjoint": not bool(
                set(train_ids) & set(calibration_ids)
            ),
            "union_matches_label_file": bool(
                set(train_ids) | set(calibration_ids) == set(label_order)
            ),
            "old_calibration_fully_retired_to_train": bool(
                parent_calibration_set <= set(train_ids)
            ),
            "new_lockbox_only_from_old_train": bool(
                set(calibration_ids) <= parent_train_set
            ),
        },
    }
    if not all(audit["checks"].values()):
        raise AssertionError("fresh split audit failed")
    write_once(args.audit_output, audit)
    print(
        f"locked split: {len(train_ids)} train / "
        f"{len(calibration_ids)} lockbox"
    )
    print(f"manifest_sha256={manifest_hash}")
    print(f"manifest={os.path.abspath(args.output)}")
    print(f"audit={os.path.abspath(args.audit_output)}")


if __name__ == "__main__":
    main()
