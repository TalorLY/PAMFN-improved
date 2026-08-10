import argparse
import gc
import hashlib
import json
import os
import re
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


CODE_FILES = (
    "main.py",
    "pipeline.py",
    "models/pamfn.py",
    "datasets.py",
    "configs/FISV_feats2.py",
    "utils/lr_scheduler.py",
    "utils/utils.py",
    "tools/make_refresh_split.py",
    "tools/analyze_direct_cue_effect.py",
    "tools/analyze_paired_train_futility.py",
    "tools/verify_paired_motion_training.py",
    "tools/lock_paired_motion_protocol.py",
    "tools/finalize_paired_motion_report.py",
    "tools/run_paired_motion_tes.sh",
)
FEATURE_FILES = (
    "data/features/FISV_rgb_UMT.npy",
    "data/features/FISV_flow_I3D.npy",
    "data/features/FISV_audio_MAST.npy",
)
DATA_MANIFEST_FILES = (
    "data/FISV/train.txt",
    "data/FISV/test.txt",
)
BRANCH_FILES = (
    "TES_rgb_UMT.pth",
    "TES_flow_I3D.pth",
    "TES_audio_MAST.pth",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Freeze code, data, split, logs, and checkpoints before the "
            "fresh TES lockbox is opened"
        )
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--reservation", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--split_audit", required=True)
    parser.add_argument("--parent_split", required=True)
    parser.add_argument("--parent_branch_dir", required=True)
    parser.add_argument("--branch_dir", required=True)
    parser.add_argument("--control_checkpoint", required=True)
    parser.add_argument("--paired_checkpoint", required=True)
    parser.add_argument("--rgb_log", required=True)
    parser.add_argument("--flow_log", required=True)
    parser.add_argument("--audio_log", required=True)
    parser.add_argument("--control_log", required=True)
    parser.add_argument("--paired_log", required=True)
    parser.add_argument("--control_train_manifest", required=True)
    parser.add_argument("--g0_train_manifest", required=True)
    parser.add_argument("--g1_train_manifest", required=True)
    parser.add_argument("--control_lockbox_csv", required=True)
    parser.add_argument("--g0_lockbox_csv", required=True)
    parser.add_argument("--g1_lockbox_csv", required=True)
    parser.add_argument("--control_lockbox_manifest", required=True)
    parser.add_argument("--g0_lockbox_manifest", required=True)
    parser.add_argument("--g1_lockbox_manifest", required=True)
    args = parser.parse_args()
    if os.path.abspath(args.output) == os.path.abspath(args.reservation):
        parser.error("output and reservation must differ")
    return args


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return {
        "path": os.path.abspath(path),
        "sha256": file_sha256(path),
        "bytes": os.path.getsize(path),
    }


def records(paths):
    return {path: file_record(path) for path in paths}


def read_log(path, required):
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()
    missing = [marker for marker in required if marker not in content]
    if missing:
        raise ValueError(f"{path} is missing markers: {missing}")
    forbidden = (
        "Traceback",
        "RuntimeError",
        "Initialized from",
        "ObservedMaxCoef",
        "Final paired calibration endpoints:",
    )
    found = [marker for marker in forbidden if marker in content]
    if found:
        raise ValueError(f"{path} contains forbidden markers: {found}")
    if re.search(r"(?i)(?<![a-z])(?:nan|[+-]?inf)(?![a-z])", content):
        raise ValueError(f"{path} contains a non-finite numeric token")
    return content


def validate_torch_checkpoint(path):
    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, torch.nn.Module):
        state = checkpoint.state_dict()
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise ValueError(f"{path} is not a model checkpoint")
    tensors = [value for value in state.values() if torch.is_tensor(value)]
    if not tensors or any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError(f"{path} has no finite tensor state")
    del tensors, state, checkpoint
    gc.collect()


def initial_digest(content, path):
    matches = re.findall(
        r"Initial paired model digest: ([0-9a-f]{64})", content
    )
    if len(matches) != 1:
        raise ValueError(f"{path} has no unique initial model digest")
    return matches[0]


def write_new(path, payload):
    if os.path.exists(path):
        raise FileExistsError(f"locked artifact already exists: {path}")
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    content = json.dumps(
        payload, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    with open(path, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def verify_split(split_path, audit_path, parent_split_path):
    with open(split_path, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    with open(audit_path, "r", encoding="utf-8") as handle:
        audit = json.load(handle)
    if split.get("dataset_family") != "FISV":
        raise ValueError("locked split is not FISV")
    if split.get("seed") != 20260719 or split.get(
            "calibration_ratio") != 0.16:
        raise ValueError("locked split seed/ratio is incorrect")
    if len(split["train_ids"]) != 336 or len(
            split["calibration_ids"]) != 64:
        raise ValueError("locked split is not 336/64")
    if set(split["train_ids"]) & set(split["calibration_ids"]):
        raise ValueError("locked split train/lockbox IDs overlap")
    if audit.get("manifest_sha256") != file_sha256(split_path):
        raise ValueError("split audit does not match manifest")
    parent_hash = file_sha256(parent_split_path)
    label_hash = file_sha256(DATA_MANIFEST_FILES[0])
    selection = split.get("selection", {})
    if selection.get("parent_split_sha256") != parent_hash or audit.get(
            "parent_split_sha256") != parent_hash:
        raise ValueError("fresh split is not bound to the parent split")
    if selection.get("label_file_sha256") != label_hash or audit.get(
            "label_file_sha256") != label_hash:
        raise ValueError("fresh split is not bound to the FISV labels")
    if not audit.get("checks") or not all(audit["checks"].values()):
        raise ValueError("split audit contains a failed check")
    counts = audit.get("counts", {})
    expected = {
        "all": 400,
        "parent_train_candidates": 320,
        "retired_parent_calibration_merged_into_train": 80,
        "new_train": 336,
        "new_lockbox": 64,
    }
    if counts != expected:
        raise ValueError(f"split audit counts differ: {counts}")

    with open(parent_split_path, "r", encoding="utf-8") as handle:
        parent = json.load(handle)
    if parent.get("dataset_family") != "FISV" or parent.get("seed") != 0 or (
            parent.get("calibration_ratio") != 0.2):
        raise ValueError("parent split is not the seed-0 FISV 80/20 split")
    parent_train = [str(value) for value in parent["train_ids"]]
    parent_cal = [str(value) for value in parent["calibration_ids"]]
    if len(parent_train) != 320 or len(parent_cal) != 80 or set(
            parent_train) & set(parent_cal):
        raise ValueError("parent split is not disjoint 320/80")

    label_rows = []
    with open(DATA_MANIFEST_FILES[0], "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError(
                    f"FISV label row {line_number} has too few columns"
                )
            label_rows.append((fields[0], float(fields[1]) / 45.0))
    label_order = [sample_id for sample_id, _ in label_rows]
    labels = dict(label_rows)
    if len(label_order) != 400 or len(labels) != 400 or set(label_order) != (
            set(parent_train) | set(parent_cal)):
        raise ValueError("FISV labels do not exactly match the parent split")
    expected_parent_cal = set(sorted(
        label_order,
        key=lambda sample_id: hashlib.sha256(
            f"0:{sample_id}".encode("utf-8")
        ).hexdigest(),
    )[:80])
    if set(parent_cal) != expected_parent_cal:
        raise ValueError("parent calibration IDs are not deterministic seed 0")

    ranked = sorted(parent_train, key=lambda value: (labels[value], value))
    selected = []
    for stratum in range(8):
        candidates = ranked[stratum * 40:(stratum + 1) * 40]
        selected.extend(sorted(
            candidates,
            key=lambda sample_id: hashlib.sha256(
                f"20260719:{stratum}:{sample_id}".encode("utf-8")
            ).hexdigest(),
        )[:8])
    selected = set(selected)
    expected_train = [
        sample_id for sample_id in label_order if sample_id not in selected
    ]
    expected_lockbox = [
        sample_id for sample_id in label_order if sample_id in selected
    ]
    if split["train_ids"] != expected_train or split[
            "calibration_ids"] != expected_lockbox:
        raise ValueError(
            "fresh split differs from the pre-registered 8x8 selection"
        )


def validate_export_manifest(
        path, role, checkpoint_path, split_path, branch_dir,
        expected_count):
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    gate = 0.0 if "g0" in role or role.startswith("control") else 1.0
    expected_motion = role.startswith("paired")
    required = {
        "schema_version": 1,
        "state": "LOCKED_PREDICTION_EXPORT_COMPLETE",
        "role": role,
        "action": "TES",
        "training_seed": 0,
        "split_seed": 20260719,
        "prediction_split": "train",
        "sample_count": expected_count,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise ValueError(f"{path} has invalid {key}: {manifest.get(key)}")
    if manifest.get("forced_gates") != {
            "motion": gate, "skeleton": 0.0}:
        raise ValueError(f"{path} has the wrong forced gates")
    model = manifest.get("model_protocol", {})
    expected_model = {
        "motion_saliency_proxy": expected_motion,
        "motion_saliency_alpha": 0.015,
        "motion_saliency_source": "delta",
        "motion_saliency_positive_only": True,
        "use_skeleton": False,
        "anchored_residual": False,
    }
    if model != expected_model:
        raise ValueError(f"{path} has the wrong model protocol")
    if manifest.get("checkpoint") != file_record(checkpoint_path):
        raise ValueError(f"{path} is not bound to the locked checkpoint")
    if manifest.get("split") != file_record(split_path):
        raise ValueError(f"{path} is not bound to the fresh split")
    expected_branches = {
        name: file_record(os.path.join(branch_dir, filename))
        for name, filename in zip(("rgb", "flow", "audio"), BRANCH_FILES)
    }
    if manifest.get("branch_checkpoints") != expected_branches:
        raise ValueError(f"{path} is not bound to the retrained branches")
    csv_record = manifest.get("prediction_csv")
    if not isinstance(csv_record, dict) or csv_record != file_record(
            csv_record.get("path", "")):
        raise ValueError(f"{path} has an invalid prediction CSV record")
    return {
        "manifest": file_record(path),
        "prediction_csv": csv_record,
    }


def main():
    args = parse_args()
    if os.path.exists(args.output) or os.path.exists(args.reservation):
        raise FileExistsError(
            "protocol was already locked/reserved; refusing repeated access"
        )
    verify_split(args.split, args.split_audit, args.parent_split)

    branch_logs = {
        "rgb": args.rgb_log,
        "flow": args.flow_log,
        "audio": args.audio_log,
    }
    branch_log_records = {}
    for name, path in branch_logs.items():
        read_log(path, (
            "Run: action=TES feats=2 seed=0 epochs=250",
            "Split: 336 train / 64 calibration",
            "Saved the final blind-training branch model",
            "Blind training complete; the held-out calibration loader was never iterated",
        ))
        branch_log_records[name] = file_record(path)

    control_content = read_log(args.control_log, (
        "Run: action=TES feats=2 seed=0 epochs=100",
        "Split: 336 train / 64 calibration",
        "mode=off_off",
        "Saved the final blind-training checkpoint",
        "Gate exposure totals: m0s0:4200",
        "Blind training complete; the held-out calibration loader was never iterated",
    ))
    paired_content = read_log(args.paired_log, (
        "Run: action=TES feats=2 seed=0 epochs=100",
        "Split: 336 train / 64 calibration",
        "mode=off_on",
        "Saved the final blind-training checkpoint",
        "Gate exposure totals: m0s0:2100 m1s0:2100",
        "Blind training complete; the held-out calibration loader was never iterated",
    ))
    control_digest = initial_digest(control_content, args.control_log)
    paired_digest = initial_digest(paired_content, args.paired_log)
    if control_digest != paired_digest:
        raise ValueError("paired scorers did not share one initial state")

    parent_branches = records([
        os.path.join(args.parent_branch_dir, filename)
        for filename in BRANCH_FILES
    ])
    trained_branches = records([
        os.path.join(args.branch_dir, filename)
        for filename in BRANCH_FILES
    ])
    for path in trained_branches:
        validate_torch_checkpoint(path)
    validate_torch_checkpoint(args.control_checkpoint)
    validate_torch_checkpoint(args.paired_checkpoint)
    train_exports = {
        "control_train": validate_export_manifest(
            args.control_train_manifest, "control_train",
            args.control_checkpoint, args.split, args.branch_dir, 336,
        ),
        "paired_g0_train": validate_export_manifest(
            args.g0_train_manifest, "paired_g0_train",
            args.paired_checkpoint, args.split, args.branch_dir, 336,
        ),
        "paired_g1_train": validate_export_manifest(
            args.g1_train_manifest, "paired_g1_train",
            args.paired_checkpoint, args.split, args.branch_dir, 336,
        ),
    }
    expected_lockbox = {
        "control_lockbox": {
            "csv": os.path.abspath(args.control_lockbox_csv),
            "manifest": os.path.abspath(args.control_lockbox_manifest),
        },
        "paired_g0_lockbox": {
            "csv": os.path.abspath(args.g0_lockbox_csv),
            "manifest": os.path.abspath(args.g0_lockbox_manifest),
        },
        "paired_g1_lockbox": {
            "csv": os.path.abspath(args.g1_lockbox_csv),
            "manifest": os.path.abspath(args.g1_lockbox_manifest),
        },
    }
    existing_lockbox = [
        path
        for record in expected_lockbox.values()
        for path in record.values()
        if os.path.exists(path)
    ]
    if existing_lockbox:
        raise FileExistsError(
            "lockbox outputs existed before reservation: "
            f"{existing_lockbox}"
        )
    payload = {
        "schema_version": 1,
        "state": "LOCKED_BEFORE_FIRST_LOCKBOX_FORWARD",
        "decision_authority": "paired_motion_decision_only",
        "action": "TES",
        "training_seed": 0,
        "split_seed": 20260719,
        "split_counts": {"train": 336, "lockbox": 64},
        "training_from_scratch": True,
        "old_checkpoint_initialization_forbidden": True,
        "lockbox_loader_iterations_during_training": 0,
        "epochs": {"modality_branches": 250, "paired_scorers": 100},
        "unique_batch_size": 16,
        "effective_forward_batch_size": 32,
        "control_endpoints": [0.0, 0.0],
        "exposed_endpoints": [0.0, 1.0],
        "endpoint_loss_weights": [0.5, 0.5],
        "motion": {
            "path": "parameter-free stage1/stage2 feature modulation",
            "alpha": 0.015,
            "source": "delta",
            "positive_only": True,
            "motion_specific_parameters": 0,
        },
        "skeleton_enabled": False,
        "official_test_evaluated": False,
        "initial_state_digest": control_digest,
        "code": records(CODE_FILES),
        "features": records(FEATURE_FILES),
        "data_manifests": records(DATA_MANIFEST_FILES),
        "parent_smoke_inputs": {
            "split": file_record(args.parent_split),
            "branches": parent_branches,
        },
        "fresh_split": file_record(args.split),
        "fresh_split_audit": file_record(args.split_audit),
        "trained_branches": trained_branches,
        "training_logs": {
            "branches": branch_log_records,
            "control": file_record(args.control_log),
            "paired": file_record(args.paired_log),
        },
        "checkpoints": {
            "control": file_record(args.control_checkpoint),
            "paired": file_record(args.paired_checkpoint),
        },
        "train_exports_locked_before_lockbox": train_exports,
        "analysis_protocol": {
            "gate_step": 1.0,
            "bootstrap_iterations": 2000,
            "bootstrap_seed": 0,
            "motion_bound": 1.0,
            "skeleton_bound": 1.0,
            "authoritative_labels": "data/FISV/train.txt TES_column/45",
        },
        "preregistered_thresholds": {
            "g0_vs_control_delta_coef_min": -0.5,
            "g0_vs_control_relative_mse_gain_min": -0.02,
            "fixed_endpoint_delta_coef_min": 0.5,
            "fixed_endpoint_relative_mse_gain_min": -0.02,
            "bootstrap_probability_min": 0.90,
            "oracle_relative_mse_gain_min": 0.02,
            "oracle_delta_coef_min": 0.0,
            "oracle_controls": [
                "train_locked_fixed_gate",
                "train_mean_direct_gated_constant",
                "best_train_fitted_gated_constant",
            ],
        },
    }
    write_new(args.output, payload)
    lock_record = file_record(args.output)
    marker = {
        "schema_version": 1,
        "state": "LOCKBOX_ACCESS_RESERVED_BEFORE_FIRST_FORWARD",
        "protocol_lock": lock_record,
        "fresh_split": file_record(args.split),
        "expected_exports": expected_lockbox,
        "repeated_access_forbidden": True,
    }
    write_new(args.reservation, marker)
    print(f"protocol_lock_sha256={lock_record['sha256']}")
    print(f"protocol_lock={os.path.abspath(args.output)}")
    print(f"lockbox_reservation={os.path.abspath(args.reservation)}")


if __name__ == "__main__":
    main()
