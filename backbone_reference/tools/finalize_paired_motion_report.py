import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analyze_direct_cue_effect import (
    VARIANTS,
    analyze_direct_effects,
    apply_paired_motion_decision,
    merge_prediction_tables,
    validate_splits,
)


BRANCH_FILES = (
    "TES_rgb_UMT.pth",
    "TES_flow_I3D.pth",
    "TES_audio_MAST.pth",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attach locked paired-training provenance to its report"
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--control_checkpoint", required=True)
    parser.add_argument("--paired_checkpoint", required=True)
    parser.add_argument("--control_log", required=True)
    parser.add_argument("--paired_log", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--split_audit", required=True)
    parser.add_argument("--branch_dir", required=True)
    parser.add_argument("--protocol_lock", required=True)
    parser.add_argument("--lockbox_reservation", required=True)
    parser.add_argument("--lockbox_open_marker", required=True)
    parser.add_argument("--control_train_manifest", required=True)
    parser.add_argument("--g0_train_manifest", required=True)
    parser.add_argument("--g1_train_manifest", required=True)
    parser.add_argument("--control_lockbox_manifest", required=True)
    parser.add_argument("--g0_lockbox_manifest", required=True)
    parser.add_argument("--g1_lockbox_manifest", required=True)
    args = parser.parse_args()
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


def write_new(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    content = json.dumps(
        payload, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    with open(path, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def read_tes_labels(path):
    labels = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < 2 or fields[0] in labels:
                raise ValueError(
                    f"invalid TES label row at {path}:{line_number}"
                )
            labels[fields[0]] = float(fields[1]) / 45.0
    if len(labels) != 400:
        raise ValueError("authoritative TES label file must have 400 IDs")
    return labels


def validate_authoritative_labels(merged, protocol_lock):
    records = protocol_lock.get("data_manifests", {})
    train_records = [
        record for record in records.values()
        if os.path.basename(record.get("path", "")) == "train.txt"
    ]
    if len(train_records) != 1:
        raise ValueError("protocol lock has no unique FISV train.txt")
    labels = read_tes_labels(train_records[0]["path"])
    for split_name, table in merged.items():
        expected = np.asarray(
            [labels[sample_id] for sample_id in table["sample_ids"]],
            dtype=np.float64,
        )
        if not np.allclose(
                table["labels"], expected, atol=1e-6, rtol=0):
            raise ValueError(
                f"{split_name} CSV labels differ from FISV TES labels"
            )


def validate_export_manifest(
        path, role, expected_csv, expected_checkpoint, split_path,
        branch_dir, expected_count):
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    split_name = "train" if role.endswith("train") else "calibration"
    gate = 1.0 if "g1" in role else 0.0
    motion = role.startswith("paired")
    required = {
        "schema_version": 1,
        "state": "LOCKED_PREDICTION_EXPORT_COMPLETE",
        "role": role,
        "action": "TES",
        "training_seed": 0,
        "split_seed": 20260719,
        "prediction_split": split_name,
        "sample_count": expected_count,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise ValueError(f"{path} has invalid {key}")
    if manifest.get("forced_gates") != {
            "motion": gate, "skeleton": 0.0}:
        raise ValueError(f"{path} has invalid forced gates")
    expected_model = {
        "motion_saliency_proxy": motion,
        "motion_saliency_alpha": 0.015,
        "motion_saliency_source": "delta",
        "motion_saliency_positive_only": True,
        "use_skeleton": False,
        "anchored_residual": False,
    }
    if manifest.get("model_protocol") != expected_model:
        raise ValueError(f"{path} has invalid model protocol")
    expected_branches = {
        name: file_record(os.path.join(branch_dir, filename))
        for name, filename in zip(("rgb", "flow", "audio"), BRANCH_FILES)
    }
    expected_records = (
        (manifest.get("prediction_csv"), file_record(expected_csv), "CSV"),
        (manifest.get("checkpoint"), file_record(expected_checkpoint),
         "checkpoint"),
        (manifest.get("split"), file_record(split_path), "split"),
        (manifest.get("branch_checkpoints"), expected_branches, "branches"),
    )
    for actual, expected, name in expected_records:
        if actual != expected:
            raise ValueError(f"{path} is not bound to expected {name}")
    return {
        "manifest": file_record(path),
        "prediction_csv": file_record(expected_csv),
    }


def assert_recomputed_report(report, merged, protocol_lock):
    analysis = protocol_lock.get("analysis_protocol", {})
    if analysis != {
            "gate_step": 1.0,
            "bootstrap_iterations": 2000,
            "bootstrap_seed": 0,
            "motion_bound": 1.0,
            "skeleton_bound": 1.0,
            "authoritative_labels": "data/FISV/train.txt TES_column/45"}:
        raise ValueError("protocol lock has an unexpected analysis contract")
    expected_thresholds = {
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
    }
    if protocol_lock.get("preregistered_thresholds") != (
            expected_thresholds):
        raise ValueError("protocol lock has unexpected decision thresholds")
    recomputed = analyze_direct_effects(
        merged["train"], merged["calibration"],
        gate_step=1.0, bootstrap=2000, seed=0,
        cue_bounds={"motion": 1.0, "skeleton": 1.0},
    )
    recomputed = apply_paired_motion_decision(recomputed)
    for key in (
            "sample_counts", "calibration_controls",
            "shared_raw_scale_metrics", "cue_effects",
            "paired_motion_decision"):
        if report.get(key) != recomputed.get(key):
            raise ValueError(
                f"report field {key} was not recomputed from its CSV inputs"
            )


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def state_dict(path):
    checkpoint = torch_load(path)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} is not a state-dict checkpoint")
    for name, value in checkpoint.items():
        if not torch.is_tensor(value) or not torch.isfinite(value).all():
            raise ValueError(f"{path}:{name} is not a finite tensor")
    return checkpoint


def read_log(path, expected_mode):
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()
    if "Traceback" in content or "RuntimeError" in content:
        raise ValueError(f"training log contains an error: {path}")
    digest_matches = re.findall(
        r"Initial paired model digest: ([0-9a-f]{64})", content
    )
    if len(digest_matches) != 1:
        raise ValueError(f"missing unique initial digest in {path}")
    required = (
        "Split: 336 train / 64 calibration",
        "Saved the final blind-training checkpoint",
        "Blind training complete; the held-out calibration loader was never iterated",
        f"mode={expected_mode}",
        (
            "Gate exposure totals: m0s0:2100 m1s0:2100"
            if expected_mode == "off_on"
            else "Gate exposure totals: m0s0:4200"
        ),
    )
    missing = [value for value in required if value not in content]
    if missing:
        raise ValueError(f"{path} is missing log markers: {missing}")
    forbidden = (
        "Final paired calibration endpoints:",
        "ObservedMaxCoef",
        "Exported 64 probe samples",
    )
    leaked = [value for value in forbidden if value in content]
    if leaked:
        raise ValueError(f"{path} accessed lockbox diagnostics: {leaked}")
    return digest_matches[0]


def verify_locked_records(value):
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= set(value):
            current = file_record(value["path"])
            if current != {
                    "path": os.path.abspath(value["path"]),
                    "sha256": value["sha256"],
                    "bytes": value["bytes"]}:
                raise ValueError(
                    f"file changed after protocol lock: {value['path']}"
                )
            return
        for nested in value.values():
            verify_locked_records(nested)
    elif isinstance(value, list):
        for nested in value:
            verify_locked_records(nested)


def validate_report_inputs(report, split):
    inputs = report.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
            "train", "calibration"}:
        raise ValueError("report is missing its ten prediction inputs")
    merged = {}
    for split_name in ("train", "calibration"):
        records = inputs[split_name]
        if set(records) != set(VARIANTS):
            raise ValueError(
                f"report {split_name} inputs do not contain five variants"
            )
        paths = {}
        for variant, record in records.items():
            if file_sha256(record["path"]) != record["sha256"]:
                raise ValueError(
                    f"report input hash changed: {record['path']}"
                )
            paths[variant] = record["path"]
        merged[split_name] = merge_prediction_tables(paths)
    validate_splits(merged["train"], merged["calibration"], 336, 64)
    if set(merged["train"]["sample_ids"]) != set(split["train_ids"]):
        raise ValueError("report train IDs do not match the locked split")
    if set(merged["calibration"]["sample_ids"]) != set(
            split["calibration_ids"]):
        raise ValueError("report lockbox IDs do not match the locked split")
    base_index = VARIANTS.index("base")
    skeleton_indices = (
        VARIANTS.index("skeleton_on"),
        VARIANTS.index("skeleton_off"),
    )
    for split_name, table in merged.items():
        base = table["predictions"][:, base_index]
        if any(
                not np.array_equal(table["predictions"][:, index], base)
                for index in skeleton_indices):
            raise ValueError(
                f"{split_name} Skeleton null differs from matched control"
            )
    return merged


def main():
    args = parse_args()
    with open(args.report, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("dataset") != "TES":
        raise ValueError("paired report dataset must be TES")
    if report.get("decision_authority") != "paired_motion_decision_only":
        raise ValueError("report is not a paired Motion decision report")
    if report.get("sample_counts") != {
            "train": 336,
            "calibration": 64,
            "train_unique_groups": 336,
            "calibration_unique_groups": 64}:
        raise ValueError("paired report sample counts are not 336/64")
    motion = report["cue_effects"]["motion"]
    if not np.isclose(
            motion["fixed_gate"]["grid_step"], 1.0,
            atol=1e-12, rtol=0):
        raise ValueError("paired report is not restricted to gate {0,1}")

    with open(args.split, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    with open(args.split_audit, "r", encoding="utf-8") as handle:
        split_audit = json.load(handle)
    if len(split["train_ids"]) != 336 or len(split["calibration_ids"]) != 64:
        raise ValueError("split is not 336/64")
    if split.get("seed") != 20260719 or split.get(
            "calibration_ratio") != 0.16:
        raise ValueError("split seed/ratio differs from the locked protocol")
    if not all(split_audit["checks"].values()):
        raise ValueError("split audit contains a failed check")
    if split_audit["manifest_sha256"] != file_sha256(args.split):
        raise ValueError("split audit does not match the split manifest")
    if os.path.exists(args.lockbox_open_marker):
        raise FileExistsError("completed lockbox marker already exists")
    merged = validate_report_inputs(report, split)
    with open(args.protocol_lock, "r", encoding="utf-8") as handle:
        protocol_lock = json.load(handle)
    with open(args.lockbox_reservation, "r", encoding="utf-8") as handle:
        reservation = json.load(handle)
    if protocol_lock.get("state") != (
            "LOCKED_BEFORE_FIRST_LOCKBOX_FORWARD"):
        raise ValueError("invalid pre-open protocol lock state")
    if reservation.get("state") != (
            "LOCKBOX_ACCESS_RESERVED_BEFORE_FIRST_FORWARD"):
        raise ValueError("invalid global lockbox reservation state")
    if reservation.get("protocol_lock") != file_record(args.protocol_lock):
        raise ValueError("reservation does not reference the protocol lock")
    if reservation.get("fresh_split") != file_record(args.split):
        raise ValueError("reservation does not reference the fresh split")
    verify_locked_records(protocol_lock)
    required_lock_sections = {
        "code", "features", "data_manifests", "fresh_split",
        "fresh_split_audit", "trained_branches", "training_logs",
        "checkpoints", "train_exports_locked_before_lockbox",
        "analysis_protocol",
    }
    missing_sections = required_lock_sections - set(protocol_lock)
    if missing_sections:
        raise ValueError(
            f"protocol lock is missing sections: {sorted(missing_sections)}"
        )
    locked_bindings = (
        (protocol_lock.get("fresh_split"), args.split, "split"),
        (protocol_lock.get("fresh_split_audit"), args.split_audit, "audit"),
        (protocol_lock.get("checkpoints", {}).get("control"),
         args.control_checkpoint, "control checkpoint"),
        (protocol_lock.get("checkpoints", {}).get("paired"),
         args.paired_checkpoint, "paired checkpoint"),
        (protocol_lock.get("training_logs", {}).get("control"),
         args.control_log, "control log"),
        (protocol_lock.get("training_logs", {}).get("paired"),
         args.paired_log, "paired log"),
    )
    for locked, path, name in locked_bindings:
        if locked != file_record(path):
            raise ValueError(f"{name} does not match the pre-open lock")

    control_initial = read_log(args.control_log, "off_off")
    paired_initial = read_log(args.paired_log, "off_on")
    if control_initial != paired_initial:
        raise ValueError(
            "off/off and off/on models did not start from identical states"
        )
    if protocol_lock.get("initial_state_digest") != control_initial:
        raise ValueError("initial state digest differs from protocol lock")

    control_state = state_dict(args.control_checkpoint)
    paired_state = state_dict(args.paired_checkpoint)
    if set(control_state) != set(paired_state):
        raise ValueError("paired checkpoints have different state-dict keys")
    for name in control_state:
        if control_state[name].shape != paired_state[name].shape:
            raise ValueError(f"checkpoint shape mismatch at {name}")
    motion_parameter_keys = [
        name for name in paired_state
        if name.startswith(("motion_encoder.", "motion_score_residual_head."))
    ]
    if motion_parameter_keys:
        raise ValueError(
            "stage Motion paired training must not add cue parameters"
        )

    current_branches = {
        os.path.abspath(os.path.join(args.branch_dir, filename)):
            file_record(os.path.join(args.branch_dir, filename))
        for filename in BRANCH_FILES
    }
    locked_branches = {
        os.path.abspath(record["path"]): record
        for record in protocol_lock["trained_branches"].values()
    }
    if locked_branches != current_branches:
        raise ValueError("branch checkpoints differ from the pre-open lock")

    inputs = report["inputs"]
    export_specs = {
        "control_train": (
            args.control_train_manifest,
            inputs["train"]["base"]["path"],
            args.control_checkpoint, 336,
        ),
        "paired_g0_train": (
            args.g0_train_manifest,
            inputs["train"]["motion_off"]["path"],
            args.paired_checkpoint, 336,
        ),
        "paired_g1_train": (
            args.g1_train_manifest,
            inputs["train"]["motion_on"]["path"],
            args.paired_checkpoint, 336,
        ),
        "control_lockbox": (
            args.control_lockbox_manifest,
            inputs["calibration"]["base"]["path"],
            args.control_checkpoint, 64,
        ),
        "paired_g0_lockbox": (
            args.g0_lockbox_manifest,
            inputs["calibration"]["motion_off"]["path"],
            args.paired_checkpoint, 64,
        ),
        "paired_g1_lockbox": (
            args.g1_lockbox_manifest,
            inputs["calibration"]["motion_on"]["path"],
            args.paired_checkpoint, 64,
        ),
    }
    exports = {
        role: validate_export_manifest(
            manifest, role, csv_path, checkpoint, args.split,
            args.branch_dir, count,
        )
        for role, (manifest, csv_path, checkpoint, count)
        in export_specs.items()
    }
    train_exports = {
        role: exports[role]
        for role in (
            "control_train", "paired_g0_train", "paired_g1_train"
        )
    }
    if train_exports != protocol_lock[
            "train_exports_locked_before_lockbox"]:
        raise ValueError("train exports differ from the pre-open lock")
    expected_reserved_exports = {
        role: {
            "csv": os.path.abspath(export_specs[role][1]),
            "manifest": os.path.abspath(export_specs[role][0]),
        }
        for role in (
            "control_lockbox", "paired_g0_lockbox",
            "paired_g1_lockbox",
        )
    }
    if reservation.get("expected_exports") != expected_reserved_exports:
        raise ValueError(
            "global reservation does not match exact checkpoint/gate roles"
        )

    validate_authoritative_labels(merged, protocol_lock)
    assert_recomputed_report(report, merged, protocol_lock)
    completed_exports = {
        role: exports[role]
        for role in (
            "control_lockbox", "paired_g0_lockbox",
            "paired_g1_lockbox",
        )
    }
    open_marker = {
        "schema_version": 2,
        "state": "LOCKBOX_EXPORTS_COMPLETED_ONCE",
        "protocol_lock": file_record(args.protocol_lock),
        "global_reservation": file_record(args.lockbox_reservation),
        "exports": completed_exports,
        "repeated_access_forbidden": True,
    }
    write_new(args.lockbox_open_marker, open_marker)

    report["paired_motion_protocol"] = {
        "schema_version": 3,
        "source": "immutable pre-open protocol lock",
        "protocol_lock": file_record(args.protocol_lock),
        "global_lockbox_reservation": file_record(
            args.lockbox_reservation
        ),
        "lockbox_open_marker": file_record(args.lockbox_open_marker),
        "locked_protocol": protocol_lock,
        "old_calibration_retired_into_training": True,
        "new_lockbox_selected_only_from_old_training": True,
        "one_forward_and_one_optimizer_step_per_unique_batch": True,
        "lockbox_evaluated_during_training": False,
        "lockbox_first_model_forward_after_protocol_lock": True,
        "checkpoint_selection_uses_lockbox": False,
        "official_test_evaluated": False,
        "initial_state_digest": control_initial,
    }
    report["warnings"].insert(
        1,
        "This fresh lockbox has one preregistered opening event for the "
        "three locked model/gate exports. Do not tune this mechanism on "
        "its result.",
    )
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(
            report, handle, indent=2, ensure_ascii=False, allow_nan=False
        )
        handle.write("\n")

    decision = report["paired_motion_decision"]
    print(f"paired_motion_status={decision['status']}")
    print(
        "authorize_next_stage_oof_autogate_experiment="
        f"{decision['authorize_next_stage_oof_autogate_experiment']}"
    )
    print(f"next_step={decision['next_step']}")
    print(f"report={os.path.abspath(args.report)}")


if __name__ == "__main__":
    main()
