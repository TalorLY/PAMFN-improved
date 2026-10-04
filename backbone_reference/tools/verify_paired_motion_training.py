import argparse
import json
import contextlib
import csv
import hashlib
import io
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import main as entrypoint
from pipeline import train_epoch, train_loop
from tools.analyze_direct_cue_effect import (
    analyze_direct_effects,
    apply_paired_motion_decision,
)


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class SpyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.records = []

    def forward(self, model_input, force_gates=None):
        motion, skeleton = force_gates if force_gates is not None else (
            None, None
        )
        self.records.append({
            "batch": int(model_input["V"].shape[0]),
            "motion": (
                None if motion is None
                else torch.as_tensor(motion).detach().clone()
            ),
            "skeleton": (
                None if skeleton is None
                else torch.as_tensor(skeleton).detach().clone()
            ),
        })
        value = model_input["V"].reshape(model_input["V"].shape[0], -1)[
            :, 0
        ]
        if motion is None:
            motion = torch.zeros_like(value)
        else:
            motion = torch.as_tensor(
                motion, dtype=value.dtype, device=value.device
            )
            if motion.ndim == 0:
                motion = motion.repeat(len(value))
        return (self.scale * (value + motion)).reshape(-1, 1), {}

    @staticmethod
    def call_loss(prediction, label, **kwargs):
        del kwargs
        return torch.mean(
            (prediction.reshape(-1) - label.reshape(-1)) ** 2
        )


def one_batch():
    video = torch.tensor([[[1.0]], [[2.0]]])
    audio = torch.zeros_like(video)
    flow = torch.zeros_like(video)
    label = torch.tensor([0.25, 0.75])
    sample_ids = ["a", "b"]
    return [(video, audio, flow, label, sample_ids)]


def verify_training_protocol():
    model = SpyModel()
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    initial = float(model.scale.detach())
    x = torch.tensor([1.0, 2.0, 2.0, 3.0])
    y = torch.tensor([0.25, 0.75, 0.25, 0.75])
    expected_gradient = float(
        torch.mean(2.0 * (initial * x - y) * x)
    )
    result = train_epoch(
        model, optimizer, None, one_batch(), 0,
        paired_motion_training="off_on",
    )
    assert len(model.records) == 1
    assert model.records[0]["batch"] == 4
    assert torch.equal(
        model.records[0]["motion"],
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
    )
    assert torch.equal(
        model.records[0]["skeleton"], torch.zeros(4)
    )
    assert optimizer.step_count == 1
    expected_scale = initial - 0.1 * expected_gradient
    assert abs(float(model.scale.detach()) - expected_scale) < 1e-6
    endpoints = result["paired_endpoints"]
    expected_off = float(torch.mean((initial * x[:2] - y[:2]) ** 2))
    expected_on = float(torch.mean((initial * x[2:] - y[2:]) ** 2))
    assert abs(endpoints["off_loss_x100"] - 100 * expected_off) < 1e-6
    assert abs(endpoints["on_loss_x100"] - 100 * expected_on) < 1e-6
    assert abs(
        result["objective_x100"]
        - 50 * (expected_off + expected_on)
    ) < 1e-6

    control = SpyModel()
    control_optimizer = CountingSGD(control.parameters(), lr=0.1)
    train_epoch(
        control, control_optimizer, None, one_batch(), 0,
        paired_motion_training="off_off",
    )
    assert len(control.records) == 1
    assert control.records[0]["batch"] == 4
    assert torch.equal(control.records[0]["motion"], torch.zeros(4))
    assert control_optimizer.step_count == 1

    legacy = SpyModel()
    legacy_optimizer = CountingSGD(legacy.parameters(), lr=0.1)
    legacy_result = train_epoch(
        legacy, legacy_optimizer, None, one_batch(), 0
    )
    assert len(legacy.records) == 1
    assert legacy.records[0]["batch"] == 2
    assert legacy.records[0]["motion"] is None
    assert legacy_result["paired_endpoints"] is None

    scheduled = SpyModel()
    scheduled_optimizer = CountingSGD(scheduled.parameters(), lr=0.1)
    train_epoch(
        scheduled, scheduled_optimizer, None, one_batch(), 0,
        gate_schedule=[(0.5, 0.0)],
    )
    assert torch.equal(
        scheduled.records[0]["motion"], torch.tensor(0.5)
    )
    print("paired batch/loss/gradient and legacy regression checks: passed")


class ForbiddenLoader:
    def __iter__(self):
        raise AssertionError("blind training iterated the lockbox")


def verify_blind_training():
    model = SpyModel()
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    messages = []
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = os.path.join(directory, "blind.pth")
        result = train_loop(
            model,
            optimizer,
            None,
            one_batch(),
            ForbiddenLoader(),
            epoch_num=1,
            evaluation_name="calibration",
            save_final_model=checkpoint,
            log_func=messages.append,
            paired_motion_training="off_on",
            calibration_final_only=True,
            blind_train=True,
        )
        assert os.path.isfile(checkpoint)
        assert all(np.isfinite(value) for value in result)
    assert any(
        "calibration loader was not iterated" in message
        for message in messages
    )
    print("blind final-checkpoint training: passed")


def parse_cli(arguments, should_pass):
    original = sys.argv
    try:
        sys.argv = ["main.py", *arguments]
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                entrypoint.parser_args()
                passed = True
            except SystemExit as error:
                if error.code not in (0, 2):
                    raise
                passed = error.code == 0
    finally:
        sys.argv = original
    assert passed is should_pass


def paired_cli(mode, motion=False):
    arguments = [
        "--gpu", "-1", "--feats", "2", "--action", "TES",
        "--multi_modality", "--seed", "0", "--split_seed", "20260719",
        "--calibration_ratio", "0.16", "--split_file", "locked.json",
        "--calibration_final_only", "--blind_train",
        "--save_final_model", "model.pth",
        "--batch_size", "16", "--paired_motion_training", mode,
    ]
    if motion:
        arguments.append("--motion_saliency_proxy")
    return arguments


def verify_cli_contract():
    parse_cli(paired_cli("off_off"), True)
    parse_cli(paired_cli("off_on", motion=True), True)
    parse_cli(paired_cli("off_on"), False)
    parse_cli(paired_cli("off_off", motion=True), False)
    parse_cli(
        paired_cli("off_on", motion=True) + ["--init_ckpt", "old.pth"],
        False,
    )
    missing_final_only = paired_cli("off_on", motion=True)
    missing_final_only.remove("--calibration_final_only")
    parse_cli(missing_final_only, False)
    missing_blind = paired_cli("off_on", motion=True)
    missing_blind.remove("--blind_train")
    parse_cli(missing_blind, False)
    parse_cli(
        paired_cli("off_on", motion=True) + ["--use_skeleton"], False
    )
    print("paired CLI fail-fast contract: passed")


def verify_prediction_manifest_writer():
    with tempfile.TemporaryDirectory() as directory:
        branch_dir = os.path.join(directory, "branches")
        os.makedirs(branch_dir)
        for filename in (
                "TES_rgb_UMT.pth", "TES_flow_I3D.pth",
                "TES_audio_MAST.pth"):
            with open(os.path.join(branch_dir, filename), "wb") as handle:
                handle.write(filename.encode("ascii"))
        checkpoint = os.path.join(directory, "paired.pth")
        split = os.path.join(directory, "split.json")
        prediction = os.path.join(directory, "prediction.csv")
        manifest = os.path.join(directory, "prediction.manifest.json")
        for path, content in (
                (checkpoint, b"checkpoint"),
                (split, b"{}"),
                (prediction, b"sample_id,label,prediction\n")):
            with open(path, "wb") as handle:
                handle.write(content)
        args = argparse.Namespace(
            prediction_role="paired_g1_train",
            prediction_split="train",
            prediction_motion_gate=1.0,
            prediction_skeleton_gate=0.0,
            action="TES",
            motion_saliency_proxy=True,
            motion_saliency_alpha=0.015,
            motion_saliency_source="delta",
            motion_saliency_positive_only=True,
            use_skeleton=False,
            anchored_residual=False,
            split_file=split,
            seed=0,
            split_seed=20260719,
            prediction_output=prediction,
            prediction_manifest_output=manifest,
            ckpt_path=checkpoint,
        )
        model_config = {
            "model_name": "pamfn_final",
            "model_conf": {
                "ckpt_dir": branch_dir,
                "rgb_ckpt_name": "UMT",
                "flow_ckpt_name": "I3D",
                "audio_ckpt_name": "MAST",
            },
        }
        entrypoint.write_prediction_manifest(
            args, model_config, {"sample_ids": ["a", "b"]}
        )
        with open(manifest, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["role"] == "paired_g1_train"
        assert set(payload["branch_checkpoints"]) == {
            "rgb", "flow", "audio"
        }
        assert all(
            os.path.isfile(record["path"])
            for record in payload["branch_checkpoints"].values()
        )
    print("prediction provenance manifest writer: passed")


def synthetic_table(prefix, labels, base, on, off):
    count = len(labels)
    return {
        "sample_ids": [f"{prefix}_{index}" for index in range(count)],
        "group_ids": [f"{prefix}_{index}" for index in range(count)],
        "labels": labels,
        "predictions": np.column_stack((base, on, off, base, base)),
        "inputs": {},
    }


def paired_report(train, calibration, bootstrap=200):
    report = analyze_direct_effects(
        train, calibration, gate_step=1.0,
        bootstrap=bootstrap, seed=0,
        cue_bounds={"motion": 1.0, "skeleton": 1.0},
    )
    return apply_paired_motion_decision(report)


def verify_paired_decisions():
    rng = np.random.RandomState(7)

    def positive(prefix, count):
        labels = rng.uniform(0.15, 0.85, count)
        off = np.clip(labels + rng.normal(0, 0.10, count), 0, 1)
        good = np.arange(count) % 4 != 0
        on = np.where(
            good,
            labels + rng.normal(0, 0.015, count),
            labels + rng.normal(0, 0.16, count),
        )
        return synthetic_table(
            prefix, labels, off, np.clip(on, 0, 1), off
        )

    report = paired_report(positive("train", 336), positive("cal", 128))
    assert report["paired_motion_decision"]["status"] == "GO_AUTOGATE"
    assert report["paired_motion_decision"]["mechanism_qualified"]
    assert report["paired_motion_decision"][
        "authorize_next_stage_oof_autogate_experiment"
    ]
    assert not report["paired_motion_decision"]["autogate_result_claimed"]
    half_grid = analyze_direct_effects(
        positive("half_train", 336), positive("half_cal", 128),
        gate_step=0.5, bootstrap=20, seed=0,
        cue_bounds={"motion": 1.0, "skeleton": 1.0},
    )
    try:
        apply_paired_motion_decision(half_grid)
    except ValueError:
        pass
    else:
        raise AssertionError("paired decision accepted an interior gate grid")

    labels = rng.uniform(0.15, 0.85, 336)
    off = np.clip(labels + rng.normal(0, 0.08, 336), 0, 1)
    constant_train = synthetic_table(
        "constant_train", labels, off, np.clip(off + 0.01, 0, 1), off
    )
    labels_cal = rng.uniform(0.15, 0.85, 128)
    off_cal = np.clip(labels_cal + rng.normal(0, 0.08, 128), 0, 1)
    constant_cal = synthetic_table(
        "constant_cal", labels_cal, off_cal,
        np.clip(off_cal + 0.01, 0, 1), off_cal
    )
    constant_report = paired_report(constant_train, constant_cal)
    assert not constant_report["paired_motion_decision"][
        "sample_routing_headroom_supported"
    ]

    bad_train = positive("bad_train", 336)
    bad_cal = positive("bad_cal", 128)
    bad_cal["predictions"][:, 0] = bad_cal["labels"]
    bad_cal["predictions"][:, 3] = bad_cal["labels"]
    bad_cal["predictions"][:, 4] = bad_cal["labels"]
    guardrail_report = paired_report(bad_train, bad_cal)
    assert not guardrail_report["paired_motion_decision"]["guardrail"][
        "supported"
    ]
    assert guardrail_report["paired_motion_decision"]["status"] == "NO_GO"
    json.dumps(report, allow_nan=False)
    print("paired GO/constant/guardrail decision controls: passed")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def verify_finalizer():
    rng = np.random.RandomState(11)

    def make_table(prefix, count):
        labels = rng.uniform(0.15, 0.85, count)
        off = np.clip(labels + rng.normal(0, 0.08, count), 0, 1)
        on = np.clip(labels + rng.normal(0, 0.04, count), 0, 1)
        return synthetic_table(prefix, labels, off, on, off)

    train_table = make_table("train", 336)
    cal_table = make_table("cal", 64)
    report = paired_report(train_table, cal_table, bootstrap=2000)
    with tempfile.TemporaryDirectory() as directory:
        report_path = os.path.join(directory, "report.json")
        split_path = os.path.join(directory, "split.json")
        audit_path = os.path.join(directory, "audit.json")
        control_path = os.path.join(directory, "control.pth")
        paired_path = os.path.join(directory, "paired.pth")
        control_log = os.path.join(directory, "control.log")
        paired_log = os.path.join(directory, "paired.log")
        branch_dir = os.path.join(directory, "branches")
        os.makedirs(branch_dir)
        split = {
            "dataset_family": "FISV",
            "seed": 20260719,
            "calibration_ratio": 0.16,
            "train_ids": [f"train_{index}" for index in range(336)],
            "calibration_ids": [f"cal_{index}" for index in range(64)],
        }
        with open(split_path, "w", encoding="utf-8") as handle:
            json.dump(split, handle)
        audit = {
            "manifest_sha256": sha256(split_path),
            "checks": {"all": True},
        }
        with open(audit_path, "w", encoding="utf-8") as handle:
            json.dump(audit, handle)
        label_path = os.path.join(directory, "train.txt")
        test_label_path = os.path.join(directory, "test.txt")
        with open(label_path, "w", encoding="utf-8") as handle:
            for table in (train_table, cal_table):
                for sample_id, label in zip(
                        table["sample_ids"], table["labels"]):
                    handle.write(f"{sample_id} {float(label) * 45.0} 0\n")
        with open(test_label_path, "w", encoding="utf-8") as handle:
            handle.write("held_out 0 0\n")
        torch.save({"weight": torch.ones(1)}, control_path)
        torch.save({"weight": torch.zeros(1)}, paired_path)
        digest = "a" * 64
        for path, mode in (
                (control_log, "off_off"), (paired_log, "off_on")):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    f"Initial paired model digest: {digest}\n"
                    "Split: 336 train / 64 calibration\n"
                    f"Paired Motion training: mode={mode}\n"
                    "Saved the final blind-training checkpoint\n"
                    "Blind training complete; the held-out calibration "
                    "loader was never iterated\n"
                    "Gate exposure totals: "
                    + (
                        "m0s0:2100 m1s0:2100\n"
                        if mode == "off_on" else "m0s0:4200\n"
                    )
                )
        for filename in (
                "TES_rgb_UMT.pth", "TES_flow_I3D.pth",
                "TES_audio_MAST.pth"):
            with open(os.path.join(branch_dir, filename), "wb") as handle:
                handle.write(filename.encode("ascii"))
        report["dataset"] = "TES"
        report["inputs"] = {"train": {}, "calibration": {}}
        for split_name, table in (
                ("train", train_table), ("calibration", cal_table)):
            for index, variant in enumerate((
                    "base", "motion_on", "motion_off",
                    "skeleton_on", "skeleton_off")):
                path = os.path.join(
                    directory, f"{split_name}_{variant}.csv"
                )
                with open(path, "w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow((
                        "sample_id", "group_id", "label", "prediction"
                    ))
                    for sample_id, group_id, label, prediction in zip(
                            table["sample_ids"], table["group_ids"],
                            table["labels"], table["predictions"][:, index]):
                        writer.writerow((
                            sample_id, group_id, float(label),
                            float(prediction)
                        ))
                report["inputs"][split_name][variant] = {
                    "path": os.path.abspath(path),
                    "sha256": sha256(path),
                }
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, allow_nan=False)

        def record(path):
            return {
                "path": os.path.abspath(path),
                "sha256": sha256(path),
                "bytes": os.path.getsize(path),
            }

        branch_records = {
            name: record(os.path.join(branch_dir, filename))
            for name, filename in zip(
                ("rgb", "flow", "audio"),
                ("TES_rgb_UMT.pth", "TES_flow_I3D.pth",
                 "TES_audio_MAST.pth"),
            )
        }
        role_specs = {
            "control_train": ("train", "base", 0.0, control_path),
            "paired_g0_train": ("train", "motion_off", 0.0, paired_path),
            "paired_g1_train": ("train", "motion_on", 1.0, paired_path),
            "control_lockbox": (
                "calibration", "base", 0.0, control_path
            ),
            "paired_g0_lockbox": (
                "calibration", "motion_off", 0.0, paired_path
            ),
            "paired_g1_lockbox": (
                "calibration", "motion_on", 1.0, paired_path
            ),
        }
        manifest_paths = {}
        for role, (split_name, variant, gate, checkpoint) in (
                role_specs.items()):
            path = os.path.join(directory, f"{role}.manifest.json")
            manifest_paths[role] = path
            csv_path = report["inputs"][split_name][variant]["path"]
            manifest = {
                "schema_version": 1,
                "state": "LOCKED_PREDICTION_EXPORT_COMPLETE",
                "role": role,
                "action": "TES",
                "training_seed": 0,
                "split_seed": 20260719,
                "prediction_split": split_name,
                "sample_count": 336 if split_name == "train" else 64,
                "forced_gates": {"motion": gate, "skeleton": 0.0},
                "model_protocol": {
                    "motion_saliency_proxy": role.startswith("paired"),
                    "motion_saliency_alpha": 0.015,
                    "motion_saliency_source": "delta",
                    "motion_saliency_positive_only": True,
                    "use_skeleton": False,
                    "anchored_residual": False,
                },
                "prediction_csv": record(csv_path),
                "checkpoint": record(checkpoint),
                "split": record(split_path),
                "branch_checkpoints": branch_records,
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)

        def export_record(role):
            split_name, variant, _, _ = role_specs[role]
            return {
                "manifest": record(manifest_paths[role]),
                "prediction_csv": record(
                    report["inputs"][split_name][variant]["path"]
                ),
            }

        protocol_path = os.path.join(directory, "protocol.json")
        reservation_path = os.path.join(directory, "reserved.json")
        marker_path = os.path.join(directory, "opened.json")
        protocol = {
            "state": "LOCKED_BEFORE_FIRST_LOCKBOX_FORWARD",
            "initial_state_digest": digest,
            "code": {"report": record(report_path)},
            "features": {"report": record(report_path)},
            "data_manifests": {
                "train": record(label_path),
                "test": record(test_label_path),
            },
            "fresh_split": record(split_path),
            "fresh_split_audit": record(audit_path),
            "trained_branches": {
                filename: record(os.path.join(branch_dir, filename))
                for filename in (
                    "TES_rgb_UMT.pth", "TES_flow_I3D.pth",
                    "TES_audio_MAST.pth"
                )
            },
            "checkpoints": {
                "control": record(control_path),
                "paired": record(paired_path),
            },
            "training_logs": {
                "control": record(control_log),
                "paired": record(paired_log),
            },
            "train_exports_locked_before_lockbox": {
                role: export_record(role)
                for role in (
                    "control_train", "paired_g0_train", "paired_g1_train"
                )
            },
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
        with open(protocol_path, "w", encoding="utf-8") as handle:
            json.dump(protocol, handle)
        with open(reservation_path, "w", encoding="utf-8") as handle:
            json.dump({
                "state": "LOCKBOX_ACCESS_RESERVED_BEFORE_FIRST_FORWARD",
                "protocol_lock": record(protocol_path),
                "fresh_split": record(split_path),
                "expected_exports": {
                    role: {
                        "csv": os.path.abspath(
                            report["inputs"][role_specs[role][0]][
                                role_specs[role][1]
                            ]["path"]
                        ),
                        "manifest": os.path.abspath(manifest_paths[role]),
                    }
                    for role in (
                        "control_lockbox", "paired_g0_lockbox",
                        "paired_g1_lockbox",
                    )
                },
            }, handle)
        command = [
            sys.executable,
            os.path.join(ROOT, "tools", "finalize_paired_motion_report.py"),
            "--report", report_path,
            "--control_checkpoint", control_path,
            "--paired_checkpoint", paired_path,
            "--control_log", control_log,
            "--paired_log", paired_log,
            "--split", split_path,
            "--split_audit", audit_path,
            "--branch_dir", branch_dir,
            "--protocol_lock", protocol_path,
            "--lockbox_reservation", reservation_path,
            "--lockbox_open_marker", marker_path,
            "--control_train_manifest", manifest_paths["control_train"],
            "--g0_train_manifest", manifest_paths["paired_g0_train"],
            "--g1_train_manifest", manifest_paths["paired_g1_train"],
            "--control_lockbox_manifest",
            manifest_paths["control_lockbox"],
            "--g0_lockbox_manifest", manifest_paths["paired_g0_lockbox"],
            "--g1_lockbox_manifest", manifest_paths["paired_g1_lockbox"],
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        with open(report_path, "r", encoding="utf-8") as handle:
            finalized = json.load(handle)
        assert finalized["paired_motion_protocol"][
            "initial_state_digest"
        ] == digest
        assert not finalized["paired_motion_protocol"][
            "official_test_evaluated"
        ]
        with open(marker_path, "r", encoding="utf-8") as handle:
            completed_marker = json.load(handle)
        assert completed_marker["state"] == "LOCKBOX_EXPORTS_COMPLETED_ONCE"
    print("paired provenance finalizer: passed")


def verify_split_policy():
    from tools.make_refresh_split import (
        read_fisv_labels,
        stable_key,
        summary,
    )

    with tempfile.TemporaryDirectory() as directory:
        label_path = os.path.join(directory, "train.txt")
        with open(label_path, "w", encoding="utf-8") as handle:
            for index in range(400):
                handle.write(f"sample_{index:03d} {index % 45} 0\n")
        labels = read_fisv_labels(label_path, "TES")
        assert len(labels) == 400
        assert stable_key(1, 2, "x") == stable_key(1, 2, "x")
        assert summary([0.0, 1.0])["mean"] == 0.5

        all_ids = [f"sample_{index:03d}" for index in range(400)]
        parent_cal_set = set(sorted(
            all_ids,
            key=lambda sample_id: hashlib.sha256(
                f"0:{sample_id}".encode("utf-8")
            ).hexdigest(),
        )[:80])
        parent_train = [
            sample_id for sample_id in all_ids
            if sample_id not in parent_cal_set
        ]
        parent_cal = [
            sample_id for sample_id in all_ids
            if sample_id in parent_cal_set
        ]
        assert len(set(parent_train) & set(parent_cal)) == 0
        parent_path = os.path.join(directory, "parent.json")
        output_path = os.path.join(directory, "refresh.json")
        audit_path = os.path.join(directory, "audit.json")
        with open(parent_path, "w", encoding="utf-8") as handle:
            json.dump({
                "dataset_family": "FISV",
                "seed": 0,
                "calibration_ratio": 0.2,
                "train_ids": parent_train,
                "calibration_ids": parent_cal,
            }, handle)
        command = [
            sys.executable,
            os.path.join(ROOT, "tools", "make_refresh_split.py"),
            "--action", "TES",
            "--label_file", label_path,
            "--parent_split", parent_path,
            "--selection_seed", "20260719",
            "--strata", "8",
            "--samples_per_stratum", "8",
            "--output", output_path,
            "--audit_output", audit_path,
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        subprocess.run(command, check=True, capture_output=True, text=True)
        with open(output_path, "r", encoding="utf-8") as handle:
            refresh = json.load(handle)
        with open(audit_path, "r", encoding="utf-8") as handle:
            audit = json.load(handle)
        assert len(refresh["train_ids"]) == 336
        assert len(refresh["calibration_ids"]) == 64
        assert set(parent_cal) <= set(refresh["train_ids"])
        assert set(refresh["calibration_ids"]) <= set(parent_train)
        assert all(audit["checks"].values())
    print("refresh split parsing/determinism primitives: passed")


def verify_protocol_lock_primitives():
    from tools.lock_paired_motion_protocol import read_log, write_new

    with tempfile.TemporaryDirectory() as directory:
        log_path = os.path.join(directory, "train.log")
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("required marker\nmetric=1.25\n")
        assert "required marker" in read_log(
            log_path, ("required marker",)
        )
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write("metric=inf\n")
        try:
            read_log(log_path, ("required marker",))
        except ValueError:
            pass
        else:
            raise AssertionError("protocol lock accepted non-finite log")
        locked = os.path.join(directory, "lock.json")
        write_new(locked, {"state": "locked"})
        try:
            write_new(locked, {"state": "changed"})
        except FileExistsError:
            pass
        else:
            raise AssertionError("protocol lock allowed overwrite")
    print("pre-open protocol lock primitives: passed")


def main():
    verify_training_protocol()
    verify_blind_training()
    verify_cli_contract()
    verify_prediction_manifest_writer()
    verify_paired_decisions()
    verify_finalizer()
    verify_split_policy()
    verify_protocol_lock_primitives()
    print("paired Motion training verification passed")


if __name__ == "__main__":
    main()
