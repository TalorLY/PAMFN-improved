import csv
import os
import sys
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import configs.FISV_feats2 as config  # noqa: E402
from models.pamfn import PAMFN  # noqa: E402
from pipeline import (  # noqa: E402
    CUE_FEATURE_COLUMNS,
    build_gate_exposure_schedule,
    cue_reliability_features,
    export_gate_grid_predictions,
    load_checkpoint,
    load_initial_checkpoint,
    make_group_split,
)


class DummyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        value = float(index) / 4
        video = torch.full((4, 2), value)
        audio = torch.zeros(4, 2)
        flow = torch.zeros(4, 2)
        skeleton = torch.zeros(4, 8)
        skeleton[:, 3] = 0.8
        skeleton[:, 7] = 0.6
        return (
            video, audio, flow, skeleton,
            torch.tensor(1.0), value, f"sample_{index}"
        )


class DummyCueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.mse = nn.MSELoss()

    def forward(self, inputs, force_gates=None):
        motion, skeleton = force_gates
        base = inputs["V"].mean(dim=(1, 2))
        prediction = base + 0.01 * motion + 0.02 * skeleton
        return prediction.unsqueeze(1), {}

    def call_loss(self, prediction, label, **kwargs):
        return self.mse(prediction.squeeze(), label.squeeze())


def expect_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def model_kwargs(**overrides):
    current = config.multi_modality
    values = {
        "model_dim": current["model_dim"],
        "fc_drop": current["fc_drop"],
        "fc_r": current["fc_r"],
        "feat_drop": current["feat_drop"],
        "K": current["K"],
        "ms_heads": current["ms_heads"],
        "cm_heads": current["cm_heads"],
        "ckpt_dir": current["ckpt_dir"],
        "rgb_ckpt_name": current["rgb_ckpt_name"],
        "flow_ckpt_name": current["flow_ckpt_name"],
        "audio_ckpt_name": current["audio_ckpt_name"],
        "dataset_name": "TES"
    }
    values.update(overrides)
    return values


def main():
    checkpoint = os.path.join(
        ROOT, "pretrained_models", "feats2", "TES_multimodal.pth"
    )
    base = PAMFN(**model_kwargs())
    load_checkpoint(base, checkpoint)

    cue_model = PAMFN(**model_kwargs(
        use_skeleton=True,
        motion_saliency_proxy=True
    ))
    missing = load_initial_checkpoint(cue_model, checkpoint)
    assert missing
    assert all(key.startswith((
        "skeleton_encoder.",
        "skeleton_score_residual_head."
    )) for key in missing)

    base.eval()
    cue_model.eval()
    torch.manual_seed(0)
    model_input = {
        "V": torch.randn(1, 16, 1024),
        "F": torch.randn(1, 16, 1024),
        "A": torch.randn(1, 16, 768),
        "S": torch.randn(1, 16, 132)
    }
    with torch.no_grad():
        base_score, _ = base(model_input)
        zero_score, _ = cue_model(
            model_input, force_gates=(0.0, 0.0)
        )
        motion_score, _ = cue_model(
            model_input, force_gates=(1.0, 0.0)
        )
    assert torch.allclose(
        base_score, zero_score, atol=1e-7, rtol=0
    )
    assert (motion_score - zero_score).abs().max().item() > 0

    cue_model.zero_grad(set_to_none=True)
    skeleton_score, _ = cue_model(
        model_input, force_gates=(0.0, 0.5)
    )
    skeleton_score.sum().backward()
    skeleton_gradient = (
        cue_model.skeleton_score_residual_head[1].weight.grad
    )
    assert skeleton_gradient is not None
    assert torch.isfinite(skeleton_gradient).all()
    assert skeleton_gradient.abs().sum().item() > 0

    sample_ids = [str(index) for index in range(400)]
    split_a = make_group_split(sample_ids, 0.2, 0)
    split_b = make_group_split(sample_ids, 0.2, 0)
    assert split_a == split_b
    assert len(split_a[0]) == 320
    assert len(split_a[1]) == 80
    assert not set(split_a[0]) & set(split_a[1])

    schedule = build_gate_exposure_schedule(100, 0)
    assert len(schedule) == 100
    assert set(schedule) == {
        (motion, skeleton)
        for motion in (0.0, 0.5, 1.0)
        for skeleton in (0.0, 0.5, 1.0)
    }
    assert schedule.count((1.0, 0.5)) >= 50

    flow = torch.tensor([[[1.0, 2.0], [2.0, 4.0], [4.0, 5.0]]])
    skeleton = torch.zeros(1, 3, 8)
    skeleton[0, :, 3] = torch.tensor([1.0, 0.4, 0.0])
    skeleton[0, :, 7] = torch.tensor([0.8, 0.2, 0.0])
    cue_features = cue_reliability_features({
        "F": flow, "S": skeleton
    })
    transformed_skeleton = skeleton.reshape(1, 3, 2, 4).clone()
    transformed_skeleton[..., :3] = (
        transformed_skeleton[..., :3] * 7.0 + 11.0
    )
    transformed_skeleton = transformed_skeleton.reshape(1, 3, 8)
    transformed_features = cue_reliability_features({
        "F": flow * 7.0,
        "S": transformed_skeleton
    })
    assert torch.allclose(
        cue_features, transformed_features, atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(
        cue_features[0, 3:],
        torch.tensor([0.4, 4.0 / 6.0, 1.0 / 3.0]),
        atol=1e-6, rtol=0
    )
    one_frame_features = cue_reliability_features({
        "F": flow[:, :1], "S": skeleton[:, :1]
    })
    assert torch.equal(
        one_frame_features[0, :3], torch.zeros(3)
    )
    two_frame_features = cue_reliability_features({
        "F": flow[:, :2], "S": skeleton[:, :2]
    })
    assert torch.equal(two_frame_features[0, 1:3], torch.zeros(2))
    zero_flow_features = cue_reliability_features({
        "F": torch.zeros_like(flow), "S": skeleton
    })
    assert torch.equal(zero_flow_features[0, :3], torch.zeros(3))
    assert torch.isfinite(zero_flow_features).all()

    visible_skeleton = skeleton.clone()
    visible_skeleton.reshape(1, 3, 2, 4)[..., 3] = 1.0
    visible_features = cue_reliability_features({
        "F": flow, "S": visible_skeleton
    })
    assert torch.equal(
        visible_features[0, 3:], torch.tensor([1.0, 0.0, 0.0])
    )
    missing_skeleton = skeleton.clone()
    missing_skeleton.reshape(1, 3, 2, 4)[..., 3] = 0.0
    missing_features = cue_reliability_features({
        "F": flow, "S": missing_skeleton
    })
    assert torch.equal(
        missing_features[0, 3:], torch.tensor([0.0, 1.0, 1.0])
    )

    expect_value_error(lambda: cue_reliability_features({"S": skeleton}))
    expect_value_error(lambda: cue_reliability_features({"F": flow}))
    expect_value_error(lambda: cue_reliability_features({
        "F": flow[:, :0], "S": skeleton[:, :0]
    }))
    invalid_flow = flow.clone()
    invalid_flow[0, 0, 0] = float("nan")
    expect_value_error(lambda: cue_reliability_features({
        "F": invalid_flow, "S": skeleton
    }))
    invalid_skeleton = skeleton.clone()
    invalid_skeleton[0, 0, 0] = float("inf")
    expect_value_error(lambda: cue_reliability_features({
        "F": flow, "S": invalid_skeleton
    }))
    expect_value_error(lambda: cue_reliability_features({
        "F": flow, "S": torch.zeros(1, 3, 6)
    }))
    invalid_visibility = skeleton.clone()
    invalid_visibility[0, 0, 3] = 1.1
    expect_value_error(lambda: cue_reliability_features({
        "F": flow, "S": invalid_visibility
    }))

    with tempfile.TemporaryDirectory() as directory:
        output = os.path.join(directory, "grid.csv")
        export_gate_grid_predictions(
            DummyCueModel(),
            DataLoader(DummyDataset(), batch_size=2, shuffle=False),
            output,
            log_func=lambda message: None
        )
        with open(output, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 4
        assert len([key for key in rows[0] if key.startswith("pred_")]) == 9
        assert all(column in rows[0] for column in CUE_FEATURE_COLUMNS)
        assert [row["sample_id"] for row in rows] == [
            f"sample_{index}" for index in range(4)
        ]
    print("official checkpoint strict load: passed")
    print("base == forced (0,0): passed")
    print("Motion residual effect: passed")
    print("Skeleton residual gradient: passed")
    print("deterministic 320/80 split: passed")
    print("balanced gate exposure schedule: passed")
    print("scale-free reliability cues: passed")
    print("nine-column sample export: passed")
    print("calibration probe verification passed")


if __name__ == "__main__":
    main()
