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

from pipeline import export_predictions  # noqa: E402


class PredictionDataset(Dataset):
    def __init__(self, duplicate_ids=False):
        self.duplicate_ids = duplicate_ids

    def __len__(self):
        return 4

    def __getitem__(self, index):
        value = float(index) / 10
        sample_id = "duplicate" if self.duplicate_ids else f"sample_{index}"
        return (
            torch.full((3, 2), value),
            torch.zeros(3, 2),
            torch.zeros(3, 2),
            torch.tensor(value),
            sample_id
        )


class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError(index)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.mse = nn.MSELoss()

    def forward(self, inputs, force_gates=None):
        if force_gates is None:
            motion, skeleton = 0.6, 0.4
        else:
            motion, skeleton = force_gates
            motion = 0.6 if motion is None else motion
            skeleton = 0.4 if skeleton is None else skeleton
        base = inputs["V"].mean(dim=(1, 2))
        prediction = base + 0.1 * motion + 0.2 * skeleton
        return prediction.unsqueeze(1), {}

    def call_loss(self, prediction, label, **kwargs):
        return self.mse(prediction.squeeze(), label.squeeze())


def expect_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def read_predictions(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert list(rows[0]) == [
        "sample_id", "group_id", "label", "prediction"
    ]
    assert [row["sample_id"] for row in rows] == [
        f"sample_{index}" for index in range(4)
    ]
    assert all(row["sample_id"] == row["group_id"] for row in rows)
    assert torch.allclose(
        torch.tensor([float(row["label"]) for row in rows]),
        torch.arange(4, dtype=torch.float32) / 10,
        atol=1e-7, rtol=0
    )
    return torch.tensor([
        float(row["prediction"]) for row in rows
    ])


def main():
    model = DummyModel()
    loader = DataLoader(
        PredictionDataset(), batch_size=2, shuffle=False
    )
    messages = []
    with tempfile.TemporaryDirectory() as directory:
        default_path = os.path.join(directory, "default.csv")
        motion_off_path = os.path.join(directory, "motion_off.csv")
        skeleton_off_path = os.path.join(directory, "skeleton_off.csv")
        all_off_path = os.path.join(directory, "all_off.csv")
        explicit_default_path = os.path.join(
            directory, "explicit_default.csv"
        )

        export_predictions(
            model, loader, default_path, log_func=messages.append
        )
        export_predictions(
            model, loader, motion_off_path, log_func=messages.append,
            force_gates=(0.0, None)
        )
        export_predictions(
            model, loader, skeleton_off_path, log_func=messages.append,
            force_gates=(None, 0.0)
        )
        export_predictions(
            model, loader, all_off_path, log_func=messages.append,
            force_gates=(0.0, 0.0)
        )
        export_predictions(
            model, loader, explicit_default_path,
            log_func=messages.append,
            force_gates=(None, None)
        )

        default = read_predictions(default_path)
        motion_off = read_predictions(motion_off_path)
        skeleton_off = read_predictions(skeleton_off_path)
        all_off = read_predictions(all_off_path)
        explicit_default = read_predictions(explicit_default_path)
        assert torch.equal(default, explicit_default)
        assert torch.allclose(default - motion_off, torch.full((4,), 0.06))
        assert torch.allclose(
            default - skeleton_off, torch.full((4,), 0.08)
        )
        assert torch.allclose(motion_off - all_off, torch.full((4,), 0.08))
        assert torch.allclose(
            skeleton_off - all_off, torch.full((4,), 0.06)
        )

        expect_value_error(lambda: export_predictions(
            model,
            DataLoader(
                PredictionDataset(duplicate_ids=True),
                batch_size=2, shuffle=False
            ),
            os.path.join(directory, "duplicate.csv"),
            log_func=lambda message: None
        ))
        expect_value_error(lambda: export_predictions(
            model,
            DataLoader(EmptyDataset(), batch_size=2, shuffle=False),
            os.path.join(directory, "empty.csv"),
            log_func=lambda message: None
        ))
        for invalid_gates in (
                (0.0,), (0.0, 0.0, 0.0), (float("nan"), None),
                (float("inf"), None), (-0.01, None), (None, 1.01)):
            expect_value_error(lambda gates=invalid_gates: export_predictions(
                model, loader,
                os.path.join(directory, "invalid.csv"),
                log_func=lambda message: None,
                force_gates=gates
            ))

    assert any("forced_gates=(0.0, None)" in value for value in messages)
    assert any("forced_gates=(None, 0.0)" in value for value in messages)
    print("prediction CSV schema and unique IDs: passed")
    print("default and forced Gate exports: passed")
    print("duplicate and empty export rejection: passed")
    print("invalid forced Gate rejection: passed")
    print("independent expert export verification passed")


if __name__ == "__main__":
    main()
