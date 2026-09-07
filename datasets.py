import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset


def zero_padding_torch(data, target_length):
    original_length = data.shape[0]
    zero_data = torch.zeros(
        (target_length - original_length,) + data.shape[1:]
    )
    return torch.cat([data, zero_data], dim=0)


def get_default_segments(dataset):
    if dataset in ("Ball", "Clubs", "Hoop", "Ribbon"):
        return 70
    if dataset in ("TES", "PCS"):
        return 130
    raise ValueError(f"{dataset} is not supported")


def read_labels(dataset_name, is_train):
    labels = []
    if dataset_name in ("Ball", "Clubs", "Hoop", "Ribbon"):
        label_path = osp.join(
            "./data/RG/",
            "train.txt" if is_train else "test.txt",
        )
        with open(label_path, "r") as handle:
            for index, line in enumerate(handle.readlines()):
                if index == 0:
                    continue
                values = line.strip().split()
                if values[0].startswith(dataset_name):
                    labels.append((values[0], float(values[3]) / 25))
    elif dataset_name in ("TES", "PCS"):
        label_path = osp.join(
            "./data/FISV",
            "train.txt" if is_train else "test.txt",
        )
        with open(label_path, "r") as handle:
            for line in handle.readlines():
                values = line.strip().split()
                score_index = 1 if dataset_name == "TES" else 2
                labels.append(
                    (values[0], float(values[score_index]) / 45)
                )
    else:
        raise ValueError(f"{dataset_name} is not supported")
    return labels


def get_train_quality_thresholds(dataset_name):
    values = np.asarray(
        [label for _, label in read_labels(dataset_name, True)],
        dtype=np.float64,
    )
    return tuple(
        float(value)
        for value in np.quantile(values, (0.25, 0.50, 0.75))
    )


def get_train_quality_centers(dataset_name, thresholds=None):
    """Return the median score in each train-only quality quartile."""
    values = np.asarray(
        [label for _, label in read_labels(dataset_name, True)],
        dtype=np.float64,
    )
    thresholds = np.asarray(
        thresholds or get_train_quality_thresholds(dataset_name),
        dtype=np.float64,
    )
    if (
            thresholds.shape != (3,)
            or not np.isfinite(thresholds).all()
            or not np.all(thresholds[:-1] < thresholds[1:])):
        raise ValueError("quality thresholds must be finite and increasing")
    grades = np.digitize(values, thresholds)
    centers = []
    for grade in range(4):
        members = values[grades == grade]
        if members.size == 0:
            raise ValueError(f"quality grade {grade + 1} is empty")
        centers.append(float(np.median(members)))
    if not np.all(np.diff(centers) > 0):
        raise ValueError("quality centers must be strictly increasing")
    return tuple(centers)


class AQADataset(Dataset):
    """Official PAMFN train-crop/full-test dataset."""

    def __init__(
            self, dataset_name, feat_dir,
            rgb_feat, flow_feat, audio_feat,
            squeeze_rgb_feat=None, squeeze_flow_feat=None,
            is_train=True, segments=-1, **kwargs):
        super().__init__()
        del kwargs
        self.is_train = is_train
        self.squeeze_rgb_feat = squeeze_rgb_feat
        self.squeeze_flow_feat = squeeze_flow_feat
        self.labels = read_labels(dataset_name, is_train)
        self.datas = self.read_data(
            dataset_name, feat_dir, rgb_feat, flow_feat, audio_feat
        )
        self.segments = (
            get_default_segments(dataset_name)
            if segments < 0 else segments
        )

    @staticmethod
    def read_data(
            dataset_name, feat_dir, rgb_feat, flow_feat, audio_feat):
        feature_dataset = (
            "FISV" if dataset_name in ("TES", "PCS")
            else dataset_name
        )
        rgb_data = np.load(
            osp.join(
                feat_dir, f"{feature_dataset}_rgb_{rgb_feat}.npy"
            ),
            allow_pickle=True,
        ).item()
        flow_data = np.load(
            osp.join(
                feat_dir, f"{feature_dataset}_flow_{flow_feat}.npy"
            ),
            allow_pickle=True,
        ).item()
        audio_data = np.load(
            osp.join(
                feat_dir, f"{feature_dataset}_audio_{audio_feat}.npy"
            ),
            allow_pickle=True,
        ).item()
        return {
            "rgb": rgb_data,
            "flow": flow_data,
            "audio": audio_data,
        }

    def squeeze_dim(self, rgb_data, flow_data, audio_data):
        if self.squeeze_rgb_feat == "mean" and rgb_data.ndim == 3:
            rgb_data = rgb_data.mean(dim=1)
        elif self.squeeze_rgb_feat == "cat" and rgb_data.ndim == 3:
            rgb_data = rgb_data.view(rgb_data.shape[0], -1)
        if self.squeeze_flow_feat == "mean" and flow_data.ndim == 3:
            flow_data = flow_data.mean(dim=1)
        elif self.squeeze_flow_feat == "cat" and flow_data.ndim == 3:
            flow_data = flow_data.view(flow_data.shape[0], -1)
        return rgb_data, flow_data, audio_data

    def __getitem__(self, index):
        sample_name, label = self.labels[index]
        rgb_data = torch.from_numpy(
            self.datas["rgb"][sample_name]
        ).float()
        flow_data = torch.from_numpy(
            self.datas["flow"][sample_name]
        ).float()
        audio_data = torch.from_numpy(
            self.datas["audio"][sample_name]
        ).float()
        rgb_data, flow_data, audio_data = self.squeeze_dim(
            rgb_data, flow_data, audio_data
        )

        if self.is_train:
            if len(rgb_data) > self.segments:
                start_index = np.random.randint(
                    len(rgb_data) - self.segments
                )
                stop_index = start_index + self.segments
                rgb_data = rgb_data[start_index:stop_index]
                audio_data = audio_data[start_index:stop_index]
                flow_data = flow_data[start_index:stop_index]
            elif len(rgb_data) < self.segments:
                rgb_data = zero_padding_torch(
                    rgb_data, self.segments
                )
                audio_data = zero_padding_torch(
                    audio_data, self.segments
                )
                flow_data = zero_padding_torch(
                    flow_data, self.segments
                )
        return rgb_data, audio_data, flow_data, label, 0

    def __len__(self):
        return len(self.labels)
