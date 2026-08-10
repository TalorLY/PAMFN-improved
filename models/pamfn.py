import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _torch_load(path):
    """Load the original pickled branch models on old and new PyTorch."""
    try:
        value = torch.load(
            path, map_location="cpu", weights_only=False
        )
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, nn.Module):
        for module in value.modules():
            if isinstance(module, nn.GELU) and not hasattr(
                    module, "approximate"):
                module.approximate = "none"
    return value


def smart_fn(fn, feat_list, func_list=False):
    if not func_list:
        return [fn(value) for value in feat_list]
    return [fn[index](value) for index, value in enumerate(feat_list)]


class BaseConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )

    def forward(self, value):
        return value + self.conv2(self.conv1(value))


class BaseModel(nn.Module):
    def __init__(self, in_dim, model_dim, drop_rate, modality="V"):
        super().__init__()
        self.modality = modality
        self.embedding = nn.Sequential(
            nn.Conv1d(in_dim, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, model_dim, 1),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(True),
            nn.Dropout(0.3),
        )
        self.stage1 = BaseConvBlock(model_dim)
        self.stage2 = BaseConvBlock(model_dim)
        self.stage3 = BaseConvBlock(model_dim)
        self.pool = nn.AvgPool1d(2, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Conv1d(model_dim, 1, 1),
            nn.Sigmoid(),
        )
        self.mse = nn.MSELoss()

    def forward(self, feats):
        value = feats[self.modality]
        if value.ndim == 4:
            value = value.mean(dim=2)
        value = value.permute(0, 2, 1).contiguous()
        value = self.embedding(value)

        stage1 = self.pool(self.stage1(value))
        stage2 = self.pool(self.stage2(stage1))
        stage3 = self.gap(self.stage3(stage2))
        score = self.fc(stage3).squeeze(dim=2)
        return score, {"feats": [value, stage1, stage2, stage3]}

    def call_loss(self, prediction, label, **kwargs):
        del kwargs
        return self.mse(prediction.squeeze(), label.squeeze())


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, model_dim, dropout=0.1):
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.linear_layers = nn.ModuleList(
            [nn.Linear(model_dim, model_dim) for _ in range(3)]
        )
        self.drop = nn.Dropout(dropout)
        # Kept for exact compatibility with the official state dict.
        self.norm = nn.LayerNorm(model_dim)

    def self_attention(
            self, query, key, value, mask=None, dropout=None):
        scores = (
            -torch.matmul(query, key.transpose(-2, -1))
            / np.sqrt(query.size(-1))
        )
        if mask is not None:
            mask = mask.unsqueeze(dim=1).repeat(
                1, self.num_heads, 1, 1
            )
            scores = scores + mask
        attention = torch.softmax(scores, dim=-1)
        if dropout is not None:
            attention = dropout(attention)
        return torch.matmul(attention, value), attention

    def forward(self, input_query, input_key, input_value, mask=None):
        batch_size = input_value.size(0)
        query, key, value = [
            layer(tensor).view(
                batch_size, -1, self.num_heads, self.head_dim
            ).transpose(1, 2)
            for layer, tensor in zip(
                self.linear_layers,
                (input_query, input_key, input_value),
            )
        ]
        value, _ = self.self_attention(
            query, key, value, mask=mask, dropout=self.drop
        )
        return value.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.head_dim
        )


class MSFusion(nn.Module):
    def __init__(self, dim, num_heads=1, dropout=0.1):
        super().__init__()
        self.fusion_s = MultiHeadAttention(
            num_heads=num_heads, model_dim=dim, dropout=dropout
        )
        self.proj = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
        )

    def forward(self, single_feats, combined_feat):
        batch_size, channels, steps = combined_feat.shape
        query = self.proj(combined_feat).permute(
            0, 2, 1
        ).contiguous().view(batch_size * steps, 1, channels)
        single_feats = torch.stack(single_feats, dim=2)
        single_feats = single_feats.permute(
            0, 3, 2, 1
        ).contiguous().view(
            batch_size * steps, single_feats.shape[2], channels
        )
        fused = self.fusion_s(
            query, single_feats, single_feats
        )
        return fused.view(
            batch_size, steps, 1, channels
        ).squeeze(2).permute(0, 2, 1).contiguous()


class CMFusion(nn.Module):
    def __init__(self, dim, kind_count, num_heads=1,
                 gap=False, dropout=0.1):
        super().__init__()
        self.K = kind_count
        self.proj1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, 1),
                nn.BatchNorm1d(dim),
                nn.ReLU(True),
                nn.Conv1d(dim, kind_count, 1),
            )
            for _ in range(3)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, 3, 1, 1),
                nn.BatchNorm1d(dim),
                nn.ReLU(True),
                nn.Conv1d(dim, dim, 3, 1, 1),
                nn.BatchNorm1d(dim),
                nn.ReLU(True),
            )
            for _ in range(kind_count)
        ])
        self.pool = (
            nn.AdaptiveAvgPool1d(1)
            if gap else nn.AvgPool1d(2, 2)
        )
        self.fusion_f = MultiHeadAttention(
            num_heads=num_heads, model_dim=dim, dropout=dropout
        )
        self.proj2 = nn.Sequential(
            nn.Conv1d(2 * dim, dim, 1),
        )
        self.policy = nn.Sequential(
            nn.Conv1d(3 * dim, kind_count, 1),
        )

    def gen_mask(self, fusion_kind):
        batch_size, kinds, steps = fusion_kind.shape
        mask = torch.zeros_like(fusion_kind)
        index = torch.arange(
            kinds, device=fusion_kind.device
        ).repeat(batch_size, 1, 1).permute(0, 2, 1).repeat(
            1, 1, steps
        )
        position_mask = index > torch.mul(
            fusion_kind, index
        ).sum(dim=1, keepdim=True)
        mask[position_mask] = -1e9
        return mask + fusion_kind - fusion_kind.detach()

    def gen_fusion_kind(self, feats, tau):
        logit = self.pool(self.policy(torch.cat(feats, dim=1)))
        fusion_kind = F.gumbel_softmax(
            logit, tau, hard=True, dim=1
        )
        return fusion_kind, self.gen_mask(fusion_kind)

    def forward(self, feats, combined_feat, single_fused, tau):
        batch_size = combined_feat.shape[0]
        attention = torch.softmax(
            torch.stack(smart_fn(self.proj1, feats, True), dim=1),
            dim=1,
        )
        attention = [
            value for value in attention.split(1, dim=2)
        ]
        fused_feats = [
            torch.mul(value, torch.stack(feats, dim=1)).sum(dim=1)
            for value in attention
        ]
        fused_feats = torch.stack([
            self.pool(self.ffn[index](value) + value)
            for index, value in enumerate(fused_feats)
        ], dim=2)
        steps = fused_feats.shape[3]
        channels = fused_feats.shape[1]
        fused_feats = fused_feats.permute(
            0, 3, 2, 1
        ).contiguous().view(
            batch_size * steps, fused_feats.shape[2], channels
        )

        query = self.proj2(torch.cat(
            [combined_feat, single_fused], dim=1
        ))
        query = -query.permute(
            0, 2, 1
        ).contiguous().view(batch_size * steps, 1, channels)
        fusion_kind, mask = self.gen_fusion_kind(feats, tau)
        mask = mask.permute(0, 2, 1).contiguous().view(
            batch_size * steps, 1, mask.shape[1]
        )
        fused_feats = self.fusion_f(
            query, fused_feats, fused_feats, mask=mask
        )
        fused_feats = fused_feats.view(
            batch_size, steps, 1, channels
        ).squeeze(2).permute(0, 2, 1).contiguous()
        return fused_feats, fusion_kind


class FusionBlock(nn.Module):
    def __init__(self, dim, kind_count, ms_heads, cm_heads, gap=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim * 3, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
        )
        self.short_cut = nn.Sequential(
            nn.Conv1d(dim * 3, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
        )
        self.s_fusion = MSFusion(dim, ms_heads)
        self.f_fusion = CMFusion(
            dim, kind_count, cm_heads, gap
        )

    def forward(self, combined, feats, single_feats, tau):
        combined = self.conv(combined) + self.short_cut(combined)
        single_fused = self.s_fusion(single_feats, combined)
        cross_fused, action = self.f_fusion(
            feats, combined, single_fused, tau
        )
        return torch.cat(
            [combined, single_fused, cross_fused], dim=1
        ), action


class PAMFN(nn.Module):
    """Official PAMFN plus optional training-only auxiliary objectives."""

    def __init__(
            self, model_dim, fc_drop, fc_r, feat_drop, K,
            ms_heads, cm_heads, ckpt_dir, rgb_ckpt_name,
            flow_ckpt_name, audio_ckpt_name, dataset_name,
            shared_rank_weight=0.0, shared_rank_margin=0.02,
            shared_rank_temperature=0.05, shared_list_weight=0.0,
            shared_list_margin=0.02, shared_list_temperature=0.05,
            ordinal_grade_weight=0.0,
            ordinal_prototype_weight=0.0,
            ordinal_quality_thresholds=(),
            ordinal_num_grades=4,
            ordinal_prototype_dim=128,
            ordinal_prototype_temperature=0.1,
            ordinal_prototype_order_margin=0.1,
            ordinal_prototype_order_scale=0.1):
        super().__init__()
        for name, value in (
                ("shared_rank_weight", shared_rank_weight),
                ("shared_rank_margin", shared_rank_margin),
                ("shared_list_weight", shared_list_weight),
                ("shared_list_margin", shared_list_margin),
                ("ordinal_grade_weight", ordinal_grade_weight),
                ("ordinal_prototype_weight",
                 ordinal_prototype_weight),
                ("ordinal_prototype_order_margin",
                 ordinal_prototype_order_margin),
                ("ordinal_prototype_order_scale",
                 ordinal_prototype_order_scale)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(shared_rank_temperature) or (
                shared_rank_temperature <= 0):
            raise ValueError(
                "shared_rank_temperature must be finite and positive"
            )
        if not np.isfinite(shared_list_temperature) or (
                shared_list_temperature <= 0):
            raise ValueError(
                "shared_list_temperature must be finite and positive"
            )
        if ordinal_prototype_dim < 1:
            raise ValueError(
                "ordinal_prototype_dim must be positive"
            )
        if ordinal_num_grades < 2:
            raise ValueError(
                "ordinal_num_grades must be at least 2"
            )
        if not np.isfinite(ordinal_prototype_temperature) or (
                ordinal_prototype_temperature <= 0):
            raise ValueError(
                "ordinal_prototype_temperature must be finite and "
                "positive"
            )
        active_objectives = sum(
            value > 0
            for value in (
                shared_rank_weight,
                shared_list_weight,
                ordinal_grade_weight,
                ordinal_prototype_weight,
            )
        )
        if active_objectives > 1:
            raise ValueError(
                "pairwise, List-wise, grade-head, and prototype "
                "supervision are mutually exclusive"
            )
        if ordinal_grade_weight > 0 or ordinal_prototype_weight > 0:
            thresholds = np.asarray(
                ordinal_quality_thresholds, dtype=np.float64
            )
            if (
                    thresholds.shape != (ordinal_num_grades - 1,)
                    or not np.isfinite(thresholds).all()
                    or not np.all(thresholds[:-1] < thresholds[1:])):
                raise ValueError(
                    "ordinal_quality_thresholds must contain "
                    "ordinal_num_grades - 1 finite increasing values"
                )

        self.shared_rank_weight = float(shared_rank_weight)
        self.shared_rank_margin = float(shared_rank_margin)
        self.shared_rank_temperature = float(
            shared_rank_temperature
        )
        self.shared_list_weight = float(shared_list_weight)
        self.shared_list_margin = float(shared_list_margin)
        self.shared_list_temperature = float(
            shared_list_temperature
        )
        self.ordinal_grade_weight = float(ordinal_grade_weight)
        self.ordinal_prototype_weight = float(
            ordinal_prototype_weight
        )
        self.ordinal_num_grades = int(ordinal_num_grades)
        self.ordinal_prototype_temperature = float(
            ordinal_prototype_temperature
        )
        self.ordinal_prototype_order_margin = float(
            ordinal_prototype_order_margin
        )
        self.ordinal_prototype_order_scale = float(
            ordinal_prototype_order_scale
        )
        self.model_r = _torch_load(osp.join(
            ckpt_dir,
            f"{dataset_name}_rgb_{rgb_ckpt_name}.pth",
        ))
        self.model_f = _torch_load(osp.join(
            ckpt_dir,
            f"{dataset_name}_flow_{flow_ckpt_name}.pth",
        ))
        self.model_a = _torch_load(osp.join(
            ckpt_dir,
            f"{dataset_name}_audio_{audio_ckpt_name}.pth",
        ))

        self.c = nn.Parameter(
            torch.zeros(model_dim * 3), requires_grad=False
        )
        self.stage1 = FusionBlock(
            model_dim, K, ms_heads, cm_heads
        )
        self.stage2 = FusionBlock(
            model_dim, K, ms_heads, cm_heads
        )
        self.stage3 = FusionBlock(
            model_dim, K, ms_heads, cm_heads, True
        )
        self.pool = nn.AvgPool1d(2, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(feat_drop)
        self.fc = nn.Sequential(
            nn.Dropout(fc_drop),
            nn.Conv1d(model_dim * 3, model_dim // fc_r, 1),
            nn.BatchNorm1d(model_dim // fc_r),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(model_dim // fc_r, 1, 1),
            nn.Sigmoid(),
        )
        self.tau = nn.Parameter(torch.ones([]) * 10)
        self.mse = nn.MSELoss()
        if (
                self.ordinal_grade_weight > 0
                or self.ordinal_prototype_weight > 0):
            self.quality_projector = nn.Linear(
                model_dim * 3, ordinal_prototype_dim
            )
            self.register_buffer(
                "ordinal_thresholds",
                torch.tensor(
                    ordinal_quality_thresholds,
                    dtype=torch.float32,
                ),
            )
        if self.ordinal_grade_weight > 0:
            self.quality_classifier = nn.Linear(
                ordinal_prototype_dim,
                self.ordinal_num_grades,
                bias=False,
            )
        if self.ordinal_prototype_weight > 0:
            self.quality_prototypes = nn.Parameter(
                torch.empty(
                    self.ordinal_num_grades,
                    ordinal_prototype_dim,
                )
            )
            self.quality_axis = nn.Parameter(
                torch.empty(ordinal_prototype_dim)
            )
            nn.init.normal_(self.quality_prototypes, std=0.02)
            nn.init.normal_(self.quality_axis, std=0.02)
    def forward(self, input_feats):
        _, feat_r = self.model_r(input_feats)
        _, feat_a = self.model_a(input_feats)
        _, feat_f = self.model_f(input_feats)

        feats = [
            [self.drop(value.detach()) for value in feat_r["feats"]],
            [self.drop(value.detach()) for value in feat_f["feats"]],
            [self.drop(value) for value in feat_a["feats"]],
        ]
        combined = self.c.repeat(
            feats[0][0].shape[0],
            feats[0][0].shape[2] // 2,
            1,
        ).permute(0, 2, 1)

        stage1, action1 = self.stage1(
            combined,
            [value[0] for value in feats],
            [value[1] for value in feats],
            self.tau,
        )
        stage1 = self.pool(stage1)
        stage2, action2 = self.stage2(
            stage1,
            [value[1] for value in feats],
            [value[2] for value in feats],
            self.tau,
        )
        stage2 = self.gap(stage2)
        stage3, action3 = self.stage3(
            stage2,
            [value[2] for value in feats],
            [value[3] for value in feats],
            self.tau,
        )
        score = self.fc(stage3).squeeze(dim=2)
        info = {"action": [action1, action2, action3]}
        if (
                self.ordinal_grade_weight > 0
                or self.ordinal_prototype_weight > 0):
            quality_embedding = F.normalize(
                self.quality_projector(stage3.squeeze(dim=2)),
                dim=1,
            )
            info["quality_embedding"] = quality_embedding
            if self.ordinal_grade_weight > 0:
                info["quality_logits"] = self.quality_classifier(
                    quality_embedding
                )
        return score, info

    def call_loss(self, prediction, label, **kwargs):
        del kwargs
        return self.mse(prediction.squeeze(), label.squeeze())

    def call_training_loss(
            self, prediction, label, **kwargs):
        return self.call_training_loss_components(
            prediction, label, **kwargs
        )["objective"]

    def call_training_loss_components(
            self, prediction, label, quality_embedding=None,
            quality_logits=None, **kwargs):
        del kwargs
        prediction = prediction.reshape(-1)
        label = label.reshape(-1)
        score_loss = self.mse(prediction, label)
        components = {
            "score_mse": score_loss,
            "objective": score_loss,
        }
        if self.shared_rank_weight > 0:
            rank_loss = self._pairwise_rank_loss(
                prediction,
                label,
                min_label_gap=self.shared_rank_margin,
                temperature=self.shared_rank_temperature,
            )
            weighted = self.shared_rank_weight * rank_loss
            components.update({
                "rank_loss": rank_loss,
                "weighted_auxiliary": weighted,
                "objective": score_loss + weighted,
            })
            return components
        if self.shared_list_weight > 0:
            list_loss = self._grouped_listmle_loss(
                prediction,
                label,
                tie_range=self.shared_list_margin,
                temperature=self.shared_list_temperature,
            )
            weighted = self.shared_list_weight * list_loss
            components.update({
                "list_loss": list_loss,
                "weighted_auxiliary": weighted,
                "objective": score_loss + weighted,
            })
            return components
        if self.ordinal_grade_weight > 0:
            if quality_logits is None:
                raise ValueError(
                    "quality_logits is required for grade-head loss"
                )
            grade_loss = self._ordinal_grade_loss(
                quality_logits, label
            )
            weighted = self.ordinal_grade_weight * grade_loss
            components.update({
                "grade_classification": grade_loss,
                "weighted_auxiliary": weighted,
                "objective": score_loss + weighted,
            })
            return components
        if self.ordinal_prototype_weight > 0:
            if quality_embedding is None:
                raise ValueError(
                    "quality_embedding is required for prototype loss"
                )
            prototype_loss, order_loss = (
                self._ordinal_prototype_loss_components(
                    quality_embedding,
                    label,
                )
            )
            auxiliary = (
                prototype_loss
                + self.ordinal_prototype_order_scale * order_loss
            )
            weighted = self.ordinal_prototype_weight * auxiliary
            components.update({
                "prototype_classification": prototype_loss,
                "prototype_order": order_loss,
                "weighted_auxiliary": weighted,
                "objective": score_loss + weighted,
            })
            return components
        return components

    def _ordinal_soft_targets(self, target, dtype):
        bins = torch.bucketize(
            target.detach(), self.ordinal_thresholds
        )
        class_ids = torch.arange(
            self.ordinal_thresholds.numel() + 1,
            device=target.device,
        ).unsqueeze(0)
        distance = (class_ids - bins.unsqueeze(1)).abs()
        soft_target = torch.where(
            distance == 0,
            torch.ones_like(distance, dtype=dtype),
            torch.where(
                distance == 1,
                torch.full_like(distance, 0.1, dtype=dtype),
                torch.zeros_like(distance, dtype=dtype),
            ),
        )
        return soft_target / soft_target.sum(
            dim=1, keepdim=True
        )

    def _ordinal_grade_loss(self, logits, target):
        soft_target = self._ordinal_soft_targets(
            target, logits.dtype
        )
        return -(
            soft_target * F.log_softmax(logits, dim=1)
        ).sum(dim=1).mean()

    def _ordinal_prototype_loss_components(
            self, embedding, target):
        prototypes = F.normalize(
            self.quality_prototypes, dim=1
        )
        logits = (
            embedding @ prototypes.transpose(0, 1)
            / self.ordinal_prototype_temperature
        )
        soft_target = self._ordinal_soft_targets(
            target, embedding.dtype
        )
        prototype_loss = -(
            soft_target * F.log_softmax(logits, dim=1)
        ).sum(dim=1).mean()

        axis = F.normalize(self.quality_axis, dim=0)
        positions = prototypes @ axis
        gaps = positions[1:] - positions[:-1]
        order_loss = F.relu(
            self.ordinal_prototype_order_margin - gaps
        ).mean()
        return prototype_loss, order_loss

    def _ordinal_prototype_loss(self, embedding, target):
        prototype_loss, order_loss = (
            self._ordinal_prototype_loss_components(
                embedding,
                target,
            )
        )
        return (
            prototype_loss
            + self.ordinal_prototype_order_scale * order_loss
        )

    def _pairwise_rank_loss(
            self, prediction, target, min_label_gap=1e-8,
            temperature=None):
        if prediction.numel() < 2:
            return prediction.sum() * 0.0
        if temperature is None:
            temperature = self.shared_rank_temperature
        target_difference = target[:, None] - target[None, :]
        prediction_difference = (
            prediction[:, None] - prediction[None, :]
        )
        upper_triangle = torch.triu(
            torch.ones_like(target_difference, dtype=torch.bool),
            diagonal=1,
        )
        valid = upper_triangle & (
            target_difference.abs() > min_label_gap
        )
        if not valid.any():
            return prediction.sum() * 0.0
        direction = target_difference[valid].sign()
        ordered_difference = prediction_difference[valid] * direction
        return F.softplus(
            -ordered_difference / temperature
        ).mean()

    def _grouped_listmle_loss(
            self, prediction, target, tie_range=0.02,
            temperature=None):
        if prediction.numel() < 2:
            return prediction.sum() * 0.0
        if temperature is None:
            temperature = self.shared_list_temperature

        order = torch.argsort(target, descending=True)
        ordered_target = target[order]
        ordered_prediction = prediction[order]
        group_scores = []
        start = 0
        item_count = ordered_target.numel()
        while start < item_count:
            end = start + 1
            while end < item_count and (
                    ordered_target[start] - ordered_target[end]
                    <= tie_range):
                end += 1
            group_scores.append(
                ordered_prediction[start:end].mean()
            )
            start = end

        if len(group_scores) < 2:
            return prediction.sum() * 0.0
        scaled_scores = torch.stack(group_scores) / temperature
        terms = [
            torch.logsumexp(scaled_scores[index:], dim=0)
            - scaled_scores[index]
            for index in range(len(group_scores) - 1)
        ]
        return torch.stack(terms).mean()
