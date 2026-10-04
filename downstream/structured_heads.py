"""Captured pose, absolute camera-pose and full-question component heads."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import normalize

from downstream.attention import QueryReader, SpatialAdapter
from downstream.captured_readers import (
    CROSS_SELF,
    SINGLE_BLOCK,
    CrossSelfQueryReader,
    SingleBlockSpatialAdapter,
)
from downstream.extended_classification import frozen_global_features


def heatmap_loss(logits, target):
    logits, target = logits.float(), target.float()
    if logits.shape != target.shape or logits.ndim != 4 or not logits.numel():
        raise ValueError("pose heatmap shape mismatch")
    if not torch.isfinite(logits).all() or torch.isinf(target).any():
        raise ValueError("nonfinite pose logits/target")
    mask = torch.isfinite(target)
    count = mask.sum((-1, -2))
    if ((count != 0) & (count != target.shape[-1] * target.shape[-2])).any():
        raise ValueError("partially missing heatmap channel")
    error = (logits - torch.nan_to_num(target)) ** 2
    return error[mask].mean() if mask.any() else logits.sum() * 0


def rotation_6d(value):
    a, b = value[..., :3].float(), value[..., 3:6].float()
    x = F.normalize(a, dim=-1)
    y = F.normalize(b - (x * b).sum(-1, keepdim=True) * x, dim=-1)
    return torch.stack([x, y, torch.cross(x, y, dim=-1)], dim=-1)


class CameraHead(nn.Module):
    def __init__(self, dim, scenes):
        super().__init__()
        if type(scenes) is not int or scenes < 1:
            raise ValueError("positive scene count required")
        self.scenes = scenes
        self.proj = nn.Linear(dim, 9 * scenes)
        nn.init.normal_(self.proj.weight, std=0.001)
        with torch.no_grad():
            self.proj.bias.copy_(
                torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).repeat(
                    scenes
                )
            )

    def forward(self, x):
        z = self.proj(x).reshape(x.shape[0], self.scenes, 9)
        return torch.cat([z[..., :3].float(), rotation_6d(z[..., 3:]).flatten(-2)], -1)


def camera_loss(outputs, target):
    if (
        outputs.ndim != 3
        or outputs.shape[-1] != 12
        or target.shape != (len(outputs), 13)
        or not len(outputs)
        or not torch.isfinite(outputs).all()
        or not torch.isfinite(target).all()
    ):
        raise ValueError("invalid camera output or target")
    scene = target[:, 0].long()
    if not ((scene == target[:, 0]) & (scene >= 0) & (scene < outputs.shape[1])).all():
        raise ValueError("invalid camera scene index")
    pred = outputs[torch.arange(len(scene), device=outputs.device), scene].float()
    return F.l1_loss(pred[:, :3], target[:, 1:4].float()) + F.l1_loss(
        pred[:, 3:], target[:, 4:].float()
    )


class QuestionHead(nn.Module):
    """Ordered UTF-8 byte GRU and nonlinear image/question/option fusion."""

    def __init__(self, dim):
        super().__init__()
        self.q_emb = nn.Embedding(257, 64, padding_idx=0)
        self.language = nn.GRU(64, 64, batch_first=True)
        self.vis = nn.Linear(dim, 256)
        self.head = nn.Sequential(nn.Linear(384, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, vis, q):
        if (
            q.ndim != 3
            or q.shape[1] < 3
            or not q.numel()
            or q.dtype != torch.long
            or not ((q >= 0) & (q <= 256)).all()
            or len(vis) != len(q)
        ):
            raise ValueError("full question and at least two options required")
        b, k, length = q.shape
        flat = q.reshape(b * k, length)
        sizes = (flat != 0).sum(1)
        valid = torch.arange(length, device=q.device)[None] < sizes[:, None]
        if (
            not torch.equal(flat != 0, valid)
            or not (sizes.reshape(b, k)[:, :2] > 0).all()
        ):
            raise ValueError("missing question/option or non-trailing padding")
        if not ((sizes.reshape(b, k)[:, 1:] > 0).sum(1) >= 2).all():
            raise ValueError("each sample requires two or more options")
        packed = nn.utils.rnn.pack_padded_sequence(
            self.q_emb(flat),
            sizes.clamp_min(1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.language(packed)
        h = hidden[-1].reshape(b, k, 64)
        v = self.vis(vis)
        features = torch.cat(
            [
                v[:, None].expand(-1, k - 1, -1),
                h[:, :1].expand(-1, k - 1, -1),
                h[:, 1:],
            ],
            -1,
        )
        return (
            self.head(features)
            .squeeze(-1)
            .masked_fill(sizes.reshape(b, k)[:, 1:] == 0, -1e4)
        )


class Probe(nn.Module):
    def __init__(self, backbone, task, outputs, reader_profile):
        super().__init__()
        if task not in ("pose", "localization", "reasoning") or reader_profile not in (
            None,
            SINGLE_BLOCK,
            CROSS_SELF,
        ):
            raise ValueError("unknown structured task or reader")
        if type(outputs) is not int or outputs < 1:
            raise ValueError("positive output count required")
        self.task = task
        self.backbone = backbone.requires_grad_(False).eval()
        channels = backbone.out_channels
        self.adapter = self.reader = None
        if reader_profile is not None:
            if task == "reasoning":
                self.reader = (
                    QueryReader
                    if reader_profile == SINGLE_BLOCK
                    else CrossSelfQueryReader
                )(channels)
            else:
                self.adapter = (
                    SingleBlockSpatialAdapter
                    if reader_profile == SINGLE_BLOCK
                    else SpatialAdapter
                )(channels)
        global_channels = getattr(backbone, "global_channels", channels)
        if task == "pose":
            self.head = nn.Conv2d(channels, outputs, 1)
            nn.init.normal_(self.head.weight, std=0.01)
            nn.init.zeros_(self.head.get_parameter("bias"))
        elif task == "localization":
            self.head = CameraHead(
                channels if self.adapter is not None else global_channels, outputs
            )
        else:
            self.head = QuestionHead(
                512 if self.reader is not None else global_channels
            )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images, questions=None):
        images = normalize(images, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        with torch.no_grad():
            if (
                self.task == "pose"
                or self.adapter is not None
                or self.reader is not None
            ):
                features = self.backbone.forward_features(images).float()
            elif callable(getattr(self.backbone, "structured_global_features", None)):
                features = self.backbone.structured_global_features(images).float()
            else:
                features = frozen_global_features(self.backbone, images)
        if self.adapter is not None:
            features = self.adapter(features)
        if self.task == "pose":
            return F.interpolate(
                self.head(features), size=(56, 56), mode="bilinear", align_corners=False
            )
        if self.adapter is not None:
            features = features.mean((2, 3))
        if self.reader is not None:
            features = self.reader(features.flatten(2).transpose(1, 2))
        return (
            self.head(features, questions)
            if self.task == "reasoning"
            else self.head(features)
        )
