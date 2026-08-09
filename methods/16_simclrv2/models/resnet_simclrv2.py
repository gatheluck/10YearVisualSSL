"""ResNet-50 encoder + 3-layer MLP projection head for SimCLR v2 (Chen et al.,
2020), ported from the lab's own paper-faithful implementation.

  f(.): ResNet-50 backbone                 -> 2048-d average-pooled feature
  g(.): Linear(2048, 2048, bias=False) -> BN -> ReLU
        -> Linear(2048, 2048, bias=False) -> BN -> ReLU
        -> Linear(2048, out_dim, bias=False) -> BN
        -> L2-normalise

The v2 change over v1 is a **3-layer** MLP head (v1 used 2 layers), with an
optional `width_multiplier` (1 = ResNet-50; 2 = wide_resnet50_2, still 2048-d).

`encoder.pt` is the ResNet-50 backbone (`encoder.*`); the projection head
(`projector.*`) is training machinery and is excluded. `get_encoder()` returns
the backbone (2048-d) for the linear probe -- the standard SSL convention of
probing the backbone, not the projection head.

This is the lab's own code, torch/torchvision only; the capture's ViT variant
(`vit_simclrv2.py`, which needs `timm`) is a separate step and is not ported.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class _Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class ResNetSimCLRv2(nn.Module):
    """ResNet-50 with the SimCLR v2 3-layer projection head."""

    def __init__(self, out_dim: int = 128, width_multiplier: int = 1):
        super().__init__()
        if width_multiplier == 1:
            resnet = tv_models.resnet50(weights=None)
            feat_dim = 2048
        elif width_multiplier == 2:
            resnet = tv_models.wide_resnet50_2(weights=None)
            feat_dim = 2048
        else:
            raise ValueError(
                f"Unsupported width_multiplier: {width_multiplier}")

        # Everything up to and including avgpool; the classification fc is dropped.
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.feat_dim = feat_dim
        # 3-layer MLP head (the SimCLR v2 change over v1's 2-layer head).
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).view(x.size(0), -1)   # [B, feat_dim]
        z = self.projector(h)
        return F.normalize(z, dim=1)

    def get_encoder(self) -> nn.Module:
        """The ResNet-50 backbone (2048-d), for downstream probing."""
        return nn.Sequential(self.encoder, _Flatten())


def build_resnet_simclrv2(out_dim: int = 128,
                          width_multiplier: int = 1) -> ResNetSimCLRv2:
    return ResNetSimCLRv2(out_dim=out_dim, width_multiplier=width_multiplier)
