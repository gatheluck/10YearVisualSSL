"""ResNet-50 encoder + 2-layer MLP projection head for SimCLR v1 (Chen et al.,
2020), ported from the lab's own paper-faithful implementation.

  f(.): ResNet-50 backbone            -> 2048-d average-pooled feature
  g(.): Linear(2048, 2048, bias=False) -> BN -> ReLU
        -> Linear(2048, out_dim, bias=False) -> BN
        -> L2-normalise

`encoder.pt` is the ResNet-50 backbone (`encoder.*`); the projection head
(`projector.*`) is training machinery and is excluded. `get_encoder()` returns
the backbone (2048-d) for the linear probe -- the standard SSL convention of
probing the backbone, not the projection head.

This is the lab's own code, torch/torchvision only; the capture's ViT variant
(`vit_simclr.py`, which needs `timm`) is a separate step and is not ported.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class _Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class ResNetSimCLR(nn.Module):
    """ResNet-50 with the SimCLR v1 projection head."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        resnet = tv_models.resnet50(weights=None)
        # Everything up to and including avgpool; the classification fc is dropped.
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # (B,2048,1,1)
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).view(x.size(0), -1)   # [B, 2048]
        z = self.projector(h)
        return F.normalize(z, dim=1)

    def get_encoder(self) -> nn.Module:
        """The ResNet-50 backbone (2048-d), for downstream probing."""
        return nn.Sequential(self.encoder, _Flatten())


def build_resnet_simclr(out_dim: int = 128) -> ResNetSimCLR:
    return ResNetSimCLR(out_dim=out_dim)
