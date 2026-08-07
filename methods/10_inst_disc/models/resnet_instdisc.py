"""ResNet-50 encoder for Instance Discrimination step 1 (Wu et al., CVPR 2018).

Architecture (the paper's exact recipe): ResNet-50 -> avgpool -> Linear(2048,
128) -> L2-normalise. The 128-d L2-normalised embedding is what the NCE loss and
the memory bank operate on.

`encoder.pt` is the ResNet-50 backbone (`encoder.*`); the 128-d projection head
(`fc.*`) is instance-discrimination machinery and is excluded. `get_encoder()`
returns the backbone (2048-d) for the linear probe -- the standard SSL
convention of probing the backbone, not the projection head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class _Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class ResNetInstDisc(nn.Module):
    """ResNet-50 with a 128-d L2-normalised projection head."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        resnet = tv_models.resnet50(weights=None)
        # Everything up to and including avgpool; the classifier fc is dropped.
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] -> L2-normalised embedding [B, feature_dim]."""
        h = self.encoder(x).view(x.size(0), -1)  # [B, 2048]
        return F.normalize(self.fc(h), dim=1)

    def get_encoder(self) -> nn.Module:
        """The ResNet-50 backbone (2048-d), for downstream probing."""
        return nn.Sequential(self.encoder, _Flatten())


def build_resnet_instdisc(feature_dim: int = 128) -> ResNetInstDisc:
    return ResNetInstDisc(feature_dim=feature_dim)
