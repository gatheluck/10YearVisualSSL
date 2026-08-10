"""ResNet-50 PIRL encoder (Misra & van der Maaten, CVPR 2020).

Ported from the lab's own PIRL code. The same ResNet-50 trunk and 2048->D
projection encode the original image and each shuffled jigsaw patch; the nine
patch embeddings are concatenated and projected to a single D-dimensional
transformed-view representation. `encoder.pt` is the ResNet-50 trunk
(``encoder.*``); the image/jigsaw projection heads are excluded.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


class _Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ResNetPIRL(nn.Module):
    """ResNet-50 with image and jigsaw projection heads."""

    def __init__(self, feature_dim: int = 128, num_patches: int = 9):
        super().__init__()
        resnet = tv_models.resnet50(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projector = nn.Linear(2048, feature_dim)
        self.jigsaw_projector = nn.Linear(feature_dim * num_patches, feature_dim)
        self.feature_dim = feature_dim
        self.num_patches = num_patches

    def encode_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).view(x.size(0), -1)

    def forward_original(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.projector(self.encode_backbone(images))
        return F.normalize(feats, dim=1)

    def forward_jigsaw(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches = patches.shape[:2]
        if num_patches != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} patches, got {num_patches}")
        flat = patches.view(batch_size * num_patches, *patches.shape[2:])
        patch_feats = self.projector(self.encode_backbone(flat))
        patch_feats = F.normalize(patch_feats, dim=1)
        patch_feats = patch_feats.view(batch_size, num_patches * self.feature_dim)
        return F.normalize(self.jigsaw_projector(patch_feats), dim=1)

    def forward(self, images: torch.Tensor, patches: torch.Tensor = None):
        image_features = self.forward_original(images)
        if patches is None:
            return image_features
        return image_features, self.forward_jigsaw(patches)

    def get_encoder(self) -> nn.Module:
        return nn.Sequential(self.encoder, _Flatten())


def build_resnet_pirl(feature_dim: int = 128, num_patches: int = 9) -> ResNetPIRL:
    return ResNetPIRL(feature_dim=feature_dim, num_patches=num_patches)
