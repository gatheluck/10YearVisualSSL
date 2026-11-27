"""ViT Step-2 PIRL model: a unified ViT-B/16 + a linear projection head, over an
image and its jigsaw-shuffled view.

Faithful to the capture's `models/vit_pirl.py`: a timm `VisionTransformer` (from
scratch) encodes both the original image and a transformed view; the transformed
view is built by **reassembling** the nine shuffled 3x3 patches back into a single
image (grid-tiled, then resized to the ViT input) and encoding it once -- keeping
the PIRL jigsaw pretext with a single backbone (unlike the native ResNet path,
which encodes each patch separately). The CLS token feeds a `Linear(embed_dim,
feature_dim)` projection, L2-normalised, for the memory-bank NCE objective. This
port's convention: the ViT trunk lives under ``self.encoder`` (num_classes=0), so
`encoder.pt` keeps only ``encoder.*`` (the projector is training machinery,
excluded) and ``get_encoder()`` returns the CLS feature for the linear probe (the
eval sizes its head to it dynamically). timm is imported lazily (only on
`arch: vit`); the ViT dimensions are configurable so a tiny model can run a CPU
smoke.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ClsFeature(nn.Module):
    def __init__(self, vit: nn.Module) -> None:
        super().__init__()
        self.vit = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit.forward_features(x)[:, 0]


class ViTPIRL(nn.Module):
    def __init__(self, feature_dim: int = 128, num_patches: int = 9,
                 image_size: int = 224, patch_size: int = 16,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0) -> None:
        super().__init__()
        from timm.models.vision_transformer import VisionTransformer
        self.encoder = VisionTransformer(
            img_size=image_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            num_classes=0, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            qkv_bias=True, norm_layer=nn.LayerNorm)
        self.projector = nn.Linear(embed_dim, feature_dim)
        self.num_patches = num_patches
        self.image_size = image_size

    def encode_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_features(x)[:, 0]

    def forward_original(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(self.encode_backbone(images)), dim=1)

    def _assemble_jigsaw(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, channels, height, width = patches.shape
        grid = int(math.sqrt(num_patches))
        if grid * grid != num_patches:
            raise ValueError(f"num_patches must be square, got {num_patches}")
        x = patches.view(batch_size, grid, grid, channels, height, width)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(
            batch_size, channels, grid * height, grid * width)
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)
        return x

    def forward_jigsaw(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.shape[1] != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} patches, got {patches.shape[1]}")
        images = self._assemble_jigsaw(patches)
        return F.normalize(self.projector(self.encode_backbone(images)), dim=1)

    def forward(self, images: torch.Tensor,
                patches: "torch.Tensor | None" = None):
        image_features = self.forward_original(images)
        if patches is None:
            return image_features
        return image_features, self.forward_jigsaw(patches)

    def get_encoder(self) -> _ClsFeature:
        return _ClsFeature(self.encoder)


def build_vit_pirl(feature_dim: int = 128, num_patches: int = 9,
                   image_size: int = 224, patch_size: int = 16,
                   embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                   mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                   attn_drop_rate: float = 0.0) -> ViTPIRL:
    return ViTPIRL(feature_dim=feature_dim, num_patches=num_patches,
                   image_size=image_size, patch_size=patch_size,
                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                   mlp_ratio=mlp_ratio, drop_rate=drop_rate,
                   attn_drop_rate=attn_drop_rate)
