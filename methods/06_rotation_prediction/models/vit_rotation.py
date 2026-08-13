"""The rotation Step-2 model: a unified ViT-B/16 with a 4-way rotation head.

Faithful to the capture's `methods/6_rotation_prediction/models/vit_rotation.py`
(timm's `VisionTransformer`, built from scratch), restructured to this port's
encoder-extraction convention so it drops into the existing adapter/probe:

- the ViT backbone lives under ``encoder`` (a timm ``VisionTransformer`` built
  with ``num_classes=0``, so it yields the CLS feature), and the pretext head is
  a separate ``head = Linear(embed_dim, num_classes)``. So ``encoder.pt`` carries
  only ``encoder.*`` (the backbone); the head is training machinery and is left
  out, exactly like the AlexNet path.
- ``get_encoder()`` returns a module whose ``forward`` is the CLS feature -- the
  same interface ``build_alexnet_rotation_model`` exposes -- so
  ``evaluate_linear_rotation.py`` probes it unchanged.

timm is imported lazily (module top level) and is only pulled in when this file
is imported, i.e. on the ``arch: vit`` path; the native AlexNet path never needs
it.
"""

from __future__ import annotations

import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class _ClsFeature(nn.Module):
    """Wraps the ViT so ``forward(x)`` returns the CLS-token feature (the probe's
    representation), matching the AlexNet encoder's image->vector interface."""

    def __init__(self, vit: VisionTransformer) -> None:
        super().__init__()
        self.vit = vit

    def forward(self, x):
        return self.vit.forward_features(x)[:, 0]


class RotationViT(nn.Module):
    """ViT-B/16 backbone (``encoder``) + a linear 4-way rotation head (``head``).

    The rotation objective is CrossEntropy over the four right-angle rotations,
    read from the CLS token -- the capture's Step-2 protocol.
    """

    def __init__(self, num_classes: int = 4, image_size: int = 224,
                 patch_size: int = 16, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0) -> None:
        super().__init__()
        self.encoder = VisionTransformer(
            img_size=image_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            num_classes=0, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            qkv_bias=True, norm_layer=nn.LayerNorm)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.head(self.encoder.forward_features(x)[:, 0])

    def get_encoder(self) -> _ClsFeature:
        return _ClsFeature(self.encoder)


def build_vit_rotation_model(num_classes: int = 4, image_size: int = 224,
                             patch_size: int = 16, embed_dim: int = 768,
                             depth: int = 12, num_heads: int = 12,
                             mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                             attn_drop_rate: float = 0.0) -> RotationViT:
    return RotationViT(num_classes=num_classes, image_size=image_size,
                       patch_size=patch_size, embed_dim=embed_dim, depth=depth,
                       num_heads=num_heads, mlp_ratio=mlp_ratio,
                       drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
