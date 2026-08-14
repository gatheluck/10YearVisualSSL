"""ViT Step-2 Instance Discrimination model: a unified ViT-B/16 + a 128-d
L2-normalised projection head.

Faithful to the capture's `models/vit_instdisc.py`: a timm `VisionTransformer`
(from scratch) reads the image, its CLS token feeds `Linear(embed_dim,
feature_dim)`, L2-normalised, for the NCE memory-bank objective. Restructured to
this port's convention: the ViT trunk lives under ``self.encoder``
(num_classes=0), so `encoder.pt` keeps only ``encoder.*`` and ``get_encoder()``
returns the CLS feature so `evaluate_linear_instdisc.py` probes it unchanged (its
head sizes to the feature dynamically). timm is imported lazily (only on
`arch: vit`); the ViT dimensions are configurable so a tiny model can run a CPU
smoke.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class _ClsFeature(nn.Module):
    def __init__(self, vit) -> None:
        super().__init__()
        self.vit = vit

    def forward(self, x):
        return self.vit.forward_features(x)[:, 0]


class ViTInstDisc(nn.Module):
    def __init__(self, feature_dim: int = 128, image_size: int = 224,
                 patch_size: int = 16, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0) -> None:
        super().__init__()
        from timm.models.vision_transformer import VisionTransformer
        self.encoder = VisionTransformer(
            img_size=image_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            num_classes=0, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            qkv_bias=True, norm_layer=nn.LayerNorm)
        self.fc = nn.Linear(embed_dim, feature_dim)

    def forward(self, x):
        cls = self.encoder.forward_features(x)[:, 0]
        return F.normalize(self.fc(cls), dim=1)

    def get_encoder(self) -> _ClsFeature:
        """The ViT trunk (CLS feature), for the linear probe."""
        return _ClsFeature(self.encoder)


def build_vit_instdisc(feature_dim: int = 128, image_size: int = 224,
                       patch_size: int = 16, embed_dim: int = 768,
                       depth: int = 12, num_heads: int = 12,
                       mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                       attn_drop_rate: float = 0.0) -> ViTInstDisc:
    return ViTInstDisc(feature_dim=feature_dim, image_size=image_size,
                       patch_size=patch_size, embed_dim=embed_dim, depth=depth,
                       num_heads=num_heads, mlp_ratio=mlp_ratio,
                       drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
