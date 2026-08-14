"""ViT Step-2 SeLa model: a unified ViT-B/16 + a single linear prototype head.

Faithful to the capture's `models/vit_sela.py`: a timm `VisionTransformer` (from
scratch) maps the image to its CLS token, and a single `Linear(embed_dim, k)`
prototype head produces the logits the SeLa self-labelling assigns over. The
prototype head is trained continuously (never reset, unlike DeepCluster). This
port's convention: the ViT trunk lives under ``self.backbone`` (num_classes=0),
so `encoder.pt` keeps only ``backbone.*`` (the prototype head is training
machinery, excluded) and ``get_features()`` returns the CLS feature for the
linear probe (the eval sizes its head to it dynamically). ``forward`` returns
`(B, k)` logits, which the shared Sinkhorn util consumes directly. timm is
imported lazily (only on `arch: vit`); the ViT dimensions are configurable so a
tiny model can run a CPU smoke.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ViTSeLa(nn.Module):
    def __init__(self, num_classes: int = 1000, image_size: int = 224,
                 patch_size: int = 16, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0) -> None:
        super().__init__()
        from timm.models.vision_transformer import VisionTransformer
        self.backbone = VisionTransformer(
            img_size=image_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            num_classes=0, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            qkv_bias=True, norm_layer=nn.LayerNorm)
        self.feature_dim = embed_dim
        self.top_layer = nn.Linear(embed_dim, num_classes)
        nn.init.normal_(self.top_layer.weight, 0, 0.01)
        nn.init.zeros_(self.top_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.top_layer(self.backbone(x))

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """The CLS backbone feature, no gradient -- for Sinkhorn and probing."""
        with torch.no_grad():
            return self.backbone(x)


def build_vit_sela(num_classes: int = 1000, image_size: int = 224,
                   patch_size: int = 16, embed_dim: int = 768, depth: int = 12,
                   num_heads: int = 12, mlp_ratio: float = 4.0,
                   drop_rate: float = 0.0, attn_drop_rate: float = 0.0) -> ViTSeLa:
    return ViTSeLa(num_classes=num_classes, image_size=image_size,
                   patch_size=patch_size, embed_dim=embed_dim, depth=depth,
                   num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate,
                   attn_drop_rate=attn_drop_rate)
