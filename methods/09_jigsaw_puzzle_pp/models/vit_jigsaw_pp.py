"""ViT Step-2 Jigsaw++ model: a unified ViT-B/16 + a permutation head (701-way).

Faithful to the capture's `models/vit_jigsaw_pp.py`: a timm `VisionTransformer`
(from scratch) reads the reassembled puzzle image and its CLS token feeds a
`LayerNorm -> Linear -> GELU -> Dropout -> Linear` classifier over the 701
permutations. Restructured to this port's convention: the ViT backbone lives
under ``encoder`` (built with `num_classes=0`, so `encoder.pt` holds only
`encoder.*`), the classifier is a separate ``head``, and ``get_encoder()``
returns the CLS feature so `evaluate_linear_jigsaw_pp.py` probes it unchanged.
timm is imported lazily (only on the `arch: vit` path).
"""

from __future__ import annotations

import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class _ClsFeature(nn.Module):
    def __init__(self, vit: VisionTransformer) -> None:
        super().__init__()
        self.vit = vit

    def forward(self, x):
        return self.vit.forward_features(x)[:, 0]


class JigsawPPViT(nn.Module):
    def __init__(self, num_classes: int = 701, image_size: int = 224,
                 patch_size: int = 16, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0,
                 hidden_dim: int = 2048, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0) -> None:
        super().__init__()
        self.encoder = VisionTransformer(
            img_size=image_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            num_classes=0, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            qkv_bias=True, norm_layer=nn.LayerNorm)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(hidden_dim, num_classes),
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.encoder.forward_features(x)[:, 0])

    def get_encoder(self) -> _ClsFeature:
        return _ClsFeature(self.encoder)


def build_vit_jigsaw_pp_model(num_classes: int = 701, image_size: int = 224,
                              patch_size: int = 16, embed_dim: int = 768,
                              depth: int = 12, num_heads: int = 12,
                              mlp_ratio: float = 4.0, hidden_dim: int = 2048,
                              drop_rate: float = 0.0,
                              attn_drop_rate: float = 0.0) -> JigsawPPViT:
    return JigsawPPViT(num_classes=num_classes, image_size=image_size,
                       patch_size=patch_size, embed_dim=embed_dim, depth=depth,
                       num_heads=num_heads, mlp_ratio=mlp_ratio,
                       hidden_dim=hidden_dim, drop_rate=drop_rate,
                       attn_drop_rate=attn_drop_rate)
