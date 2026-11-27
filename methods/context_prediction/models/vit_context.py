"""ViT Step-2 context-prediction model: a shared ViT-B/16 + an 8-way head.

Faithful to the capture's `models/vit_context.py`: the two patches (center and a
neighbour) share one timm `VisionTransformer` (from scratch); each patch's CLS
token is taken, the two are concatenated, and a
`LayerNorm -> Linear -> GELU -> Dropout -> Linear -> GELU -> Dropout -> Linear`
head predicts which of the 8 relative positions the neighbour sits in
(CrossEntropy). Restructured to this port's convention: the ViT backbone lives
under ``encoder`` (built with `num_classes=0`), the head is separate, and
``get_encoder()`` returns a single-patch CLS-feature module so
`evaluate_linear_official.py` probes it. `encoder.pt` carries the encoder's own
keys (the adapter strips the `encoder.` prefix). timm is imported lazily.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class _ClsFeature(nn.Module):
    """`forward(x)` = the ViT CLS-token feature for one patch/image."""

    def __init__(self, vit: VisionTransformer) -> None:
        super().__init__()
        self.vit = vit

    def forward(self, x):
        return self.vit.forward_features(x)[:, 0]


class ContextViT(nn.Module):
    def __init__(self, num_classes: int = 8, image_size: int = 224,
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
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(hidden_dim, num_classes),
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def _cls(self, x):
        return self.encoder.forward_features(x)[:, 0]

    def forward(self, first_patch, second_patch):
        combined = torch.cat([self._cls(first_patch), self._cls(second_patch)],
                             dim=1)
        return self.head(combined)

    def get_encoder(self) -> _ClsFeature:
        return _ClsFeature(self.encoder)


def build_vit_context_model(num_classes: int = 8, image_size: int = 224,
                            patch_size: int = 16, embed_dim: int = 768,
                            depth: int = 12, num_heads: int = 12,
                            mlp_ratio: float = 4.0, hidden_dim: int = 2048,
                            drop_rate: float = 0.0,
                            attn_drop_rate: float = 0.0) -> ContextViT:
    return ContextViT(num_classes=num_classes, image_size=image_size,
                      patch_size=patch_size, embed_dim=embed_dim, depth=depth,
                      num_heads=num_heads, mlp_ratio=mlp_ratio,
                      hidden_dim=hidden_dim, drop_rate=drop_rate,
                      attn_drop_rate=attn_drop_rate)
