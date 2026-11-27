"""ViT Step-2 SimCLR v2 model: a unified ViT-B/16 + a 3-layer projection head.

Faithful to the capture's `models/vit_simclrv2.py`: a timm `VisionTransformer`
(from scratch) reads the image and its CLS token feeds a 3-layer MLP projector
(`Linear(768,2048,bias=F)->BN->ReLU->Linear(2048,2048,bias=F)->BN->ReLU->
Linear(2048,out_dim,bias=F)`); the projected vector is L2-normalised for NT-Xent.
Restructured to this port's convention: the ViT backbone lives under ``encoder``
(num_classes=0), the projector is separate (`encoder.pt` keeps only `encoder.*`),
and ``get_encoder()`` returns the CLS feature so `evaluate_linear_simclrv2.py`
probes it unchanged. timm is imported lazily (only on the `arch: vit` path).
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer


class _ClsFeature(nn.Module):
    def __init__(self, vit: VisionTransformer) -> None:
        super().__init__()
        self.vit = vit

    def forward(self, x):
        return self.vit.forward_features(x)[:, 0]


class SimCLRv2ViT(nn.Module):
    def __init__(self, out_dim: int = 128, image_size: int = 224,
                 patch_size: int = 16, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0) -> None:
        super().__init__()
        self.encoder = VisionTransformer(
            img_size=image_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            num_classes=0, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            qkv_bias=True, norm_layer=nn.LayerNorm)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, out_dim, bias=False),
        )

    def forward(self, x):
        z = self.projector(self.encoder.forward_features(x)[:, 0])
        return F.normalize(z, dim=1)

    def get_encoder(self) -> _ClsFeature:
        return _ClsFeature(self.encoder)


def build_vit_simclrv2(out_dim: int = 128, image_size: int = 224,
                       patch_size: int = 16, embed_dim: int = 768, depth: int = 12,
                       num_heads: int = 12, mlp_ratio: float = 4.0,
                       drop_rate: float = 0.0,
                       attn_drop_rate: float = 0.0) -> SimCLRv2ViT:
    return SimCLRv2ViT(out_dim=out_dim, image_size=image_size,
                       patch_size=patch_size, embed_dim=embed_dim, depth=depth,
                       num_heads=num_heads, mlp_ratio=mlp_ratio,
                       drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
