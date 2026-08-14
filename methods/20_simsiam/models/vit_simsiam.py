"""ViT Step-2 SimSiam model: a unified ViT-B/16 backbone + 3-layer projector +
2-layer predictor.

Faithful to the capture's `models/simsiam_vit.py`: a timm `VisionTransformer`
(from scratch) reads the image, its CLS token feeds a 3-layer MLP projector
(`[Linear+BN+ReLU] x2 + Linear + BN(affine=False)`) and a 2-layer bottleneck
predictor (`Linear+BN+ReLU + Linear`). `forward` returns `(p1, p2, z1, z2)` with
the **stop-gradient on the projector outputs z, not on the predictor outputs p**
(SimSiam's defining detail). The loss is the shared negative-cosine
`simsiam_loss`. This port's convention: the ViT trunk lives under
``self.backbone`` (num_classes=0), so `encoder.pt` keeps only ``backbone.*`` and
``get_encoder()`` returns that trunk (its CLS feature) for the linear probe. timm
is imported lazily (only on `arch: vit`), and the ViT dimensions are configurable
so a tiny model can run a CPU smoke.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimSiamViT(nn.Module):
    def __init__(self, dim: int = 768, pred_dim: int = 192,
                 image_size: int = 224, patch_size: int = 16,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0) -> None:
        super().__init__()
        from timm.models.vision_transformer import VisionTransformer
        self.backbone = VisionTransformer(
            img_size=image_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            num_classes=0, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            qkv_bias=True, norm_layer=nn.LayerNorm)

        # 3-layer projector (last BN has affine=False), on the CLS feature.
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False),
        )

        # 2-layer bottleneck predictor.
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        h1 = self.backbone(x1)          # (B, embed_dim) -- CLS token
        h2 = self.backbone(x2)
        z1 = self.projector(h1)         # (B, dim)
        z2 = self.projector(h2)
        p1 = self.predictor(z1)         # (B, dim)
        p2 = self.predictor(z2)
        # Stop-gradient on z, not on p.
        return p1, p2, z1.detach(), z2.detach()

    def get_encoder(self) -> nn.Module:
        """The ViT trunk (CLS feature), for the linear probe."""
        return self.backbone


def build_simsiam_vit(dim: int = 768, pred_dim: int = 192,
                      image_size: int = 224, patch_size: int = 16,
                      embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                      mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                      attn_drop_rate: float = 0.0) -> SimSiamViT:
    return SimSiamViT(dim=dim, pred_dim=pred_dim, image_size=image_size,
                      patch_size=patch_size, embed_dim=embed_dim, depth=depth,
                      num_heads=num_heads, mlp_ratio=mlp_ratio,
                      drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
