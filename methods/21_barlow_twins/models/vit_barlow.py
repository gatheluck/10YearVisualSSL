"""ViT Step-2 Barlow Twins model: a unified ViT-B/16 backbone + 3-layer MLP
projector, with the cross-correlation (redundancy-reduction) loss in-model.

Faithful to the capture's `models/barlow_vit.py`: a timm `VisionTransformer`
(from scratch) reads the image, its CLS token feeds the projector; the loss
brings the empirical cross-correlation of the two views' batch-normalised
projections toward the identity (on-diagonal -> 1, off-diagonal -> 0, weighted by
lambd). The loss is architecture-agnostic and reuses the native path's
``off_diagonal`` and ``_build_projector`` unchanged. This port's convention: the
ViT trunk lives under ``self.backbone`` (num_classes=0), so `encoder.pt` keeps
only ``backbone.*`` and ``get_encoder()`` returns that trunk (its CLS feature)
for the linear probe (the eval sizes its head to embed_dim). timm is imported
lazily (only on `arch: vit`); the ViT dimensions are configurable so a tiny model
can run a CPU smoke.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:  # package context (models.vit_barlow) vs standalone test load (load_from)
    from .barlow_resnet import _build_projector, off_diagonal
except ImportError:  # pragma: no cover - exercised only by the standalone load
    from models.barlow_resnet import _build_projector, off_diagonal


class BarlowTwinsViT(nn.Module):
    def __init__(self, projector: str = "2048-2048-2048", lambd: float = 0.0051,
                 image_size: int = 224, patch_size: int = 16,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0) -> None:
        super().__init__()
        self.lambd = lambd
        from timm.models.vision_transformer import VisionTransformer
        self.backbone = VisionTransformer(
            img_size=image_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            num_classes=0, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            qkv_bias=True, norm_layer=nn.LayerNorm)
        self.projector = _build_projector(embed_dim, projector)
        proj_dim = int(projector.split("-")[-1])
        self.bn = nn.BatchNorm1d(proj_dim, affine=False)

    def forward(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(z1.size(0))
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        return on_diag + self.lambd * off_diag

    def get_encoder(self) -> nn.Module:
        """The ViT trunk (CLS feature), for the linear probe."""
        return self.backbone


def build_barlow_vit(projector: str = "2048-2048-2048", lambd: float = 0.0051,
                     image_size: int = 224, patch_size: int = 16,
                     embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                     mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                     attn_drop_rate: float = 0.0) -> BarlowTwinsViT:
    return BarlowTwinsViT(projector=projector, lambd=lambd,
                          image_size=image_size, patch_size=patch_size,
                          embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                          mlp_ratio=mlp_ratio, drop_rate=drop_rate,
                          attn_drop_rate=attn_drop_rate)
