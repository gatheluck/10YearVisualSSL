"""ViT Step-2 CMC model: two unified ViT-B/16 branches (L and ab) + two 3-layer
MLP projectors.

Faithful to the capture's `models/vit_cmc.py`: an RGB image is converted to Lab
(by the dataset) and split into L (1-channel) and ab (2-channel); a timm
`VisionTransformer` per view (from scratch, `in_chans` 1 and 2) maps each to its
CLS token, then a 3-layer MLP projector to an L2-normalised `feat_dim` embedding.
The two embeddings feed the cross-view NCE memory-bank objective. This port's
convention: the two ViT trunks live under ``encoder_l`` / ``encoder_ab``
(num_classes=0), so `encoder.pt` keeps only ``encoder_l.*`` / ``encoder_ab.*``
(the projectors are training machinery, excluded), and ``get_encoder()`` returns
the two CLS features concatenated (the representation the linear probe reads; the
eval sizes its head to it dynamically). timm is imported lazily (only on
`arch: vit`); the ViT dimensions are configurable so a tiny model can run a CPU
smoke.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _projector(embed_dim: int, hidden_dim: int, feat_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(embed_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, feat_dim),
    )


class _ClsConcat(nn.Module):
    """Both branches' CLS features, concatenated -- the linear-probe feature."""

    def __init__(self, model: "ViTCMC") -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_l, cls_ab = self.model.get_features(x)
        return torch.cat([cls_l, cls_ab], dim=1)


class ViTCMC(nn.Module):
    def __init__(self, feat_dim: int = 256, hidden_dim: int = 2048,
                 image_size: int = 224, patch_size: int = 16,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0) -> None:
        super().__init__()
        from timm.models.vision_transformer import VisionTransformer
        vit = dict(img_size=image_size, patch_size=patch_size,
                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                   mlp_ratio=mlp_ratio, num_classes=0, drop_rate=drop_rate,
                   attn_drop_rate=attn_drop_rate, qkv_bias=True,
                   norm_layer=nn.LayerNorm)
        self.encoder_l = VisionTransformer(in_chans=1, **vit)
        self.encoder_ab = VisionTransformer(in_chans=2, **vit)
        self.proj_l = _projector(embed_dim, hidden_dim, feat_dim)
        self.proj_ab = _projector(embed_dim, hidden_dim, feat_dim)

    @staticmethod
    def _cls(vit: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return vit.forward_features(x)[:, 0]

    def forward(self, x: torch.Tensor):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = F.normalize(self.proj_l(self._cls(self.encoder_l, l)), dim=1)
        feat_ab = F.normalize(self.proj_ab(self._cls(self.encoder_ab, ab)), dim=1)
        return feat_l, feat_ab

    def get_features(self, x: torch.Tensor):
        """The two branches' raw CLS features (pre-projection)."""
        l, ab = torch.split(x, [1, 2], dim=1)
        return self._cls(self.encoder_l, l), self._cls(self.encoder_ab, ab)

    def get_encoder(self) -> _ClsConcat:
        return _ClsConcat(self)


def build_vit_cmc(feat_dim: int = 256, hidden_dim: int = 2048,
                  image_size: int = 224, patch_size: int = 16,
                  embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                  mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                  attn_drop_rate: float = 0.0) -> ViTCMC:
    return ViTCMC(feat_dim=feat_dim, hidden_dim=hidden_dim, image_size=image_size,
                  patch_size=patch_size, embed_dim=embed_dim, depth=depth,
                  num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate,
                  attn_drop_rate=attn_drop_rate)
