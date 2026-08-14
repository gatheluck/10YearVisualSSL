"""ViT Step-2 SwAV model: a unified ViT-B/16 encoder + projection head +
prototypes, with the multi-crop forward the native ResNet path uses.

Faithful to the capture's `models/vit_swav.py`: a timm `VisionTransformer` (from
scratch, ``dynamic_img_size=True`` so it accepts the 224 global and 96 local
crops), its CLS token through a 2-layer MLP projector (`Linear -> BN -> ReLU ->
Linear`), L2-normalised, and learnable prototypes (`Linear(out_dim, K, bias=
False)`, column-normalised). Restructured to mirror this port's `ResNetSwAV`
interface exactly -- ``forward(list_of_crops)`` groups the crops by resolution,
runs the backbone per group, concatenates, projects and scores against the
prototypes, returning ``(embeddings, scores)`` -- so the port's multi-crop
`train_epoch`, `swav_loss` and `distributed_sinkhorn` are reused unchanged. The
ViT trunk lives under ``self.encoder`` (num_classes=0), so `encoder.pt` keeps
only ``encoder.*`` (the projection head and prototypes are training machinery,
excluded) and ``get_encoder()`` returns the CLS feature for the linear probe (the
eval sizes its head to it dynamically). timm is imported lazily (only on
`arch: vit`); the ViT dimensions are configurable so a tiny model can run a CPU
smoke.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ClsFeature(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)


class ViTSwAV(nn.Module):
    def __init__(self, out_dim: int = 128, hidden_mlp: int = 2048,
                 nmb_prototypes: int = 3000, image_size: int = 224,
                 patch_size: int = 16, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0) -> None:
        super().__init__()
        from timm.models.vision_transformer import VisionTransformer
        self.encoder = VisionTransformer(
            img_size=image_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            num_classes=0, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            qkv_bias=True, norm_layer=nn.LayerNorm, dynamic_img_size=True)
        self.feat_dim = embed_dim
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_mlp),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_mlp, out_dim),
        )
        self.prototypes = nn.Linear(out_dim, nmb_prototypes, bias=False)

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).view(x.size(0), -1)

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection_head(h), dim=1)

    @torch.no_grad()
    def normalize_prototypes(self) -> None:
        w = F.normalize(self.prototypes.weight.data.clone(), dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        crop_sizes = torch.tensor([crop.shape[-1] for crop in x])
        idx_crops = torch.cumsum(
            torch.unique_consecutive(crop_sizes, return_counts=True)[1], 0)
        start_idx, output = 0, []
        for end_idx in idx_crops:
            end_idx = int(end_idx)
            output.append(self.forward_encoder(torch.cat(x[start_idx:end_idx])))
            start_idx = end_idx
        z = self.forward_head(torch.cat(output, dim=0))
        return z, self.prototypes(z)

    def get_encoder(self) -> nn.Module:
        return _ClsFeature(self.encoder)


def build_vit_swav(out_dim: int = 128, hidden_mlp: int = 2048,
                   nmb_prototypes: int = 3000, image_size: int = 224,
                   patch_size: int = 16, embed_dim: int = 768, depth: int = 12,
                   num_heads: int = 12, mlp_ratio: float = 4.0,
                   drop_rate: float = 0.0, attn_drop_rate: float = 0.0) -> ViTSwAV:
    return ViTSwAV(out_dim=out_dim, hidden_mlp=hidden_mlp,
                   nmb_prototypes=nmb_prototypes, image_size=image_size,
                   patch_size=patch_size, embed_dim=embed_dim, depth=depth,
                   num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate,
                   attn_drop_rate=attn_drop_rate)
