"""ViT Step-2 BYOL model: unified ViT-B/16 online/target encoders + projector +
predictor, with an EMA target network.

Faithful to the capture's `models/vit_byol.py`: a timm `VisionTransformer` (from
scratch) is the online encoder (CLS token); an online projector and predictor
(reused from the native path) feed the symmetric negative-cosine BYOL loss
against an EMA **target** encoder+projector's projection. This port's convention:
the online ViT trunk lives under ``self.online_encoder`` (num_classes=0), so
`encoder.pt` keeps only ``online_encoder.*`` and ``encode()`` returns that trunk's
CLS feature for the linear probe (the eval sizes its head dynamically, so no eval
change is needed). The target BN modules stay in training mode (batch stats) and
the target parameters are frozen -- BYOL's target-network semantics, which in this
single-process port need nothing beyond ``model.train()`` (no SyncBatchNorm as
under DDP). timm is imported lazily (only on `arch: vit`); the ViT dimensions are
configurable so a tiny model can run a CPU smoke.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

try:  # package context (models.vit_byol) vs standalone test load (load_from)
    from .resnet50_byol import PredictionMLP, ProjectionMLP
except ImportError:  # pragma: no cover - exercised only by the standalone load
    from models.resnet50_byol import PredictionMLP, ProjectionMLP


class BYOLViT(nn.Module):
    def __init__(self, encoder_dim: int = 768, proj_hidden_dim: int = 4096,
                 proj_output_dim: int = 256, pred_hidden_dim: int = 4096,
                 pred_output_dim: int = 256, image_size: int = 224,
                 patch_size: int = 16, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0) -> None:
        super().__init__()
        from timm.models.vision_transformer import VisionTransformer
        # The online encoder IS the ViT trunk (CLS feature), so encoder.pt is
        # online_encoder.* and encode() = online_encoder(x).
        self.online_encoder = VisionTransformer(
            img_size=image_size, patch_size=patch_size, embed_dim=encoder_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            num_classes=0, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            qkv_bias=True, norm_layer=nn.LayerNorm)
        self.online_projector = ProjectionMLP(encoder_dim, proj_hidden_dim,
                                              proj_output_dim)
        self.predictor = PredictionMLP(proj_output_dim, pred_hidden_dim,
                                       pred_output_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False
        self.encoder_dim = encoder_dim

    @torch.no_grad()
    def update_target_network(self, tau: float) -> None:
        for op, tp in zip(self.online_encoder.parameters(),
                          self.target_encoder.parameters()):
            tp.data.mul_(tau).add_(op.data, alpha=1.0 - tau)
        for op, tp in zip(self.online_projector.parameters(),
                          self.target_projector.parameters()):
            tp.data.mul_(tau).add_(op.data, alpha=1.0 - tau)

    def forward_online(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.online_projector(self.online_encoder(x)))

    @torch.no_grad()
    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_projector(self.target_encoder(x))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        p1 = self.forward_online(x1)
        p2 = self.forward_online(x2)
        z1 = self.forward_target(x1)
        z2 = self.forward_target(x2)
        return p1, p2, z1, z2

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """The online ViT trunk's CLS feature, for the linear probe."""
        return self.online_encoder(x)


def build_byol_vit(encoder_dim: int = 768, proj_hidden_dim: int = 4096,
                   proj_output_dim: int = 256, pred_hidden_dim: int = 4096,
                   pred_output_dim: int = 256, image_size: int = 224,
                   patch_size: int = 16, depth: int = 12, num_heads: int = 12,
                   mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                   attn_drop_rate: float = 0.0) -> BYOLViT:
    return BYOLViT(encoder_dim=encoder_dim, proj_hidden_dim=proj_hidden_dim,
                   proj_output_dim=proj_output_dim,
                   pred_hidden_dim=pred_hidden_dim,
                   pred_output_dim=pred_output_dim, image_size=image_size,
                   patch_size=patch_size, depth=depth, num_heads=num_heads,
                   mlp_ratio=mlp_ratio, drop_rate=drop_rate,
                   attn_drop_rate=attn_drop_rate)
