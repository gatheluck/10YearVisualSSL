"""ViT backbone for the DINOv2 unified Step-2 pretraining (iBOT mask-token support).

Ported from the capture's `methods/28_dinov2/models/dinov2_vit.py`. Wraps timm's
VisionTransformer so masked patch positions can be replaced with a learnable
[MASK] token before the blocks (iBOT; Zhou et al. 2021).

The capture built the fixed `vit_base_patch16_224`; here the ViT dims are threaded
through `timm.create_model` (via **model_kwargs) so a small hermetic CPU smoke can
build a tiny ViT at a lower resolution. The real recipe passes ViT-B/16 dims.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DINOv2Backbone(nn.Module):
    """A timm ViT with a learnable iBOT [MASK] token.

    `get_cls_token(x, mask)` -> (B, D); `get_all_tokens(x, mask)` -> (cls, patches).
    """

    def __init__(self, arch: str = "vit_base_patch16_224", pretrained: bool = False,
                 **model_kwargs):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            arch, pretrained=pretrained, num_classes=0, global_pool="",
            dynamic_img_size=True, **model_kwargs)

        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_embed.patch_size
        if isinstance(self.patch_size, (tuple, list)):
            self.patch_size = self.patch_size[0]

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.patch_embed(x)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone._pos_embed(x)

    def _blocks(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "patch_drop"):
            x = self.backbone.patch_drop(x)
        if hasattr(self.backbone, "norm_pre"):
            x = self.backbone.norm_pre(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        return x

    def forward(self, x: torch.Tensor, mask: "torch.Tensor | None" = None) -> torch.Tensor:
        """x: (B,3,H,W); mask: (B,N) bool, True = masked patch. Returns (B,1+N,C)."""
        B = x.shape[0]
        patches = self._embed(x)
        if patches.dim() == 4:
            _, H, W, C = patches.shape
            patches_flat = patches.view(B, H * W, C)
        else:
            H = W = C = None
            patches_flat = patches
        N = patches_flat.shape[1]

        if mask is not None:
            mask_tokens = self.mask_token.expand(B, N, -1)
            m = mask.to(patches_flat.device).unsqueeze(-1).to(patches_flat.dtype)
            patches_flat = patches_flat * (1.0 - m) + mask_tokens * m

        if patches.dim() == 4:
            patches = patches_flat.view(B, H, W, C)
        else:
            patches = patches_flat

        tokens = self._pos_embed(patches)
        tokens = self._blocks(tokens)
        return tokens

    def get_cls_token(self, x: torch.Tensor,
                      mask: "torch.Tensor | None" = None) -> torch.Tensor:
        return self.forward(x, mask)[:, 0]

    def get_all_tokens(self, x: torch.Tensor, mask: "torch.Tensor | None" = None):
        tokens = self.forward(x, mask)
        return tokens[:, 0], tokens[:, 1:]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """All tokens, unmasked -- so a linear probe can read the CLS at [:, 0]."""
        return self.forward(x, mask=None)


def build_dinov2_backbone(arch: str, pretrained: bool = False,
                          **model_kwargs) -> DINOv2Backbone:
    return DINOv2Backbone(arch=arch, pretrained=pretrained, **model_kwargs)
