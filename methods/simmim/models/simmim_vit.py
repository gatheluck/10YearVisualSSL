"""SimMIM with a ViT-B/16 encoder -- the unified Step-2 backbone (Xie et al.,
2022; arXiv:2111.09886).

SimMIM's step 1 is genuinely Swin-based (see ``simmim_swinb.py``); the capture's
unified Step 2 plugs the same masked-image-modelling objective into a **ViT-B/16**
encoder instead. This is a port of the capture's ``models/simmim_vit.py``.

Masking is pixel-space, identical to the Swin variant: masked pixels are replaced
by a learned per-channel mask token before the patch embed, the FULL token grid
(patches + CLS) is encoded, a single linear head predicts ``3 x patch^2`` pixel
values per patch, and an L1 loss is taken on the masked patches only. ``encoder.pt``
is the bare timm ViT (``encoder.*``); the mask token and the prediction head are
training machinery and are excluded. For linear probing the representation is the
CLS token (``forward_features(x)[:, 0]``), the capture's own choice.

``build_vit_encoder`` is the single construction path for the encoder, shared by
the SimMIM wrapper and the adapter's ``load_encoder`` so the ViT is built one way.
The capture built the fixed ``vit_base_patch16_224``; here the ViT dims are
threaded through to ``timm.create_model`` so a small hermetic CPU smoke can run a
tiny ViT at a lower resolution (the real recipe passes ViT-B/16 dims).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_vit_encoder(img_size: int = 224, patch_size: int = 16,
                      embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                      mlp_ratio: float = 4.0, drop_path_rate: float = 0.0):
    """The bare timm ViT encoder (no classifier, all tokens returned).

    One construction path for both the SimMIM wrapper and the adapter's
    ``load_encoder``. ``global_pool=""`` makes ``forward_features`` return the full
    ``(B, N+1, D)`` token sequence, so the CLS token (index 0) is available to the
    probe. timm is imported lazily so the native Swin path never triggers it here.
    """
    import timm
    return timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="",
        img_size=int(img_size), patch_size=int(patch_size),
        embed_dim=int(embed_dim), depth=int(depth), num_heads=int(num_heads),
        mlp_ratio=float(mlp_ratio), drop_path_rate=float(drop_path_rate))


def build_simmim_vit(img_size: int = 224, patch_size: int = 16,
                     mask_patch_size: int = 16, embed_dim: int = 768,
                     depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0,
                     drop_path_rate: float = 0.1):
    return SimMIMViT(
        img_size=img_size, patch_size=patch_size, mask_patch_size=mask_patch_size,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate)


class SimMIMViT(nn.Module):
    """SimMIM masked-image-modelling wrapper around a ViT-B/16 encoder.

    Masking is applied in pixel-space before the patch embedding: masked pixels are
    replaced by a learned per-channel mask token, and ALL tokens (masked included)
    are processed by the ViT.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 mask_patch_size: int = 16, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0,
                 drop_path_rate: float = 0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_patch_size = mask_patch_size
        self.encoder_dim = embed_dim

        self.encoder = build_vit_encoder(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate)

        # Learnable per-channel mask token, broadcast to fill masked pixels.
        self.mask_token = nn.Parameter(torch.zeros(1, 3, 1, 1))
        # Prediction head: embed_dim -> 3 x mask_patch^2 pixel values per patch.
        self.decoder = nn.Linear(embed_dim, 3 * mask_patch_size * mask_patch_size,
                                 bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder.weight, std=0.02)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """x: (B,3,H,W) normalised; mask: (B,H,W) float, 1=masked pixel.
        Returns (scalar L1 loss on masked patches, prediction (B,n,n,3*p^2))."""
        B, _, H, W = x.shape
        p = self.mask_patch_size
        n = H // p

        m = mask.unsqueeze(1).float()                      # (B,1,H,W)
        x_in = x * (1.0 - m) + self.mask_token * m         # (B,3,H,W)

        all_tokens = self.encoder.forward_features(x_in)   # (B, N+1, D)
        patch_tokens = all_tokens[:, 1:]                   # drop CLS
        patch_tokens = patch_tokens.reshape(B, n, n, self.encoder_dim)
        pred = self.decoder(patch_tokens)                  # (B,n,n,3*p*p)

        loss = self._l1_loss(pred, x, mask, n, p)
        return loss, pred

    def _l1_loss(self, pred, target, mask, n, p):
        B = pred.shape[0]
        t = target.reshape(B, 3, n, p, n, p)
        t = t.permute(0, 2, 4, 1, 3, 5).reshape(B, n, n, 3 * p * p)
        mp = mask.reshape(B, n, p, n, p)[:, :, 0, :, 0].float()   # (B,n,n)
        loss_per = F.l1_loss(pred, t, reduction="none").mean(-1)  # (B,n,n)
        return (loss_per * mp).sum() / mp.sum().clamp(min=1.0)

    def get_encoder(self):
        """The bare ViT encoder, for linear probing (probe its CLS token)."""
        return self.encoder
