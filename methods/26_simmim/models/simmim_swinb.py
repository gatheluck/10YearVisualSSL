"""SimMIM with a Swin-B encoder (Xie et al., 2022; arXiv:2111.09886).

Ported from the lab's own SimMIM code. Masking is applied after patch embedding:
the masked patch tokens are replaced by a learned mask token, the full token grid
is encoded by Swin, and a 1x1 conv + PixelShuffle decoder reconstructs pixels; an
L1 loss is taken only on the masked pixels.

SimMIM's step 1 is genuinely Swin-based, so timm supplies the SwinTransformer
(``build_swin_encoder`` -- shared by the model and the adapter's ``load_encoder``
so the encoder is constructed one way). The Swin is built from scratch (no
pretrained download), so the run stays hermetic. ``encoder.pt`` is the bare Swin
encoder; the learned mask token and the decoder are excluded.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_swin_encoder(img_size: int = 192, patch_size: int = 4,
                       window_size: int = 6, embed_dim: int = 128,
                       depths: tuple = (2, 2, 18, 2),
                       num_heads: tuple = (4, 8, 16, 32),
                       drop_path_rate: float = 0.0) -> nn.Module:
    """The bare timm Swin encoder. One construction path for both the SimMIM
    model and the linear-eval loader (drop_path_rate does not change the
    state_dict keys, so the loader may leave it at 0)."""
    try:
        from timm.models.swin_transformer import SwinTransformer
    except ImportError as e:  # pragma: no cover - environment guard
        raise ImportError("timm>=0.9 required for SwinTransformer") from e
    return SwinTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        num_classes=0,          # no classification head
        global_pool="",         # return feature tokens, not pooled
        embed_dim=embed_dim,
        depths=tuple(depths),
        num_heads=tuple(num_heads),
        window_size=window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=drop_path_rate,
    )


def build_simmim_swinb(
    img_size: int = 192,
    patch_size: int = 4,
    window_size: int = 6,
    embed_dim: int = 128,
    depths: tuple = (2, 2, 18, 2),
    num_heads: tuple = (4, 8, 16, 32),
    mask_patch_size: int = 32,
    drop_path_rate: float = 0.0,
) -> "SimMIMSwinB":
    return SimMIMSwinB(
        img_size=img_size,
        patch_size=patch_size,
        window_size=window_size,
        embed_dim=embed_dim,
        depths=tuple(depths),
        num_heads=tuple(num_heads),
        mask_patch_size=mask_patch_size,
        drop_path_rate=drop_path_rate,
    )


class SimMIMSwinB(nn.Module):
    """SimMIM masked image modelling wrapper around a Swin-B encoder."""

    def __init__(
        self,
        img_size: int = 192,
        patch_size: int = 4,
        window_size: int = 6,
        embed_dim: int = 128,
        depths: tuple = (2, 2, 18, 2),
        num_heads: tuple = (4, 8, 16, 32),
        mask_patch_size: int = 32,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        depths = tuple(depths)
        self.img_size = img_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = patch_size
        self.encoder_stride = patch_size * (2 ** (len(depths) - 1))
        self.in_chans = 3
        # Last-stage channel dim: embed_dim * 2^(num_stages-1) (128*8 = 1024).
        self.encoder_dim = embed_dim * (2 ** (len(depths) - 1))

        self.encoder = build_swin_encoder(
            img_size=img_size, patch_size=patch_size, window_size=window_size,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            drop_path_rate=drop_path_rate)

        # Embed-dim mask token applied to Swin patch embeddings.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        # Official SimMIM-style lightweight decoder.
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder_dim,
                out_channels=(self.encoder_stride ** 2) * self.in_chans,
                kernel_size=1,
            ),
            nn.PixelShuffle(self.encoder_stride),
        )

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """x: (B,3,H,W); mask: (B, H/patch, W/patch) with 1 = masked patch token.
        Returns (scalar L1 loss on masked pixels, reconstructed image)."""
        target = x

        tokens = self.encoder.patch_embed(target)
        if tokens.ndim != 4:
            raise RuntimeError(
                "Expected timm Swin patch_embed to return BHWC tokens; "
                f"got shape {tuple(tokens.shape)}")

        expected_mask_shape = tokens.shape[1:3]
        if mask.shape[-2:] != expected_mask_shape:
            raise ValueError(
                "SimMIM Swin-B expects a patch-grid mask with shape "
                f"(B, {expected_mask_shape[0]}, {expected_mask_shape[1]}), "
                f"got {tuple(mask.shape)}")

        w = mask.unsqueeze(-1).type_as(tokens)
        tokens = tokens * (1.0 - w) + self.mask_token.type_as(tokens) * w

        feats = self.encoder.layers(tokens)
        feats = self.encoder.norm(feats)
        feats = feats.permute(0, 3, 1, 2).contiguous()

        pred = self.decoder(feats)
        loss = self._l1_loss(pred, target, mask)
        return loss, pred

    def _l1_loss(self, pred: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise RuntimeError(
                f"Decoder output shape {tuple(pred.shape)} does not match "
                f"target shape {tuple(target.shape)}")
        pixel_mask = mask.repeat_interleave(self.model_patch_size, 1) \
                         .repeat_interleave(self.model_patch_size, 2)
        pixel_mask = pixel_mask.unsqueeze(1).type_as(pred)
        loss_recon = F.l1_loss(pred, target, reduction="none")
        return (loss_recon * pixel_mask).sum() / (pixel_mask.sum() + 1e-5) \
            / self.in_chans

    def get_encoder(self) -> nn.Module:
        """The bare Swin encoder, for linear probing."""
        return self.encoder

    @torch.jit.ignore
    def no_weight_decay(self):
        skip = {"mask_token"}
        if hasattr(self.encoder, "no_weight_decay"):
            skip |= {f"encoder.{name}" for name in self.encoder.no_weight_decay()}
        return skip

    @torch.no_grad()
    def encode_global(self, x: torch.Tensor) -> torch.Tensor:
        """Global mean-pooled features for linear probing: (B, encoder_dim). No
        mask token -- just the clean image through the encoder."""
        feats = self.encoder.forward_features(x)   # (B, N, C) or (B, H, W, C)
        return feats.reshape(feats.size(0), -1, feats.size(-1)).mean(1)
