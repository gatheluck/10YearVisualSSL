"""ViT Step-2 split-brain model: two half-width ViT-B/16 cross-channel branches.

Faithful to the capture's `model.py` ViT path: the split-brain autoencoder keeps
its two cross-channel branches, but each branch's backbone is a from-scratch
half-width ViT-B/16 (timm `VisionTransformer`, `embed_dim=384`, `num_heads=6`)
instead of an AlexNet. ``net1`` reads the L channel (`in_chans=1`) and predicts
the quantised ab channels (313 bins); ``net2`` reads the ab channels
(`in_chans=2`) and predicts the quantised L channel (50 bins). Each branch is a
ViT trunk under ``self.encoder`` + a small conv decoder that upsamples the patch
grid (x4) to per-pixel class logits.

This port's convention: the two ViT trunks live under ``net1.encoder`` /
``net2.encoder``, so `encoder.pt` keeps exactly ``net1.encoder.*`` /
``net2.encoder.*`` -- the same prefixes the native AlexNet path uses -- and the
two decoders are pretext machinery, excluded. ``extract_features(l, ab)``
concatenates both branches' CLS embeddings (384 + 384 = 768-d), the
representation the linear probe reads. timm is imported lazily (only on
`arch: vit`); the ViT dimensions are configurable so a tiny model can run a CPU
smoke.
"""

from __future__ import annotations

import torch
import torch.nn as nn

AB_TARGET_CLASSES = 313
L_TARGET_CLASSES = 50


class SplitBrainViT(nn.Module):
    """One cross-channel branch: a ViT encoder + a conv decoder that predicts
    ``out_classes`` per pixel (patch grid upsampled x4)."""

    def __init__(self, in_channels: int, out_classes: int, img_size: int = 224,
                 patch_size: int = 16, embed_dim: int = 384, depth: int = 12,
                 num_heads: int = 6, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        from timm.models.vision_transformer import VisionTransformer
        self.encoder = VisionTransformer(
            img_size=img_size, patch_size=patch_size, in_chans=in_channels,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, num_classes=0, qkv_bias=True,
            norm_layer=nn.LayerNorm)
        self.embed_dim = embed_dim
        self.grid = img_size // patch_size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2,
                               padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_classes, kernel_size=1))

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        # timm VisionTransformer.forward_features returns [B, 1+N, D] (CLS + patches).
        return self.encoder.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._tokens(x)
        patch = tokens[:, 1:, :]                          # drop CLS
        B = x.shape[0]
        spatial = patch.transpose(1, 2).reshape(B, self.embed_dim,
                                                self.grid, self.grid)
        return self.decoder(spatial)                     # [B, out, grid*4, grid*4]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._tokens(x)[:, 0]                      # CLS, embed_dim-d


class SplitBrainViTModel(nn.Module):
    """The two half-width ViT cross-channel branches (L->ab and ab->L)."""

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 embed_dim: int = 384, depth: int = 12, num_heads: int = 6,
                 mlp_ratio: float = 4.0) -> None:
        super().__init__()
        kw = dict(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                  depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.net1 = SplitBrainViT(1, AB_TARGET_CLASSES, **kw)   # L  -> ab bins
        self.net2 = SplitBrainViT(2, L_TARGET_CLASSES, **kw)    # ab -> L bins

    def forward(self, l_input: torch.Tensor, ab_input: torch.Tensor
                ) -> "tuple[torch.Tensor, torch.Tensor]":
        return self.net1(l_input), self.net2(ab_input)

    def extract_features(self, l_input: torch.Tensor, ab_input: torch.Tensor
                         ) -> torch.Tensor:
        """The concatenated CLS embeddings of both branches (2*embed_dim), for
        downstream probing."""
        return torch.cat([self.net1.extract_features(l_input),
                          self.net2.extract_features(ab_input)], dim=1)


def build_split_brain_vit(img_size: int = 224, patch_size: int = 16,
                          embed_dim: int = 384, depth: int = 12,
                          num_heads: int = 6,
                          mlp_ratio: float = 4.0) -> SplitBrainViTModel:
    return SplitBrainViTModel(img_size=img_size, patch_size=patch_size,
                              embed_dim=embed_dim, depth=depth,
                              num_heads=num_heads, mlp_ratio=mlp_ratio)
