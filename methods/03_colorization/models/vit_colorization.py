"""ViT Step-2 colorization model: a self-contained ViT-B/16 + a CNN decoder.

Faithful to the capture's `models/colorization_vit.py`: a from-scratch Vision
Transformer (hand-written patch embed / attention / MLP blocks -- **no timm**)
maps the grayscale L channel (`in_chans=1`) to patch tokens, and a lightweight
CNN decoder upsamples the patch grid back to a per-pixel `num_bins`-way ab
classification (the same 313-bin colorization pretext the native CNN path uses).

This port's convention: the ViT trunk lives under ``self.encoder`` (patch embed,
CLS token, position embedding, blocks, final norm), so `encoder.pt` keeps only
``encoder.*`` -- the same prefix the native CNN path uses -- and the decoder is
pretext machinery, excluded. ``get_encoder()`` returns a module whose forward is
the CLS feature (embed_dim), so `evaluate_linear_colorization.py` -- which calls
``model.get_encoder()`` -- works unchanged for either arch. The decoder is fixed
to the patch-16 x16 upsample (4 transpose-conv stages), so its output resolution
equals the crop size exactly when ``patch_size == 16`` (the unified ViT-B/16).
The ViT dimensions are configurable so a tiny model can run a CPU smoke.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Grayscale image (L channel) to patch embedding."""

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 1, embed_dim: int = 768) -> None:
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # [B, N, D]


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int,
                 drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0,
                 attn_drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _ViTTrunk(nn.Module):
    """The ViT-B/16 encoder: patch embed + CLS + pos-embed + blocks + norm.

    Lives under ``ColorizationViT.encoder`` so `encoder.pt` is exactly this.
    ``forward`` returns the normed token sequence [B, 1+N, D]; the model reads
    the patch tokens for the decoder and the CLS token for the probe.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 1, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0,
                 drop_rate: float = 0.1, attn_drop_rate: float = 0.1) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                  drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class _ClsFeature(nn.Module):
    """Wraps the trunk so its forward is the CLS feature (for the probe)."""

    def __init__(self, trunk: _ViTTrunk) -> None:
        super().__init__()
        self.encoder = trunk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)[:, 0]


class ColorizationViT(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 num_bins: int = 313, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0,
                 drop_rate: float = 0.1, attn_drop_rate: float = 0.1) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.encoder = _ViTTrunk(
            img_size=img_size, patch_size=patch_size, in_chans=1,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate)
        self.decoder = self._build_decoder(embed_dim, num_bins)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                    nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _build_decoder(embed_dim: int, num_bins: int) -> nn.Sequential:
        """14x14 (patch grid) -> 224x224 via four x2 transpose-conv stages, i.e.
        an x16 upsample; with patch_size=16 the output resolution equals the
        crop size."""
        return nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, num_bins, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(x)                    # [B, 1+N, D]
        patch = tokens[:, 1:, :]                    # drop CLS
        hp = wp = self.img_size // self.patch_size
        spatial = patch.transpose(1, 2).reshape(x.shape[0], self.embed_dim,
                                                hp, wp)
        return self.decoder(spatial)               # [B, num_bins, H, W]

    def get_encoder(self) -> nn.Module:
        """A frozen feature extractor: the CLS embedding (embed_dim)."""
        return _ClsFeature(self.encoder)


def build_vit_colorization(img_size: int = 224, patch_size: int = 16,
                           num_bins: int = 313, embed_dim: int = 768,
                           depth: int = 12, num_heads: int = 12,
                           mlp_ratio: float = 4.0, drop_rate: float = 0.1,
                           attn_drop_rate: float = 0.1) -> ColorizationViT:
    return ColorizationViT(
        img_size=img_size, patch_size=patch_size, num_bins=num_bins,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
