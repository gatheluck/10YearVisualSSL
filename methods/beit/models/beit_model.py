"""BEiT pre-training model (Bao et al., 2021; arXiv:2106.08254).

Ported from the lab's own BEiT code: a ViT-Base/16 backbone with LayerScale
blocks and a MIM prediction head (embed_dim -> vocab_size) that predicts DALL-E
dVAE visual tokens at masked positions. `encoder.pt` is the backbone trunk
(patch_embed, cls_token, pos_embed, blocks, norm); the shared mask_token and the
MIM head are training machinery and are excluded. The port adds a generic
``build_beit`` (explicit dims) so a small hermetic CPU smoke can run a tiny ViT.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


class DropPath(nn.Module):
    """Stochastic depth per-sample (main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.floor(
            torch.rand(shape, dtype=x.dtype, device=x.device) + keep)
        return x / keep * random_tensor


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
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
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    """Transformer block with LayerScale (CaiT-style, used in BEiT)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, init_values: float = 0.1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), act_layer(), nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim), nn.Dropout(drop))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class BEiT(nn.Module):
    """BEiT pre-training model (ViT backbone + MIM head). forward(x, mask) returns
    the vocab logits at the masked patch positions."""

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, vocab_size: int = 8192, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0, drop_path_rate: float = 0.1,
                 init_values: float = 0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], init_values=init_values, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W); mask: (B, num_patches) bool (True = masked). Returns
        (num_masked_total, vocab_size)."""
        B = x.shape[0]
        x_emb = self.patch_embed(x)
        mask_expanded = mask.unsqueeze(-1).expand_as(x_emb)
        x_emb = torch.where(mask_expanded, self.mask_token.expand_as(x_emb), x_emb)
        cls = self.cls_token.expand(B, -1, -1)
        x_emb = torch.cat([cls, x_emb], dim=1)
        x_emb = x_emb + self.pos_embed
        for blk in self.blocks:
            x_emb = blk(x_emb)
        x_emb = self.norm(x_emb)
        patch_out = x_emb[:, 1:, :]
        logits_all = self.head(patch_out)
        return logits_all[mask]

    def get_encoder(self) -> "BEiTEncoder":
        return BEiTEncoder(self)


class BEiTEncoder(nn.Module):
    """Frozen BEiT encoder for linear probing: mean of patch tokens (CLS
    excluded)."""

    def __init__(self, beit: BEiT):
        super().__init__()
        self.patch_embed = beit.patch_embed
        self.cls_token = beit.cls_token
        self.pos_embed = beit.pos_embed
        self.blocks = beit.blocks
        self.norm = beit.norm
        self.embed_dim = beit.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_emb = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x_emb = torch.cat([cls, x_emb], dim=1)
        x_emb = x_emb + self.pos_embed
        for blk in self.blocks:
            x_emb = blk(x_emb)
        x_emb = self.norm(x_emb)
        return x_emb[:, 1:, :].mean(dim=1)


def build_beit(img_size: int = 224, patch_size: int = 16, vocab_size: int = 8192,
               embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
               mlp_ratio: float = 4.0, drop_path_rate: float = 0.1,
               init_values: float = 0.1) -> BEiT:
    """Construct a BEiT from explicit dims (the port's config-driven path)."""
    return BEiT(img_size=img_size, patch_size=patch_size, in_chans=3,
                vocab_size=vocab_size, embed_dim=embed_dim, depth=depth,
                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                drop_path_rate=drop_path_rate, init_values=init_values)


def build_beit_base(vocab_size: int = 8192, drop_path_rate: float = 0.1,
                    init_values: float = 0.1) -> BEiT:
    """BEiT-Base/16 as described in arXiv:2106.08254."""
    return build_beit(img_size=224, patch_size=16, vocab_size=vocab_size,
                      embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                      drop_path_rate=drop_path_rate, init_values=init_values)
