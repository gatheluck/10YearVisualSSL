"""
Vision Transformer (ViT) backbone for iBOT.

Supports ViT-S/16 and ViT-B/16 with:
  - Learnable [MASK] token for masked image modeling
  - Returns both [CLS] token and patch tokens

Based on:
  - An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)
  - iBOT: Image BERT Pre-Training with Online Tokenizer (Zhou et al., 2021)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class PatchEmbed(nn.Module):
    """Image to patch embedding."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # B, N, C


class DropPath(nn.Module):
    """Stochastic depth per sample."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features    = out_features    or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn  = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp   = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """
    ViT backbone for iBOT.

    Args:
        img_size    : Input image resolution (224 or 96 for local crops).
        patch_size  : Patch size (16).
        embed_dim   : Token embedding dimension.
        depth       : Number of transformer blocks.
        num_heads   : Number of attention heads.
        mlp_ratio   : MLP hidden-dim / embed_dim.
        use_mask_token: If True, create a learnable [MASK] token for MIM.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_mask_token=False,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop    = nn.Dropout(p=drop_rate)

        # Learnable [MASK] token — only allocated for student network
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_mask_token else None

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.mask_token is not None:
            nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias,   0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, h, w):
        """Interpolate positional embeddings to handle different resolutions."""
        npatch = x.shape[1] - 1  # exclude CLS
        N      = self.pos_embed.shape[1] - 1
        if npatch == N and h == w:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        sqrt_N = int(N ** 0.5)
        h0 = h // self.patch_size
        w0 = w // self.patch_size
        # Match DINO/iBOT positional interpolation. The +0.1 avoids floating
        # point rounding issues when using bicubic scale factors.
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, sqrt_N, sqrt_N, dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / sqrt_N, w0 / sqrt_N),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x, mask=None):
        """
        Args:
            x    : [B, C, H, W] input images.
            mask : [B, N] boolean mask; True = masked (student only).
                   If None, no masking is applied (teacher forward pass).

        Returns:
            cls_token  : [B, D] class token
            patch_tokens: [B, N, D] patch tokens (before norm or after)
        """
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # [B, N, D]

        # Apply [MASK] token at masked positions (student only)
        if mask is not None and self.mask_token is not None:
            # mask: [B, N], True = this patch is masked
            mask_tokens = self.mask_token.expand(B, x.shape[1], -1)  # [B, N, D]
            mask_expanded = mask.unsqueeze(-1).to(x.dtype)            # [B, N, 1]
            x = x * (1 - mask_expanded) + mask_tokens * mask_expanded

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]

        # Add positional embedding (with interpolation for local crops)
        x = x + self.interpolate_pos_encoding(x, H, W)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        cls_token    = x[:, 0]       # [B, D]
        patch_tokens = x[:, 1:]      # [B, N, D]
        return cls_token, patch_tokens

    def get_intermediate_layers(self, x, n=1, mask=None):
        """Return normalized token outputs from the last `n` transformer blocks."""
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        if mask is not None and self.mask_token is not None:
            mask_tokens = self.mask_token.expand(B, x.shape[1], -1)
            mask_expanded = mask.unsqueeze(-1).to(x.dtype)
            x = x * (1 - mask_expanded) + mask_tokens * mask_expanded

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.interpolate_pos_encoding(x, H, W)
        x = self.pos_drop(x)

        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


# ── Factory functions ────────────────────────────────────────────────────────

def vit_small(patch_size=16, **kwargs):
    """ViT-Small/16: embed_dim=384, depth=12, heads=6."""
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        **kwargs,
    )


def vit_base(patch_size=16, **kwargs):
    """ViT-Base/16: embed_dim=768, depth=12, heads=12."""
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        **kwargs,
    )


def vit_large(patch_size=16, **kwargs):
    """ViT-Large/16: embed_dim=1024, depth=24, heads=16."""
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        **kwargs,
    )


VIT_CONFIGS = {
    "vit_small": vit_small,
    "vit_base":  vit_base,
    "vit_large": vit_large,
}
