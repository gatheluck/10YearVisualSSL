"""Vision Transformer (ViT) for I-JEPA (Assran et al., 2023; arXiv:2301.08243).

Ported from the lab's own I-JEPA code (which follows facebookresearch/ijepa):

  - VisionTransformer: context / target encoder.
      * No CLS token -- linear eval uses the mean of the patch tokens.
      * forward(x)           -> [B, N, D]  (full image, all N patches)
      * forward(x, mask_ids) -> [B, n, D]  (only the selected patch indices)
  - VisionTransformerPredictor: a narrow ViT predictor that takes context tokens
    + positional queries at target positions and returns predicted target reps.

The capture ships vit_base / vit_large / vit_huge; the port adds vit_tiny so a
small hermetic CPU smoke can run, and builds the predictor with
``pred_num_heads = max(1, pred_dim // 64)`` so a narrow smoke predictor still has
at least one attention head. This is self-contained (no timm).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, hd]
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 drop: float = 0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0., attn_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden, dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                          # [B, D, gh, gw]
        x = x.flatten(2).transpose(1, 2)          # [B, N, D]
        return x


class VisionTransformer(nn.Module):
    """ViT encoder used for both the context encoder and the target encoder."""

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0.):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def no_weight_decay(self):
        return {"pos_embed"}

    def forward(self, x: torch.Tensor,
                mask_ids: "torch.Tensor | None" = None) -> torch.Tensor:
        """x: [B,C,H,W]; mask_ids: [B, n_keep] patch indices to keep (context
        encoder path) or None for all patches (target encoder path)."""
        tokens = self.patch_embed(x)                         # [B, N, D]
        tokens = tokens + self.pos_embed                     # add positional emb

        if mask_ids is not None:
            idx = mask_ids.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
            tokens = torch.gather(tokens, dim=1, index=idx)

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward -> mean-pooled feature for linear evaluation."""
        tokens = self.forward(x)            # [B, N, D]
        return tokens.mean(dim=1)           # [B, D]


class VisionTransformerPredictor(nn.Module):
    """I-JEPA predictor: a narrow ViT that takes context encoder tokens and
    predicts representations at target (masked) positions."""

    def __init__(self, num_patches: int, encoder_dim: int,
                 pred_dim: int = 384, pred_depth: int = 6,
                 pred_num_heads: int = 12, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True):
        super().__init__()
        self.num_patches = num_patches
        self.pred_dim = pred_dim
        self.encoder_dim = encoder_dim

        self.encoder_to_pred = nn.Linear(encoder_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, pred_dim), requires_grad=True)

        self.blocks = nn.ModuleList([
            Block(pred_dim, pred_num_heads, mlp_ratio, qkv_bias)
            for _ in range(pred_depth)])
        self.norm = nn.LayerNorm(pred_dim, eps=1e-6)
        self.pred_to_encoder = nn.Linear(pred_dim, encoder_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {"mask_token", "pos_embed"}

    def forward(self, context_tokens: torch.Tensor,
                context_ids: torch.Tensor,
                target_ids: torch.Tensor) -> torch.Tensor:
        """context_tokens: [B, n_ctx, encoder_dim]; context_ids: [B, n_ctx];
        target_ids: [B, n_tgt]. Returns [B, n_tgt, encoder_dim]."""
        B = context_tokens.shape[0]
        n_tgt = target_ids.shape[1]

        ctx = self.encoder_to_pred(context_tokens)          # [B, n_ctx, pred_dim]
        ctx_pos = self.pos_embed.expand(B, -1, -1)          # [B, N, pred_dim]
        ctx_idx = context_ids.unsqueeze(-1).expand(-1, -1, self.pred_dim)
        ctx = ctx + torch.gather(ctx_pos, 1, ctx_idx)

        mask_tokens = self.mask_token.expand(B, n_tgt, -1).clone()
        tgt_idx = target_ids.unsqueeze(-1).expand(-1, -1, self.pred_dim)
        mask_tokens = mask_tokens + torch.gather(ctx_pos, 1, tgt_idx)

        tokens = torch.cat([ctx, mask_tokens], dim=1)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        pred = tokens[:, ctx.shape[1]:, :]                  # [B, n_tgt, pred_dim]
        pred = self.pred_to_encoder(pred)                   # [B, n_tgt, encoder_dim]
        return pred


def vit_tiny(img_size: int = 224, patch_size: int = 16, **kwargs) -> VisionTransformer:
    """ViT-Tiny (port addition for the hermetic smoke): embed_dim=48, depth=2."""
    return VisionTransformer(img_size=img_size, patch_size=patch_size,
                             embed_dim=48, depth=2, num_heads=3, **kwargs)


def vit_base(img_size: int = 224, patch_size: int = 16, **kwargs) -> VisionTransformer:
    """ViT-B/16: embed_dim=768, depth=12, heads=12."""
    return VisionTransformer(img_size=img_size, patch_size=patch_size,
                             embed_dim=768, depth=12, num_heads=12, **kwargs)


def vit_large(img_size: int = 224, patch_size: int = 16, **kwargs) -> VisionTransformer:
    """ViT-L/16: embed_dim=1024, depth=24, heads=16."""
    return VisionTransformer(img_size=img_size, patch_size=patch_size,
                             embed_dim=1024, depth=24, num_heads=16, **kwargs)


def vit_huge(img_size: int = 224, patch_size: int = 14, **kwargs) -> VisionTransformer:
    """ViT-H/14: embed_dim=1280, depth=32, heads=16."""
    return VisionTransformer(img_size=img_size, patch_size=patch_size,
                             embed_dim=1280, depth=32, num_heads=16, **kwargs)


_ENCODER_REGISTRY = {
    "vit_tiny":  (vit_tiny,  16),
    "vit_base":  (vit_base,  16),
    "vit_large": (vit_large, 16),
    "vit_huge":  (vit_huge,  14),
}

# encoder_dim, default patch, default pred_dim
_DIM_MAP = {
    "vit_tiny":  (48,   16, 32),
    "vit_base":  (768,  16, 192),
    "vit_large": (1024, 16, 256),
    "vit_huge":  (1280, 14, 384),
}


def build_ijepa_encoder(model_name: str, img_size: int = 224,
                        patch_size: "int | None" = None) -> VisionTransformer:
    """Build a context/target encoder by name: vit_tiny/base/large/huge."""
    if model_name not in _ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from {list(_ENCODER_REGISTRY)}")
    fn, default_patch = _ENCODER_REGISTRY[model_name]
    ps = patch_size if patch_size is not None else default_patch
    return fn(img_size=img_size, patch_size=ps)


def build_ijepa_predictor(model_name: str, img_size: int = 224,
                          patch_size: "int | None" = None,
                          pred_dim: "int | None" = None,
                          pred_depth: int = 6) -> VisionTransformerPredictor:
    """Build the predictor matching the given encoder."""
    if model_name not in _DIM_MAP:
        raise ValueError(f"Unknown model: {model_name}")
    enc_dim, default_patch, default_pred_dim = _DIM_MAP[model_name]
    ps = patch_size if patch_size is not None else default_patch
    pd = pred_dim if pred_dim is not None else default_pred_dim
    grid = img_size // ps
    num_patches = grid * grid
    return VisionTransformerPredictor(
        num_patches=num_patches,
        encoder_dim=enc_dim,
        pred_dim=pd,
        pred_depth=pred_depth,
        pred_num_heads=max(1, pd // 64),  # keep head_dim ~= 64, at least 1 head
    )
