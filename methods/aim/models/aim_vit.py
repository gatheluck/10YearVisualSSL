"""
AIM: Autoregressive Image Models (El-Nouby et al., ICML 2024).

Architecture details (from Appendix D of arXiv:2401.08541):
  - ViT backbone with PREFIX-LM attention during pre-training
      * Prefix tokens  (0 .. S-1) : fully bidirectional among themselves
      * Suffix tokens  (S .. K-1) : causal (attend to prefix + prev suffix)
      * At evaluation              : fully bidirectional (no mask)
  - No [CLS] token
  - 2D sinusoidal positional embeddings (added to trunk input AND to head input)
  - No bias in linear layers
  - No LayerScale, no stochastic depth, no QK-Norm
  - MLP prediction head : 12 residual MLP blocks (each: LN → Linear → GELU → Linear)
  - Loss target: the input sequence shifted one patch left, per the official
    next-patch raster-order objective; per-patch normalised pixels, MSE loss
  - Training precision: bfloat16

Model configurations
  AIM-0.6B  ViT-H/14  embed=1280  depth=32  heads=16  (Step 1)
  AIM-Base  ViT-B/16  embed=768   depth=12  heads=12  (Step 2)
"""
from __future__ import annotations

import math
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: 2-D sinusoidal positional embedding (same as MAE / original ViT)
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """
    Returns: (grid_size**2, embed_dim) sinusoidal positional embeddings.
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2-D sincos PE"
    half = embed_dim // 2

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid   = np.meshgrid(grid_w, grid_h)   # NOTE: meshgrid(w, h) → shape (2, H, W)
    grid   = np.stack(grid, axis=0)        # (2, H, W)
    grid   = grid.reshape(2, -1)           # (2, N)   N = grid_size²

    def sincos_1d(pos: np.ndarray, d: int) -> np.ndarray:
        """1-D sinusoidal embedding of shape (len(pos), d)."""
        omega = np.arange(d // 2, dtype=np.float32) / (d // 2)
        omega = 1.0 / (10000 ** omega)
        out   = np.outer(pos, omega)          # (N, d/2)
        return np.concatenate([np.sin(out), np.cos(out)], axis=-1)  # (N, d)

    emb_h = sincos_1d(grid[1], half)   # (N, embed/2)
    emb_w = sincos_1d(grid[0], half)   # (N, embed/2)
    return np.concatenate([emb_h, emb_w], axis=-1)  # (N, embed)


# ---------------------------------------------------------------------------
# Utility: prefix-LM attention mask
# ---------------------------------------------------------------------------

def make_prefix_causal_mask(seq_len: int, prefix_len: int,
                             device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Prefix-LM additive attention bias (0 = attend, -inf = block).

    Pattern
    -------
    * All tokens attend to ALL prefix tokens   (columns 0 .. prefix_len-1  → 0)
    * Suffix tokens attend causally to suffix  (lower-triangular within suffix block → 0)
    * Prefix tokens do NOT attend to suffix    (upper-right prefix→suffix block → -inf)

    Shape: (seq_len, seq_len)
    """
    # Start from a causal mask (upper triangular = -inf, rest = 0)
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )
    # Override prefix-prefix block to be fully bidirectional
    if prefix_len > 0:
        mask[:prefix_len, :prefix_len] = 0.0
    return mask   # (L, L)


def next_patch_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    prefix_len: int,
) -> torch.Tensor:
    """AIM raster-order next-patch loss after an unrestricted prefix."""
    num_patches = pred.shape[1]
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")
    if not 1 <= prefix_len < num_patches:
        raise ValueError(
            f"prefix_len must be in [1, {num_patches - 1}], got {prefix_len}"
        )
    return F.mse_loss(
        pred[:, prefix_len - 1 : num_patches - 1].float(),
        target[:, prefix_len:num_patches].float(),
    )


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image → non-overlapping patch tokens, no CLS token."""

    def __init__(self, img_size: int = 224, patch_size: int = 14,
                 in_chans: int = 3, embed_dim: int = 1280):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size  = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Linear(in_chans * patch_size * patch_size, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)  →  (B, N, embed_dim)"""
        B, C, H, W = x.shape
        P = self.patch_size
        # Unfold into patches: (B, C, H/P, P, W/P, P) → (B, N, C*P*P)
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 1, 3, 5)   # (B, Hg, Wg, C, P, P)
        x = x.reshape(B, -1, C * P * P)   # (B, N, C*P*P)
        return self.proj(x)                # (B, N, embed_dim)


# ---------------------------------------------------------------------------
# Standard Transformer components (no bias throughout)
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    """Feed-forward block used inside transformer blocks."""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 act_layer=nn.GELU):
        super().__init__()
        hidden = hidden_features or in_features * 4
        self.fc1  = nn.Linear(in_features, hidden,     bias=False)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden,     in_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """Multi-head self-attention, optionally accepting an additive bias mask."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=False)
        self.proj      = nn.Linear(dim, dim,     bias=False)

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x          : (B, N, D)
        attn_mask  : (N, N) additive bias, 0 = attend, -inf = block  [optional]
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each: (B, H, N, Dh)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, N, N)
        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(0)   # broadcast over B, H
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        return self.proj(x)


class Block(nn.Module):
    """Transformer block: Pre-LN, Attention, Pre-LN, MLP, both with residual."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = Mlp(dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# AIM MLP prediction head
# ---------------------------------------------------------------------------

class MLPHeadBlock(nn.Module):
    """One residual MLP block used in the AIM prediction head."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, int(dim * mlp_ratio), bias=False)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(int(dim * mlp_ratio), dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class AIMPredictor(nn.Module):
    """
    Per-patch MLP prediction head.

    Projects backbone features (embed_dim → head_dim), applies num_blocks
    residual MLP blocks, then projects to output (patch_pixels).
    2-D sinusoidal PE is added at the head input (before blocks) following
    the paper.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 num_blocks: int = 12, head_dim: int = 2048,
                 mlp_ratio: float = 4.0, num_patches: int = 256):
        super().__init__()
        self.in_proj  = nn.Linear(in_dim,   head_dim, bias=False)
        self.blocks   = nn.ModuleList([
            MLPHeadBlock(head_dim, mlp_ratio=mlp_ratio)
            for _ in range(num_blocks)
        ])
        self.norm     = nn.LayerNorm(head_dim)
        self.out_proj = nn.Linear(head_dim, out_dim,  bias=False)

        # Positional embedding for the head (fixed sinusoidal, added to each patch)
        grid_size = int(num_patches ** 0.5)
        pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(head_dim, grid_size).astype(np.float32)
        ).unsqueeze(0)  # (1, N, head_dim)
        self.register_buffer("pos_embed", pos_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, in_dim)  →  (B, N, out_dim)"""
        x = self.in_proj(x) + self.pos_embed          # (B, N, head_dim)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.out_proj(x)                        # (B, N, out_dim)


# ---------------------------------------------------------------------------
# Full AIM ViT model
# ---------------------------------------------------------------------------

class AIMViT(nn.Module):
    """
    Autoregressive Image Model ViT backbone + prediction head.

    Pre-training
    ~~~~~~~~~~~~
    * forward(x, prefix_len)  → (loss, pred, target)
    * A random prefix_len is sampled each call if not supplied.

    Feature extraction (evaluation)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * forward_features(x, layer_ids)  →  list of (B, N, D) tensors
    * All tokens attend bidirectionally (no mask).
    """

    def __init__(
        self,
        img_size:    int   = 224,
        patch_size:  int   = 14,
        in_chans:    int   = 3,
        embed_dim:   int   = 1280,
        depth:       int   = 32,
        num_heads:   int   = 16,
        mlp_ratio:   float = 4.0,
        # Head config
        head_depth:  int   = 12,
        head_dim:    int   = 2048,
        head_mlp_ratio: float = 4.0,
        # Prefix fraction for training (fraction of patches used as prefix)
        prefix_fraction_range: Tuple[float, float] = (0.1, 0.5),
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.patch_size  = patch_size
        self.prefix_fraction_range = prefix_fraction_range

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.grid_size   = self.patch_embed.grid_size

        # Fixed sinusoidal positional embedding (trunk)
        pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(embed_dim, self.grid_size).astype(np.float32)
        ).unsqueeze(0)  # (1, N, embed_dim)
        self.register_buffer("pos_embed", pos_embed)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Prediction head (patch pixel regression)
        patch_pixels = in_chans * patch_size * patch_size
        self.predictor = AIMPredictor(
            in_dim=embed_dim, out_dim=patch_pixels,
            num_blocks=head_depth, head_dim=head_dim,
            mlp_ratio=head_mlp_ratio, num_patches=self.num_patches,
        )

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """(B, C, H, W) → (B, N, C*P*P)  (raster scan)"""
        B, C, H, W = x.shape
        P = patch_size
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * P * P)
        return x

    @staticmethod
    def normalize_patches(patches: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Per-patch pixel normalisation (He et al., MAE).
        patches: (B, N, C*P*P)
        """
        mean = patches.mean(dim=-1, keepdim=True)
        var  = patches.var(dim=-1, keepdim=True)
        return (patches - mean) / (var + eps).sqrt()

    # ------------------------------------------------------------------
    # Trunk forward (shared between pre-training and evaluation)
    # ------------------------------------------------------------------

    def _run_trunk(self, x: torch.Tensor,
                   attn_mask: Optional[torch.Tensor] = None
                   ) -> List[torch.Tensor]:
        """
        Returns list of intermediate features (one per block).
        x: (B, N, D)  (after patch embed + PE)
        """
        feats = []
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
            feats.append(x)
        return feats   # length = depth, each (B, N, D)

    # ------------------------------------------------------------------
    # Pre-training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        imgs:       torch.Tensor,
        prefix_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        imgs: (B, C, H, W)

        Returns
        -------
        loss   : scalar MSE loss over suffix patches
        pred   : (B, N, patch_pixels)  predictions for ALL patches
        target : (B, N, patch_pixels)  per-patch normalised pixels
        """
        B = imgs.shape[0]
        K = self.num_patches

        # Sample prefix length
        if prefix_len is None:
            lo = max(1, int(self.prefix_fraction_range[0] * K))
            hi = max(lo + 1, int(self.prefix_fraction_range[1] * K))
            prefix_len = torch.randint(lo, hi, (1,)).item()
        prefix_len = int(prefix_len)

        # Patch targets (per-patch normalised pixels, float32 for loss)
        target = self.normalize_patches(self.patchify(imgs.float(), self.patch_size))

        # Patch embedding + PE
        x = self.patch_embed(imgs) + self.pos_embed   # (B, K, D)

        # Prefix-LM attention mask
        attn_mask = make_prefix_causal_mask(K, prefix_len, device=x.device)

        # Transformer trunk → take final layer output
        all_feats = self._run_trunk(x, attn_mask=attn_mask)
        x = self.norm(all_feats[-1])    # (B, K, D)

        # MLP prediction head
        pred = self.predictor(x)        # (B, K, patch_pixels)

        # Official AIM predicts the next patch. Position i may attend to its own
        # input patch, so comparing pred[i] with target[i] leaks the answer.
        # The last prefix position predicts the first suffix patch.
        loss = next_patch_mse_loss(pred, target, prefix_len)

        return loss, pred, target

    # ------------------------------------------------------------------
    # Feature extraction for evaluation (no causal mask, multi-layer)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_features(
        self,
        imgs:      torch.Tensor,
        layer_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Extract patch features with fully bidirectional attention.

        layer_ids : which transformer layers to average (0-indexed).
                    Default: last 6 layers (paper practice for generative models).

        Returns: (B, N, D)  averaged multi-layer patch features.
        """
        K = self.num_patches
        if layer_ids is None:
            layer_ids = list(range(len(self.blocks) - 6, len(self.blocks)))

        x = self.patch_embed(imgs) + self.pos_embed   # (B, K, D)
        all_feats = self._run_trunk(x, attn_mask=None)  # bidirectional

        selected = torch.stack(
            [self.norm(all_feats[i]) for i in layer_ids], dim=0
        )   # (num_layers, B, K, D)
        return selected.mean(dim=0)   # (B, K, D)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def aim_base(img_size: int = 224, patch_size: int = 16, **kwargs) -> AIMViT:
    """AIM ViT-Base (Step 2 unified comparison, trained from scratch on IN-1k)."""
    return AIMViT(
        img_size=img_size, patch_size=patch_size,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
        head_depth=12, head_dim=2048, head_mlp_ratio=4.0,
        **kwargs,
    )


def aim_huge(img_size: int = 224, patch_size: int = 14, **kwargs) -> AIMViT:
    """AIM ViT-H/14 (≈ AIM-0.6B, Step 1 architecture for weight-loading verification)."""
    return AIMViT(
        img_size=img_size, patch_size=patch_size,
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4.0,
        head_depth=12, head_dim=2048, head_mlp_ratio=4.0,
        **kwargs,
    )


MODEL_REGISTRY = {
    "aim_base_patch16": aim_base,
    "aim_huge_patch14": aim_huge,
}
