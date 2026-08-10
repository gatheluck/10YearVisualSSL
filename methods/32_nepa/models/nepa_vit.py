"""Next-Embedding Predictive Autoregression (NEPA) Vision Transformer.

Ported from the lab's own NEPA code (Xu et al., 2025; arXiv:2512.16922):
  - Standard ViT backbone with causal attention masking
  - Patch embedding layer f (Conv2d)
  - Autoregressive transformer predictor h
  - Stop-gradient on target embeddings (SimSiam style)
  - Negative cosine similarity loss over the shifted patch sequence
  - EMA model for evaluation (decay 0.9999)
  - 2D RoPE, LayerScale (init 1e-5), GeLU or optional SwiGLU, QK-Norm

NEPA ships its own ViT (no timm). `encoder.pt` is the EMA model. The port adds
``build_nepa_model`` (constructs from explicit dims, so a small hermetic CPU smoke
can run a tiny ViT) alongside the capture's build_nepa_vit_base / _large.
"""

from __future__ import annotations

import math
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2-D Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

def _build_rope_freqs(
    head_dim: int, grid_h: int, grid_w: int, base: float = 100.0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None, training: bool = False,
    shift: Optional[float] = None, jitter: Optional[float] = None,
    rescale: Optional[float] = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build NEPA-style 2D RoPE (cos, sin) of shape (T, head_dim). Coordinates are
    patch centres normalised to [-1, 1] and optionally jittered during training."""
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
    inv_freq = 1.0 / (base ** torch.arange(0, 1, 4 / head_dim,
                                           dtype=torch.float32, device=device))

    coords_h = torch.arange(0.5, grid_h, dtype=torch.float32, device=device) / grid_h
    coords_w = torch.arange(0.5, grid_w, dtype=torch.float32, device=device) / grid_w
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    coords = 2.0 * coords - 1.0

    if training:
        if shift is not None:
            shift_hw = torch.empty((1, 2), device=device,
                                   dtype=torch.float32).uniform_(-shift, shift)
            coords = coords + shift_hw
        if jitter is not None:
            jitter_range = math.log(jitter)
            jitter_hw = torch.empty((1, 2), device=device, dtype=torch.float32
                                    ).uniform_(-jitter_range, jitter_range).exp()
            coords = coords * jitter_hw
        if rescale is not None:
            rescale_range = math.log(rescale)
            rescale_hw = torch.empty(1, device=device, dtype=torch.float32
                                     ).uniform_(-rescale_range, rescale_range).exp()
            coords = coords * rescale_hw

    angles = 2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]
    angles = angles.flatten(1, 2).tile(2)
    return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D RoPE to patch-token queries/keys (prefix tokens are left alone)."""
    num_patches = cos.shape[-2]
    num_prefix_tokens = q.shape[-2] - num_patches
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    if num_prefix_tokens > 0:
        q_prefix, q_patches = q.split((num_prefix_tokens, num_patches), dim=-2)
        k_prefix, k_patches = k.split((num_prefix_tokens, num_patches), dim=-2)
        q_patches = q_patches * cos + _rotate_half(q_patches) * sin
        k_patches = k_patches * cos + _rotate_half(k_patches) * sin
        q = torch.cat((q_prefix, q_patches), dim=-2)
        k = torch.cat((k_prefix, k_patches), dim=-2)
    else:
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
    return q, k


# ---------------------------------------------------------------------------
# QK-Norm: per-head LayerNorm (no affine)
# ---------------------------------------------------------------------------

class QKNorm(nn.Module):
    """LayerNorm without learnable parameters applied per-head to Q and K."""

    def __init__(self, head_dim: int, eps: float = 1e-12):
        super().__init__()
        self.norm = nn.LayerNorm(head_dim, eps=eps, elementwise_affine=False,
                                 bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class SwiGLU(nn.Module):
    """SwiGLU activation (hidden_dim ~= 2/3 * 4 * embed_dim to match a 4x MLP)."""

    def __init__(self, in_dim: int, hidden_dim: int,
                 out_dim: Optional[int] = None):
        super().__init__()
        out_dim = out_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


# ---------------------------------------------------------------------------
# Causal Self-Attention with RoPE and QK-Norm
# ---------------------------------------------------------------------------

class NEPAAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, causal: bool = True,
                 use_qk_norm: bool = True, layer_norm_eps: float = 1e-12):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.q_norm = (QKNorm(self.head_dim, eps=layer_norm_eps)
                       if use_qk_norm else nn.Identity())
        self.k_norm = (QKNorm(self.head_dim, eps=layer_norm_eps)
                       if use_qk_norm else nn.Identity())

    def forward(self, x: torch.Tensor, rope_cos: Optional[torch.Tensor] = None,
                rope_sin: Optional[torch.Tensor] = None,
                causal_override: Optional[bool] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, heads, T, head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rope(q, k, rope_cos, rope_sin)

        use_causal = self.causal if causal_override is None else causal_override
        x = F.scaled_dot_product_attention(q, k, v, is_causal=use_causal)

        x = x.transpose(1, 2).reshape(B, T, C)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Transformer Block with LayerScale
# ---------------------------------------------------------------------------

class NEPABlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 causal: bool = True, layerscale_init: float = 1e-5,
                 use_qk_norm: bool = True, use_swiglu: bool = True,
                 layer_norm_eps: float = 1e-12):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.attn = NEPAAttention(embed_dim, num_heads, causal=causal,
                                  use_qk_norm=use_qk_norm,
                                  layer_norm_eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        if use_swiglu:
            swiglu_hidden = int(embed_dim * mlp_ratio * 2 / 3)
            swiglu_hidden = (swiglu_hidden + 63) // 64 * 64
            self.mlp = SwiGLU(embed_dim, swiglu_hidden)
        else:
            hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
            )

        self.ls1 = nn.Parameter(torch.ones(embed_dim) * layerscale_init)
        self.ls2 = nn.Parameter(torch.ones(embed_dim) * layerscale_init)

    def forward(self, x: torch.Tensor, rope_cos: Optional[torch.Tensor] = None,
                rope_sin: Optional[torch.Tensor] = None,
                causal_override: Optional[bool] = None) -> torch.Tensor:
        x = x + self.ls1 * self.attn(self.norm1(x), rope_cos, rope_sin,
                                     causal_override)
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Patch Embedding (embedding layer f)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 14,
                 in_chans: int = 3, embed_dim: int = 768, use_norm: bool = False,
                 layer_norm_eps: float = 1e-12):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)
        self.norm = (nn.LayerNorm(embed_dim, eps=layer_norm_eps)
                     if use_norm else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                   # (B, D, gh, gw)
        x = x.flatten(2).transpose(1, 2)   # (B, T, D)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Full NEPA Model
# ---------------------------------------------------------------------------

class NEPAModel(nn.Module):
    """NEPA: f = PatchEmbed, h = causal NEPABlocks; loss = negative cosine of
    z_hat[:, :-1] against a stop-grad shifted target z[:, 1:]. Eval uses the EMA."""

    def __init__(self, img_size: int = 224, patch_size: int = 14,
                 in_chans: int = 3, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0,
                 layerscale_init: float = 1e-5, use_qk_norm: bool = True,
                 use_swiglu: bool = True, use_rope: bool = True,
                 use_cls_token: bool = True, patch_embed_norm: bool = False,
                 layer_norm_eps: float = 1e-12, rope_theta: float = 100.0,
                 pos_embed_shift: Optional[float] = None,
                 pos_embed_jitter: Optional[float] = None,
                 pos_embed_rescale: Optional[float] = 2.0,
                 ema_decay: float = 0.9999):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_rope = use_rope
        self.use_cls_token = use_cls_token
        self.rope_theta = rope_theta
        self.pos_embed_shift = pos_embed_shift
        self.pos_embed_jitter = pos_embed_jitter
        self.pos_embed_rescale = pos_embed_rescale
        self.ema_decay = ema_decay

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim,
                                      use_norm=patch_embed_norm,
                                      layer_norm_eps=layer_norm_eps)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = (nn.Parameter(torch.empty(1, 1, embed_dim))
                          if use_cls_token else None)

        self.blocks = nn.ModuleList([
            NEPABlock(embed_dim=embed_dim, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, causal=True,
                      layerscale_init=layerscale_init, use_qk_norm=use_qk_norm,
                      use_swiglu=use_swiglu, layer_norm_eps=layer_norm_eps)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        self.ema_model: Optional["NEPAModel"] = None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def setup_ema(self):
        """Initialise the EMA replica (call once after model creation)."""
        self.ema_model = copy.deepcopy(self)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_model.eval()

    @torch.no_grad()
    def update_ema(self):
        if self.ema_model is None:
            return
        d = self.ema_decay
        for ema_p, p in zip(self.ema_model.parameters(), self.parameters()):
            ema_p.data.mul_(d).add_(p.data, alpha=1.0 - d)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Patch embeddings z = f(x): (B, T, D)."""
        z = self.patch_embed(x)
        if self.cls_token is not None:
            cls = self.cls_token.expand(z.shape[0], -1, -1)
            z = torch.cat((cls, z), dim=1)
        return z

    def predict(self, z: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """Run the autoregressive predictor h on z. Returns z_hat: (B, T, D)."""
        x = z
        cos = sin = None
        if self.use_rope:
            num_prefix = 1 if self.use_cls_token else 0
            num_patches = z.shape[1] - num_prefix
            grid_size = int(math.sqrt(num_patches))
            if grid_size * grid_size != num_patches:
                raise ValueError(
                    f"NEPA RoPE expects a square patch grid, got {num_patches} "
                    "patches")
            cos, sin = _build_rope_freqs(
                head_dim=self.embed_dim // self.blocks[0].attn.num_heads,
                grid_h=grid_size, grid_w=grid_size, base=self.rope_theta,
                dtype=z.dtype, device=z.device, training=self.training,
                shift=self.pos_embed_shift, jitter=self.pos_embed_jitter,
                rescale=self.pos_embed_rescale)

        for blk in self.blocks:
            x = blk(x, cos, sin, causal_override=causal)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward: embed then predict (pretraining path). Returns z_hat."""
        z = self.embed(x)
        return self.predict(z, causal=True)

    def nepa_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Negative cosine similarity of z_hat[:, :-1] vs stop-grad z[:, 1:]."""
        z = self.embed(x)
        z_hat = self.predict(z, causal=True)
        pred = z_hat[:, :-1, :]
        target = z[:, 1:, :].detach()          # stop-gradient on the target
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        return -(pred * target).sum(dim=-1).mean()

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, use_ema: bool = True,
                         pool: str = "avg", causal: bool = True) -> torch.Tensor:
        """Features for linear probing. pool: 'avg' (paper), 'last', or 'embed'."""
        model = (self.ema_model if (use_ema and self.ema_model is not None)
                 else self)
        model.eval()
        if pool == "embed":
            return model.embed(x).mean(dim=1)
        z = model.embed(x)
        z_hat = model.predict(z, causal=causal)
        if pool == "avg":
            return z_hat.mean(dim=1)
        elif pool == "last":
            return z_hat[:, -1, :]
        raise ValueError(f"Unknown pool: {pool}")


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

def build_nepa_model(embed_dim: int, depth: int, num_heads: int,
                     img_size: int = 224, patch_size: int = 14,
                     mlp_ratio: float = 4.0, use_swiglu: bool = False,
                     use_qk_norm: bool = True, use_rope: bool = True,
                     use_cls_token: bool = True, patch_embed_norm: bool = False,
                     layerscale_init: float = 1e-5,
                     layer_norm_eps: float = 1e-12, rope_theta: float = 100.0,
                     pos_embed_shift: Optional[float] = None,
                     pos_embed_jitter: Optional[float] = None,
                     pos_embed_rescale: Optional[float] = 2.0,
                     ema_decay: float = 0.9999) -> NEPAModel:
    """Construct a NEPAModel from explicit dims (the port's config-driven path)."""
    return NEPAModel(
        img_size=img_size, patch_size=patch_size, in_chans=3,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        mlp_ratio=mlp_ratio, layerscale_init=layerscale_init,
        use_qk_norm=use_qk_norm, use_swiglu=use_swiglu, use_rope=use_rope,
        use_cls_token=use_cls_token, patch_embed_norm=patch_embed_norm,
        layer_norm_eps=layer_norm_eps, rope_theta=rope_theta,
        pos_embed_shift=pos_embed_shift, pos_embed_jitter=pos_embed_jitter,
        pos_embed_rescale=pos_embed_rescale, ema_decay=ema_decay)


def build_nepa_vit_base(img_size: int = 224, patch_size: int = 14,
                        **kwargs) -> NEPAModel:
    """ViT-B (embed_dim=768, depth=12, heads=12) for NEPA."""
    return build_nepa_model(embed_dim=768, depth=12, num_heads=12,
                            img_size=img_size, patch_size=patch_size, **kwargs)


def build_nepa_vit_large(img_size: int = 224, patch_size: int = 14,
                         **kwargs) -> NEPAModel:
    """ViT-L (embed_dim=1024, depth=24, heads=16) for NEPA."""
    return build_nepa_model(embed_dim=1024, depth=24, num_heads=16,
                            img_size=img_size, patch_size=patch_size, **kwargs)
