"""ViT Step-2 CPC model: a unified ViT-B/16 patch grid + a column-wise GRU
context + InfoNCE.

Faithful to the capture's `models/cpc_vit.py`: a timm `VisionTransformer` (from
scratch) reads the image; its patch tokens (the CLS token dropped) are reshaped
into the ViT's patch grid and used as the CPC z-grid. A column-wise GRU
autoregresses top-to-bottom to give the context c-grid, and `pred_steps` linear
predictors score an InfoNCE loss (`cpc_loss_fast`) that predicts future rows'
z-vectors from the context.

Restructured to this port's convention: the ViT trunk lives under
``self.encoder`` (num_classes=0, global_pool=""), so `encoder.pt` keeps only
``encoder.*`` -- the same prefix as the native visual-CPC-2018 patch encoder --
and ``get_encoder()`` returns a module whose forward is the ViT's CLS feature,
for the linear probe. timm is imported lazily (only on `arch: vit`), and the ViT
dimensions are configurable so a tiny model can run a CPU smoke.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ColumnGRU(nn.Module):
    """A GRU run down each column of the z-grid; the context at row r is the GRU
    state after rows 0..r-1 (a zero row is prepended, so row 0 has no context)."""

    def __init__(self, z_dim: int, c_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(z_dim, c_dim, batch_first=True)

    def forward(self, z_grid: torch.Tensor) -> torch.Tensor:
        B, rows, cols, z_dim = z_grid.shape
        z = z_grid.permute(0, 2, 1, 3).reshape(B * cols, rows, z_dim)
        h_out, _ = self.gru(z)
        zeros = torch.zeros(B * cols, 1, h_out.size(-1), device=z.device,
                            dtype=z.dtype)
        c = torch.cat([zeros, h_out[:, :-1, :]], dim=1)
        return c.reshape(B, cols, rows, -1).permute(0, 2, 1, 3)


class _ViTCLSEncoder(nn.Module):
    """Wraps the ViT so its forward is the CLS feature -- the linear-probe head."""

    def __init__(self, vit: nn.Module) -> None:
        super().__init__()
        self.vit = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit.forward_features(x)[:, 0, :]


class CPCViT(nn.Module):
    """CPC on a ViT patch grid: ViT tokens -> z-grid, column GRU -> context,
    linear predictors -> InfoNCE."""

    def __init__(self, z_dim: int = 768, c_dim: int = 768, pred_steps: int = 5,
                 image_size: int = 224, patch_size: int = 16,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0) -> None:
        super().__init__()
        if z_dim != embed_dim:
            raise ValueError(
                f"z_dim ({z_dim}) must equal embed_dim ({embed_dim}): the "
                "z-grid is the ViT's patch tokens, so its channel count is the "
                "ViT hidden dim")
        from timm.models.vision_transformer import VisionTransformer
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.pred_steps = pred_steps
        self.encoder = VisionTransformer(
            img_size=image_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            num_classes=0, global_pool="", drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, qkv_bias=True,
            norm_layer=nn.LayerNorm)
        self.grid_size = self.encoder.patch_embed.grid_size
        self.context = _ColumnGRU(z_dim, c_dim)
        self.predictors = nn.ModuleList(
            [nn.Linear(c_dim, z_dim, bias=False) for _ in range(pred_steps)])

    def encode_grid(self, x: torch.Tensor):
        B = x.size(0)
        rows, cols = self.grid_size
        tokens = self.encoder.forward_features(x)
        patch_tokens = tokens[:, 1:, :]
        z_grid = patch_tokens.reshape(B, rows, cols, self.z_dim)
        c_grid = self.context(z_grid)
        return z_grid, c_grid

    def forward(self, x: torch.Tensor):
        return self.encode_grid(x)

    def get_encoder(self) -> nn.Module:
        """The ViT trunk (CLS feature), for the linear probe."""
        return _ViTCLSEncoder(self.encoder)

    def cpc_loss_fast(self, z_grid: torch.Tensor, c_grid: torch.Tensor,
                      temperature: float = 0.07) -> torch.Tensor:
        B, rows, cols, z_dim = z_grid.shape
        device = z_grid.device
        losses = []
        for k in range(1, self.pred_steps + 1):
            max_anchor = rows - k
            if max_anchor <= 0:
                continue
            c_anchor = c_grid[:, :max_anchor, :, :]
            z_target = z_grid[:, k:k + max_anchor, :, :]
            Wc = self.predictors[k - 1](
                c_anchor.reshape(-1, self.c_dim)).reshape(
                    B, max_anchor, cols, z_dim)
            Wc = F.normalize(Wc, dim=-1)
            z_tgt_n = F.normalize(z_target, dim=-1)
            n = max_anchor * cols
            Wc_flat = Wc.reshape(B, n, z_dim)
            zt_flat = z_tgt_n.reshape(B, n, z_dim)
            full_scores = torch.bmm(
                Wc_flat,
                zt_flat.reshape(1, B * n, z_dim).expand(B, -1, -1).transpose(1, 2)
            ) / temperature
            labels = (torch.arange(n, device=device).unsqueeze(0).expand(B, -1)
                      + torch.arange(B, device=device).unsqueeze(1) * n)
            loss_k = F.cross_entropy(full_scores.reshape(B * n, B * n),
                                     labels.reshape(B * n))
            losses.append(loss_k)
        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=device)


def build_cpc_vit(z_dim: int = 768, c_dim: int = 768, pred_steps: int = 5,
                  img_size: int = 224, patch_size: int = 16,
                  embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                  mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                  attn_drop_rate: float = 0.0) -> CPCViT:
    return CPCViT(z_dim=z_dim, c_dim=c_dim, pred_steps=pred_steps,
                  image_size=img_size, patch_size=patch_size,
                  embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                  mlp_ratio=mlp_ratio, drop_rate=drop_rate,
                  attn_drop_rate=attn_drop_rate)
