"""Shared BASIC5_ATTENTIVE_v1 readers; task recipes remain separate.

Callers select the canonical final tokens and remove special tokens. This module
never invents a spatial grid or a temporal position policy. Query attention
without positional information is permutation invariant, including for video.
"""
from __future__ import annotations

import torch
from torch import nn


def validate_adaptation(cfg):
    """Keep experimental readers out of legacy/table-producing task recipes."""
    from downstream.spatial_backbones import requires_component_profile
    if (requires_component_profile(cfg.get("backbone", {}).get("kind"))
            and cfg.get("profile") != "capture_basic5_components"):
        raise ValueError("this backbone requires capture_basic5_components")
    adaptation = cfg.get("adaptation", "frozen")
    if adaptation not in ("frozen", "attentive", "finetune"):
        raise ValueError("config.adaptation: expected frozen, attentive or finetune")
    if adaptation != "frozen" and cfg.get("profile") != "capture_basic5_components":
        raise ValueError(f"{adaptation} adaptation requires capture_basic5_components")
    if adaptation == "finetune":
        from downstream.spatial_backbones import supports_trainable
        if not supports_trainable(cfg.get("backbone", {}).get("kind", "vit")):
            raise ValueError("finetune requires a verified trainable provider")
    return adaptation


def frozen_spatial_features(backbone, x, adapter=None):
    """Stop encoder gradients while preserving the adapter's autograd graph."""
    with torch.no_grad():
        spatial = backbone.forward_features(x)
    return adapter(spatial) if adapter is not None else spatial


def task_spatial_features(backbone, x, adapter=None, *, adaptation="frozen"):
    """The FT path preserves the caller's autograd context, including eval."""
    if adaptation == "finetune":
        return backbone.forward_features(x)
    return frozen_spatial_features(backbone, x, adapter)


def clip_attentive_gradients(model, adaptation):
    """Captured AP and FT clip all trainable parameters, including the encoder."""
    if adaptation in ("attentive", "finetune"):
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)


class _AttentionBlock(nn.Module):
    """One pre-LN attention/MLP block, eight heads, MLP ratio four, no dropout."""

    def __init__(self, width: int, *, cross_attention: bool = False):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width) if cross_attention else nn.Identity()
        self.attention = nn.MultiheadAttention(width, 8, dropout=0, batch_first=True)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))

    def forward(self, queries, context=None):
        q = self.query_norm(queries)
        kv = q if context is None else self.context_norm(context)
        x = queries + self.attention(q, kv, kv, need_weights=False)[0]
        return x + self.mlp(self.mlp_norm(x))


class QueryReader(nn.Module):
    """Project [B, N, C] tokens to 512, cross-attend 32 queries, mean-pool.

The classifier is deliberately external: tasks retain their baseline head.
Widths/query counts are fixed across backbones, not tuning parameters.
"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.input_projection = nn.Linear(in_channels, 512)
        self.queries = nn.Parameter(torch.randn(32, 512) / (512 ** .5))
        self.block = _AttentionBlock(512, cross_attention=True)

    def forward(self, tokens):
        if (tokens.ndim != 3 or tokens.shape[1] == 0
                or tokens.shape[2] != self.in_channels):
            raise ValueError("query reader requires nonempty tokens [B, N, C]")
        context = self.input_projection(tokens)
        queries = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        return self.block(queries, context).mean(dim=1)


class SpatialAdapter(nn.Module):
    """Residual C -> 256 -> attention/MLP -> C on a real spatial grid.

Reuse the same instance at each compatible scale. Different channel widths
require an explicit upstream mapping; they cannot silently get separate readers.
"""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.input_projection = nn.Conv2d(channels, 256, kernel_size=1)
        self.block = _AttentionBlock(256)
        self.output_projection = nn.Conv2d(256, channels, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, spatial):
        if (spatial.ndim != 4 or spatial.shape[1] != self.channels
                or min(spatial.shape[2:]) < 1):
            raise ValueError("adapter requires real nonempty spatial features [B, C, H, W]")
        projected = self.input_projection(spatial)
        tokens = self.block(projected.flatten(2).transpose(1, 2))
        residual = self.output_projection(tokens.transpose(1, 2).reshape_as(projected))
        return spatial + residual


class AttentiveSpatialBackbone(nn.Module):
    """Frozen spatial provider plus trainable shared adapter, usable by a head."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.out_channels = backbone.out_channels
        self.adapter = SpatialAdapter(self.out_channels)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward_features(self, x):
        return frozen_spatial_features(self.backbone, x, self.adapter)

    def forward(self, x):
        return self.forward_features(x)
