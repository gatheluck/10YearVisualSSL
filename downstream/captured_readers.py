"""Explicit reader variants observed in experimental sources.

These variants are not interchangeable protocol corrections. Callers must name
one; the selected profile is retained in results and the hashed configuration.
"""
from __future__ import annotations

import torch
from torch import nn
from downstream.attention import QueryReader, SpatialAdapter, _AttentionBlock

SINGLE_BLOCK = 'captured_single_block_v1'
CROSS_SELF = 'captured_cross_self_v1'


class CrossSelfQueryReader(nn.Module):
    """Normalized cross attention followed by self attention and an MLP."""
    def __init__(self, channels):
        super().__init__()
        self.input_projection = nn.Linear(channels, 512)
        self.input_norm = nn.LayerNorm(512)
        self.queries = nn.Parameter(torch.zeros(1, 32, 512))
        nn.init.trunc_normal_(self.queries, std=.02)
        self.query_norm = nn.LayerNorm(512)
        self.context_norm = nn.LayerNorm(512)
        self.cross = nn.MultiheadAttention(512, 8, dropout=0., batch_first=True)
        self.block = _AttentionBlock(512)
        self.block.apply(SpatialAdapter._init_dense_block)
        self.output_norm = nn.LayerNorm(512)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1] < 1 or tokens.shape[2] != self.input_projection.in_features:
            raise ValueError('reader requires nonempty [B, N, C] tokens')
        context = self.context_norm(self.input_norm(self.input_projection(tokens)))
        queries = self.query_norm(self.queries.expand(tokens.shape[0], -1, -1))
        queries = queries + self.cross(queries, context, context, need_weights=False)[0]
        return self.output_norm(self.block(queries)).mean(1)


class SingleBlockSpatialAdapter(nn.Module):
    """One residual block; preserve the reference initialization and residual count."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.input_projection = nn.Conv2d(channels, 256, 1)
        self.block = _AttentionBlock(256)
        self.output_projection = nn.Conv2d(256, channels, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != self.channels or min(features.shape[-2:]) < 1:
            raise ValueError('adapter requires real nonempty spatial features [B, C, H, W]')
        x = self.input_projection(features)
        tokens = self.block(x.flatten(2).transpose(1, 2))
        return features + self.output_projection(tokens.transpose(1, 2).reshape_as(x))


def _selected(backbone, profile):
    expected = getattr(backbone, 'reader_profile', None)
    if expected != profile:
        raise ValueError(f'reader_profile must explicitly match provider recipe: {expected}')
    if profile not in (None, SINGLE_BLOCK, CROSS_SELF):
        raise ValueError('unknown reader_profile')
    return profile


def query_reader(backbone, profile=None):
    selected = _selected(backbone, profile)
    cls = CrossSelfQueryReader if selected == CROSS_SELF else QueryReader
    return cls(backbone.out_channels)


def spatial_adapter(backbone, profile=None):
    selected = _selected(backbone, profile)
    cls = SingleBlockSpatialAdapter if selected == SINGLE_BLOCK else SpatialAdapter
    return cls(backbone.out_channels)
