"""Franca's nested (Matryoshka) projection head (Franca; the method's own
contribution over the DINOv2 backbone it shares).

Ported from the capture's `methods/36_franca/train_step2_vit.py` (the head is
defined inline there). Each nested level `d` in `nesting_dims` takes the first `d`
dims of the backbone feature, projects and MLPs them, and predicts a prototype
count scaled by `d / max(nesting_dims)` -- a coarse-to-fine set of assignments.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MatryoshkaHead(nn.Module):
    def __init__(self, in_dim: int, nesting_dims: list, out_dim: int,
                 hidden_dim: int, bottleneck_dim: int, nlayers: int) -> None:
        super().__init__()
        self.nesting_dims = list(nesting_dims)
        self.projections = nn.ModuleList(
            [nn.Linear(dim, dim) for dim in self.nesting_dims])
        self.mlps = nn.ModuleList(
            [self._build_mlp(dim, hidden_dim, bottleneck_dim, nlayers)
             for dim in self.nesting_dims])
        max_dim = self.nesting_dims[-1]
        self.last_layers = nn.ModuleList([
            nn.utils.weight_norm(
                nn.Linear(bottleneck_dim, int(out_dim * dim / max_dim), bias=False))
            for dim in self.nesting_dims])
        self.apply(self._init_weights)
        for layer in self.last_layers:
            nn.init.constant_(layer.weight_g, 1.0)

    @staticmethod
    def _build_mlp(in_dim: int, hidden_dim: int, bottleneck_dim: int,
                   nlayers: int) -> nn.Sequential:
        layers: list = []
        prev = in_dim
        for _ in range(max(nlayers - 1, 0)):
            layers.extend([nn.Linear(prev, hidden_dim), nn.GELU()])
            prev = hidden_dim
        layers.append(nn.Linear(prev, bottleneck_dim))
        layers.append(nn.GELU())
        return nn.Sequential(*layers)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple:
        outputs = []
        for dim, proj, mlp, last in zip(self.nesting_dims, self.projections,
                                        self.mlps, self.last_layers):
            h = proj(x[..., :dim])
            h = mlp(h)
            outputs.append(last(h))
        return tuple(outputs)

    def bottleneck_features(self, x: torch.Tensor) -> tuple:
        features = []
        for dim, proj, mlp in zip(self.nesting_dims, self.projections, self.mlps):
            h = proj(x[..., :dim])
            h = mlp(h)
            features.append(h)
        return tuple(features)
