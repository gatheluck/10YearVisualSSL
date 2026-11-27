"""DINOv3 projection heads.

The released implementation uses a three-layer MLP ending in a bottleneck,
normalizes the bottleneck feature, and applies a bias-free prototype layer.
There are no LayerNorms or learned logit scales in the head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        out_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
        use_bn: bool = False,
        mlp_bias: bool = True,
    ):
        super().__init__()
        self.mlp = _build_mlp(
            nlayers=max(1, nlayers),
            in_dim=in_dim,
            bottleneck_dim=bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.init_weights()

    def init_weights(self) -> None:
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        return self.last_layer(F.normalize(x, dim=-1, p=2, eps=eps))


class IBOTHead(DINOHead):
    """A separate DINO head applied to masked patch tokens."""


class TokenTypeAffine(nn.Module):
    """Identity-initialized calibration; raw backbone features stay unchanged."""

    def __init__(self, dimension: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dimension))
        self.beta = nn.Parameter(torch.zeros(dimension))

    def forward(self, tokens):
        return tokens * self.gamma + self.beta


def build_head_mlp(in_dim, hidden_dim, bottleneck_dim):
    mlp = _build_mlp(3, in_dim, bottleneck_dim, hidden_dim, False, True)
    mlp.apply(DINOHead._init_weights)
    return mlp


def build_prototypes(bottleneck_dim, out_dim):
    layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
    DINOHead._init_weights(layer)
    return layer


def project_tokens(mlp, prototypes, tokens):
    hidden = mlp(tokens)
    eps = 1e-6 if hidden.dtype == torch.float16 else 1e-12
    return prototypes(F.normalize(hidden, dim=-1, p=2, eps=eps))


def _build_mlp(
    nlayers: int,
    in_dim: int,
    bottleneck_dim: int,
    hidden_dim: int,
    use_bn: bool,
    bias: bool,
) -> nn.Sequential | nn.Linear:
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)

    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim, bias=bias)]
    if use_bn:
        layers.append(nn.BatchNorm1d(hidden_dim))
    layers.append(nn.GELU())
    for _ in range(nlayers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
    layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
    return nn.Sequential(*layers)
