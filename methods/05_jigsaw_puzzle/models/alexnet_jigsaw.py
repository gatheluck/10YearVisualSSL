"""AlexNet-based Jigsaw Puzzle model (Noroozi & Favaro, ECCV 2016).

Ported from the lab's own implementation. The Context-Free Network (CFN) is an
AlexNet-like per-tile encoder whose FC layers are 1x1 convolutions, so it cannot
use a tile's absolute position; the same encoder is applied siamese-style to all
9 tiles, their features are concatenated, and an FC head predicts which of a fixed
permutation set was applied.

`encoder.pt` is the shared CFN encoder (`encoder.*`); the permutation classifier
is pretext machinery and is excluded. `get_encoder()` returns the encoder for the
linear probe.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CFNAlexNet(nn.Module):
    """Context-Free Network: AlexNet conv stack + 1x1-conv "FC" layers, pooled to
    a fixed 512-d feature per tile."""

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 1x1 convs instead of FC, so position is not used.
        self.cfn = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._initialize_weights()

    def _initialize_weights(self):
        # He (fan-in) init for the conv+CFN ReLU stack. std=0.01 everywhere
        # collapses the signal through 7 layers (activation std -> ~2e-6), so the
        # loss stays at chance forever; fan-in init keeps the forward variance
        # stable so the network can learn.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] tile -> feature [B, 512]."""
        x = self.features(x)
        x = self.cfn(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class JigsawPuzzleAlexNet(nn.Module):
    """The full pretext model: a shared CFN encoder over 9 tiles + a permutation
    classifier."""

    def __init__(self, num_classes: int = 100, dropout: float = 0.5):
        super().__init__()
        self.encoder = CFNAlexNet(dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512 * 9, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        self._initialize_classifier()

    def _initialize_classifier(self):
        # He init for the hidden FC (so gradients flow); small std for the output
        # FC (so initial predictions are near-uniform).
        layers = [m for m in self.classifier.modules() if isinstance(m, nn.Linear)]
        for i, m in enumerate(layers):
            if i == len(layers) - 1:
                nn.init.normal_(m.weight, mean=0, std=0.01)
            else:
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        """tiles: [B, 9, 3, H, W] -> logits [B, num_classes]."""
        b, t = tiles.size(0), tiles.size(1)
        flat = tiles.view(b * t, *tiles.shape[2:])
        feats = self.encoder(flat).view(b, t, -1)
        return self.classifier(feats.view(b, -1))

    def get_encoder(self) -> nn.Module:
        """The shared CFN encoder, for downstream probing."""
        return self.encoder


def build_alexnet_jigsaw_model(num_classes: int = 1000,
                               dropout: float = 0.5) -> JigsawPuzzleAlexNet:
    return JigsawPuzzleAlexNet(num_classes=num_classes, dropout=dropout)
