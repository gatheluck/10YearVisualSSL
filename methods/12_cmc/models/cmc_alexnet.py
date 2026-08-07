"""Visual CMC AlexNet model (Tian et al., 2019), ported from the lab's own
paper-faithful implementation.

An RGB image is converted to CIE Lab and split into its L (1-channel) and ab
(2-channel) views. Two half-size AlexNet branches -- ``encoder_l`` and
``encoder_ab``, channels halved vs. standard AlexNet -- map each view to an
L2-normalised ``feat_dim``-d embedding. ``forward(x, layer)`` returns the two
branches' features at any layer 1-8; ``layer 8`` is the L2-normalised embedding
used by the NCE loss, ``layer 6`` (fc6) is the layer the linear probe reads.

``encoder.pt`` is this two-branch encoder (``encoder_l.*`` / ``encoder_ab.*``);
the NCE memory banks live in the separate ``NCEAverage`` module and are excluded.
``get_encoder()`` returns a backbone that maps a Lab image to the layer-6
features of both branches, concatenated -- the representation the probe reads.

The lab's fc6 is ``Linear(128*6*6, 2048)``, hard-coding a 6x6 conv5 map (the
paper's 224px input). This port inserts an ``AdaptiveMaxPool2d((6,6))`` before
fc6: a no-op at 224px (conv5 is already 6x6) that lets a small hermetic CPU smoke
run at a smaller input. The paper's 224px geometry is still what the shipped
config asks for.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class Normalize(nn.Module):
    """L2-normalisation layer."""

    def __init__(self, power: int = 2):
        super().__init__()
        self.power = power

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        return x.div(norm.clamp(min=1e-8))


class AlexNetHalf(nn.Module):
    """Single-branch half-size AlexNet for one color-space view.

    Args:
        in_channels: 1 for the L channel, 2 for the ab channels.
        feat_dim:    output feature dimension (default 128).
    """

    def __init__(self, in_channels: int = 1, feat_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 11, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(48, 128, 5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 192, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(192, 192, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        # A no-op at the paper's 224px input (conv5 is already 6x6); lets a
        # smaller hermetic smoke reach fc6. See the module docstring.
        self.spatial_pool = nn.AdaptiveMaxPool2d((6, 6))
        self.fc6 = nn.Sequential(
            nn.Linear(128 * 6 * 6, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
        )
        self.fc8 = nn.Linear(2048, feat_dim)
        self.l2norm = Normalize(2)

    def forward(self, x: torch.Tensor, layer: int = 8) -> torch.Tensor:
        """Forward up to the given layer index (1-8)."""
        if layer <= 0:
            return x
        x = self.conv1(x)
        if layer == 1:
            return x
        x = self.conv2(x)
        if layer == 2:
            return x
        x = self.conv3(x)
        if layer == 3:
            return x
        x = self.conv4(x)
        if layer == 4:
            return x
        x = self.conv5(x)
        if layer == 5:
            return x
        x = self.spatial_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        if layer == 6:
            return x
        x = self.fc7(x)
        if layer == 7:
            return x
        x = self.fc8(x)
        x = self.l2norm(x)
        return x


class _CMCLayer6Feature(nn.Module):
    """Maps a Lab image to the layer-6 features of both branches, concatenated
    -- the representation the linear probe reads. Runs from the encoder weights
    ``encoder.pt`` carries."""

    def __init__(self, model: "AlexNetCMC", layer: int = 6):
        super().__init__()
        self.model = model
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_l, feat_ab = self.model(x, self.layer)
        feat_l = feat_l.flatten(1)
        feat_ab = feat_ab.flatten(1)
        return torch.cat([feat_l, feat_ab], dim=1)


class AlexNetCMC(nn.Module):
    """Two-branch AlexNet for Contrastive Multiview Coding.

    Input:  Lab image tensor (B, 3, H, W).
    Output: (feat_l, feat_ab) at the requested layer.
    """

    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.feat_dim = feat_dim
        self.encoder_l = AlexNetHalf(in_channels=1, feat_dim=feat_dim)
        self.encoder_ab = AlexNetHalf(in_channels=2, feat_dim=feat_dim)

    def forward(self, x: torch.Tensor, layer: int = 8
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.encoder_l(l, layer)
        feat_ab = self.encoder_ab(ab, layer)
        return feat_l, feat_ab

    def get_encoder(self, layer: int = 6) -> nn.Module:
        """A backbone mapping a Lab image to the concatenated layer-6 features of
        both branches, for downstream probing."""
        return _CMCLayer6Feature(self, layer)


def build_cmc_from_config(config: Dict) -> AlexNetCMC:
    model_cfg = config.get("model", {})
    return AlexNetCMC(feat_dim=int(model_cfg.get("feat_dim", 128)))
