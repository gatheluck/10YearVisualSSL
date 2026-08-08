"""Split-Brain Autoencoder model (Zhang, Isola & Efros, CVPR 2017), ported from
the lab's own implementation.

Two cross-channel AlexNet sub-networks: ``net1`` maps the L channel (1-ch) to the
quantised ab channels (313 bins), ``net2`` maps the ab channels (2-ch) to the
quantised L channel (50 bins) -- each an encoder + a small deconv decoder that
outputs per-pixel class logits.

``encoder.pt`` is the two sub-network encoders (``net1.encoder.*`` /
``net2.encoder.*``); the decoders are pretext machinery and are excluded.
``extract_features(l, ab)`` concatenates both encoders' spatially-averaged
features (256 + 256 = 512-d) -- the representation the linear probe reads.

The lab's model also carries a ViT step-2 branch (timm); it is excluded, as in
every port, so this module has no timm dependency.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

AB_TARGET_CLASSES = 313
L_TARGET_CLASSES = 50


class SplitBrainAlexNet(nn.Module):
    """One cross-channel branch: an AlexNet encoder + a deconv decoder that
    predicts ``out_classes`` per pixel."""

    def __init__(self, in_channels: int, out_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.encoder(x), dim=[2, 3])


class SplitBrainModel(nn.Module):
    """The two cross-channel branches (L->ab and ab->L)."""

    def __init__(self):
        super().__init__()
        self.net1 = SplitBrainAlexNet(1, AB_TARGET_CLASSES)   # L -> ab bins
        self.net2 = SplitBrainAlexNet(2, L_TARGET_CLASSES)    # ab -> L bins

    def forward(self, l_input: torch.Tensor, ab_input: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.net1(l_input), self.net2(ab_input)

    def extract_features(self, l_input: torch.Tensor, ab_input: torch.Tensor
                         ) -> torch.Tensor:
        """The concatenated, spatially-averaged features of both encoders
        (512-d), for downstream probing."""
        return torch.cat([self.net1.extract_features(l_input),
                          self.net2.extract_features(ab_input)], dim=1)


def build_split_brain_from_config(config: Dict) -> SplitBrainModel:
    # The AlexNet split-brain has no size/width knobs; the ab (313) and L (50)
    # bin counts are fixed constants of the protocol.
    return SplitBrainModel()
