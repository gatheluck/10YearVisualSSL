"""Colorization CNN (Zhang, Isola & Efros, ECCV 2016), ported from the lab's own
implementation.

The L (lightness) channel of a Lab image is the input; a VGG-style CNN with
BatchNorm and dilated convolutions predicts the ab colour channels, quantised
into ``num_bins`` (313) in-gamut bins, as a per-pixel classification.

The lab's model names its layers flat (conv1_1 ... conv8_3, conv_out). This port
groups them into ``encoder`` (conv1-7, the trunk `get_features` probes),
``decoder`` (conv8, upsampling) and ``head`` (the 1x1 bin classifier), so
``encoder.pt`` -- the trunk -- is a clean ``encoder.*`` prefix; the layer shapes,
order and computation are unchanged. ``get_encoder()`` returns the trunk with a
global-average-pool, the 512-d representation the linear probe reads.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def _conv_bn_relu(in_ch, out_ch, stride=1, padding=1, dilation=1):
    return [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                      padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]


class _Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


def _build_encoder() -> nn.Sequential:
    layers = []
    # conv1 (downsample /2)
    layers += _conv_bn_relu(1, 64)
    layers += _conv_bn_relu(64, 64, stride=2)
    # conv2 (downsample /2)
    layers += _conv_bn_relu(64, 128)
    layers += _conv_bn_relu(128, 128, stride=2)
    # conv3 (downsample /2)
    layers += _conv_bn_relu(128, 256)
    layers += _conv_bn_relu(256, 256)
    layers += _conv_bn_relu(256, 256, stride=2)
    # conv4
    layers += _conv_bn_relu(256, 512)
    layers += _conv_bn_relu(512, 512)
    layers += _conv_bn_relu(512, 512)
    # conv5 (dilated)
    layers += _conv_bn_relu(512, 512, padding=2, dilation=2)
    layers += _conv_bn_relu(512, 512, padding=2, dilation=2)
    layers += _conv_bn_relu(512, 512, padding=2, dilation=2)
    # conv6 (dilated)
    layers += _conv_bn_relu(512, 512, padding=2, dilation=2)
    layers += _conv_bn_relu(512, 512, padding=2, dilation=2)
    layers += _conv_bn_relu(512, 512, padding=2, dilation=2)
    # conv7
    layers += _conv_bn_relu(512, 512)
    layers += _conv_bn_relu(512, 512)
    layers += _conv_bn_relu(512, 512)
    return nn.Sequential(*layers)


def _build_decoder() -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        *_conv_bn_relu(256, 256),
        *_conv_bn_relu(256, 256),
    )


class ColorizationCNN(nn.Module):
    """VGG-style colorization network: L -> per-pixel distribution over ab bins."""

    def __init__(self, num_bins: int = 313):
        super().__init__()
        self.num_bins = num_bins
        self.encoder = _build_encoder()          # -> (B, 512, H/8, W/8)
        self.decoder = _build_decoder()          # -> (B, 256, H/4, W/4)
        self.head = nn.Conv2d(256, num_bins, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear",
                                    align_corners=False)

    def forward(self, l: torch.Tensor) -> torch.Tensor:
        h = self.encoder(l)
        d = self.decoder(h)
        return self.upsample(self.head(d))       # (B, num_bins, H, W)

    def get_encoder(self) -> nn.Module:
        """The conv trunk with global-average-pool -> 512-d, for probing."""
        return nn.Sequential(self.encoder, nn.AdaptiveAvgPool2d((1, 1)),
                             _Flatten())


def build_colorization_cnn(config: Dict) -> ColorizationCNN:
    model_cfg = config.get("model", {})
    return ColorizationCNN(num_bins=int(model_cfg.get("num_bins", 313)))
