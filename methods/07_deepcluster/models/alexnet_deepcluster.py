"""AlexNet-BN with a Sobel front-end for DeepCluster (Caron et al., ECCV 2018),
ported from the lab's own implementation.

Differences from standard AlexNet (as in the original DeepCluster repo):
  - BatchNorm replaces LRN.
  - Input is a 2-channel Sobel-gradient image (fixed, no gradient), not raw RGB.
  - The classification `top_layer` is decoupled and reset each epoch (its output
    dimension is the number of clusters k).
  - The feature used for clustering / linear eval is fc7 (4096-d).

`encoder.pt` is the backbone (`features.*` + `classifier.*`); the `top_layer`
(the reset-each-epoch k-way head) and the fixed `sobel_layer` are excluded (the
Sobel filter is rebuilt deterministically on load).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SobelFilter(nn.Module):
    """Convert RGB to a 2-channel Sobel-gradient representation (fixed weights)."""

    def __init__(self):
        super().__init__()
        grayscale = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        grayscale.weight.data.fill_(1.0 / 3.0)
        self.grayscale = grayscale
        sobel = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        sobel.weight.data[0, 0] = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel.weight.data[1, 0] = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
        self.sobel = sobel
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sobel(self.grayscale(x))


class AlexNetDeepCluster(nn.Module):
    """AlexNet-BN for DeepCluster."""

    def __init__(self, num_classes: int = 10000, sobel: bool = True):
        super().__init__()
        self.sobel_layer = SobelFilter() if sobel else None
        in_channels = 2 if sobel else 3
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
        )
        self.top_layer = nn.Linear(4096, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _backbone(self, x: torch.Tensor,
                  before_final_relu: bool = False) -> torch.Tensor:
        if self.sobel_layer is not None:
            x = self.sobel_layer(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if before_final_relu:
            layers = list(self.classifier.children())
            if layers and isinstance(layers[-1], nn.ReLU):
                layers = layers[:-1]
            for layer in layers:
                x = layer(x)
        else:
            x = self.classifier(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._backbone(x)
        if self.top_layer is not None:
            x = self.top_layer(x)
        return x

    def get_features(self, x: torch.Tensor,
                     before_final_relu: bool = False) -> torch.Tensor:
        """fc7 features (used for clustering and linear eval)."""
        with torch.no_grad():
            return self._backbone(x, before_final_relu=before_final_relu)

    def reset_top_layer(self, num_classes: int, device: "torch.device",
                        seed: "int | None" = None):
        """Reinitialise the top_layer weights in place (same ``seed`` -> same
        weights)."""
        assert num_classes == self.top_layer.out_features, (
            f"num_classes {num_classes} != top_layer.out_features "
            f"{self.top_layer.out_features}")
        if seed is not None:
            torch.manual_seed(seed)
        nn.init.normal_(self.top_layer.weight.data, 0, 0.01)
        nn.init.constant_(self.top_layer.bias.data, 0)


def build_alexnet_deepcluster(sobel: bool = True,
                              num_classes: int = 10000) -> AlexNetDeepCluster:
    return AlexNetDeepCluster(num_classes=num_classes, sobel=sobel)
