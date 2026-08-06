"""AlexNet-BN rotation-prediction model (Gidaris et al., ICLR 2018).

Ported from the lab's own implementation, which follows the official RotNet
AlexNet-BN (gidariss/FeatureLearningRotNet `architectures/AlexNet.py`). The
pretext is a 4-class problem: predict which of {0, 90, 180, 270} degrees was
applied to the image.

`encoder.pt` is the AlexNet-BN feature extractor (`encoder.*`): the five
convolutional blocks and the two FC blocks, producing a 4096-d feature per
image. The 4-class rotation head (`classifier.*`) is pretext machinery and is
excluded. `get_encoder()` returns the encoder for the linear probe.

One thing is added in the port: an `AdaptiveAvgPool2d((6, 6))` before the FC
block so the encoder accepts any input size. At the paper's 224px input the
pool5 map is already 6x6, so the adaptive pool is an identity there; it only
matters for the smaller inputs a CPU smoke uses.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class AlexNetRotationEncoder(nn.Module):
    """The AlexNet-BN feature extractor: five conv blocks + two FC blocks,
    mapping an image to a 4096-d feature. The rotation head is not part of it."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # Added by the port: makes the encoder accept any input size; identity at
        # the paper's 224px input, where pool5 already produces a 6x6 map.
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc_block = nn.Sequential(
            Flatten(),
            nn.Linear(6 * 6 * 256, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = 4096

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] -> feature [B, 4096]."""
        x = self.features(x)
        x = self.pool(x)
        return self.fc_block(x)


class RotationAlexNet(nn.Module):
    """The full pretext model: the AlexNet-BN encoder + a 4-class rotation
    head."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.encoder = AlexNetRotationEncoder()
        self.classifier = nn.Linear(self.encoder.feature_dim, num_classes)
        self.num_classes = num_classes
        self._initialize_weights()

    def _initialize_weights(self):
        # The official RotNet AlexNet-BN init: fan-out He for convs, unit BN, and
        # small-std normal for the linear layers.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] -> rotation logits [B, num_classes]."""
        return self.classifier(self.encoder(x))

    def get_encoder(self) -> nn.Module:
        """The AlexNet-BN encoder, for downstream probing."""
        return self.encoder


def build_alexnet_rotation_model(num_classes: int = 4) -> RotationAlexNet:
    return RotationAlexNet(num_classes=num_classes)
