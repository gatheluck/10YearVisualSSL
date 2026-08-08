"""AlexNet for the Jigsaw++ knowledge-transfer stage (Noroozi et al., CVPR 2018,
"Boosting Self-Supervised Learning via Knowledge Transfer"), ported from the
lab's own implementation.

Stage (d) of the pipeline: k-means over the VGG16 pretext's conv4 features gives
pseudo-labels, and a **standard AlexNet** is trained to classify them ("we use
the standard AlexNet and ImageNet classification settings to train the
pseudo-label classifier network"). This AlexNet is the knowledge-transfer output
the linear probe reads.

`encoder.pt` for this stage is the AlexNet conv trunk (`features.*`); the
classification head (`classifier.*`) is training machinery and is excluded.
`get_encoder()` returns features + avgpool -> 9216-d, the representation the probe
reads (`arch: alexnet_cluster_cls`).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class AlexNetClusterCls(nn.Module):
    """Standard AlexNet with a configurable pseudo-label head."""

    def __init__(self, num_classes: int = 2000, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride=4, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x.flatten(1))

    def get_encoder(self) -> nn.Module:
        """The conv trunk + avgpool -> 9216-d, for downstream probing."""
        return nn.Sequential(self.features, self.avgpool, _Flatten())


def build_alexnet_cluster_cls_model(num_classes: int = 2000,
                                    dropout: float = 0.5) -> AlexNetClusterCls:
    return AlexNetClusterCls(num_classes=num_classes, dropout=dropout)
