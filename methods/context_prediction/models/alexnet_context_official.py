"""
Official-style AlexNet for Doersch et al. Context Prediction.

The legacy local model used grouped AlexNet convs, ImageNet-style BN in early
layers, and adaptive pooling to 6x6.  The released deepcontext Caffe model does
not: it keeps LRN after conv1/conv2, removes groups, and uses BatchNorm without
scale/shift only from conv3 onward and on fc6/fc7.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OfficialContextAlexNetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=5)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=1.0)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=1.0)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(384, affine=False)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(384, affine=False)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256, affine=False)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc6 = nn.Linear(256 * 2 * 2, 4096)
        self.bn6 = nn.BatchNorm1d(4096, affine=False)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def conv_stack(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x), inplace=True)
        x = self.norm1(self.pool1(x))
        x = F.relu(self.conv2(x), inplace=True)
        x = self.norm2(self.pool2(x))
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = F.relu(self.bn5(self.conv5(x)), inplace=True)
        return self.pool5(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stack(x)
        if x.shape[-2:] != (2, 2):
            x = F.adaptive_avg_pool2d(x, (2, 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.bn6(self.fc6(x)), inplace=True)
        return x


class OfficialContextPredictionAlexNet(nn.Module):
    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        self.encoder = OfficialContextAlexNetEncoder()
        self.fc7 = nn.Linear(4096 * 2, 4096)
        self.bn7 = nn.BatchNorm1d(4096, affine=False)
        self.fc8 = nn.Linear(4096, 4096)
        self.fc9 = nn.Linear(4096, num_classes)
        self._initialize_classifier()

    def _initialize_classifier(self) -> None:
        for module in (self.fc7, self.fc8, self.fc9):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, first_patch: torch.Tensor, second_patch: torch.Tensor) -> torch.Tensor:
        first = self.encoder(first_patch)
        second = self.encoder(second_patch)
        x = torch.cat([first, second], dim=1)
        x = F.relu(self.bn7(self.fc7(x)), inplace=True)
        x = F.relu(self.fc8(x), inplace=True)
        return self.fc9(x)

    def get_encoder(self) -> nn.Module:
        return self.encoder


def build_official_context_alexnet(num_classes: int = 8) -> OfficialContextPredictionAlexNet:
    return OfficialContextPredictionAlexNet(num_classes=num_classes)
