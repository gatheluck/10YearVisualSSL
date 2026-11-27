"""ResNet for SeLa (Asano et al., ICLR 2020; arXiv:1911.05371), ported from the
lab's own paper-faithful implementation.

A ResNet backbone (the official ResNetV2-50 by default, or a torchvision
ResNet-50) with `num_heads` linear **prototype heads** (`top_layer`). During
step 1 the heads predict Sinkhorn pseudo-labels; unlike DeepCluster the heads are
**not reset** each epoch and there is no Sobel front-end. The backbone produces a
2048-d avg-pooled feature.

`encoder.pt` is the backbone (`backbone.*`); the prototype heads (`top_layer.*`)
are training machinery and are excluded. `get_features()` returns the 2048-d
feature for the linear probe. torch/torchvision only; the capture's ViT variant
(`vit_sela.py`, which needs `timm`) is a separate step and is not ported.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class PreActBottleneck(nn.Module):
    """Official self-label ResNetV2 pre-activation bottleneck."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1, expansion=4):
        super().__init__()
        self.expansion = expansion
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1,
                               bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        return out + shortcut


class PreActResNetBackbone(nn.Module):
    """Feature trunk from the official `yukimasano/self-label` ResNetV2."""

    def __init__(self, block=PreActBottleneck, layers=(3, 4, 6, 3), expansion=4):
        super().__init__()
        self.in_planes = 16 * expansion
        self.features = nn.Sequential(
            nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(block, 16 * expansion, layers[0], stride=1,
                             expansion=4),
            self._make_layer(block, 2 * 16 * expansion, layers[1], stride=2,
                             expansion=4),
            self._make_layer(block, 4 * 16 * expansion, layers[2], stride=2,
                             expansion=4),
            self._make_layer(block, 8 * 16 * expansion, layers[3], stride=2,
                             expansion=4),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.feature_dim = 512 * expansion

    def _make_layer(self, block, planes, num_blocks, stride, expansion):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(block(self.in_planes, planes, block_stride, expansion))
            self.in_planes = planes * expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


class ResNetSeLa(nn.Module):
    """ResNet backbone + linear prototype heads for SeLa."""

    def __init__(self, num_classes: int = 3000, num_heads: int = 1,
                 arch: str = "resnetv2"):
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.arch = arch

        if arch == "resnetv2":
            self.backbone = PreActResNetBackbone()
            self.feature_dim = self.backbone.feature_dim
        elif arch in ("resnet50", "resnetv1"):
            resnet = tvm.resnet50(weights=None)
            self.backbone = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
                resnet.avgpool)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported SeLa architecture: {arch}")

        # Prototype heads (continuously trained, never reset).
        self.top_layer = nn.ModuleList([
            nn.Linear(self.feature_dim, num_classes) for _ in range(num_heads)])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        if x.dim() > 2:
            x = x.flatten(1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits for training: (B, K) for one head, else (B, H, K)."""
        feat = self._get_backbone_features(x)
        logits = [head(feat) for head in self.top_layer]
        if self.num_heads == 1:
            return logits[0]
        return torch.stack(logits, dim=1)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """The 2048-d backbone feature, no gradient -- for Sinkhorn and probing."""
        with torch.no_grad():
            return self._get_backbone_features(x)


def create_resnet_sela(config: dict) -> ResNetSeLa:
    clustering = config.get("clustering", {})
    model_cfg = config.get("model", {})
    return ResNetSeLa(
        num_classes=clustering["k"],
        num_heads=clustering.get("num_heads", 1),
        arch=model_cfg.get("arch", model_cfg.get("type", "resnetv2")))
