"""VGG16-based Jigsaw++ model for the pretext task (Noroozi et al., CVPR 2018).

Ported from the lab's own implementation. A shared VGG16 encoder processes each
of the 9 tiles siamese-style; conv4_3 features are adaptively max-pooled to
4x4x512 and passed through an FC layer to a 1024-d per-tile feature; the 9
features are concatenated and an FC head predicts which permutation was applied.

`encoder.pt` is the shared VGG16 encoder (`encoder.*`); the permutation
classifier is pretext machinery and is excluded. `get_encoder()` returns the
encoder for the linear probe.

The paper's knowledge-transfer stages (cluster the conv4 features with faiss-GPU
into pseudo-labels, then train an AlexNet on them) are a separate faiss-GPU
pipeline and are not part of this port; only the VGG16 pretext (stage a) is
brought across. So the capture's `get_conv4_features*` clustering helpers and the
torchvision-weight loader are dropped here.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VGG16Encoder(nn.Module):
    """VGG16 backbone as the tile encoder: blocks 1-4, adaptive max-pool to 4x4,
    and an FC layer, giving a 1024-d feature per tile/image."""

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        # Block 1: 2 conv, pool
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Block 2: 2 conv, pool
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Block 3: 3 conv, pool
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Block 4: 3 conv (no pool; the paper takes features at conv4)
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
        )
        # Adaptive max-pool to 4x4 (paper: "max-pooled to 4x4x512"); this also
        # lets the encoder accept any tile size, so a CPU smoke can use a small
        # tile.
        self.pool4 = nn.AdaptiveMaxPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * 4 * 512, 1024),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        # Fan-in He init for the conv+FC ReLU stack. A blanket std=0.01 keeps the
        # VGG Jigsaw++ pretext at chance for tens of epochs; fan-in init keeps the
        # forward variance stable so the network can learn.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] tile -> feature [B, 1024]."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool4(x)
        x = x.flatten(1)
        return self.fc(x)

    def get_conv4_features(self, x: torch.Tensor) -> torch.Tensor:
        """Raw conv4 features before the FC head, for the knowledge-transfer
        clustering: [B, 3, H, W] -> pool4 -> [B, 512, 4, 4] -> [B, 8192]."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool4(x)
        return x.flatten(1)


class VGG16JigsawPP(nn.Module):
    """The full pretext model: a shared VGG16 encoder over 9 tiles + a
    permutation classifier."""

    def __init__(self, num_classes: int = 701, dropout: float = 0.5):
        super().__init__()
        self.num_tiles = 9
        self.encoder = VGG16Encoder(dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(9 * 1024, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )
        self._init_classifier()

    def _init_classifier(self):
        # He init for the hidden FC (so gradients flow); very small std for the
        # output FC. With the Kaiming-initialized 4096-d hidden layer, a larger
        # output std produced huge initial logits on real ImageNet batches and
        # the optimizer collapsed to the uniform ln(num_classes) point.
        layers = [m for m in self.classifier.modules() if isinstance(m, nn.Linear)]
        for idx, m in enumerate(layers):
            if idx == len(layers) - 1:
                nn.init.normal_(m.weight, 0, 0.001)
            else:
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="relu")
            nn.init.constant_(m.bias, 0)

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        """tiles: [B, 9, 3, H, W] -> logits [B, num_classes]."""
        b, n = tiles.shape[:2]
        flat = tiles.view(b * n, *tiles.shape[2:])
        feats = self.encoder(flat).view(b, n * 1024)
        return self.classifier(feats)

    def get_encoder(self) -> nn.Module:
        """The shared VGG16 encoder, for downstream probing."""
        return self.encoder


def build_vgg16_jigsaw_pp_model(num_classes: int = 701,
                                dropout: float = 0.5) -> VGG16JigsawPP:
    return VGG16JigsawPP(num_classes=num_classes, dropout=dropout)
