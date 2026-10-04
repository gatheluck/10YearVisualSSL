"""Shared frozen pair features for captured flow and first-mask readers."""

import torch
from torch import nn
from torchvision.transforms.functional import normalize

from downstream.attention import SpatialAdapter
from downstream.captured_readers import (
    CROSS_SELF,
    SINGLE_BLOCK,
    SingleBlockSpatialAdapter,
)


class FrozenSpatialPair(nn.Module):
    def __init__(self, backbone, reader_profile):
        super().__init__()
        if reader_profile not in (None, SINGLE_BLOCK, CROSS_SELF):
            raise ValueError("unknown paired spatial reader profile")
        self.backbone = backbone.requires_grad_(False).eval()
        self.adapter = (
            (
                SingleBlockSpatialAdapter
                if reader_profile == SINGLE_BLOCK
                else SpatialAdapter
            )(backbone.out_channels)
            if reader_profile
            else None
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def features(self, images):
        with torch.no_grad():
            value = self.backbone.forward_features(
                normalize(images, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ).float()
        return self.adapter(value) if self.adapter is not None else value
