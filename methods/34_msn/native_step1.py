"""Native Step-1 feature protocol."""
import torch
from torch import nn
class NativeFeatures(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self, images, **kwargs):
        features = self.backbone.forward_blocks(images, num_blocks=1)
        return features, None
