"""Native Step-1 feature protocol."""
import torch
from torch import nn
class NativeFeatures(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self, images, **kwargs):
        left, right = self.backbone(images, 5)
        features = torch.cat((left, right), dim=1)
        return torch.nn.functional.adaptive_max_pool2d(features, (6, 6)).flatten(1)
