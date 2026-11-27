"""Native Step-1 feature protocol."""
import torch
from torch import nn
class NativeFeatures(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self, images, **kwargs):
        video = images.unsqueeze(2).repeat(1, 1, 16, 1, 1)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=images.is_cuda):
            tokens = self.backbone(video)
            features = torch.nn.functional.normalize(tokens.mean(dim=1).float(), dim=1)
        # The native evaluator caches normalized features in float16.
        return features.half().float().unsqueeze(1)
