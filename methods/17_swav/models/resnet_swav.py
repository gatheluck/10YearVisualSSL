"""
ResNet-50 encoder + projection head + prototypes for SwAV Step 1.
Strictly follows Caron et al. (2020) NeurIPS:
  f(·): ResNet-50 backbone  ->  2048-dim average-pooled features
  g(·): Linear(2048,2048) -> BN -> ReLU -> Linear(2048, out_dim) -> L2-normalise
  C   : Prototype weight matrix [K x out_dim], column-normalised (no bias)

Multi-crop: 2×224 global + 6×96 local crops (handled in data module).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class _Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ResNetSwAV(nn.Module):
    """ResNet-50 backbone with SwAV projection head and learnable prototypes."""

    def __init__(self, out_dim: int = 128, hidden_mlp: int = 2048,
                 nmb_prototypes: int = 3000):
        super().__init__()
        resnet = tv_models.resnet50(weights=None)
        # Strip the classification FC; keep everything through avgpool
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # -> [B,2048,1,1]
        self.projection_head = nn.Sequential(
            nn.Linear(2048, hidden_mlp),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_mlp, out_dim),
        )
        # Prototype layer: no bias, weights are L2-normalised each step
        self.prototypes = nn.Linear(out_dim, nmb_prototypes, bias=False)

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features [B, 2048]."""
        return self.encoder(x).view(x.size(0), -1)

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised projected embeddings [B, out_dim]."""
        z = self.projection_head(h)
        return F.normalize(z, dim=1)

    @torch.no_grad()
    def normalize_prototypes(self):
        """L2-normalize prototype rows, matching the official SwAV update."""
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def _forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_encoder(x)
        return h

    def forward(self, x):
        """Return concatenated embeddings and prototype scores for one/many crops."""
        if not isinstance(x, list):
            x = [x]

        crop_sizes = torch.tensor([crop.shape[-1] for crop in x])
        idx_crops = torch.cumsum(torch.unique_consecutive(crop_sizes, return_counts=True)[1], 0)
        start_idx = 0
        output = []
        for end_idx in idx_crops:
            end_idx = int(end_idx)
            h = self._forward_tensor(torch.cat(x[start_idx:end_idx]))
            output.append(h)
            start_idx = end_idx

        z = self.forward_head(torch.cat(output, dim=0))
        p = self.prototypes(z)
        return z, p

    def get_encoder(self) -> nn.Module:
        """Backbone + flatten for linear evaluation (no projection)."""
        return nn.Sequential(self.encoder, _Flatten())


def build_resnet_swav(out_dim: int = 128, hidden_mlp: int = 2048,
                      nmb_prototypes: int = 3000) -> ResNetSwAV:
    return ResNetSwAV(out_dim=out_dim, hidden_mlp=hidden_mlp,
                      nmb_prototypes=nmb_prototypes)
