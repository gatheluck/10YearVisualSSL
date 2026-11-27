"""
KoLeo Regularization Loss.

Reference: Sablayrolles et al., 2018 (arXiv:1806.03198)
           "Spreading Vectors for Similarity Search"
           Used in DINOv2 and DINOv3 as a diversity regularizer.

KoLeo encourages the representations within a mini-batch to spread uniformly
in the embedding space by maximizing the minimum pairwise distances:

    L_KoLeo = -(1/B) * sum_i log(min_{j≠i} ||z_i - z_j||_2)

where z_i are L2-normalized feature vectors.

In DINOv3, one loss is applied to each global crop over the local process
batch. The two crop losses are summed by the trainer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko regularization on one crop's CLS batch."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, D) L2-normalized or unnormalized CLS token features.

        Returns:
            loss: scalar KoLeo regularization loss.
        """
        # L2 normalize
        z = F.normalize(z.float(), dim=-1)  # (B, D)

        B = z.shape[0]
        if B < 2:
            return z.sum() * 0.0

        with torch.no_grad():
            similarities = z @ z.T
            similarities.fill_diagonal_(float("-inf"))
            nearest = similarities.argmax(dim=1)
        distances = torch.linalg.vector_norm(z - z[nearest], dim=-1)
        return -torch.log(distances.float() + self.eps).mean()
