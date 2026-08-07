"""NCE loss with a memory bank for Instance Discrimination (Wu et al., CVPR 2018).

Non-parametric noise-contrastive estimation over a memory bank that stores one
L2-normalised embedding per training instance (Section 3.3):

    h(i, v) = exp(v . m_i / tau) * N / (exp(v . m_i / tau) * N + m * Z)
    L = -log h(i, v_i) - sum_j log(1 - h(j, v_i))

with Z a running estimate of the partition function. The lab wrapper carries
DistributedDataParallel all-reduce/all-gather branches for multi-GPU training;
they are dropped here for the single-process port, so the memory bank and its
momentum update run on one device.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NCELoss(nn.Module):
    """NCE loss with a momentum memory bank.

    Args:
        num_samples:   total training images N.
        feature_dim:   embedding dimension (128).
        temperature:   tau (default 0.07).
        momentum:      memory-bank update momentum (default 0.5).
        num_negatives: noise samples m per positive (default 4096).
    """

    def __init__(self, num_samples: int, feature_dim: int = 128,
                 temperature: float = 0.07, momentum: float = 0.5,
                 num_negatives: int = 4096):
        super().__init__()
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.momentum = momentum
        self.num_negatives = num_negatives

        stdv = 1.0 / math.sqrt(feature_dim / 3.0)
        memory = torch.rand(num_samples, feature_dim).mul_(2 * stdv).add_(-stdv)
        memory = F.normalize(memory, dim=1)
        self.register_buffer("memory", memory)
        self.register_buffer("Z", torch.tensor(-1.0))

    def forward(self, features: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        """features: L2-normalised embeddings [B, d]; indices: [B]."""
        B, N, m = features.size(0), self.num_samples, self.num_negatives
        device = features.device

        # column 0 = positive (this instance), columns 1..m = random negatives.
        idx = torch.zeros(B, 1 + m, dtype=torch.long, device=device)
        idx[:, 0] = indices
        idx[:, 1:] = torch.randint(0, N, (B, m), device=device)

        weight = self.memory[idx.view(-1)].view(B, 1 + m,
                                                self.feature_dim).detach()
        out = torch.exp(
            torch.bmm(weight, features.unsqueeze(2)).squeeze(2)
            / self.temperature)  # [B, 1+m]

        with torch.no_grad():
            batch_Z = out.detach().float().mean().mul(N)
            if self.Z.item() < 0:
                self.Z.copy_(batch_Z)
            else:
                self.Z.mul_(0.5).add_(batch_Z, alpha=0.5)

        c = m * self.Z.item() / N
        loss_pos = -torch.log(out[:, 0] / (out[:, 0] + c) + 1e-7)
        loss_neg = -torch.log(c / (out[:, 1:] + c) + 1e-7).sum(dim=1)
        return (loss_pos + loss_neg).mean()

    @torch.no_grad()
    def update_memory(self, features: torch.Tensor,
                      indices: torch.Tensor) -> None:
        """Momentum-update the memory bank rows for the batch's instances."""
        feats = features.detach()
        unique_idx, inverse = torch.unique(indices, sorted=True,
                                           return_inverse=True)
        feature_sum = torch.zeros(unique_idx.numel(), self.feature_dim,
                                  device=feats.device, dtype=feats.dtype)
        feature_sum.index_add_(0, inverse, feats)
        counts = torch.zeros(unique_idx.numel(), device=feats.device,
                             dtype=feats.dtype)
        counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=feats.dtype))
        mean_features = feature_sum / counts.unsqueeze(1)

        updated = F.normalize(
            self.momentum * self.memory[unique_idx]
            + (1 - self.momentum) * mean_features, dim=1)
        self.memory[unique_idx] = updated
