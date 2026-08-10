"""Memory-bank NCE loss for PIRL (Misra & van der Maaten, CVPR 2020).

Ported from the lab's own code. A cross-entropy NCE against a momentum-updated
memory bank with one row per training image (the instance-discrimination bank
shape). The DDP all_gather in the memory update is kept but inert single-process.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class PIRLMemoryBankNCE(nn.Module):
    """Cross-entropy NCE against a momentum-updated memory bank."""

    def __init__(self, num_samples: int, feature_dim: int = 128,
                 temperature: float = 0.07, momentum: float = 0.5,
                 num_negatives: int = 32000):
        super().__init__()
        self.num_samples = int(num_samples)
        self.feature_dim = int(feature_dim)
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.num_negatives = int(num_negatives)

        stdv = 1.0 / math.sqrt(feature_dim / 3.0)
        memory = torch.rand(num_samples, feature_dim).mul_(2 * stdv).add_(-stdv)
        self.register_buffer("memory", F.normalize(memory, dim=1))

    def forward(self, query: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(query).all():
            raise FloatingPointError("Non-finite query features before NCE")

        batch_size = query.size(0)
        device = query.device
        neg = torch.randint(low=0, high=self.num_samples,
                            size=(batch_size, self.num_negatives), device=device)
        bank_indices = torch.cat([indices.view(-1, 1), neg], dim=1)
        keys = self.memory[bank_indices.reshape(-1)].view(
            batch_size, 1 + self.num_negatives, self.feature_dim).detach()
        logits = torch.bmm(keys, query.unsqueeze(2)).squeeze(2) / self.temperature
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite NCE logits")
        targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite NCE loss")
        return loss

    @torch.no_grad()
    def update_memory(self, features: torch.Tensor, indices: torch.Tensor,
                      distributed: bool = False):
        if distributed and dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered_features = [torch.zeros_like(features)
                                 for _ in range(world_size)]
            gathered_indices = [torch.zeros_like(indices)
                                for _ in range(world_size)]
            dist.all_gather(gathered_features, features.detach())
            dist.all_gather(gathered_indices, indices)
            features = torch.cat(gathered_features, dim=0)
            indices = torch.cat(gathered_indices, dim=0)
        else:
            features = features.detach()

        if not torch.isfinite(features).all():
            raise FloatingPointError("Non-finite features during memory update")

        self.memory[indices] = F.normalize(
            self.momentum * self.memory[indices]
            + (1.0 - self.momentum) * features, dim=1)
