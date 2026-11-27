"""NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for SimCLR v2
(Chen et al., 2020), ported from the lab's own implementation.

For a batch of N images, two augmented views give 2N projections. Each view's
positive is the other view of the same image; the other 2N-2 views are negatives.
The loss is the temperature-scaled cross-entropy that pulls the positive pair
together and pushes the negatives apart.

The lab wrapper carries a DistributedDataParallel `all_gather_with_grad` branch
(so negatives span all ranks); it is kept but guarded by
``dist.is_initialized()``, so it is inert in this single-process port.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class _MultiplyGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, scale):
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """Gather global values while retaining autograd for this rank's slice.
    Inert (returns the tensor unchanged) when not running under DDP."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    rank = dist.get_rank()
    with torch.no_grad():
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor.contiguous())
    gathered[rank] = tensor          # reconnect gradient for the local portion
    return torch.cat(gathered, dim=0)


class NTXentLoss(nn.Module):
    """NT-Xent loss (SimCLR v2). Single-process here; the DDP gather is inert.

    Args:
        temperature: softmax temperature tau (the shipped config uses 0.1).
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """z1, z2: [N, D] L2-normalised projections of the two views.
        Returns the scalar NT-Xent loss."""
        z1 = all_gather_with_grad(F.normalize(z1, dim=1))
        z2 = all_gather_with_grad(F.normalize(z2, dim=1))
        N = z1.size(0)

        # Concatenate both views: [2N, D]
        z = torch.cat([z1, z2], dim=0)

        # Pairwise cosine similarity (unit vectors) scaled by tau: [2N, 2N]
        sim = torch.mm(z, z.t()) / self.temperature

        # Mask out self-similarity on the diagonal
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive labels: row i -> positive at i+N, row i+N -> positive at i
        labels = torch.cat([
            torch.arange(N, 2 * N, device=z.device),
            torch.arange(N,        device=z.device),
        ], dim=0)

        loss = F.cross_entropy(sim, labels)
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                # DDP averages parameter gradients; each rank owns one autograd
                # slice of this replicated global loss, so compensate the average.
                loss = _MultiplyGradient.apply(loss, world_size)
        return loss
