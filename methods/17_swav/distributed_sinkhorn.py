"""Distributed Sinkhorn and global collapse diagnostics for SwAV."""

import math

import torch
import torch.distributed as dist


def _resolve_world_size(world_size):
    actual = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if world_size is None:
        return actual
    if world_size != actual:
        raise ValueError(f"world_size={world_size} does not match process group size {actual}")
    return actual


@torch.no_grad()
def distributed_sinkhorn(out, niters=3, eps=0.05, world_size=None):
    """Balance local score columns against the global distributed batch.

    Columns belong to different samples on each rank, so only the global scalar
    total and K row sums are reduced. Local columns are normalized in place.
    """
    if out.ndim != 2:
        raise ValueError(f"expected [B_local, K] scores, got shape {tuple(out.shape)}")
    if niters <= 0 or eps <= 0:
        raise ValueError("niters and eps must be positive")

    world_size = _resolve_world_size(world_size)
    logits = out.detach().float() / eps
    global_max = logits.max()
    if world_size > 1:
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
    Q = torch.exp(logits - global_max).t().contiguous()
    K, B_local = Q.shape
    B_global = B_local * world_size

    total = Q.sum()
    if world_size > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    Q /= total.clamp_min(1e-12)

    for _ in range(niters):
        row_sums = Q.sum(dim=1, keepdim=True)
        if world_size > 1:
            dist.all_reduce(row_sums, op=dist.ReduceOp.SUM)
        Q /= row_sums.clamp_min(1e-12)
        Q /= K

        Q /= Q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        Q /= B_global

    Q *= B_global
    return Q.t()


@torch.no_grad()
def collapse_diagnostics(assignments, embeddings, world_size=None):
    """Return global assignment-usage and representation-collapse metrics."""
    world_size = _resolve_world_size(world_size)
    q = torch.cat([value.detach().float() for value in assignments], dim=0)
    if q.ndim != 2 or q.size(0) == 0:
        raise ValueError("assignments must contain non-empty [B, K] tensors")

    prototype_mass = q.sum(dim=0)
    sample_count = torch.tensor(float(q.size(0)), device=q.device)
    confidence_sum = q.max(dim=1).values.sum()
    if world_size > 1:
        dist.all_reduce(prototype_mass, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(confidence_sum, op=dist.ReduceOp.SUM)

    usage = prototype_mass / sample_count.clamp_min(1.0)
    entropy = -(usage * usage.clamp_min(1e-12).log()).sum()
    K = q.size(1)
    normalized_entropy = entropy / math.log(K) if K > 1 else entropy.new_tensor(1.0)
    active_fraction = (usage >= (0.1 / K)).float().mean()

    z = torch.cat([value.detach().float() for value in embeddings], dim=0)
    feature_sum = z.sum(dim=0)
    feature_sq_sum = z.square().sum(dim=0)
    feature_count = torch.tensor(float(z.size(0)), device=z.device)
    if world_size > 1:
        dist.all_reduce(feature_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(feature_sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(feature_count, op=dist.ReduceOp.SUM)
    feature_mean = feature_sum / feature_count.clamp_min(1.0)
    feature_var = feature_sq_sum / feature_count.clamp_min(1.0) - feature_mean.square()

    return {
        "assignment_entropy": normalized_entropy,
        "assignment_perplexity": entropy.exp(),
        "active_prototype_fraction": active_fraction,
        "max_prototype_mass": usage.max(),
        "assignment_confidence": confidence_sum / sample_count.clamp_min(1.0),
        "embedding_std": feature_var.clamp_min(0).sqrt().mean(),
    }
