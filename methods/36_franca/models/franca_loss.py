"""Franca's Sinkhorn-Knopp DINO/iBOT losses (the method's contribution over
DINOv2's centering). Ported from the capture's `train_step2_vit.py`.

Instead of DINOv2's EMA centering, Franca balances the teacher assignments with a
Sinkhorn-Knopp normalisation, applied independently per nested (Matryoshka) level.
The `dist` calls are guarded, so the maths is identical on a single process.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_knopp_teacher(teacher_logits: torch.Tensor, teacher_temp: float,
                           n_iterations: int = 3,
                           distributed: bool = True) -> torch.Tensor:
    """Globally balanced assignments for one Matryoshka level."""
    if teacher_logits.ndim != 2:
        raise ValueError("teacher logits must have shape [samples, prototypes]")
    use_dist = distributed and dist.is_available() and dist.is_initialized()
    local_samples, n_prototypes = teacher_logits.shape
    total_samples = torch.tensor(float(local_samples),
                                 device=teacher_logits.device, dtype=torch.float32)
    if use_dist:
        dist.all_reduce(total_samples)
    if total_samples.item() <= 0:
        raise ValueError("Sinkhorn requires at least one teacher sample")
    if local_samples:
        global_max = teacher_logits.detach().float().max()
    else:
        global_max = torch.tensor(float("-inf"), device=teacher_logits.device)
    if use_dist:
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
    scaled = ((teacher_logits.detach().float() - global_max) / teacher_temp).clamp(min=-80.0)
    assignments = torch.exp(scaled).t()
    normalizer = assignments.sum()
    if use_dist:
        dist.all_reduce(normalizer)
    assignments /= normalizer.clamp_min(torch.finfo(assignments.dtype).tiny)
    for _ in range(n_iterations):
        row_sums = assignments.sum(dim=1, keepdim=True)
        if use_dist:
            dist.all_reduce(row_sums)
        assignments /= row_sums.clamp_min(torch.finfo(assignments.dtype).tiny)
        assignments /= n_prototypes
        assignments /= assignments.sum(dim=0, keepdim=True).clamp_min(
            torch.finfo(assignments.dtype).tiny)
        assignments /= total_samples
    assignments *= total_samples
    return assignments.t()


class FrancaDinoLoss(nn.Module):
    def __init__(self, out_dims: list, student_temp: float,
                 sinkhorn_iterations: int = 3) -> None:
        super().__init__()
        self.out_dims = tuple(out_dims)
        self.student_temp = student_temp
        self.sinkhorn_iterations = sinkhorn_iterations

    def forward(self, student_cls: list, teacher_cls: list,
                temp: float) -> torch.Tensor:
        total = torch.tensor(0.0, device=student_cls[0][0].device)
        terms_per_level = 0
        n_levels = len(student_cls[0])
        for level in range(n_levels):
            teacher_sizes = [outputs[level].shape[0] for outputs in teacher_cls]
            teacher_targets = sinkhorn_knopp_teacher(
                torch.cat([outputs[level] for outputs in teacher_cls], dim=0),
                teacher_temp=temp, n_iterations=self.sinkhorn_iterations,
            ).split(teacher_sizes)
            level_terms = 0
            for t_idx, target in enumerate(teacher_targets):
                for s_idx, s_tuple in enumerate(student_cls):
                    if s_idx == t_idx:
                        continue
                    total -= (target * F.log_softmax(
                        s_tuple[level] / self.student_temp, dim=-1)).sum(dim=-1).mean()
                    level_terms += 1
            if level == 0:
                terms_per_level = level_terms
            elif level_terms != terms_per_level:
                raise RuntimeError("DINO crop term count changed across nested levels")
        return total / max(terms_per_level, 1)


class FrancaIBOTLoss(nn.Module):
    def __init__(self, out_dims: list, student_temp: float,
                 sinkhorn_iterations: int = 3) -> None:
        super().__init__()
        self.out_dims = tuple(out_dims)
        self.student_temp = student_temp
        self.sinkhorn_iterations = sinkhorn_iterations

    def forward(self, student_patches: list, teacher_patches: list, masks: list,
                temp: float) -> torch.Tensor:
        total = torch.tensor(0.0, device=student_patches[0][0].device)
        n_levels = len(student_patches[0])
        for level in range(n_levels):
            teacher_sizes = [outputs[level].shape[0] for outputs in teacher_patches]
            teacher_targets = sinkhorn_knopp_teacher(
                torch.cat([outputs[level] for outputs in teacher_patches], dim=0),
                teacher_temp=temp, n_iterations=self.sinkhorn_iterations,
            ).split(teacher_sizes)
            for s_tuple, target, mask in zip(student_patches, teacher_targets, masks):
                mask = mask.to(s_tuple[level].device)
                s_log = F.log_softmax(s_tuple[level] / self.student_temp, dim=-1)
                patch_loss = -(target * s_log).sum(dim=-1)
                weights = (
                    (1.0 / mask.float().sum(-1).clamp(min=1.0)).unsqueeze(-1)
                    .expand_as(mask)[mask].to(patch_loss.device, dtype=patch_loss.dtype))
                total = total + (patch_loss * weights).sum() / max(mask.shape[0], 1)
        return total / max(len(student_patches), 1)
