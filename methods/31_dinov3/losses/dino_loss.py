"""DINO class-token loss with released DINOv3 Sinkhorn semantics."""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def sinkhorn_knopp(
    teacher_output: torch.Tensor,
    teacher_temp: float,
    n_iters: int = 3,
) -> torch.Tensor:
    """Balance assignments jointly over all local teacher samples and ranks."""
    if teacher_output.ndim != 2:
        raise ValueError("teacher_output must have shape [samples, prototypes]")
    if teacher_output.shape[0] == 0:
        return teacher_output.float()

    teacher_output = teacher_output.float()
    q = torch.exp(teacher_output / teacher_temp).t()
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    batch_size = q.shape[1] * world_size
    n_prototypes = q.shape[0]

    sum_q = q.sum()
    if world_size > 1:
        dist.all_reduce(sum_q)
    q /= sum_q

    for _ in range(n_iters):
        row_sum = q.sum(dim=1, keepdim=True)
        if world_size > 1:
            dist.all_reduce(row_sum)
        q /= row_sum
        q /= n_prototypes
        q /= q.sum(dim=0, keepdim=True)
        q /= batch_size

    q *= batch_size
    return q.t()


class DINOLoss(nn.Module):
    def __init__(
        self,
        n_crops_global: int = 2,
        n_crops_local: int = 8,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        sk_n_iters: int = 3,
    ):
        super().__init__()
        self.n_global = n_crops_global
        self.n_local = n_crops_local
        self.n_crops = n_crops_global + n_crops_local
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.sk_n_iters = sk_n_iters

    def teacher_targets(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Run one Sinkhorn problem over both global crops, as released."""
        if teacher_logits.shape[0] % self.n_global:
            raise ValueError("teacher batch is not divisible by n_crops_global")
        batch_size = teacher_logits.shape[0] // self.n_global
        targets = sinkhorn_knopp(
            teacher_logits.detach(),
            teacher_temp=self.teacher_temp,
            n_iters=self.sk_n_iters,
        )
        return targets.reshape(self.n_global, batch_size, -1)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        local_loss_weight: float = 1.0,
    ) -> torch.Tensor:
        if student_logits.shape[0] % self.n_crops:
            raise ValueError("student batch is not divisible by the number of crops")
        batch_size = student_logits.shape[0] // self.n_crops
        if teacher_logits.shape[0] != batch_size * self.n_global:
            raise ValueError("student and teacher crop batches do not match")

        teacher_probs = self.teacher_targets(teacher_logits)
        student_log_probs = F.log_softmax(
            student_logits.float().reshape(self.n_crops, batch_size, -1) / self.student_temp,
            dim=-1,
        )

        if not 0.0 <= local_loss_weight <= 1.0:
            raise ValueError("local_loss_weight must be in [0, 1]")

        loss = -torch.einsum("sbk,tbk->st", student_log_probs, teacher_probs)
        diagonal = min(self.n_crops, self.n_global)
        pair_weights = loss.new_ones(loss.shape)
        pair_weights.diagonal()[:diagonal] = 0.0
        pair_weights[self.n_global :] *= local_loss_weight
        n_terms = batch_size * (self.n_crops * self.n_global - diagonal)
        return (loss * pair_weights).sum() / n_terms
