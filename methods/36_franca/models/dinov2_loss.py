"""DINOv2 loss functions (Oquab et al., 2023; arXiv:2304.07193).

Ported from the capture's DINOv2 loss module (the DINOv2-port sibling):
  1. DINOLoss  -- cross-view knowledge distillation with EMA centering.
  2. iBOTLoss  -- masked-patch cross-entropy (Zhou et al., 2021).
  3. KoLeoLoss -- nearest-neighbour spread regularisation.
The `dist` all-gather / all-reduce helpers are guarded, so they are inert on a
single process; the maths is identical to the multi-GPU run.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


@torch.no_grad()
def _all_gather(tensor: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    world = dist.get_world_size()
    gathered = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


@torch.no_grad()
def _all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


class DINOLoss(nn.Module):
    """DINO knowledge-distillation loss with EMA centering."""

    def __init__(self, out_dim: int, student_temp: float = 0.1,
                 teacher_temp: float = 0.04, center_momentum: float = 0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_cls: list, teacher_cls: list) -> torch.Tensor:
        teacher_out = [
            F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
            for t in teacher_cls
        ]
        total_loss = torch.tensor(0.0, device=student_cls[0].device)
        n_pairs = 0
        for t_idx, t_soft in enumerate(teacher_out):
            for s_idx, s in enumerate(student_cls):
                if s_idx == t_idx:
                    continue
                loss = -(t_soft * F.log_softmax(s / self.student_temp, dim=-1)).sum(dim=-1).mean()
                total_loss = total_loss + loss
                n_pairs += 1
        self._update_center(teacher_cls)
        return total_loss / max(n_pairs, 1)

    @torch.no_grad()
    def _update_center(self, teacher_cls: list):
        batch_center = torch.cat(teacher_cls, dim=0).mean(dim=0, keepdim=True)
        batch_center = _all_reduce_mean(batch_center)
        self.center = self.center * self.center_momentum + batch_center * (1.0 - self.center_momentum)


class iBOTLoss(nn.Module):
    """iBOT masked image modelling loss on patch tokens."""

    def __init__(self, out_dim: int, student_temp: float = 0.1,
                 teacher_temp: float = 0.04, center_momentum: float = 0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, out_dim))

    def forward(self, student_patches: list, teacher_patches: list,
                masks: list) -> torch.Tensor:
        total_loss = student_patches[0].new_zeros(())
        total_samples = 0
        masked_teacher_logits = []
        for s_patch, t_patch, mask in zip(student_patches, teacher_patches, masks):
            mask = mask.to(device=s_patch.device, dtype=torch.bool)
            num_masked = int(mask.sum().item())
            if s_patch.ndim == 3:
                s_masked = s_patch[mask]
                t_masked = t_patch[mask]
            elif s_patch.ndim == 2:
                if s_patch.shape[0] != num_masked or t_patch.shape[0] != num_masked:
                    raise ValueError("masked iBOT logits do not match the supplied mask")
                s_masked = s_patch
                t_masked = t_patch
            else:
                raise ValueError("iBOT patch logits must be rank 2 or rank 3")

            if num_masked:
                center = self.center.reshape(1, -1)
                t_soft = F.softmax((t_masked - center) / self.teacher_temp, dim=-1).detach()
                s_log = F.log_softmax(s_masked / self.student_temp, dim=-1)
                patch_loss = -(t_soft * s_log).sum(dim=-1)
                inverse_mask_count = (
                    mask.sum(dim=-1).clamp(min=1).float().reciprocal().unsqueeze(-1)
                    .expand_as(mask)[mask]
                )
                total_loss = total_loss + (patch_loss * inverse_mask_count).sum()
                masked_teacher_logits.append(t_masked.detach())
            total_samples += mask.shape[0]
        self._update_center(masked_teacher_logits)
        return total_loss / max(total_samples, 1)

    @torch.no_grad()
    def _update_center(self, masked_teacher_logits: list):
        patch_sum = torch.zeros_like(self.center.reshape(-1))
        patch_count = torch.zeros((), device=self.center.device, dtype=torch.long)
        for logits in masked_teacher_logits:
            patch_sum.add_(logits.sum(dim=0).to(patch_sum.dtype))
            patch_count.add_(logits.shape[0])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(patch_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(patch_count, op=dist.ReduceOp.SUM)
        if patch_count.item() == 0:
            return
        batch_center = (patch_sum / patch_count).view_as(self.center)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1.0 - self.center_momentum)


class KoLeoLoss(nn.Module):
    """KoLeo regularisation (Sablayrolles et al., 2018): push each sample away
    from its nearest neighbour in the batch."""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def forward(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = F.normalize(x, eps=eps, p=2, dim=-1)
            dots = x @ x.T
            n = x.shape[0]
            dots.view(-1)[:: n + 1].fill_(-1)
            nn_idx = dots.argmax(dim=1)
            distances = self.pdist(x, x[nn_idx])
            return -torch.log(distances + eps).mean()
