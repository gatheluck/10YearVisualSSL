"""iBOT masking and patch-token loss with DINOv3 semantics."""

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class MaskingGenerator:
    """Generate an exact-count mask as a union of randomly placed blocks."""

    def __init__(
        self,
        input_size: tuple[int, int],
        min_num_patches: int = 4,
        max_num_patches: int | None = None,
        min_aspect: float = 0.3,
        max_aspect: float | None = None,
    ):
        self.height, self.width = input_size
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches or self.height * self.width
        max_aspect = max_aspect or 1.0 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        return low + torch.rand(()).item() * (high - low)

    def _mask_block(self, mask: torch.Tensor, max_mask_patches: int) -> int:
        if max_mask_patches < self.min_num_patches:
            return 0
        for _ in range(10):
            target_area = self._uniform(self.min_num_patches, max_mask_patches)
            aspect = math.exp(self._uniform(*self.log_aspect_ratio))
            height = int(round(math.sqrt(target_area * aspect)))
            width = int(round(math.sqrt(target_area / aspect)))
            if not (0 < height < self.height and 0 < width < self.width):
                continue

            top = int(torch.randint(0, self.height - height + 1, ()).item())
            left = int(torch.randint(0, self.width - width + 1, ()).item())
            region = mask[top : top + height, left : left + width]
            delta = int((~region).sum().item())
            if 0 < delta <= max_mask_patches:
                region.fill_(True)
                return delta
        return 0

    def __call__(self, num_masking_patches: int) -> torch.Tensor:
        total = self.height * self.width
        if not 0 <= num_masking_patches <= total:
            raise ValueError(f"num_masking_patches must be in [0, {total}]")

        mask = torch.zeros(self.height, self.width, dtype=torch.bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            remaining = min(
                num_masking_patches - mask_count,
                self.max_num_patches,
            )
            delta = self._mask_block(mask, remaining)
            if delta == 0:
                break
            mask_count += delta

        # Block placement can stall near the exact target. Complete from random
        # unmasked positions so every requested mask count is exact.
        missing = num_masking_patches - int(mask.sum().item())
        if missing:
            available = (~mask.flatten()).nonzero(as_tuple=False).flatten()
            chosen = available[torch.randperm(available.numel())[:missing]]
            mask.flatten()[chosen] = True
        return mask


def generate_block_mask(
    batch_size: int,
    n_patches_h: int,
    n_patches_w: int,
    mask_ratio_min: float = 0.1,
    mask_ratio_max: float = 0.5,
    mask_probability: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Mask exactly ``floor(batch_size * probability)`` crop samples.

    Ratios are sampled from equal-width strata spanning the configured range.
    The resulting rows are shuffled so masking is not tied to crop order.
    """
    if not 0.0 <= mask_probability <= 1.0:
        raise ValueError("mask_probability must be in [0, 1]")
    if not 0.0 <= mask_ratio_min < mask_ratio_max <= 1.0:
        raise ValueError("mask ratios must satisfy 0 <= min < max <= 1")

    n_tokens = n_patches_h * n_patches_w
    n_samples_masked = int(batch_size * mask_probability)
    generator = MaskingGenerator((n_patches_h, n_patches_w))
    ratio_edges = torch.linspace(mask_ratio_min, mask_ratio_max, n_samples_masked + 1)

    masks: list[torch.Tensor] = []
    for index in range(n_samples_masked):
        low = float(ratio_edges[index])
        high = float(ratio_edges[index + 1])
        ratio = low + torch.rand(()).item() * (high - low)
        target_count = int(n_tokens * ratio)
        masks.append(generator(target_count).flatten())
    masks.extend(
        torch.zeros(n_tokens, dtype=torch.bool)
        for _ in range(n_samples_masked, batch_size)
    )

    if not masks:
        return torch.zeros((0, n_tokens), dtype=torch.bool, device=device)
    order = torch.randperm(batch_size)
    return torch.stack(masks, dim=0)[order].to(device=device)


@torch.no_grad()
def sinkhorn_knopp_patches(
    teacher_output: torch.Tensor,
    teacher_temp: float,
    n_iters: int = 3,
) -> torch.Tensor:
    """Balance all masked patch assignments, including unequal rank counts."""
    teacher_output = teacher_output.float()
    q = torch.exp(teacher_output / teacher_temp).t()
    n_prototypes = q.shape[0]
    distributed = dist.is_available() and dist.is_initialized()

    total_patches = torch.tensor(
        float(q.shape[1]), device=q.device, dtype=torch.float32
    )
    if distributed:
        dist.all_reduce(total_patches)
    if total_patches.item() == 0:
        return teacher_output

    sum_q = q.sum()
    if distributed:
        dist.all_reduce(sum_q)
    q /= sum_q
    for _ in range(n_iters):
        row_sum = q.sum(dim=1, keepdim=True)
        if distributed:
            dist.all_reduce(row_sum)
        q /= row_sum
        q /= n_prototypes
        q /= q.sum(dim=0, keepdim=True)
        q /= total_patches
    q *= total_patches
    return q.t()


class IBOTLoss(nn.Module):
    def __init__(
        self,
        n_crops_global: int = 2,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        sk_n_iters: int = 3,
    ):
        super().__init__()
        self.n_global = n_crops_global
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.sk_n_iters = sk_n_iters

    def forward(
        self,
        student_patch_logits: torch.Tensor,
        teacher_patch_logits: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        if student_patch_logits.shape != teacher_patch_logits.shape:
            raise ValueError("student and teacher patch logits must have equal shape")
        if not masks.any():
            return student_patch_logits.sum() * 0.0

        if student_patch_logits.ndim == 2:
            if student_patch_logits.shape[0] != int(masks.sum().item()):
                raise ValueError("masked logits count does not match masks")
            student_masked = student_patch_logits
            teacher_masked = teacher_patch_logits.detach()
        elif student_patch_logits.ndim == 3:
            if masks.shape != student_patch_logits.shape[:2]:
                raise ValueError("masks must match the patch-token dimensions")
            student_masked = student_patch_logits[masks]
            teacher_masked = teacher_patch_logits.detach()[masks]
        else:
            raise ValueError("patch logits must have shape [M, K] or [B, N, K]")
        teacher_probs = sinkhorn_knopp_patches(
            teacher_masked,
            teacher_temp=self.teacher_temp,
            n_iters=self.sk_n_iters,
        )
        patch_loss = -(
            teacher_probs
            * F.log_softmax(student_masked.float() / self.student_temp, dim=-1)
        ).sum(dim=-1)

        per_image_count = masks.sum(dim=-1).clamp(min=1)
        masks_weight = (
            per_image_count.reciprocal().unsqueeze(-1).expand_as(masks)[masks]
        )
        return (patch_loss * masks_weight).sum() / masks.shape[0]
