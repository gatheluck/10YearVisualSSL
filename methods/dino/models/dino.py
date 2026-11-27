"""DINO: self-distillation with no labels (Caron et al., 2021; arXiv:2104.14294).

Ported from the lab's own DINO code.

  Student : ViT backbone + DINOHead (receives all n_crops views)
  Teacher : EMA copy of the student  (receives only the 2 global views)

Loss: cross-entropy between the teacher (centred + sharpened) and student
(sharpened) logits, averaged over all (teacher_view, student_view) pairs where
the views differ. An online centre (EMA) prevents collapse; the teacher momentum
follows a cosine schedule m_init -> 1.0.

The port threads ``img_size`` (the capture hard-coded the ViT's 224 default) so
the same code runs a small hermetic CPU smoke. The DDP all_reduce in the centre
update is kept but inert single-process.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from .vision_transformer import build_vit, get_embed_dim
from .dino_head import DINOHead


class MultiCropWrapper(nn.Module):
    """Forward multiple crops (possibly different sizes) through a backbone +
    head. Crops of the same spatial resolution are concatenated into one batch."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, crops: "list[torch.Tensor]") -> torch.Tensor:
        if not isinstance(crops, list):
            crops = [crops]

        idx_crops: "list[tuple[int, int]]" = []
        start = 0
        last_size = crops[0].shape[-1]
        for i, crop in enumerate(crops[1:], start=1):
            if crop.shape[-1] != last_size:
                idx_crops.append((start, i))
                start = i
                last_size = crop.shape[-1]
        idx_crops.append((start, len(crops)))

        output_parts: "list[torch.Tensor]" = []
        for s, e in idx_crops:
            inp = torch.cat(crops[s:e], dim=0)
            feat = self.backbone(inp)
            out = self.head(feat)
            output_parts.append(out)

        return torch.cat(output_parts, dim=0)


class DINOLoss(nn.Module):
    """Cross-entropy between the teacher (centred, sharpened) and the student
    (sharpened). The centre is updated online (EMA) so all output dimensions
    contribute equally (prevents collapse to one dimension)."""

    def __init__(
        self,
        out_dim: int,
        n_crops: int,
        n_global_crops: int = 2,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.n_crops = n_crops
        self.n_global_crops = n_global_crops
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        teacher_temp: float,
    ) -> torch.Tensor:
        s_chunks = (student_out / self.student_temp).chunk(self.n_crops)
        t_probs = F.softmax(
            (teacher_out - self.center) / teacher_temp, dim=-1
        ).detach()
        t_chunks = t_probs.chunk(self.n_global_crops)

        total_loss = torch.tensor(0.0, device=student_out.device)
        n_terms = 0
        for iq, q in enumerate(t_chunks):
            for iv, v in enumerate(s_chunks):
                if iv == iq:
                    continue  # skip same-view pairs
                total_loss += torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1).mean()
                n_terms += 1

        total_loss /= n_terms
        self._update_center(teacher_out)
        return total_loss

    @torch.no_grad()
    def _update_center(self, teacher_out: torch.Tensor) -> None:
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center /= dist.get_world_size()
        self.center = (
            self.center * self.center_momentum
            + batch_center * (1.0 - self.center_momentum)
        )


class DINO(nn.Module):
    """Student + teacher + DINO loss in a single module. The teacher is an EMA
    copy of the student; gradients only flow through the student."""

    def __init__(
        self,
        arch: str = "vit_small",
        out_dim: int = 65536,
        n_local_crops: int = 8,
        student_temp: float = 0.1,
        teacher_temp_init: float = 0.04,
        teacher_temp_final: float = 0.04,
        teacher_temp_warmup_epochs: int = 0,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        use_bn_in_head: bool = False,
        norm_last_layer: bool = True,
        drop_path_rate: float = 0.0,
        img_size: int = 224,
    ):
        super().__init__()
        self.n_global_crops = 2
        self.n_local_crops = n_local_crops
        self.n_crops = self.n_global_crops + n_local_crops

        self.teacher_temp_init = teacher_temp_init
        self.teacher_temp_final = teacher_temp_final
        self.teacher_temp_warmup_epochs = teacher_temp_warmup_epochs

        embed_dim = get_embed_dim(arch)

        self.student = MultiCropWrapper(
            build_vit(arch, img_size=img_size, drop_path_rate=drop_path_rate),
            DINOHead(
                in_dim=embed_dim,
                out_dim=out_dim,
                use_bn=use_bn_in_head,
                norm_last_layer=norm_last_layer,
                hidden_dim=hidden_dim,
                bottleneck_dim=bottleneck_dim,
            ),
        )

        self.teacher = MultiCropWrapper(
            build_vit(arch, img_size=img_size, drop_path_rate=0.0),
            DINOHead(
                in_dim=embed_dim,
                out_dim=out_dim,
                use_bn=use_bn_in_head,
                norm_last_layer=False,  # teacher head does not freeze weight_g
                hidden_dim=hidden_dim,
                bottleneck_dim=bottleneck_dim,
            ),
        )
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Initialise teacher = student.
        self.teacher.load_state_dict(self.student.state_dict())

        self.loss_fn = DINOLoss(
            out_dim=out_dim,
            n_crops=self.n_crops,
            n_global_crops=self.n_global_crops,
            student_temp=student_temp,
            teacher_temp=teacher_temp_init,
        )

    def get_teacher_temp(self, epoch: float) -> float:
        """Linear teacher-temperature schedule over fractional epochs."""
        if (
            self.teacher_temp_warmup_epochs <= 0
            or self.teacher_temp_init == self.teacher_temp_final
            or epoch >= self.teacher_temp_warmup_epochs
        ):
            return self.teacher_temp_final
        ratio = epoch / self.teacher_temp_warmup_epochs
        return self.teacher_temp_init + ratio * (
            self.teacher_temp_final - self.teacher_temp_init
        )

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """EMA update: teacher <- m*teacher + (1-m)*student."""
        for ps, pt in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            pt.data.mul_(momentum).add_((1.0 - momentum) * ps.data)

    def forward(self, crops: "list[torch.Tensor]", epoch: float = 0) -> torch.Tensor:
        student_out = self.student(crops)
        with torch.no_grad():
            teacher_out = self.teacher(crops[: self.n_global_crops])
        teacher_temp = self.get_teacher_temp(epoch)
        return self.loss_fn(student_out, teacher_out, teacher_temp)

    def get_backbone(self) -> nn.Module:
        """The student backbone. encoder.pt ships the teacher backbone (the
        representation DINO is known for; the capture's linear eval defaults to
        the teacher), so the adapter extracts teacher.backbone.* directly."""
        return self.student.backbone

    def cancel_last_layer_gradients(self) -> None:
        """Zero the gradients of the student head's last layer (freeze_last)."""
        for p in self.student.head.last_layer.parameters():
            p.grad = None


def build_dino(
    arch: str = "vit_small",
    out_dim: int = 65536,
    n_local_crops: int = 8,
    student_temp: float = 0.1,
    teacher_temp_init: float = 0.04,
    teacher_temp_final: float = 0.04,
    teacher_temp_warmup_epochs: int = 0,
    hidden_dim: int = 2048,
    bottleneck_dim: int = 256,
    use_bn_in_head: bool = False,
    norm_last_layer: bool = True,
    drop_path_rate: float = 0.0,
    img_size: int = 224,
) -> DINO:
    return DINO(
        arch=arch,
        out_dim=out_dim,
        n_local_crops=n_local_crops,
        student_temp=student_temp,
        teacher_temp_init=teacher_temp_init,
        teacher_temp_final=teacher_temp_final,
        teacher_temp_warmup_epochs=teacher_temp_warmup_epochs,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        use_bn_in_head=use_bn_in_head,
        norm_last_layer=norm_last_layer,
        drop_path_rate=drop_path_rate,
        img_size=img_size,
    )
