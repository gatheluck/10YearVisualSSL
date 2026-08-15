"""Franca student-teacher model for the unified Step 2 (Franca; arXiv:2507.14137).

Ported from the capture's `methods/36_franca/train_step2_vit.py` (the model is
defined inline there). It reuses the DINOv2 ViT backbone (`DINOv2Backbone`, shared
with the DINOv2 port) but plugs in Franca's nested Matryoshka heads (separate DINO and
iBOT heads) and is trained with the Sinkhorn-Knopp losses (`franca_loss`). The
teacher is an EMA copy of the student; `encoder.pt` is the teacher backbone.

The ViT dims are threaded to timm (as the DINOv2 port does) so a hermetic CPU smoke can
build a tiny ViT; the real recipe passes ViT-B/16 dims.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dinov2_vit import DINOv2Backbone
from .franca_head import MatryoshkaHead


class FrancaStep2Model(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m = cfg["model"]
        embed_dim = int(m["embed_dim"])
        nesting_dims = [int(x) for x in m["nesting_dims"]]
        out_dim = int(cfg["head"]["out_dim"])
        hidden_dim = int(cfg["head"]["hidden_dim"])
        bottleneck_dim = int(cfg["head"]["bottleneck_dim"])
        nlayers = int(cfg["head"]["nlayers"])
        arch = m["arch"]
        # Thread the ViT dims to timm so the config declares what ran (and a smoke
        # can build a tiny ViT). The real recipe is ViT-B/16 (embed_dim 768).
        bb_kwargs = {"patch_size": int(m["patch_size"]), "embed_dim": embed_dim,
                     "depth": int(m["depth"]), "num_heads": int(m["num_heads"]),
                     "mlp_ratio": float(m.get("mlp_ratio", 4.0))}
        if "img_size" in m:
            bb_kwargs["img_size"] = int(m["img_size"])

        self.student_bb = DINOv2Backbone(arch, pretrained=False, **bb_kwargs)
        self.teacher_bb = DINOv2Backbone(arch, pretrained=False, **bb_kwargs)
        head_args = (embed_dim, nesting_dims, out_dim, hidden_dim, bottleneck_dim, nlayers)
        self.student_dino_head = MatryoshkaHead(*head_args)
        self.teacher_dino_head = MatryoshkaHead(*head_args)
        self.student_ibot_head = MatryoshkaHead(*head_args)
        self.teacher_ibot_head = MatryoshkaHead(*head_args)
        self._init_teacher()

    def _pairs(self):
        return [(self.student_bb, self.teacher_bb),
                (self.student_dino_head, self.teacher_dino_head),
                (self.student_ibot_head, self.teacher_ibot_head)]

    def _init_teacher(self) -> None:
        for src, tgt in self._pairs():
            for ps, pt in zip(src.parameters(), tgt.parameters()):
                pt.data.copy_(ps.data)
                pt.requires_grad_(False)
        self._set_teacher_eval()

    def _set_teacher_eval(self) -> None:
        for name in ("teacher_bb", "teacher_dino_head", "teacher_ibot_head"):
            module = getattr(self, name, None)
            if module is not None:
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._set_teacher_eval()
        return self

    def forward(self, global_crops: list, local_crops: list, masks: list):
        global_cls, global_patch, global_backbone_cls = [], [], []
        for crop, mask in zip(global_crops, masks):
            cls, patches = self.student_bb.get_all_tokens(crop, mask=mask)
            global_backbone_cls.append(cls)
            global_cls.append(self.student_dino_head(cls))
            global_patch.append(self.student_ibot_head(patches[mask]))
        local_cls = [self.student_dino_head(self.student_bb.get_cls_token(crop))
                     for crop in local_crops]
        return global_cls, global_patch, local_cls, global_backbone_cls

    def forward_student(self, global_crops: list, local_crops: list, masks: list):
        global_cls, global_patch, local_cls, _ = self.forward(
            global_crops, local_crops, masks)
        return global_cls, global_patch, local_cls

    @torch.no_grad()
    def forward_teacher(self, global_crops: list, masks: list):
        cls_out, patch_out = [], []
        for crop, mask in zip(global_crops, masks):
            cls, patches = self.teacher_bb.get_all_tokens(crop, mask=None)
            cls_out.append(self.teacher_dino_head(cls))
            patch_out.append(self.teacher_ibot_head(patches[mask]))
        return cls_out, patch_out

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        for src, tgt in self._pairs():
            for ps, pt in zip(src.parameters(), tgt.parameters()):
                pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)

    def student_parameters(self):
        yield from self.student_bb.parameters()
        yield from self.student_dino_head.parameters()
        yield from self.student_ibot_head.parameters()

    def get_teacher_backbone(self):
        return self.teacher_bb


def build_franca_model(cfg: dict) -> FrancaStep2Model:
    return FrancaStep2Model(cfg)
