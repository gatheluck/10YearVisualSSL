"""DINOv2 student-teacher model (Oquab et al., 2023; arXiv:2304.07193).

Ported from the capture's `methods/28_dinov2/models/dinov2_model.py`. The student
is a ViT backbone + shared DINO/iBOT head (official default); the teacher is an EMA
copy (no gradients). The DINO/iBOT prototype head is a 3-layer MLP -> 256
bottleneck -> L2-norm -> weight-norm Linear to 65,536 prototypes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov2_vit import DINOv2Backbone


def _build_mlp(in_dim: int, bottleneck_dim: int, hidden_dim: int, nlayers: int) -> nn.Sequential:
    layers: list = []
    prev = in_dim
    for _ in range(nlayers - 1):
        layers += [nn.Linear(prev, hidden_dim), nn.GELU()]
        prev = hidden_dim
    layers.append(nn.Linear(prev, bottleneck_dim))
    return nn.Sequential(*layers)


class DINOHead(nn.Module):
    """DINOv2 projection head: MLP -> L2-norm -> weight-norm prototype layer."""

    def __init__(self, in_dim: int, out_dim: int = 65536, hidden_dim: int = 2048,
                 bottleneck_dim: int = 256, nlayers: int = 3):
        super().__init__()
        self.mlp = _build_mlp(in_dim, bottleneck_dim, hidden_dim, nlayers)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False))
        nn.init.constant_(self.last_layer.weight_g, 1.0)
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.last_layer(self.forward_features(x))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(x), dim=-1)


class DINOv2Model(nn.Module):
    """Container for the DINOv2 student and its EMA teacher."""

    def __init__(self, cfg: dict):
        super().__init__()
        arch = cfg["model"]["arch"]
        embed_dim = cfg["model"]["embed_dim"]
        out_dim = cfg["dino"]["out_dim"]
        ibot_dim = cfg["ibot"]["out_dim"]
        self.ibot_separate_head = bool(cfg["ibot"].get("separate_head", True))
        # The ViT dims are threaded to timm so the config declares what ran (and a
        # hermetic CPU smoke can build a tiny ViT). img_size comes from the global
        # crop size; dynamic_img_size lets the smaller local crops share the ViT.
        bb_kwargs = {"patch_size": int(cfg["model"]["patch_size"]),
                     "embed_dim": int(embed_dim),
                     "depth": int(cfg["model"]["depth"]),
                     "num_heads": int(cfg["model"]["num_heads"]),
                     "mlp_ratio": float(cfg["model"].get("mlp_ratio", 4.0))}
        if "img_size" in cfg["model"]:
            bb_kwargs["img_size"] = int(cfg["model"]["img_size"])
        head_kwargs = {
            "hidden_dim": cfg["dino"].get("head_hidden_dim", 2048),
            "bottleneck_dim": cfg["dino"].get("head_bottleneck_dim", 256),
            "nlayers": cfg["dino"].get("head_nlayers", 3),
        }

        self.student_bb = DINOv2Backbone(arch, pretrained=False, **bb_kwargs)
        self.teacher_bb = DINOv2Backbone(arch, pretrained=False, **bb_kwargs)

        self.student_dino_head = DINOHead(embed_dim, out_dim, **head_kwargs)
        self.teacher_dino_head = DINOHead(embed_dim, out_dim, **head_kwargs)

        if self.ibot_separate_head:
            ibot_head_kwargs = {
                "hidden_dim": cfg["ibot"].get("head_hidden_dim", 2048),
                "bottleneck_dim": cfg["ibot"].get("head_bottleneck_dim", 256),
                "nlayers": cfg["ibot"].get("head_nlayers", 3),
            }
            self.student_ibot_head = DINOHead(embed_dim, ibot_dim, **ibot_head_kwargs)
            self.teacher_ibot_head = DINOHead(embed_dim, ibot_dim, **ibot_head_kwargs)
        else:
            if ibot_dim != out_dim:
                raise ValueError("shared DINO/iBOT head requires matching output dimensions")
            self.student_ibot_head = None
            self.teacher_ibot_head = None

        self._init_teacher()

    def _init_teacher(self):
        for src, tgt in self._ema_pairs():
            for ps, pt in zip(src.parameters(), tgt.parameters()):
                pt.data.copy_(ps.data)
                pt.requires_grad = False

    def _ema_pairs(self):
        pairs = [(self.student_bb, self.teacher_bb),
                 (self.student_dino_head, self.teacher_dino_head)]
        if getattr(self, "ibot_separate_head", True):
            pairs.append((self.student_ibot_head, self.teacher_ibot_head))
        return pairs

    def _student_patch_head(self):
        return self.student_ibot_head if getattr(self, "ibot_separate_head", True) else self.student_dino_head

    def _teacher_patch_head(self):
        return self.teacher_ibot_head if getattr(self, "ibot_separate_head", True) else self.teacher_dino_head

    @staticmethod
    def _project_masked_patches(head, patches, mask):
        if mask is None:
            return head(patches)
        return head(patches[mask.to(device=patches.device, dtype=torch.bool)])

    def forward(self, global_crops: list, local_crops: list, masks: list):
        global_cls, global_ptok, global_backbone_cls = [], [], []
        first_cls = None
        for crop_idx, (crop, mask) in enumerate(zip(global_crops, masks)):
            cls, patches = self.student_bb.get_all_tokens(crop, mask=mask)
            global_backbone_cls.append(cls)
            if crop_idx == 0:
                first_cls = cls
            global_cls.append(self.student_dino_head(cls))
            global_ptok.append(
                self._project_masked_patches(self._student_patch_head(), patches, mask))
        local_cls = [
            self.student_dino_head(self.student_bb.get_cls_token(crop, mask=None))
            for crop in local_crops
        ]
        return global_cls, global_ptok, local_cls, first_cls, global_backbone_cls

    def forward_student(self, global_crops: list, local_crops: list, masks: list):
        global_cls, global_ptok, local_cls, _, _ = self.forward(
            global_crops, local_crops, masks)
        return global_cls, global_ptok, local_cls

    @torch.no_grad()
    def forward_teacher(self, global_crops: list, masks: "list | None" = None):
        teacher_cls, teacher_patches = [], []
        if masks is None:
            masks = [None] * len(global_crops)
        for crop, mask in zip(global_crops, masks):
            cls, patches = self.teacher_bb.get_all_tokens(crop, mask=None)
            teacher_cls.append(self.teacher_dino_head(cls))
            teacher_patches.append(
                self._project_masked_patches(self._teacher_patch_head(), patches, mask))
        return teacher_cls, teacher_patches

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        for src, tgt in self._ema_pairs():
            for ps, pt in zip(src.parameters(), tgt.parameters()):
                pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)

    def student_parameters(self):
        yield from self.student_bb.parameters()
        yield from self.student_dino_head.parameters()
        if getattr(self, "ibot_separate_head", True):
            yield from self.student_ibot_head.parameters()

    def student_projection_heads(self):
        heads = [self.student_dino_head]
        if getattr(self, "ibot_separate_head", True):
            heads.append(self.student_ibot_head)
        return heads

    def get_student_backbone(self) -> DINOv2Backbone:
        return self.student_bb

    def get_teacher_backbone(self) -> DINOv2Backbone:
        return self.teacher_bb


def build_dinov2_model(cfg: dict) -> DINOv2Model:
    return DINOv2Model(cfg)
