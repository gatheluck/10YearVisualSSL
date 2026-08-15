"""DINOv2 unified Step-2 models (Oquab et al., 2023). Ported from the capture's
own DINOv2 implementation; the ViT is timm-based, the losses and the
student-teacher wrapper are torch-only. Used only by the from-scratch Step-2
pretraining; the eval-only Step-1 path builds the official backbone from
third_party/dinov2 instead."""

from .dinov2_vit import DINOv2Backbone, build_dinov2_backbone
from .dinov2_loss import DINOLoss, iBOTLoss, KoLeoLoss
from .dinov2_model import DINOv2Model, DINOHead, build_dinov2_model

__all__ = ["DINOv2Backbone", "build_dinov2_backbone",
           "DINOLoss", "iBOTLoss", "KoLeoLoss",
           "DINOv2Model", "DINOHead", "build_dinov2_model"]
