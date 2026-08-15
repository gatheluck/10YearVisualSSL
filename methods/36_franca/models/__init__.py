"""Franca unified Step-2 models (Franca; arXiv:2507.14137).

Reuses the DINOv2 ViT backbone (shared design with the DINOv2 port) and adds Franca's
own nested Matryoshka heads and Sinkhorn-Knopp losses. Used only by the from-scratch
Step-2 pretraining; the eval-only Step-1 path builds the official Franca backbone
from third_party/franca instead."""

from .dinov2_vit import DINOv2Backbone, build_dinov2_backbone
from .dinov2_loss import KoLeoLoss
from .franca_head import MatryoshkaHead
from .franca_loss import FrancaDinoLoss, FrancaIBOTLoss, sinkhorn_knopp_teacher
from .franca_model import FrancaStep2Model, build_franca_model

__all__ = ["DINOv2Backbone", "build_dinov2_backbone", "KoLeoLoss",
           "MatryoshkaHead", "FrancaDinoLoss", "FrancaIBOTLoss",
           "sinkhorn_knopp_teacher", "FrancaStep2Model", "build_franca_model"]
