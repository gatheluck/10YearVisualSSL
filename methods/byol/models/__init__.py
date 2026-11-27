"""BYOL models (Grill et al., 2020): the native ResNet-50 path, plus the unified
ViT-B/16 Step-2 variant (arch: vit; imported lazily as it needs timm). The EMA
target update, symmetric negative-cosine BYOLLoss and cosine tau schedule are
shared by both paths."""

from .resnet50_byol import (BYOLResNet50, BYOLLoss, LARS, ProjectionMLP,
                            PredictionMLP, LinearClassifier,
                            build_byol_resnet50, build_lars_optimizer,
                            compute_ema_tau)

__all__ = ["BYOLResNet50", "BYOLLoss", "LARS", "ProjectionMLP", "PredictionMLP",
           "LinearClassifier", "build_byol_resnet50", "build_lars_optimizer",
           "compute_ema_tau", "build_byol_vit"]


def build_byol_vit(*args, **kwargs):
    """Lazy accessor for the ViT-B/16 BYOL model (needs timm)."""
    from .vit_byol import build_byol_vit as _build
    return _build(*args, **kwargs)
