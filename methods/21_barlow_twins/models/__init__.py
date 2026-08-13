"""Barlow Twins models: the native ResNet-50 path, plus the unified ViT-B/16
Step-2 variant (arch: vit; imported lazily as it needs timm). The
cross-correlation loss, ``off_diagonal`` and ``_build_projector`` are shared by
both paths."""

from .barlow_resnet import BarlowTwinsResNet, build_barlow_resnet

__all__ = ["BarlowTwinsResNet", "build_barlow_resnet", "build_barlow_vit"]


def build_barlow_vit(*args, **kwargs):
    """Lazy accessor for the ViT-B/16 Barlow Twins model (needs timm)."""
    from .vit_barlow import build_barlow_vit as _build
    return _build(*args, **kwargs)
