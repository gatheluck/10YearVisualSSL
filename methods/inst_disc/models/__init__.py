"""Instance-discrimination models: the native ResNet-50 path, plus the unified
ViT-B/16 Step-2 variant (arch: vit; imported lazily as it needs timm)."""

from __future__ import annotations

from .resnet_instdisc import ResNetInstDisc, build_resnet_instdisc

__all__ = ["ResNetInstDisc", "build_resnet_instdisc", "build_vit_instdisc"]


def build_vit_instdisc(*args, **kwargs):
    """Lazy accessor for the ViT-B/16 Instance Discrimination model (needs timm)."""
    from .vit_instdisc import build_vit_instdisc as _build
    return _build(*args, **kwargs)
