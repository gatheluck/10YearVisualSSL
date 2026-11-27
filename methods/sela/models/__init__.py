"""SeLa models (Asano et al., 2020): the native ResNetv2 path, plus the unified
ViT-B/16 Step-2 variant (arch: vit; imported lazily as it needs timm)."""

from .resnet_sela import (PreActResNetBackbone, ResNetSeLa, create_resnet_sela)

__all__ = ["PreActResNetBackbone", "ResNetSeLa", "create_resnet_sela",
           "build_vit_sela"]


def build_vit_sela(*args, **kwargs):
    """Lazy accessor for the ViT-B/16 SeLa model (needs timm)."""
    from .vit_sela import build_vit_sela as _build
    return _build(*args, **kwargs)
