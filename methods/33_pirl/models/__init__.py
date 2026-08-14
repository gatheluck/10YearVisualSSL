"""PIRL models (Misra & van der Maaten, CVPR 2020): the native ResNet-50 path,
plus the unified ViT-B/16 Step-2 variant (arch: vit; imported lazily as it needs
timm)."""

from .resnet_pirl import ResNetPIRL, build_resnet_pirl

__all__ = ["ResNetPIRL", "build_resnet_pirl", "build_vit_pirl"]


def build_vit_pirl(*args, **kwargs):
    """Lazy accessor for the ViT-B/16 PIRL model (needs timm)."""
    from .vit_pirl import build_vit_pirl as _build
    return _build(*args, **kwargs)
