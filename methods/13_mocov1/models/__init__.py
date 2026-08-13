"""MoCo v1 models (He et al., 2019): the native ResNet-50 path, plus the unified
ViT-B/16 Step-2 variant (arch: vit; imported lazily as it needs timm)."""

from .resnet_moco import MoCoResNet, ResNetEncoder, build_moco_resnet

__all__ = ["MoCoResNet", "ResNetEncoder", "build_moco_resnet",
           "build_moco_vit"]


def build_moco_vit(*args, **kwargs):
    """Lazy accessor for the ViT-B/16 MoCo v1 model (needs timm)."""
    from .vit_moco import build_moco_vit as _build
    return _build(*args, **kwargs)
