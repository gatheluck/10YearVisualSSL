"""MoCo v2 models (Chen et al., 2020): the native ResNet-50 path, plus the
unified ViT-B/16 Step-2 variant (arch: vit; imported lazily as it needs timm)."""

from .resnet_mocov2 import MoCoV2ResNet, ResNetEncoderV2, build_mocov2_resnet

__all__ = ["MoCoV2ResNet", "ResNetEncoderV2", "build_mocov2_resnet",
           "build_mocov2_vit"]


def build_mocov2_vit(*args, **kwargs):
    """Lazy accessor for the ViT-B/16 MoCo v2 model (needs timm)."""
    from .vit_mocov2 import build_mocov2_vit as _build
    return _build(*args, **kwargs)
