"""DeepCluster models (Caron et al., 2018): the native AlexNet-BN path, plus the
unified ViT-B/16 Step-2 variant (arch: vit; imported lazily as it needs timm)."""

from .alexnet_deepcluster import (AlexNetDeepCluster, SobelFilter,
                                  build_alexnet_deepcluster)

__all__ = ["AlexNetDeepCluster", "SobelFilter", "build_alexnet_deepcluster",
           "build_vit_deepcluster"]


def build_vit_deepcluster(*args, **kwargs):
    """Lazy accessor for the ViT-B/16 DeepCluster model (needs timm)."""
    from .vit_deepcluster import build_vit_deepcluster as _build
    return _build(*args, **kwargs)
