"""MoCo v3 ViT model (Chen et al., 2021). The capture's step 2 (also ViT) is
excluded from this port; timm supplies the VisionTransformer base class."""

from .vit_mocov3 import (MoCoV3, VisionTransformerMoCo, ViTFeatureExtractor,
                         build_mocov3_vit, build_vit_backbone)

__all__ = ["MoCoV3", "VisionTransformerMoCo", "ViTFeatureExtractor",
           "build_mocov3_vit", "build_vit_backbone"]
