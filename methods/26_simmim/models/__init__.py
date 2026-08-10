"""SimMIM Swin-B model (Xie et al., 2022). Self-contained around timm's
SwinTransformer; the capture's step 2 (ViT) is excluded, as in every port."""

from .simmim_swinb import SimMIMSwinB, build_simmim_swinb, build_swin_encoder

__all__ = ["SimMIMSwinB", "build_simmim_swinb", "build_swin_encoder"]
