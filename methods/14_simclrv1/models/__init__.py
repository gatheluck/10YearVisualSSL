"""SimCLR v1 ResNet-50 model (Chen et al., 2020). The capture's ViT variant
(step 2, which needs timm) is excluded from this port."""

from .resnet_simclr import ResNetSimCLR, build_resnet_simclr

__all__ = ["ResNetSimCLR", "build_resnet_simclr"]
