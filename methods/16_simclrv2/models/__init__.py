"""SimCLR v2 ResNet-50 model (Chen et al., 2020). The capture's ViT variant
(step 2, which needs timm) is excluded from this port."""

from .resnet_simclrv2 import ResNetSimCLRv2, build_resnet_simclrv2

__all__ = ["ResNetSimCLRv2", "build_resnet_simclrv2"]
