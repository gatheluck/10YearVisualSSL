"""SimCLR v2 models (Chen et al., 2020). The native ResNet-50 model is here; the
unified ViT-B/16 Step-2 model lives in vit_simclrv2.py, imported lazily (only on
the arch: vit path) so the native path needs no timm."""

from .resnet_simclrv2 import ResNetSimCLRv2, build_resnet_simclrv2

__all__ = ["ResNetSimCLRv2", "build_resnet_simclrv2"]
