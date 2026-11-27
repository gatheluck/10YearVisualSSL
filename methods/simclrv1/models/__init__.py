"""SimCLR v1 models (Chen et al., 2020). The native ResNet-50 model is here; the
unified ViT-B/16 Step-2 model lives in vit_simclr.py, imported lazily (only on the
arch: vit path) so the native path needs no timm."""

from .resnet_simclr import ResNetSimCLR, build_resnet_simclr

__all__ = ["ResNetSimCLR", "build_resnet_simclr"]
