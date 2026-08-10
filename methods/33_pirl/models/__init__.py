"""PIRL ResNet-50 model (Misra & van der Maaten, CVPR 2020). Self-contained
(torchvision ResNet-50); the capture's step 2 (ViT) is excluded, as in every port."""

from .resnet_pirl import ResNetPIRL, build_resnet_pirl

__all__ = ["ResNetPIRL", "build_resnet_pirl"]
