"""The instance-discrimination model, in one place. Only the ResNet-50 step-1
model is brought across; the capture's ViT (step 2) is excluded like every
method's step 2."""

from __future__ import annotations

from .resnet_instdisc import ResNetInstDisc, build_resnet_instdisc

__all__ = ["ResNetInstDisc", "build_resnet_instdisc"]
