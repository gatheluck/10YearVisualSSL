"""The rotation model, in one place. Only the AlexNet-BN step-1 model is brought
across; the capture's ViT (step 2) is excluded like every method's step 2."""

from __future__ import annotations

from .alexnet_rotation import (AlexNetRotationEncoder, RotationAlexNet,
                               build_alexnet_rotation_model)

__all__ = ["AlexNetRotationEncoder", "RotationAlexNet",
           "build_alexnet_rotation_model"]
