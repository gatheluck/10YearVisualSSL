"""Split-Brain Autoencoder models (Zhang et al., 2017): the native AlexNet path,
plus the unified ViT-B/16 Step-2 variant (arch: vit; the two half-width ViT
branches need timm, so their builder is imported lazily)."""

from .split_brain import (SplitBrainModel, SplitBrainAlexNet,
                          build_split_brain_from_config,
                          AB_TARGET_CLASSES, L_TARGET_CLASSES)

__all__ = ["SplitBrainModel", "SplitBrainAlexNet", "build_split_brain_from_config",
           "AB_TARGET_CLASSES", "L_TARGET_CLASSES", "build_split_brain_vit"]


def build_split_brain_vit(*args, **kwargs):
    """Lazy accessor for the dual half-ViT split-brain model (needs timm)."""
    from .vit_split_brain import build_split_brain_vit as _build
    return _build(*args, **kwargs)
