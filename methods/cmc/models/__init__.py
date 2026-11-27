"""CMC models: the native two-branch AlexNet, plus the unified ViT-B/16 Step-2
variant (arch: vit; imported lazily as it needs timm)."""

from .cmc_alexnet import (AlexNetCMC, AlexNetHalf, Normalize,
                          build_cmc_from_config)

__all__ = ["AlexNetCMC", "AlexNetHalf", "Normalize", "build_cmc_from_config",
           "build_vit_cmc"]


def build_vit_cmc(*args, **kwargs):
    """Lazy accessor for the two-branch ViT-B/16 CMC model (needs timm)."""
    from .vit_cmc import build_vit_cmc as _build
    return _build(*args, **kwargs)
