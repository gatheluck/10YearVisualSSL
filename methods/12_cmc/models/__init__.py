"""CMC AlexNet model (Tian et al., 2019)."""

from .cmc_alexnet import (AlexNetCMC, AlexNetHalf, Normalize,
                          build_cmc_from_config)

__all__ = ["AlexNetCMC", "AlexNetHalf", "Normalize", "build_cmc_from_config"]
