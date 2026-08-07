"""NCE memory-bank machinery for CMC (Tian et al., 2019)."""

from .alias_multinomial import AliasMethod
from .nce_average import NCEAverage
from .nce_criterion import NCECriterion, NCESoftmaxLoss

__all__ = ["AliasMethod", "NCEAverage", "NCECriterion", "NCESoftmaxLoss"]
