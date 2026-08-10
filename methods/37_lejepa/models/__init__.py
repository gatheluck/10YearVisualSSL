"""LeJEPA model + SIGReg objective (Balestriero & LeCun, 2025). Self-contained
(a timm ViT backbone, NOT a submodule); SIGReg is reimplemented locally. Step 2
(the captured ViT fine-tune) is excluded."""

from .lejepa_model import (LeJEPABackbone, LeJEPAEncoder, ProjectionMLP,
                           build_backbone, build_lejepa)
from .sigreg import SIGReg, differentiable_mean_across_ranks

__all__ = ["LeJEPAEncoder", "LeJEPABackbone", "ProjectionMLP", "build_backbone",
           "build_lejepa", "SIGReg", "differentiable_mean_across_ranks"]
