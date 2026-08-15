"""AIM unified Step-2 model (El-Nouby et al., 2024). The lab's own from-scratch
re-implementation of AIM (torch-only, following arXiv:2401.08541 Appendix D) --
prefix-LM ViT trunk + MLP prediction head, next-patch pixel MSE. Used only by the
from-scratch Step-2 pretraining; the eval-only Step-1 path builds the official
AIM-600M backbone from third_party/ml-aim instead."""

from .aim_vit import AIMViT, aim_base, aim_huge, MODEL_REGISTRY

__all__ = ["AIMViT", "aim_base", "aim_huge", "MODEL_REGISTRY"]
