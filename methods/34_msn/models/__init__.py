"""MSN model construction (Assran et al., 2022). The ViT is the pinned
facebookresearch/msn upstream (third_party/msn), imported not copied."""

from .msn_model import build_msn_backbone, build_msn_model

__all__ = ["build_msn_model", "build_msn_backbone"]
