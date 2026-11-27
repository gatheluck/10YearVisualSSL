"""NEPA ViT model (Xu et al., 2025). Self-contained (its own nepa_vit.py with 2D
RoPE / QK-norm / causal AR predictor, NOT timm). The same NEPAModel serves both
the native step 1 (ViT-B/14) and the unified ViT-B/16 Step 2 (patch_size 16)."""

from .nepa_vit import (
    NEPAModel,
    build_nepa_model,
    build_nepa_vit_base,
    build_nepa_vit_large,
)

__all__ = [
    "NEPAModel",
    "build_nepa_model",
    "build_nepa_vit_base",
    "build_nepa_vit_large",
]
