"""I-JEPA ViT encoder + predictor (Assran et al., 2023). Self-contained (its own
vision_transformer.py, NOT timm); the capture's step 2 (ViT-B) is excluded."""

from .vision_transformer import (
    VisionTransformer,
    VisionTransformerPredictor,
    vit_tiny,
    vit_base,
    vit_large,
    vit_huge,
    build_ijepa_encoder,
    build_ijepa_predictor,
)

__all__ = [
    "VisionTransformer",
    "VisionTransformerPredictor",
    "vit_tiny",
    "vit_base",
    "vit_large",
    "vit_huge",
    "build_ijepa_encoder",
    "build_ijepa_predictor",
]
