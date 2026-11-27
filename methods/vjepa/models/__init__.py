"""V-JEPA model construction (Bardes et al., 2024). The ViT + predictor are the
pinned facebookresearch/jepa upstream (third_party/jepa), imported not copied."""

from .vjepa_model import build_vjepa, build_vjepa_encoder

__all__ = ["build_vjepa", "build_vjepa_encoder"]
