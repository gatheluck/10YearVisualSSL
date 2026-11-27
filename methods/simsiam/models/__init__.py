"""SimSiam models: the native ResNet-50 path, plus the unified ViT-B/16 Step-2
variant (arch: vit; imported lazily as it needs timm). `simsiam_loss` (negative
cosine similarity with stop-gradient) is shared by both paths."""

from .simsiam_resnet import SimSiamResNet, build_simsiam_resnet, simsiam_loss

__all__ = ["SimSiamResNet", "build_simsiam_resnet", "simsiam_loss",
           "build_simsiam_vit"]


def build_simsiam_vit(*args, **kwargs):
    """Lazy accessor for the ViT-B/16 SimSiam model (needs timm)."""
    from .vit_simsiam import build_simsiam_vit as _build
    return _build(*args, **kwargs)
