"""Context Encoder models (Pathak et al., 2016): the native AlexNet inpainting
path, plus the unified ViT-B/16 Step-2 variant (arch: vit). The ViT encoder-
decoder needs timm, so its builder is imported lazily; the native AlexNet path
never imports it."""

from .context_encoder import (
    ContextEncoderAlexNet,
    Discriminator,
    create_model,
)

__all__ = [
    'ContextEncoderAlexNet',
    'Discriminator',
    'create_model',
    'build_vit_context_encoder',
]


def build_vit_context_encoder(*args, **kwargs):
    """Lazy accessor for the ViT-B/16 context-encoder model (needs timm)."""
    from .vit_context_encoder import build_vit_context_encoder as _build
    return _build(*args, **kwargs)
