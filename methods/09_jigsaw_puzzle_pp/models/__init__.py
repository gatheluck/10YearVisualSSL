"""The jigsaw++ model, in one place. Only the VGG16 pretext (step 1) model is
brought across; the capture's AlexNet cluster-classification (the faiss
knowledge-transfer path) and its ViT (step 2) are excluded from this port."""

from __future__ import annotations

from .vgg_jigsaw_pp import (VGG16Encoder, VGG16JigsawPP,
                            build_vgg16_jigsaw_pp_model)

__all__ = ["VGG16Encoder", "VGG16JigsawPP", "build_vgg16_jigsaw_pp_model"]
