"""The jigsaw++ models. The VGG16 pretext (step 1) encoder, and -- for the
knowledge-transfer stage -- the standard AlexNet trained on k-means pseudo-labels
(Noroozi et al.'s "Boosting SSL via Knowledge Transfer"). The capture's ViT
(step 2) is excluded from this port."""

from __future__ import annotations

from .vgg_jigsaw_pp import (VGG16Encoder, VGG16JigsawPP,
                            build_vgg16_jigsaw_pp_model)
from .alexnet_cluster_cls import (AlexNetClusterCls,
                                  build_alexnet_cluster_cls_model)

__all__ = ["VGG16Encoder", "VGG16JigsawPP", "build_vgg16_jigsaw_pp_model",
           "AlexNetClusterCls", "build_alexnet_cluster_cls_model"]
