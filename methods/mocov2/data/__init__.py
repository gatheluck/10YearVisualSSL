"""MoCo v2 two-view dataset (Chen et al., 2020)."""

from .mocov2_dataset import MoCoV2Dataset, IMAGENET_MEAN, IMAGENET_STD

__all__ = ["MoCoV2Dataset", "IMAGENET_MEAN", "IMAGENET_STD"]
