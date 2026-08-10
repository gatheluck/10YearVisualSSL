"""PIRL jigsaw dataset (Misra & van der Maaten, CVPR 2020)."""

from .pirl_dataset import (ImageNetPIRLDataset, JigsawViewTransform,
                           build_pirl_loader)

__all__ = ["ImageNetPIRLDataset", "JigsawViewTransform", "build_pirl_loader"]
