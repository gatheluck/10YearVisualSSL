"""Franca unified Step-2 data pipeline (multi-crop + cyclic inverse-block masking)."""

from .franca_dataset import (
    CyclicMaskGenerator, FrancaMultiCropDataset, build_dinov2_augmentation,
    collate_franca_batch, generate_global_masks, get_franca_dataloader,
)

__all__ = ["CyclicMaskGenerator", "FrancaMultiCropDataset",
           "build_dinov2_augmentation", "collate_franca_batch",
           "generate_global_masks", "get_franca_dataloader"]
