"""ImageNet data loader for NEPA (Xu et al., 2025).

Ported from the lab's own code. Step 1 augmentation (official NEPA): a
RandomResizedCrop + RandomHorizontalFlip, no colour jitter. The unified Step 2
augmentation adds a ColorJitter (RandomResizedCrop scale (0.2, 1.0) + flip +
ColorJitter). Both use the official NEPA normalisation (mean/std = 0.5). The port
drops the DistributedSampler (single-process) and threads a seeded generator so a
run is reproducible.
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


_NEPA_MEAN = [0.5, 0.5, 0.5]
_NEPA_STD = [0.5, 0.5, 0.5]


def _step1_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_NEPA_MEAN, std=_NEPA_STD),
    ])


def _step2_transform(img_size: int = 224) -> transforms.Compose:
    """Unified Step 2 augmentation: RandomResizedCrop + flip + ColorJitter."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, scale=(0.2, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,
                               hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=_NEPA_MEAN, std=_NEPA_STD),
    ])


def val_transform(img_size: int = 224) -> transforms.Compose:
    """Deterministic resize + centre crop (for linear eval), NEPA normalisation."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_NEPA_MEAN, std=_NEPA_STD),
    ])


def get_nepa_dataloader(data_path: str, augmentation: str, batch_size: int,
                        num_workers: int = 8, img_size: int = 224,
                        seed: int = 0):
    """Single-process NEPA pretraining DataLoader. Loads from ``data_path/train``.
    ``augmentation`` is 'step1' (native) or 'step2' (unified ViT-B/16)."""
    if augmentation == "step1":
        tf = _step1_transform(img_size)
    elif augmentation == "step2":
        tf = _step2_transform(img_size)
    else:
        raise ValueError(f"Unknown augmentation: {augmentation!r}. Choose "
                         "'step1' (native) or 'step2' (unified).")
    dataset = datasets.ImageFolder(os.path.join(data_path, "train"), transform=tf)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
        generator=torch.Generator().manual_seed(seed))
    return loader, dataset
