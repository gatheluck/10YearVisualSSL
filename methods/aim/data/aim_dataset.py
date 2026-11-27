"""
Data loading and augmentation for AIM pre-training and evaluation.

Pre-training augmentation (Step 1 original AIM):
  RandomResizedCrop(224, scale=[0.4, 1.0], ratio=[0.75, 1.33], bicubic)
  RandomHorizontalFlip(p=0.5)
  Normalise with ImageNet mean/std

Step 2 unified augmentation (Type 1):
  RandomResizedCrop(224, scale=[0.08, 1.0], bicubic)
  RandomHorizontalFlip(p=0.5)
  ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
  Normalise with ImageNet mean/std

Attentive-probe training augmentation (from paper Appendix D):
  RandomResizedCrop(224, scale=[0.08, 1.0], bicubic)
  RandomHorizontalFlip(p=0.5)
  ColorJitter(0.3)
  AutoAugment (rand-m9-mstd0.5-inc1)
  Normalise

Validation augmentation:
  Resize(256, bicubic) → CenterCrop(224) → Normalise
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _normalise():
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def pretrain_transforms_step1(img_size: int = 224) -> transforms.Compose:
    """Step 1 original AIM pre-training augmentation (arXiv:2401.08541 Appendix D)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.4, 1.0),
            ratio=(0.75, 1.3333),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        _normalise(),
    ])


def pretrain_transforms_step2(img_size: int = 224) -> transforms.Compose:
    """Step 2 unified SSL pre-training augmentation (Type 1)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.08, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1,
        ),
        transforms.ToTensor(),
        _normalise(),
    ])


def probe_train_transforms(img_size: int = 224) -> transforms.Compose:
    """Attentive probe training augmentation (paper Appendix D)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.08, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        _normalise(),
    ])


def val_transforms(img_size: int = 224) -> transforms.Compose:
    """Standard validation augmentation."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        _normalise(),
    ])


def get_pretrain_loader(
    data_path:   str,
    batch_size:  int,
    img_size:    int   = 224,
    num_workers: int   = 8,
    distributed: bool  = False,
    step:        int   = 1,
    drop_last:   bool  = True,
    persistent_workers: bool = True,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """ImageNet-1k training DataLoader for AIM pre-training."""
    tfm = pretrain_transforms_step1(img_size) if step == 1 else pretrain_transforms_step2(img_size)
    dataset = datasets.ImageFolder(data_path, transform=tfm)

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0 and persistent_workers),
    )
    return loader, sampler


def get_probe_loader(
    data_path:   str,
    batch_size:  int,
    split:       str  = "train",
    img_size:    int  = 224,
    num_workers: int  = 8,
    distributed: bool = False,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """ImageNet-1k DataLoader for attentive probe training/evaluation."""
    tfm     = probe_train_transforms(img_size) if split == "train" else val_transforms(img_size)
    dataset = datasets.ImageFolder(data_path, transform=tfm)

    sampler = DistributedSampler(dataset, shuffle=(split == "train")) if distributed else None
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == "train"),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
    )
    return loader, sampler
