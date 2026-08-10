"""ImageNet-1k data loader for I-JEPA (Assran et al., 2023).

Ported from the lab's own code. Step 1 augmentation (official I-JEPA):
RandomResizedCrop (scale 0.3-1.0), no flip / jitter / blur. The port drops the
DistributedSampler (single-process) and threads a seeded generator so a run is
reproducible. The MultiBlockMaskCollator is passed as ``collate_fn``.
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_transform(augmentation: str, img_size: int = 224,
                     use_horizontal_flip: bool = False) -> transforms.Compose:
    """augmentation: 'step1' (I-JEPA pretrain), 'step2' (unified), 'eval'."""
    norm = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    if augmentation == "step1":
        steps = [transforms.RandomResizedCrop(
            img_size, scale=(0.3, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC)]
        if use_horizontal_flip:
            steps.append(transforms.RandomHorizontalFlip())
        steps += [transforms.ToTensor(), norm]
        return transforms.Compose(steps)

    if augmentation == "step2":
        return transforms.Compose([
            transforms.RandomResizedCrop(
                img_size, scale=(0.2, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.2, hue=0.1),
            transforms.ToTensor(), norm])

    if augmentation == "eval":
        return transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(), norm])

    raise ValueError(f"Unknown augmentation type: {augmentation!r}. "
                     "Choose 'step1', 'step2', or 'eval'.")


def get_imagenet_loader(data_path: str, split: str = "train",
                        augmentation: str = "step1", img_size: int = 224,
                        batch_size: int = 128, num_workers: int = 8,
                        collate_fn=None, drop_last: bool = True,
                        use_horizontal_flip: bool = False, seed: int = 0):
    """Single-process ImageNet DataLoader. ``collate_fn`` is the
    MultiBlockMaskCollator for SSL. Loads from ``data_path/split``."""
    folder = os.path.join(data_path, split)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Dataset directory not found: {folder}")

    dataset = datasets.ImageFolder(
        folder,
        transform=_build_transform(augmentation, img_size, use_horizontal_flip))

    shuffle = (split == "train")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, drop_last=drop_last, collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(seed))
    return loader, dataset
