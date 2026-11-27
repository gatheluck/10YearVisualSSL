"""MoCo v1 dataset (He et al., 2019), ported from the lab's own implementation.

Returns two independently-augmented views (q, k) of the same image plus its
label. The step-1 augmentation is the exact MoCo v1 recipe: RandomResizedCrop(
scale 0.2-1.0), RandomGrayscale(0.2), ColorJitter(0.4,0.4,0.4,0.4),
RandomHorizontalFlip -- NO GaussianBlur and NO MLP head (those are v2).
"""

from __future__ import annotations

from typing import Tuple

import torch
from torchvision import datasets, transforms

import adapterlib

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _step1_augmentation(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,
                               hue=0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class MoCoDataset(datasets.ImageFolder):
    """ImageFolder returning ``(q_view, k_view, label)`` for MoCo pretraining."""

    def __init__(self, root: str, mode: str = "step1", image_size: int = 224):
        super().__init__(adapterlib.dataset_split_dir(root, "train"),
                         transform=None)
        if mode != "step1":
            raise ValueError(f"unknown mode {mode!r}; this port ports step 1")
        self.moco_transform = _step1_augmentation(image_size)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path, label = self.samples[index]
        img = self.loader(path)
        q = self.moco_transform(img)
        k = self.moco_transform(img)
        return q, k, label
