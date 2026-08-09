"""MoCo v2 dataset (Chen et al., 2020), ported from the lab's own implementation.

Returns two independently-augmented views (q, k) of the same image plus its
label. The step-1 augmentation is the exact MoCo v2 recipe (the FB `--aug-plus`
flag): RandomResizedCrop(scale 0.2-1.0), ColorJitter(0.4,0.4,0.4,**0.1**) with
p=0.8 (v1 used hue 0.4, always-on), RandomGrayscale(0.2), **GaussianBlur**
(kernel 23, sigma 0.1-2.0, p=0.5 -- the v2 addition), RandomHorizontalFlip, then
ImageNet normalisation.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _step1_augmentation(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.4, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class MoCoV2Dataset(datasets.ImageFolder):
    """ImageFolder returning ``(q_view, k_view, label)`` for MoCo v2 pretraining."""

    def __init__(self, root: str, image_size: int = 224):
        super().__init__(root, transform=None)
        self.moco_transform = _step1_augmentation(image_size)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path, label = self.samples[index]
        img = self.loader(path)
        q = self.moco_transform(img)
        k = self.moco_transform(img)
        return q, k, label
