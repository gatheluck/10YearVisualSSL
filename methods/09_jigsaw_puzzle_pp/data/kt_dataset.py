"""Datasets for the Jigsaw++ knowledge-transfer stage (Noroozi et al., 2018).

`build_kt_dataset` is a plain ImageFolder (RGB, ImageNet-normalised) used both to
extract the VGG16 conv4 features for clustering and to train the AlexNet.
`KTPseudoLabelDataset` wraps it, replacing labels with the k-means pseudo-labels
(aligned by dataset index).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import adapterlib

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _transform(image_size: int, train: bool) -> transforms.Compose:
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), norm])
    resize = int(round(image_size * 256 / 224))
    return transforms.Compose([
        transforms.Resize(resize), transforms.CenterCrop(image_size),
        transforms.ToTensor(), norm])


def build_kt_dataset(data_root: str, image_size: int = 224,
                     train: bool = True) -> datasets.ImageFolder:
    """An ImageFolder yielding ``(rgb_tensor, class_label)`` in a stable order."""
    return datasets.ImageFolder(
        adapterlib.dataset_split_dir(data_root, "train"),
        transform=_transform(image_size, train))


class KTPseudoLabelDataset(Dataset):
    """Wraps a base dataset, replacing its labels with cluster pseudo-labels."""

    def __init__(self, base: Dataset, pseudo_labels):
        self.base = base
        self.pseudo_labels = np.asarray(pseudo_labels).astype(np.int64)
        if len(self.pseudo_labels) != len(self.base):
            raise ValueError(
                f"pseudo_labels ({len(self.pseudo_labels)}) does not match the "
                f"dataset ({len(self.base)})")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img, _ = self.base[index]
        return img, int(self.pseudo_labels[index])
