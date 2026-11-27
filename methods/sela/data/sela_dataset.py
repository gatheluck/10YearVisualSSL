"""Dataset utilities for SeLa (Asano et al., 2020), ported from the lab's own
implementation.

The training loader must return ``(image, label, index)`` so Sinkhorn
assignments map back to the right images. Augmentation is the official
`self-label augs=3` recipe (RandomResizedCrop, RandomGrayscale, ColorJitter,
HorizontalFlip) with ImageNet normalisation; the probe uses the deterministic
resize + centre-crop pipeline.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import adapterlib

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_sela_train_transform(image_size: int = 224) -> transforms.Compose:
    """Official self-label `augs=3` ImageNet augmentation."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic resize + centre crop (no augmentation), ImageNet norm."""
    resize = int(round(image_size * 256 / 224))
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class IndexedImageFolder(Dataset):
    """Wraps torchvision ImageFolder to return ``(image, label, index)``.

    The index is used to map Sinkhorn assignments back to each image.
    """

    def __init__(self, root: str, transform=None):
        self.dataset = datasets.ImageFolder(
            adapterlib.dataset_split_dir(root, "train"), transform=transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        return img, label, idx

    @property
    def classes(self):
        return self.dataset.classes


def create_indexed_train_loader(train_path: str, image_size: int,
                                batch_size: int, num_workers: int, seed: int = 0):
    """The SeLa train loader: augmented images + sample indices."""
    dataset = IndexedImageFolder(
        train_path, transform=get_sela_train_transform(image_size))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=False,
        generator=torch.Generator().manual_seed(seed))
    return loader, dataset
