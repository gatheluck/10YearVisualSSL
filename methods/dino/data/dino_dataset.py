"""DINO multi-crop augmentation and dataset (Caron et al., 2021).

Ported from the lab's own DINO code (paper Appendix A):

  Global crops (n=2, scale 0.4-1.0):
    view 1 : GaussianBlur always (p=1.0)
    view 2 : GaussianBlur p=0.1, Solarization p=0.2
  Local crops (n=n_local, scale 0.05-0.4): GaussianBlur p=0.5
  Common to all views: RandomResizedCrop (BICUBIC), HFlip (p=0.5),
    ColorJitter (p=0.8), RandomGrayscale (p=0.2), ImageNet normalise.

The port drops the DistributedSampler (single-process) and threads a seeded
generator so a run is reproducible; ``global_size`` / ``local_size`` are threaded
so a small hermetic CPU smoke can run at a lower resolution.
"""

from __future__ import annotations

import random
from typing import Callable

from PIL import Image, ImageFilter, ImageOps

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import adapterlib


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class GaussianBlur:
    """Apply PIL GaussianBlur with probability p, sigma sampled in [min, max]."""

    def __init__(self, p: float = 0.5,
                 sigma_min: float = 0.1, sigma_max: float = 2.0):
        self.p = p
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img: "Image.Image") -> "Image.Image":
        if random.random() < self.p:
            sigma = random.uniform(self.sigma_min, self.sigma_max)
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class Solarization:
    """Apply PIL solarization with probability p."""

    def __init__(self, p: float):
        self.p = p

    def __call__(self, img: "Image.Image") -> "Image.Image":
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


def _common_aug() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.2, hue=0.1)],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
    ])


def global_aug_1(img_size: int = 224,
                 scale: "tuple[float, float]" = (0.4, 1.0)) -> Callable:
    """Global view 1: always blur."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, scale=scale,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        _common_aug(),
        GaussianBlur(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def global_aug_2(img_size: int = 224,
                 scale: "tuple[float, float]" = (0.4, 1.0)) -> Callable:
    """Global view 2: blur p=0.1, solarization p=0.2."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, scale=scale,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        _common_aug(),
        GaussianBlur(p=0.1),
        Solarization(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def local_aug(img_size: int = 96,
              scale: "tuple[float, float]" = (0.05, 0.4)) -> Callable:
    """Local crops: blur p=0.5."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, scale=scale,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        _common_aug(),
        GaussianBlur(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


class DINODataset(datasets.ImageFolder):
    """ImageFolder returning multi-crop views for DINO: ``(crops, label)`` where
    ``crops`` is a list of ``2 + n_local_crops`` tensors."""

    def __init__(
        self,
        root: str,
        n_local_crops: int = 8,
        global_size: int = 224,
        local_size: int = 96,
        global_scale: "tuple[float, float]" = (0.4, 1.0),
        local_scale: "tuple[float, float]" = (0.05, 0.4),
    ):
        super().__init__(adapterlib.dataset_split_dir(root, "train"),
                         transform=None)
        self.n_local_crops = n_local_crops
        self.tf_global1 = global_aug_1(global_size, global_scale)
        self.tf_global2 = global_aug_2(global_size, global_scale)
        self.tf_local = local_aug(local_size, local_scale)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = self.loader(path)
        crops = [self.tf_global1(img), self.tf_global2(img)]
        for _ in range(self.n_local_crops):
            crops.append(self.tf_local(img))
        return crops, label


def multicrop_collate(batch):
    """list[(crops_list, label)] -> (list[Tensor(B,3,H,W)] length n_crops,
    Tensor(B,) labels)."""
    crops_by_view = list(zip(*[item[0] for item in batch]))
    crops_by_view = [torch.stack(c) for c in crops_by_view]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return crops_by_view, labels


def get_dino_dataloader(
    data_path: str,
    n_local_crops: int = 8,
    batch_size: int = 128,
    num_workers: int = 8,
    global_size: int = 224,
    local_size: int = 96,
    global_scale: "tuple[float, float]" = (0.4, 1.0),
    local_scale: "tuple[float, float]" = (0.05, 0.4),
    seed: int = 0,
):
    """Build the DINO multi-crop training data loader (single-process). Yields
    ``(list[Tensor], Tensor)`` per batch."""
    dataset = DINODataset(
        root=data_path,
        n_local_crops=n_local_crops,
        global_size=global_size,
        local_size=local_size,
        global_scale=global_scale,
        local_scale=local_scale,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=multicrop_collate,
        generator=torch.Generator().manual_seed(seed),
    )
    return loader, dataset
