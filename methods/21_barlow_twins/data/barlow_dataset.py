"""
Barlow Twins dataset: returns two asymmetrically-augmented views of each image.

Step 1 augmentation (exact Barlow Twins paper / original repo):
  View 1 (transform):
    RandomResizedCrop(224, BICUBIC)
    RandomHorizontalFlip(p=0.5)
    RandomApply([ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8)
    RandomGrayscale(p=0.2)
    GaussianBlur(p=1.0)
    Solarization(p=0.0)
    ToTensor + ImageNet normalise

  View 2 (transform_prime):
    RandomResizedCrop(224, BICUBIC)
    RandomHorizontalFlip(p=0.5)
    RandomApply([ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8)
    RandomGrayscale(p=0.2)
    GaussianBlur(p=0.1)
    Solarization(p=0.2)
    ToTensor + ImageNet normalise

Step 2 augmentation (unified settings, type 1 — symmetric):
  Both views use the same pipeline (RandomResizedCrop, HFlip, ColorJitter, Grayscale).
"""

import random
from typing import Tuple

from PIL import Image, ImageFilter, ImageOps
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── Custom transforms ────────────────────────────────────────────────────────

class GaussianBlur:
    """Apply PIL GaussianBlur with probability p (sigma uniformly in [0.1, 2.0])."""

    def __init__(self, p: float):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        return img


class Solarization:
    """Apply PIL solarize with probability p."""

    def __init__(self, p: float):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


# ─── Augmentation pipelines ───────────────────────────────────────────────────

def _step1_view1(img_size: int = 224) -> transforms.Compose:
    """View 1: GaussianBlur always applied, no solarization."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.2, hue=0.1)],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(p=1.0),
        Solarization(p=0.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _step1_view2(img_size: int = 224) -> transforms.Compose:
    """View 2: GaussianBlur rarely, solarization occasionally."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.2, hue=0.1)],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(p=0.1),
        Solarization(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _step2_symmetric(img_size: int = 224) -> transforms.Compose:
    """Unified Step 2 augmentation (type 1) — applied to both views."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.4, hue=0.1)],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ─── Dataset ──────────────────────────────────────────────────────────────────

class BarlowDataset(datasets.ImageFolder):
    """
    ImageFolder subclass returning two views (y1, y2) per image.

    For Step 1, the two transforms are asymmetric (different blur/solarize
    probabilities) exactly matching the official Barlow Twins code.
    For Step 2, both transforms are the same symmetric pipeline.
    """

    def __init__(
        self,
        root: str,
        transform1: transforms.Compose,
        transform2: transforms.Compose,
    ):
        super().__init__(root, transform=None)
        self.transform1 = transform1
        self.transform2 = transform2

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path, label = self.samples[index]
        img = self.loader(path)
        y1 = self.transform1(img)
        y2 = self.transform2(img)
        return y1, y2, label


def get_barlow_dataloader(
    data_path: str,
    augmentation: str = "step1",
    batch_size: int = 256,
    num_workers: int = 8,
    img_size: int = 224,
    distributed: bool = False,
) -> Tuple[DataLoader, BarlowDataset]:
    """
    Build Barlow Twins data loader for SSL pre-training.

    Args:
        data_path   : path to ImageNet train split (parent/train/)
        augmentation: "step1" (asymmetric exact paper) or "step2" (unified symmetric)
        batch_size  : per-GPU batch size
        num_workers : DataLoader workers
        img_size    : crop size (224)
        distributed : whether to use DistributedSampler
    """
    if augmentation == "step1":
        t1 = _step1_view1(img_size)
        t2 = _step1_view2(img_size)
    elif augmentation == "step2":
        t1 = _step2_symmetric(img_size)
        t2 = _step2_symmetric(img_size)
    else:
        raise ValueError(f"Unknown augmentation mode: {augmentation!r}")

    dataset = BarlowDataset(data_path, transform1=t1, transform2=t2)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, dataset
