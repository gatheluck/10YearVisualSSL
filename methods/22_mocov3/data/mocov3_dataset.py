"""MoCo v3 dataset (Chen et al., 2021), ported from the lab's own implementation.

Returns two asymmetrically-augmented views (view1, view2) of the same image plus
its label:
  View 1 (strong): RandomResizedCrop + ColorJitter(p=0.8) + Grayscale(p=0.2)
                   + GaussianBlur(p=1.0) + HFlip
  View 2 (weaker): ... + GaussianBlur(p=0.1) + Solarize(p=0.2) + HFlip
Both end with ToTensor + ImageNet normalisation.
"""

from __future__ import annotations

from typing import Tuple

import PIL.Image
import PIL.ImageFilter
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class GaussianBlur:
    """Gaussian blur with a random sigma (BYOL / MoCo v3)."""

    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img: "PIL.Image.Image") -> "PIL.Image.Image":
        sigma = torch.empty(1).uniform_(self.sigma_min, self.sigma_max).item()
        return img.filter(PIL.ImageFilter.GaussianBlur(radius=sigma))


class Solarize:
    """Solarize: invert pixel values above a threshold."""

    def __init__(self, threshold: int = 128):
        self.threshold = threshold

    def __call__(self, img: "PIL.Image.Image") -> "PIL.Image.Image":
        return TF.solarize(img, self.threshold)


def _view1_augmentation(img_size: int = 224, crop_min: float = 0.08):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(crop_min, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
                               p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur(0.1, 2.0)], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _view2_augmentation(img_size: int = 224, crop_min: float = 0.08):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(crop_min, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
                               p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur(0.1, 2.0)], p=0.1),
        transforms.RandomApply([Solarize(128)], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_val_transform(img_size: int = 224):
    """Deterministic resize + centre crop (no augmentation), ImageNet norm."""
    return transforms.Compose([
        transforms.Resize(int(round(img_size * 256 / 224))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


class MoCoV3Dataset(datasets.ImageFolder):
    """ImageFolder returning ``(view1, view2, label)`` for MoCo v3 pretraining."""

    def __init__(self, root: str, img_size: int = 224, crop_min: float = 0.08):
        super().__init__(root, transform=None)
        self.view1 = _view1_augmentation(img_size, crop_min)
        self.view2 = _view2_augmentation(img_size, crop_min)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path, label = self.samples[index]
        img = self.loader(path)
        return self.view1(img), self.view2(img), label


def get_mocov3_dataloader(data_path, img_size, batch_size, num_workers,
                          crop_min=0.08, seed=0):
    """DataLoader for MoCo v3 pretraining: yields (view1, view2, label)."""
    dataset = MoCoV3Dataset(data_path, img_size=img_size, crop_min=crop_min)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
        generator=torch.Generator().manual_seed(seed))
    return loader, dataset
