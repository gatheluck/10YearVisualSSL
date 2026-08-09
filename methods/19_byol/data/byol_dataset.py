"""Dataset and augmentation for BYOL (Grill et al., 2020), ported from the lab's
own implementation.

BYOL uses **asymmetric** augmentation for the two views (Appendix B):
  View 1: RandomResizedCrop + Flip + ColorJitter + Grayscale + GaussianBlur(p=1.0)
  View 2: ... + GaussianBlur(p=0.1) + Solarize(p=0.2)
Both end with ToTensor + ImageNet normalisation. The dataset yields
``((view1, view2), label)``.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GaussianBlur:
    """Gaussian blur with a kernel ~10% of the image size (BYOL / SimCLR)."""

    def __init__(self, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, x):
        sigma = torch.empty(1).uniform_(self.sigma_min, self.sigma_max).item()
        kernel_size = int(0.1 * x.size[0])
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(kernel_size, 3)
        return transforms.functional.gaussian_blur(x, kernel_size, sigma)


class Solarize:
    """Solarize: invert pixel values above a threshold."""

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, x):
        return transforms.functional.solarize(x, self.threshold)


class BYOLTwoViewTransform:
    """Two asymmetric BYOL views of one image."""

    def __init__(self, img_size=224, augmentation="byol"):
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                              saturation=0.2, hue=0.1)
        base = [
            transforms.RandomResizedCrop(
                img_size, scale=(0.08, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        if augmentation in ("byol", "original"):
            self.transform1 = transforms.Compose(base + [
                transforms.RandomApply([GaussianBlur(0.1, 2.0)], p=1.0),
                transforms.ToTensor(), normalize])
            self.transform2 = transforms.Compose(base + [
                transforms.RandomApply([GaussianBlur(0.1, 2.0)], p=0.1),
                transforms.RandomApply([Solarize(128)], p=0.2),
                transforms.ToTensor(), normalize])
        else:
            raise ValueError(f"Unknown augmentation type: {augmentation}")

    def __call__(self, x):
        return self.transform1(x), self.transform2(x)


def get_linear_eval_transform(img_size=224, mode="val"):
    """Deterministic val transform (resize + centre crop, ImageNet norm)."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return transforms.Compose([
        transforms.Resize(int(round(img_size * 256 / 224))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(), normalize])


def get_byol_dataloader(data_path, batch_size=512, num_workers=8, img_size=224,
                        augmentation="byol", seed=0):
    """DataLoader for BYOL pre-training: yields ((view1, view2), label)."""
    dataset = ImageFolder(
        data_path,
        transform=BYOLTwoViewTransform(img_size=img_size,
                                       augmentation=augmentation))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
        generator=torch.Generator().manual_seed(seed))
    return loader, dataset
