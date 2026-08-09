"""SimCLR v2 two-view dataset (Chen et al., 2020), ported from the lab's own
implementation.

Returns two independently-augmented views (view1, view2) of the same image plus
its label. The augmentation is the exact SimCLR recipe (Appendix B):
RandomResizedCrop(scale 0.08-1.0, bicubic), RandomHorizontalFlip, ColorJitter
(brightness/contrast/saturation 0.8s, hue 0.2s) with p=0.8, RandomGrayscale(0.2),
GaussianBlur(p=0.5, sigma 0.1-2.0, kernel ~10% of the image). SimCLR uses **no**
ImageNet mean/std normalisation -- the pipeline ends at ToTensor.

The lab wrapper returns ``((view1, view2), label)`` via a TwoViewTransform on an
ImageFolder; this port returns ``(view1, view2, label)`` so the trainer reads
each view directly, the same shape as the other two-view ports.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


def get_simclrv2_augmentation(img_size: int = 224,
                            color_jitter_strength: float = 1.0
                            ) -> transforms.Compose:
    """SimCLR augmentation pipeline (Appendix B, Chen et al. 2020)."""
    s = color_jitter_strength
    # GaussianBlur kernel must be odd; ~10% of the image size.
    kernel = max(3, int(round(img_size * 0.1)) | 1)   # OR 1 to force odd
    color_jitter = transforms.ColorJitter(
        brightness=0.8 * s, contrast=0.8 * s, saturation=0.8 * s,
        hue=min(0.2 * s, 0.5))
    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, scale=(0.08, 1.0),
            interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=kernel, sigma=(0.1, 2.0))],
            p=0.5),
        transforms.ToTensor(),
    ])


class SimCLRv2Dataset(datasets.ImageFolder):
    """ImageFolder returning ``(view1, view2, label)`` for SimCLR pretraining."""

    def __init__(self, root: str, image_size: int = 224,
                 color_jitter_strength: float = 1.0):
        super().__init__(root, transform=None)
        self.simclrv2_transform = get_simclrv2_augmentation(
            image_size, color_jitter_strength)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path, label = self.samples[index]
        img = self.loader(path)
        view1 = self.simclrv2_transform(img)
        view2 = self.simclrv2_transform(img)
        return view1, view2, label
