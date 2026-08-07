"""ImageNet dataset for the visual CPC 2018 protocol, ported from the lab's own
implementation.

Each image becomes a grid of overlapping patches (7x7 of 64x64 patches from a
256x256 crop in the paper). During pretraining, each patch receives an
independent random sub-crop padded back to the patch size.

The capture hard-codes a 7x7 grid check; this port relaxes it to any grid of at
least 2x2 so a small hermetic CPU smoke can run (the paper's 7x7 geometry is
still what the shipped config asks for). The DistributedSampler branch is dropped
for the single-process port.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PatchCropPad:
    """Random crop from every patch, padded back to the patch size."""

    def __init__(self, crop_size: int = 60, patch_size: int = 64):
        self.crop_size = crop_size
        self.patch_size = patch_size

    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        if self.crop_size >= self.patch_size:
            return patch
        max_off = self.patch_size - self.crop_size
        top = random.randint(0, max_off)
        left = random.randint(0, max_off)
        patch = patch[:, top:top + self.crop_size, left:left + self.crop_size]
        pad_top = random.randint(0, max_off)
        pad_left = random.randint(0, max_off)
        return F.pad(patch, (pad_left, max_off - pad_left,
                             pad_top, max_off - pad_top))


class VisualCPC2018Dataset(Dataset):
    """ImageFolder wrapper returning a grid of patches per image."""

    def __init__(self, root: str, mode: str = "train", image_size: int = 256,
                 source_size: int = 300, patch_size: int = 64,
                 patch_crop_size: int = 60, stride: int = 32):
        self.mode = mode
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.patch_aug = (PatchCropPad(patch_crop_size, patch_size)
                          if mode == "train" else None)

        normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        resize = transforms.Resize(
            (source_size, source_size),
            interpolation=transforms.InterpolationMode.BILINEAR)
        if mode == "train":
            transform = transforms.Compose([
                resize,
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                resize,
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])

        self.base = datasets.ImageFolder(root, transform=transform)
        grid = (image_size - patch_size) // stride + 1
        if grid < 2:
            raise ValueError(
                f"visual CPC needs at least a 2x2 grid, got {grid}x{grid} for "
                f"image_size={image_size}, patch_size={patch_size}, "
                f"stride={stride}")
        self.grid_size = grid

    @property
    def classes(self):
        return self.base.classes

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image, label = self.base[index]
        return self._extract_grid(image), label

    def _extract_grid(self, image: torch.Tensor) -> torch.Tensor:
        rows = []
        for r in range(self.grid_size):
            cols = []
            for c in range(self.grid_size):
                top, left = r * self.stride, c * self.stride
                patch = image[:, top:top + self.patch_size,
                              left:left + self.patch_size]
                if self.patch_aug is not None:
                    patch = self.patch_aug(patch)
                cols.append(patch)
            rows.append(torch.stack(cols, dim=0))
        return torch.stack(rows, dim=0)
