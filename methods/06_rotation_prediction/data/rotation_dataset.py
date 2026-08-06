"""Rotation-prediction dataset (Gidaris et al., ICLR 2018), ported from the lab's
own implementation.

Each image is returned as its four right-angle rotations {0, 90, 180, 270} with
labels [0, 1, 2, 3], matching the official RotNet ImageNet dataloader. A collate
that flattens the [B, 4, ...] batch into [B*4, ...] is provided so one original
batch becomes four rotated training samples.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from torchvision.datasets import ImageFolder

# The four right-angle rotations, in the label order the paper uses.
ROTATIONS = (0, 90, 180, 270)


def rotate(img: torch.Tensor, angle: int) -> torch.Tensor:
    """Rotate a [C, H, W] tensor counter-clockwise by a right angle."""
    if angle == 0:
        return img
    if angle == 90:
        return torch.flip(img.transpose(1, 2), [1])
    if angle == 180:
        return torch.flip(img, [1, 2])
    if angle == 270:
        return torch.flip(img.transpose(1, 2), [2])
    raise ValueError(f"invalid rotation angle: {angle}; expected a right angle")


class RotationDataset(Dataset):
    """For each image, return its four rotations stacked as [4, C, H, W] and the
    labels [0, 1, 2, 3]."""

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 normalize: bool = True):
        self.dataset = ImageFolder(root)
        self.transform = transform
        self.normalize = normalize
        self.num_classes = len(ROTATIONS)
        self._normalize = transforms.Normalize(mean=self.IMAGENET_MEAN,
                                                std=self.IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, _ = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        rotated = [rotate(img, angle) for angle in ROTATIONS]
        if self.normalize:
            rotated = [self._normalize(x) for x in rotated]
        labels = torch.arange(self.num_classes, dtype=torch.long)
        return torch.stack(rotated, dim=0), labels


def rotation_collate(batch):
    """Flatten a batch of [4, C, H, W] items into [B*4, C, H, W] samples and
    their [B*4] labels, so each original image contributes four training
    examples."""
    images, labels = default_collate(batch)
    bsz, rot, c, h, w = images.shape
    return images.view(bsz * rot, c, h, w), labels.view(bsz * rot)
