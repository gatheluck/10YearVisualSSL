"""ImageNet dataloader for PIRL step 1 (Misra & van der Maaten, CVPR 2020).

Ported from the lab's own code. Each sample returns an ordinary image view and a
jigsaw-style transformed view (nine shuffled 3x3 patches) plus the stable
ImageFolder index used by the memory bank. The port drops the DistributedSampler
(single-process) and threads a seeded generator so a run is reproducible.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import adapterlib


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class JigsawViewTransform:
    """Build a PIRL-style transformed view from shuffled 3x3 patches."""

    def __init__(self, resize: int = 256, crop_size: int = 255,
                 grid_size: int = 3, patch_size: int = 64, train: bool = True):
        if crop_size % grid_size != 0:
            raise ValueError("crop_size must be divisible by grid_size")
        cell_size = crop_size // grid_size
        if patch_size > cell_size:
            raise ValueError("patch_size must be <= crop_size / grid_size")

        self.resize = resize
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.cell_size = cell_size
        self.train = train
        self.resize_op = transforms.Resize(
            resize, interpolation=transforms.InterpolationMode.BILINEAR)
        self.crop_op = (transforms.RandomCrop(crop_size) if train
                        else transforms.CenterCrop(crop_size))
        self.color = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def _patch_crop_box(self):
        slack = self.cell_size - self.patch_size
        if self.train:
            dx = random.randint(0, slack) if slack > 0 else 0
            dy = random.randint(0, slack) if slack > 0 else 0
        else:
            dx = slack // 2
            dy = slack // 2
        return dx, dy

    def __call__(self, image: "Image.Image") -> torch.Tensor:
        image = self.crop_op(self.resize_op(image))

        patches = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                left = col * self.cell_size
                top = row * self.cell_size
                cell = image.crop((left, top, left + self.cell_size,
                                   top + self.cell_size))
                dx, dy = self._patch_crop_box()
                patch = cell.crop((dx, dy, dx + self.patch_size,
                                   dy + self.patch_size))
                if self.train:
                    patch = self.color(patch)
                patches.append(self.to_tensor(patch))

        if self.train:
            random.shuffle(patches)
        return torch.stack(patches, dim=0)


class ImageNetPIRLDataset(datasets.ImageFolder):
    """ImageFolder returning (image_view, jigsaw_view, index, class_label)."""

    def __init__(self, root: str, image_size: int = 224,
                 jigsaw_resize: int = 256, jigsaw_crop_size: int = 255,
                 jigsaw_grid_size: int = 3, jigsaw_patch_size: int = 64,
                 train: bool = True):
        self.train = train
        self.image_transform = self._build_image_transform(image_size, train)
        self.jigsaw_transform = JigsawViewTransform(
            resize=jigsaw_resize, crop_size=jigsaw_crop_size,
            grid_size=jigsaw_grid_size, patch_size=jigsaw_patch_size,
            train=train)
        super().__init__(root=adapterlib.dataset_split_dir(root, "train"))

    @staticmethod
    def _build_image_transform(image_size: int, train: bool):
        normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        return transforms.Compose([
            transforms.Resize(256,
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = self.loader(path).convert("RGB")
        return (self.image_transform(image), self.jigsaw_transform(image),
                index, target)


def build_pirl_loader(data_path: str, cfg: dict, train: bool = True,
                      batch_size: int = 256, num_workers: Optional[int] = None,
                      seed: int = 0):
    """Single-process PIRL DataLoader yielding (image, patches, index, label)."""
    dataset = ImageNetPIRLDataset(
        root=data_path, image_size=cfg.get("image_size", 224),
        jigsaw_resize=cfg.get("jigsaw_resize", 256),
        jigsaw_crop_size=cfg.get("jigsaw_crop_size", 255),
        jigsaw_grid_size=cfg.get("jigsaw_grid_size", 3),
        jigsaw_patch_size=cfg.get("jigsaw_patch_size", 64), train=train)

    nw = cfg.get("num_workers", 8) if num_workers is None else num_workers
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=nw,
        pin_memory=True, drop_last=train,
        generator=torch.Generator().manual_seed(seed))
    return loader, dataset
