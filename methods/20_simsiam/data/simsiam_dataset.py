"""
SimSiam dataset: returns TWO independently-augmented views of the same image.

Step 1 augmentation (strict SimSiam / SimCLR-style, from facebookresearch/simsiam):
  RandomResizedCrop(224, scale=(0.2, 1.0))
  RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
  RandomGrayscale(p=0.2)
  RandomApply([GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5)
  RandomHorizontalFlip()
  ToTensor + ImageNet normalisation

  Note on order: ColorJitter → Grayscale → GaussianBlur → HFlip matches the
  original loader.py / main_simsiam.py in facebookresearch/simsiam exactly.

Step 2 augmentation (unified benchmark, type 1):
  RandomResizedCrop(224, scale=(0.2, 1.0))
  RandomHorizontalFlip()
  RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
  RandomGrayscale(p=0.2)
  ToTensor + ImageNet normalisation
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _step1_augmentation(img_size: int = 224) -> transforms.Compose:
    """
    Exact SimSiam augmentation from facebookresearch/simsiam main_simsiam.py.

    GaussianBlur kernel_size=23 covers the sigma range [0.1, 2.0] well for 224px.
    torchvision GaussianBlur uniformly samples sigma in the given range.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
        ], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _step2_augmentation(img_size: int = 224) -> transforms.Compose:
    """Unified Step 2 augmentation (type 1): no GaussianBlur."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


class SimSiamDataset(datasets.ImageFolder):
    """
    ImageFolder that returns (v1, v2, label) triples.

    Both views are independently drawn from the same augmentation pipeline,
    applied to the same image with different random seeds.
    """

    def __init__(self, root: str, transform: transforms.Compose):
        super().__init__(root, transform=None)
        self.simsiam_transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path, label = self.samples[index]
        img = self.loader(path)
        v1 = self.simsiam_transform(img)
        v2 = self.simsiam_transform(img)
        return v1, v2, label


def get_simsiam_dataloader(
    data_path: str,
    augmentation: str = "step1",
    batch_size: int = 64,
    num_workers: int = 8,
    img_size: int = 224,
    distributed: bool = False,
) -> Tuple[DataLoader, SimSiamDataset]:
    """
    Build SimSiam data loader for SSL pre-training.

    Args:
        data_path   : path to ImageNet train split  (…/ImageNet/train)
        augmentation: "step1" (SimSiam paper) or "step2" (unified)
        batch_size  : per-GPU batch size
        num_workers : DataLoader workers
        img_size    : crop size (224)
        distributed : use DistributedSampler for DDP
    """
    if augmentation == "step1":
        aug = _step1_augmentation(img_size)
    elif augmentation == "step2":
        aug = _step2_augmentation(img_size)
    else:
        raise ValueError(f"Unknown augmentation: {augmentation!r}")

    dataset = SimSiamDataset(data_path, transform=aug)
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
