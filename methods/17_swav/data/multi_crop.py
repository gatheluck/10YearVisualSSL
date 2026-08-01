"""
Multi-crop augmentation dataset for SwAV.
Follows Caron et al. (2020) NeurIPS exactly:
  Global crops (×2): 224×224, scale [0.14, 1.0]
    - RandomResizedCrop, HFlip, ColorJitter(0.8,0.8,0.8,0.2,p=0.8),
      Grayscale(p=0.2), GaussianBlur(p=0.5)
  Local crops (×6):  96×96,  scale [0.05, 0.14]
    - RandomResizedCrop, HFlip, ColorJitter(0.8,0.8,0.8,0.2,p=0.8),
      Grayscale(p=0.2), GaussianBlur(p=0.5)

Returns a list of tensors: [global_1, global_2, local_1, ..., local_6]
"""

import random
from PIL import Image, ImageFilter

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class GaussianBlur:
    """Apply Gaussian blur with random radius."""

    def __init__(self, radius_min: float = 0.1, radius_max: float = 2.0):
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img: Image.Image) -> Image.Image:
        radius = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _make_transform(crop_size: int, min_scale: float, max_scale: float,
                    color_jitter_strength: float = 1.0) -> transforms.Compose:
    """Build a single-crop augmentation pipeline."""
    s = color_jitter_strength
    color_jitter = transforms.ColorJitter(
        brightness=0.8 * s, contrast=0.8 * s, saturation=0.8 * s, hue=0.2 * s
    )
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
    )
    return transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=(min_scale, max_scale)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur(0.1, 2.0)], p=0.5),
        transforms.ToTensor(),
        normalize,
    ])


class MultiCropDataset(Dataset):
    """
    Wraps an ImageFolder dataset and returns multi-crop views.

    size_crops     : list of crop sizes,  e.g. [224, 96]
    nmb_crops      : number of crops per size, e.g. [2, 6]
    min_scale_crops: min scale per size,   e.g. [0.14, 0.05]
    max_scale_crops: max scale per size,   e.g. [1.0,  0.14]
    """

    def __init__(
        self,
        data_path: str,
        size_crops: list,
        nmb_crops: list,
        min_scale_crops: list,
        max_scale_crops: list,
        color_jitter_strength: float = 1.0,
    ):
        assert len(size_crops) == len(nmb_crops) == len(min_scale_crops) == len(max_scale_crops)
        self.dataset = datasets.ImageFolder(data_path)
        self.trans = []
        for size, min_s, max_s in zip(size_crops, min_scale_crops, max_scale_crops):
            self.trans.append(
                _make_transform(size, min_s, max_s, color_jitter_strength)
            )
        self.nmb_crops = nmb_crops

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        multi_crops = []
        for transform, n in zip(self.trans, self.nmb_crops):
            multi_crops.extend([transform(img) for _ in range(n)])
        return multi_crops, label


def get_swav_dataloader(
    data_path: str,
    size_crops: list,
    nmb_crops: list,
    min_scale_crops: list,
    max_scale_crops: list,
    batch_size: int,
    num_workers: int = 8,
    color_jitter_strength: float = 1.0,
    distributed: bool = True,
):
    dataset = MultiCropDataset(
        data_path=data_path,
        size_crops=size_crops,
        nmb_crops=nmb_crops,
        min_scale_crops=min_scale_crops,
        max_scale_crops=max_scale_crops,
        color_jitter_strength=color_jitter_strength,
    )
    # **Conditional, and it was not.** The captured loader always built a
    # DistributedSampler, which needs an initialised process group -- so this
    # method could not run on one process at all, on a GPU or otherwise. Every
    # other method's loader already guards this the same way.
    sampler = (torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=True) if distributed else None)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler
