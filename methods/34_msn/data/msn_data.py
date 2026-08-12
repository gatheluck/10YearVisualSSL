"""MSN multi-view data pipeline (Assran et al., 2022; arXiv:2204.07141).

Reimplements the official `facebookresearch/msn` `make_transforms` recipe: each
image becomes `rand_views` large random-resized crops (rand_size, scale 0.3-1.0)
and `focal_views` small focal crops (focal_size, scale 0.05-0.3), each with
horizontal flip, colour distortion (ColorJitter + grayscale), Gaussian blur and
ImageNet normalisation.

Why reimplemented rather than imported: the upstream `make_transforms` uses a
custom `GaussianBlur` that passes a `torch.Tensor` radius to
`PIL.ImageFilter.GaussianBlur`, which the pinned Pillow (12.x) rejects. This port
uses `torchvision.transforms.GaussianBlur` (a faithful equivalent, sigma 0.1-2.0)
so the pipeline runs under the pinned environment. The upstream MSN model and loss
are used as-is (imported from third_party/msn); only this augmentation is rewritten.

The port drops the DistributedSampler (single-process) and threads a seeded
DataLoader generator so a run is reproducible.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import adapterlib

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _color_distortion(s: float = 1.0) -> transforms.Compose:
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return transforms.Compose([
        transforms.RandomApply([jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ])


def _view_transform(size: int, crop_scale, color_jitter: float,
                    blur_kernel: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=tuple(crop_scale)),
        transforms.RandomHorizontalFlip(),
        _color_distortion(s=color_jitter),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0))],
            p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


class MSNMultiViewTransform:
    """Returns rand_views large crops + focal_views focal crops, as a list."""

    def __init__(self, rand_size: int = 224, focal_size: int = 96,
                 rand_crop_scale=(0.3, 1.0), focal_crop_scale=(0.05, 0.3),
                 color_jitter: float = 1.0, rand_views: int = 1,
                 focal_views: int = 10):
        self.rand_views = int(rand_views)
        self.focal_views = int(focal_views)
        rand_k = max(3, (int(rand_size) // 20) | 1)
        focal_k = max(3, (int(focal_size) // 20) | 1)
        self.rand_transform = _view_transform(rand_size, rand_crop_scale,
                                              color_jitter, rand_k)
        self.focal_transform = _view_transform(focal_size, focal_crop_scale,
                                               color_jitter, focal_k)

    def __call__(self, img):
        views = [self.rand_transform(img) for _ in range(self.rand_views)]
        views += [self.focal_transform(img) for _ in range(self.focal_views)]
        return views


def val_transform(img_size: int = 224) -> transforms.Compose:
    """Deterministic resize + centre crop (for linear eval), ImageNet norm."""
    resize = int(round(img_size * 256 / 224))
    return transforms.Compose([
        transforms.Resize(resize,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def get_msn_dataloader(data_path: str, batch_size: int, rand_size: int = 224,
                       focal_size: int = 96, rand_crop_scale=(0.3, 1.0),
                       focal_crop_scale=(0.05, 0.3), color_jitter: float = 1.0,
                       rand_views: int = 1, focal_views: int = 10,
                       num_workers: int = 8, seed: int = 0):
    """Single-process MSN pretraining DataLoader.

    Yields (views, labels) where views is a list of (rand_views + focal_views)
    batched tensors -- torch's default_collate transposes the per-sample view
    lists into that shape, so no custom collate is needed.
    """
    transform = MSNMultiViewTransform(
        rand_size=rand_size, focal_size=focal_size,
        rand_crop_scale=rand_crop_scale, focal_crop_scale=focal_crop_scale,
        color_jitter=color_jitter, rand_views=rand_views, focal_views=focal_views)
    dataset = datasets.ImageFolder(
        adapterlib.dataset_split_dir(data_path, "train"), transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
        generator=torch.Generator().manual_seed(seed))
    return loader, dataset
