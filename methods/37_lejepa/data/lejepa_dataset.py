"""LeJEPA multi-view dataset + augmentation (Balestriero & LeCun, 2025).

Ported from the lab's own LeJEPA code. Each sample returns ``views`` independently
augmented crops of the same image, stacked as [V,C,H,W]; the collate then yields a
batch [N,V,C,H,W]. The augmentation is the standard SSL recipe (random resized
crop, colour jitter, grayscale, Gaussian blur, solarize, horizontal flip,
ImageNet normalisation). The port drops the DistributedSampler (single-process)
and threads a seeded DataLoader generator so a run is reproducible. ``val_transform``
is the deterministic resize + centre crop used for linear evaluation.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(img_size: int = 224, crop_scale=(0.08, 1.0),
                          color_jitter=(0.8, 0.8, 0.8, 0.2),
                          color_jitter_p: float = 0.8, grayscale_p: float = 0.2,
                          blur_p: float = 0.5, blur_kernel: int = 0,
                          solarize_p: float = 0.2, hflip_p: float = 0.5):
    if blur_kernel <= 0:
        blur_kernel = max(3, (img_size // 10) | 1)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, scale=tuple(crop_scale),
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomApply([transforms.ColorJitter(*color_jitter)],
                               p=float(color_jitter_p)),
        transforms.RandomGrayscale(p=float(grayscale_p)),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0))],
            p=float(blur_p)),
        transforms.RandomApply([transforms.RandomSolarize(threshold=128)],
                               p=float(solarize_p)),
        transforms.RandomHorizontalFlip(p=float(hflip_p)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def val_transform(img_size: int = 224) -> transforms.Compose:
    resize = int(round(img_size / 0.875))
    return transforms.Compose([
        transforms.Resize(resize,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MultiViewImageFolder(datasets.ImageFolder):
    """ImageFolder that returns ``views`` augmented crops of each image, stacked
    as [V,C,H,W], plus the label."""

    def __init__(self, root: str, view_transform, views: int):
        super().__init__(root)
        self.view_transform = view_transform
        self.views = int(views)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        img = self.loader(path)
        views = torch.stack([self.view_transform(img) for _ in range(self.views)])
        return views, target


def get_lejepa_dataloader(data_path: str, batch_size: int, views: int = 4,
                          num_workers: int = 0, img_size: int = 224,
                          seed: int = 0, **aug):
    """Single-process multi-view DataLoader (yields views [N,V,C,H,W], labels)."""
    dataset = MultiViewImageFolder(
        data_path, build_train_transform(img_size=img_size, **aug), views=views)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
        generator=torch.Generator().manual_seed(seed))
    return loader, dataset
