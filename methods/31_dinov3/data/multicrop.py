"""
Multi-crop data augmentation for DINOv3 / DINOv2 training.

DINOv3 uses:
  2 global crops : 256x256 (adapted to 224x224 for ImageNet-1k Step 2)
                   scale = (0.32, 1.0),  strong color augmentation
  8 local crops  : 112x112 (adapted to 96x96 for Step 2)
                   scale = (0.05, 0.32), same color augmentation

For Step 2 (unified settings, ImageNet-1k):
  global crop: 224x224, scale=(0.32, 1.0)
  local  crop: 96x96,   scale=(0.05, 0.32)
  augmentations: RandomResizedCrop, RandomHFlip, ColorJitter, RandomGrayscale,
                 GaussianBlur, Solarization (for second global crop only)

Reference: DINOv2 (Oquab et al., 2024), Algorithm 1 / Appendix A.
"""

import random
from typing import List, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import ImageFolder

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class GaussianBlur:
    """Apply Gaussian blur with random kernel size and sigma."""
    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(self.radius_min, self.radius_max)
            img = TF.gaussian_blur(img, kernel_size=9, sigma=sigma)
        return img


class Solarization:
    """Apply solarization (invert pixels above threshold) with probability p."""
    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = TF.solarize(img, threshold=128)
        return img


def _make_color_jitter() -> T.ColorJitter:
    return T.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.2,
        hue=0.1,
    )


def _global_crop_transform(
    global_size: int,
    scale: Tuple[float, float],
    apply_solarization: bool = False,
    blur_probability: float = 1.0,
) -> T.Compose:
    """Build augmentation pipeline for a single global crop."""
    transforms = [
        T.RandomResizedCrop(
            global_size,
            scale=scale,
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([_make_color_jitter()], p=0.8),
        T.RandomGrayscale(p=0.2),
        GaussianBlur(p=blur_probability),
    ]
    if apply_solarization:
        transforms.append(Solarization(p=0.2))
    transforms += [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(transforms)


def _local_crop_transform(
    local_size: int,
    scale: Tuple[float, float],
) -> T.Compose:
    """Build augmentation pipeline for a single local crop."""
    return T.Compose([
        T.RandomResizedCrop(
            local_size,
            scale=scale,
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([_make_color_jitter()], p=0.8),
        T.RandomGrayscale(p=0.2),
        GaussianBlur(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class MultiCropAugmentation:
    """
    Multi-crop augmentation pipeline.

    Returns n_global + n_local views for each image.
    Global crops: strong augmentation with optional solarization.
    Local  crops: moderate augmentation.

    Args:
        global_size     : Spatial resolution of global crops (224 for Step 2).
        local_size      : Spatial resolution of local crops  (96  for Step 2).
        global_scale    : (min, max) scale range for global crops.
        local_scale     : (min, max) scale range for local  crops.
        n_global        : Number of global crops (2 in DINOv3).
        n_local         : Number of local  crops (8 in DINOv3).
        return_gram_teacher_crops: Return undistorted companions that share
                                   each global crop's geometry.
    """

    def __init__(
        self,
        global_size: int = 224,
        local_size: int = 96,
        global_scale: Tuple[float, float] = (0.32, 1.0),
        local_scale: Tuple[float, float] = (0.05, 0.32),
        n_global: int = 2,
        n_local: int = 8,
        return_gram_teacher_crops: bool = False,
    ):
        self.n_global = n_global
        self.n_local = n_local
        self.return_gram_teacher_crops = return_gram_teacher_crops

        self.global_geometric_transforms = [
            T.Compose(
                [
                    T.RandomResizedCrop(
                        global_size,
                        scale=global_scale,
                        interpolation=T.InterpolationMode.BICUBIC,
                    ),
                    T.RandomHorizontalFlip(p=0.5),
                ]
            )
            for _ in range(n_global)
        ]
        self.normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.global_distortions = [
            T.Compose(
                [
                    T.RandomApply([_make_color_jitter()], p=0.8),
                    T.RandomGrayscale(p=0.2),
                    GaussianBlur(p=0.1 if index == 1 else 1.0),
                    *([Solarization(p=0.2)] if index == 1 else []),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
            for index in range(n_global)
        ]

        # Retained as public full transforms for diagnostics and parity checks.
        self.global_transforms: List[T.Compose] = []
        for i in range(n_global):
            self.global_transforms.append(
                _global_crop_transform(global_size, global_scale,
                                       apply_solarization=(i == 1),
                                       blur_probability=0.1 if i == 1 else 1.0)
            )

        self.local_transform = _local_crop_transform(local_size, local_scale)

    def __call__(self, image):
        """
        Args:
            image: PIL Image.
        Returns a view list, or a dictionary containing that list and clean
        global companions for a separate Gram teacher.
        """
        views = []
        gram_teacher_views = []
        for geometric, distortion in zip(
            self.global_geometric_transforms, self.global_distortions
        ):
            geometric_crop = geometric(image)
            views.append(distortion(geometric_crop))
            if self.return_gram_teacher_crops:
                gram_teacher_views.append(self.normalize(geometric_crop))
        for _ in range(self.n_local):
            views.append(self.local_transform(image))
        if self.return_gram_teacher_crops:
            return {
                "views": views,
                "gram_teacher_views": gram_teacher_views,
            }
        return views


def collate_multicrop(batch: list):
    """
    Custom collate: transpose list-of-views into view-list-of-batches.

    Input  batch: list of (views_list, label) where views_list has length V.
    Output: (views, labels)
        views : list of V tensors each (B, C, H, W)
        labels: (B,) long tensor
    """
    has_gram_teacher_views = isinstance(batch[0][0], dict)
    views_per_sample = [
        item[0]["views"] if has_gram_teacher_views else item[0]
        for item in batch
    ]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    n_views = len(views_per_sample[0])
    views = [
        torch.stack([views_per_sample[i][v] for i in range(len(batch))])
        for v in range(n_views)
    ]
    if not has_gram_teacher_views:
        return views, labels

    gram_views_per_sample = [item[0]["gram_teacher_views"] for item in batch]
    n_gram_views = len(gram_views_per_sample[0])
    gram_teacher_views = [
        torch.stack(
            [gram_views_per_sample[index][view] for index in range(len(batch))]
        )
        for view in range(n_gram_views)
    ]
    return views, gram_teacher_views, labels


def get_multicrop_dataloader(
    data_path: str,
    augmentation: MultiCropAugmentation,
    batch_size: int,
    num_workers: int = 8,
    distributed: bool = False,
    drop_last: bool = True,
    pin_memory: bool = True,
    seed: int | None = None,
) -> DataLoader:
    """
    Build DataLoader for multi-crop SSL pre-training.

    Args:
        data_path    : Path to ImageNet train split (ImageFolder structure).
        augmentation : MultiCropAugmentation instance.
        batch_size   : Per-GPU batch size.
        num_workers  : DataLoader workers per GPU.
        distributed  : Use DistributedSampler if True.
        drop_last    : Drop incomplete last batch.
        pin_memory   : Pin memory for GPU transfer.

    Returns:
        DataLoader with collate_multicrop.
    """
    dataset = ImageFolder(root=data_path, transform=augmentation)

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=drop_last)
        shuffle = False
    else:
        shuffle = True

    generator = None
    if seed is not None and sampler is None:
        generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_multicrop,
        generator=generator,
    )
    return loader
