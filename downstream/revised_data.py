"""Fixed-224 readers for the explicit revised LoRA/DPT experiment profile.

The portable backbones accept ImageNet-normalized tensors. Normalize once here;
the inspected experimental readers returned raw RGB to their own adapters.
ADE augmentation intentionally uses Python RNG; ImageNet uses torchvision RNG.
These readers do not change the historical task readers' defaults.
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as F

_NORMALIZE = T.Normalize((.485, .456, .406), (.229, .224, .225))


def _split(split: str) -> bool:
    if split not in ('train', 'val'):
        raise ValueError('split must be train or val')
    return split == 'train'


def build_imagenet(root: Path, split: str) -> ImageFolder:
    """Build a sorted class-folder reader with the revised crop recipe."""
    train = _split(split)
    geometry = [T.RandomResizedCrop(
        224, scale=(.08, 1.), ratio=(.75, 4/3),
        interpolation=T.InterpolationMode.BICUBIC), T.RandomHorizontalFlip(.5)
    ] if train else [T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                     T.CenterCrop(224)]
    return ImageFolder(Path(root) / split,
                       transform=T.Compose([*geometry, T.ToTensor(), _NORMALIZE]))


class RevisedADE20K(Dataset):
    """ADE images with synchronized Python-RNG geometry and 150-class labels."""

    def __init__(self, root: Path, split: str):
        self.train = _split(split)
        tag = 'training' if self.train else 'validation'
        image_dir = Path(root) / 'images' / tag
        self.images = sorted(image_dir.glob('*.jpg'))
        if not self.images:
            raise FileNotFoundError(f'no ADE20K images under {image_dir}')
        self.masks = [Path(root) / 'annotations' / tag / image.with_suffix('.png').name
                      for image in self.images]
        for mask in self.masks:
            if not mask.is_file():
                raise FileNotFoundError(f'missing ADE20K mask: {mask}')

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        with Image.open(self.images[index]) as source:
            image = source.convert('RGB')
        with Image.open(self.masks[index]) as source:
            mask = source.copy()
        size = (224, 224)
        crop = None
        flip = False
        if self.train:
            scale = random.uniform(.5, 2.)
            size = (max(224, round(image.width * scale)),
                    max(224, round(image.height * scale)))
            top = random.randint(0, size[1] - 224)
            left = random.randint(0, size[0] - 224)
            crop = (left, top, left + 224, top + 224)
            flip = random.random() < .5
        image = image.resize(size, Image.Resampling.BICUBIC)
        mask = mask.resize(size, Image.Resampling.NEAREST)
        if crop is not None:
            image, mask = image.crop(crop), mask.crop(crop)
        if flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        labels = torch.from_numpy(np.array(mask, dtype=np.int64)) - 1
        labels[(labels < 0) | (labels >= 150)] = 255
        return _NORMALIZE(F.to_tensor(image)), labels
