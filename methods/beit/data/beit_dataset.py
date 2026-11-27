"""BEiT dual-view dataset (Bao et al., 2021), ported from the lab's code.

Each sample returns a 224px patch view (ImageNet-normalised, for the ViT), a
112px token view (DALL-E map_pixels-normalised once, for the dVAE) sharing the
same crop + flip, and a blockwise mask. The port drops the DistributedSampler
(single-process) and threads a seeded generator so a run is reproducible.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from PIL import Image

import adapterlib

from .masking import BEiTMaskingGenerator

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _map_pixels_tensor(x: torch.Tensor) -> torch.Tensor:
    return 0.8 * x + 0.1


def val_transform(img_size: int = 224) -> transforms.Compose:
    """Deterministic resize + centre crop (for linear eval), ImageNet norm."""
    return transforms.Compose([
        transforms.Resize(int(round(img_size * 256 / 224)),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


class BEiTDualTransform:
    """Same crop + flip -> a 224 patch view and a 112 token view."""

    def __init__(self, img_size: int = 224, token_size: int = 112,
                 crop_scale: tuple = (0.67, 1.0),
                 color_jitter_strength: float = 0.4):
        self.img_size = img_size
        self.token_size = token_size
        self.crop_scale = crop_scale
        self.color_jitter = transforms.ColorJitter(
            brightness=color_jitter_strength, contrast=color_jitter_strength,
            saturation=color_jitter_strength, hue=0.0)
        self.to_tensor = transforms.ToTensor()
        self.patch_norm = transforms.Normalize(mean=_IMAGENET_MEAN,
                                               std=_IMAGENET_STD)

    def __call__(self, img: "Image.Image"):
        img = self.color_jitter(img)
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img, scale=self.crop_scale, ratio=(3.0 / 4.0, 4.0 / 3.0))
        patch_pil = TF.resized_crop(
            img, i, j, h, w, size=(self.img_size, self.img_size),
            interpolation=TF.InterpolationMode.BICUBIC)
        token_pil = TF.resized_crop(
            img, i, j, h, w, size=(self.token_size, self.token_size),
            interpolation=TF.InterpolationMode.LANCZOS)
        if torch.rand(1).item() > 0.5:
            patch_pil = TF.hflip(patch_pil)
            token_pil = TF.hflip(token_pil)
        patch_t = self.patch_norm(self.to_tensor(patch_pil))
        token_t = _map_pixels_tensor(self.to_tensor(token_pil))
        return patch_t, token_t


class BEiTPretrainDataset(torch.utils.data.Dataset):
    """ImageFolder + BEiT dual-view augmentation + masking. Returns
    (patch_img, token_img, mask, label)."""

    def __init__(self, root: str, img_size: int = 224, patch_size: int = 16,
                 token_size: int = 112, num_masking_patches: int = 75,
                 min_masking_patches: int = 16, crop_scale: tuple = (0.67, 1.0)):
        self.dataset = datasets.ImageFolder(
            adapterlib.dataset_split_dir(root, "train"))
        self.dual_transform = BEiTDualTransform(
            img_size=img_size, token_size=token_size, crop_scale=crop_scale)
        side = img_size // patch_size
        self.masking_gen = BEiTMaskingGenerator(
            input_size=(side, side), num_masking_patches=num_masking_patches,
            min_num_patches=min_masking_patches)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        patch_img, token_img = self.dual_transform(img)
        mask = torch.from_numpy(self.masking_gen())
        return patch_img, token_img, mask, label


def beit_collate_fn(batch):
    patch_imgs = torch.stack([b[0] for b in batch])
    token_imgs = torch.stack([b[1] for b in batch])
    masks = torch.stack([b[2] for b in batch])
    labels = torch.tensor([b[3] for b in batch])
    return patch_imgs, token_imgs, masks, labels


def get_beit_dataloader(data_path: str, batch_size: int, num_workers: int = 8,
                        img_size: int = 224, patch_size: int = 16,
                        token_size: int = 112, num_masking_patches: int = 75,
                        min_masking_patches: int = 16, seed: int = 0):
    """Single-process BEiT pretraining DataLoader (yields patch/token/mask/label)."""
    dataset = BEiTPretrainDataset(
        root=data_path, img_size=img_size, patch_size=patch_size,
        token_size=token_size, num_masking_patches=num_masking_patches,
        min_masking_patches=min_masking_patches)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, collate_fn=beit_collate_fn,
        generator=torch.Generator().manual_seed(seed))
    return loader, dataset
