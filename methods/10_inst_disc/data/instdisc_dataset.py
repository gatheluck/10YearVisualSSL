"""ImageNet dataset wrapper for Instance Discrimination.

Each item is `(image, dataset_index, label)`: the dataset index is what the NCE
memory bank is keyed on, so every image is its own class. The lab wrapper's
DistributedSampler branch is dropped for the single-process port.
"""

from __future__ import annotations

from torchvision import datasets, transforms

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def get_instdisc_transforms(mode: str = "train", img_size: int = 224):
    """Augmentation following Wu et al. (2018):
      train: RandomResizedCrop + RandomGrayscale + ColorJitter + HFlip
      val:   Resize(256) + CenterCrop(img_size)."""
    norm = transforms.Normalize(mean=_MEAN, std=_STD)
    if mode == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        norm,
    ])


class ImageFolderWithIndex(datasets.ImageFolder):
    """ImageFolder that returns (image, dataset_index, label)."""

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return img, index, label
