"""
Dataset loaders for Context Encoder step 1 (AlexNet inpainting) and the linear
evaluation.

Trimmed during the port: the captured file also carried `InpaintingDatasetViT`
(the step 2 loader) and a Caffe BGR preprocessing path used only by the official
Caffe feature evaluation. Neither is part of this port, so both were dropped
along with an unused `numpy` import.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


class InpaintingDataset(Dataset):
    """
    Dataset for context encoder inpainting task (AlexNet-based)
    Creates a centered square mask in the image and returns the masked input,
    the original, the target (the centre region) and the binary mask.
    """
    def __init__(self, root, split='train', img_size=227, mask_size=128, transform=None):
        self.img_size = img_size
        self.mask_size = mask_size

        # Calculate mask position (center region)
        self.mask_start = (img_size - mask_size) // 2
        self.mask_end = self.mask_start + mask_size

        data_dir = os.path.join(root, split)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(img_size) if split == 'train' else transforms.CenterCrop(img_size),
                transforms.RandomHorizontalFlip() if split == 'train' else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        self.dataset = datasets.ImageFolder(data_dir, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]

        # Extract the center region to predict
        target = img[:, self.mask_start:self.mask_end, self.mask_start:self.mask_end].clone()

        # Create masked input (set center region to 0)
        masked_img = img.clone()
        masked_img[:, self.mask_start:self.mask_end, self.mask_start:self.mask_end] = 0

        # Create binary mask (1 for masked region, 0 for visible)
        mask = torch.zeros(1, self.img_size, self.img_size)
        mask[:, self.mask_start:self.mask_end, self.mask_start:self.mask_end] = 1

        return {
            'image': img,
            'masked_image': masked_img,
            'target': target,
            'mask': mask
        }


class ImageNetLinearProbe(Dataset):
    """ImageNet dataset for linear probing evaluation (torch preprocessing)."""
    def __init__(self, root, split='train', img_size=224, preprocess='torch'):
        if preprocess != 'torch':
            raise ValueError(
                f"preprocess={preprocess!r}: only 'torch' is available in this "
                "port (the Caffe path belonged to the official feature eval, "
                "which was not brought across)")
        data_dir = os.path.join(root, split)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if split == 'train':
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize
            ])
        self.dataset = datasets.ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def create_dataloader(dataset_type, root, split='train', batch_size=32,
                      num_workers=4, model_type='alexnet', **kwargs):
    """Factory for the step-1 inpainting loader and the linear-probe loader."""
    if dataset_type == 'inpainting':
        dataset = InpaintingDataset(root, split=split, **kwargs)
    elif dataset_type == 'linear_probe':
        dataset = ImageNetLinearProbe(root, split=split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    shuffle = (split == 'train')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle
    )
    return dataloader
