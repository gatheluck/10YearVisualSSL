"""
Dataset and DataLoader for VAE training on ImageNet
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_vae_dataloader(
    data_path,
    batch_size=256,
    num_workers=8,
    img_size=224,
    augmentation_type='simple',
    distributed=False,
    world_size=1,
    rank=0
):
    """
    Create DataLoader for VAE training
    
    Args:
        data_path: Path to dataset (ImageNet or MNIST)
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        img_size: Image size (default: 224)
        augmentation_type: Type of augmentation ('simple' or 'advanced')
        distributed: Whether using distributed training
        world_size: Number of GPUs
        rank: GPU rank
    
    Returns:
        train_loader: DataLoader for training
    """
    
    # Detect if we're using MNIST or ImageNet
    is_mnist = 'MNIST' in data_path or os.path.exists(os.path.join(data_path, 'MNIST'))
    
    # No augmentation (original MNIST paper - no augmentation at all)
    if augmentation_type == 'none':
        if is_mnist and img_size == 28:
            # For MNIST at original resolution, no resizing needed
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                # MNIST is already grayscale, convert to 3-channel for consistency
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                # Normalize to [0, 1] for VAE reconstruction
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                # For MNIST, convert grayscale to 3-channel
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                # Normalize to [0, 1] for VAE reconstruction
            ])
    
    # Data augmentation for Step 1 (minimal augmentation, following original VAE paper)
    elif augmentation_type == 'minimal':
        transform_list = [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
        if is_mnist:
            transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x))
        train_transform = transforms.Compose(transform_list)
    
    # Data augmentation for Step 1 (simple augmentation)
    elif augmentation_type == 'simple':
        transform_list = [
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
        ]
        if is_mnist:
            transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x))
        train_transform = transforms.Compose(transform_list)
    
    # Data augmentation for Step 2 (advanced augmentation)
    elif augmentation_type == 'advanced':
        transform_list = [
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]
        if is_mnist:
            transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x))
        train_transform = transforms.Compose(transform_list)
    
    else:
        raise ValueError(f"Unknown augmentation_type: {augmentation_type}")
    
    # Create dataset
    if is_mnist:
        # Use MNIST dataset
        train_dataset = datasets.MNIST(
            root=data_path,
            train=True,
            download=False,  # Already downloaded
            transform=train_transform
        )
        print(f"MNIST dataset loaded with {len(train_dataset)} images")
    else:
        # Use ImageFolder for ImageNet
        train_dir = os.path.join(data_path, 'train')
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        print(f"ImageNet dataset created with {len(train_dataset)} images")
    
    # Create sampler for distributed training
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader, train_sampler if distributed else None


if __name__ == '__main__':
    # Test the dataloader
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to ImageNet dataset')
    args = parser.parse_args()
    
    # Create dataloader
    train_loader, _ = get_vae_dataloader(
        data_path=args.data_path,
        batch_size=4,
        num_workers=2,
        augmentation_type='simple'
    )
    
    print(f"DataLoader created with {len(train_loader)} batches")
    
    # Test one batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        break
    
    print("DataLoader test passed!")

