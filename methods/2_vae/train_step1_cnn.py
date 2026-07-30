"""
Training script for VAE with CNN architecture (Step 1)
Following the original VAE paper (Kingma & Welling, 2013) MNIST settings
Adapted for ImageNet with multi-GPU support (8x H200)
"""

import os
import random
import sys
import time
import argparse
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vae_cnn import VAE_CNN, vae_loss
from data.vae_dataset import get_vae_dataloader


def is_main(args):
    return getattr(args, 'local_rank', 0) == 0


def assert_finite_tensor(name, tensor):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite tensor detected: {name}")


def assert_finite_model(model):
    raw_model = model.module if hasattr(model, 'module') else model
    for name, param in raw_model.named_parameters():
        assert_finite_tensor(f"parameter {name}", param.data)


def assert_finite_state_dict(state_dict):
    for name, value in state_dict.items():
        if torch.is_tensor(value):
            assert_finite_tensor(f"checkpoint tensor {name}", value)


def beta_for_epoch(config, epoch):
    beta = float(config['training']['beta'])
    warmup_epochs = int(config['training'].get('kl_warmup_epochs', 0))
    if warmup_epochs <= 0:
        return beta
    return beta * min(1.0, float(epoch) / float(warmup_epochs))


def parse_args():
    parser = argparse.ArgumentParser(description='Train VAE with CNN architecture (Step 1)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=0)
    # Note: local_rank is now read from environment variable LOCAL_RANK (set by torchrun)
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_distributed(args):
    """Setup distributed training"""
    if args.distributed:
        # Read local_rank from environment variable (set by torchrun)
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize process group
        dist.init_process_group(backend='nccl')
        
        # Get world size
        args.world_size = dist.get_world_size()
        
        # Set device
        torch.cuda.set_device(args.local_rank)
        
        if args.local_rank == 0:
            print(f"Distributed training initialized: {args.world_size} GPUs")
    else:
        args.local_rank = 0
        args.world_size = 1


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(state, checkpoint_dir, epoch_or_filename, is_best=False):
    """Save checkpoint (only on rank 0)"""
    if 'model_state_dict' in state:
        assert_finite_state_dict(state['model_state_dict'])
    loss = state.get('loss')
    if loss is not None and not torch.isfinite(torch.tensor(float(loss))):
        raise RuntimeError(f"Refusing to save non-finite checkpoint loss: {loss}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if isinstance(epoch_or_filename, int):
        filename = f'checkpoint_epoch_{epoch_or_filename}.pth'
    else:
        filename = str(epoch_or_filename)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    tmp_path = checkpoint_path + '.tmp'
    torch.save(state, tmp_path)
    os.replace(tmp_path, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
        torch.save(state, best_path)
        print(f"Best checkpoint saved to {best_path}")


def train_one_epoch(model, train_loader, optimizer, device, epoch, config, writer, args):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_div = 0.0
    
    start_time = time.time()
    
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)
        
        # Forward pass
        recon, original, mu, logvar = model(images)
        
        # Compute loss
        beta = beta_for_epoch(config, epoch)
        loss, recon_loss, kl_div = vae_loss(
            recon, original, mu, logvar, 
            beta=beta,
            logvar_clamp=tuple(config['training'].get('logvar_clamp', [-10.0, 10.0])),
        )
        assert_finite_tensor("vae loss", loss)
        assert_finite_tensor("reconstruction loss", recon_loss)
        assert_finite_tensor("kl divergence", kl_div)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert_finite_tensor(f"gradient {name}", param.grad)
        
        # Gradient clipping (if enabled)
        if config['training'].get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
        
        optimizer.step()
        assert_finite_model(model)
        
        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_div += kl_div.item()
        
        # Print progress (only on rank 0)
        if args.local_rank == 0 and (batch_idx + 1) % config['training']['print_freq'] == 0:
            avg_loss = total_loss / (batch_idx + 1)
            avg_recon = total_recon_loss / (batch_idx + 1)
            avg_kl = total_kl_div / (batch_idx + 1)
            
            print(f"Epoch [{epoch}/{config['training']['epochs']}] "
                  f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {avg_loss:.4f} "
                  f"Recon: {avg_recon:.4f} "
                  f"KL: {avg_kl:.4f} "
                  f"Beta: {beta:.4f}")
    
    # Calculate epoch statistics
    epoch_loss = total_loss / len(train_loader)
    epoch_recon = total_recon_loss / len(train_loader)
    epoch_kl = total_kl_div / len(train_loader)
    epoch_time = time.time() - start_time
    
    # Log to tensorboard (only on rank 0)
    if args.local_rank == 0 and writer is not None:
        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/recon_loss', epoch_recon, epoch)
        writer.add_scalar('train/kl_div', epoch_kl, epoch)
        writer.add_scalar('train/epoch_time', epoch_time, epoch)
    
    if args.local_rank == 0:
        print(f"\nEpoch [{epoch}/{config['training']['epochs']}] completed in {epoch_time:.2f}s")
        print(f"Average Loss: {epoch_loss:.4f}, Recon: {epoch_recon:.4f}, KL: {epoch_kl:.4f}\n")
    
    return epoch_loss


def make_deterministic() -> None:
    """Ask torch for reproducible kernels.

    Added during the port, for the same reason as in the first method: without
    it torch may choose a kernel by timing, and two runs of one config on one
    machine can differ. `warn_only` keeps an operation with no deterministic
    implementation warning in the run's own output instead of aborting.
    """
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(args, config=None):
    """The original body, with the config allowed to arrive as a value.

    A pure extraction apart from that. The captured config carries an absolute
    path on the cluster as `output.checkpoint_dir`; the contract says the
    adapter writes only under `--out`, so the adapter builds the config and
    passes it here rather than pointing at a file it would have to rewrite.
    """
    setup_distributed(args)

    seed = int(getattr(args, "seed", 0))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    make_deterministic()
    if config is None:
        config = load_config(args.config)
    if args.local_rank == 0:
        print("Configuration loaded:")
        print(yaml.dump(config, default_flow_style=False))
    
    # Set device
    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
    if args.local_rank == 0:
        print(f"Using device: {device}")
        if args.distributed:
            print(f"Distributed training: {args.world_size} GPUs")
            print(f"Effective batch size: {config['data']['batch_size']} × {args.world_size} = {config['data']['batch_size'] * args.world_size}")
    
    # Create model
    model = VAE_CNN(
        latent_dim=config['model']['latent_dim'],
        input_channels=3,
        image_size=config['data']['img_size']
    ).to(device)
    
    # Wrap model with DDP
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # Count parameters (only on rank 0)
    if args.local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer (following original VAE paper settings)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        betas=config['training'].get('betas', (0.9, 0.999)),
        weight_decay=config['training'].get('weight_decay', 0.0)
    )
    
    # Learning rate scheduler (constant as in original paper)
    scheduler = None
    if config['training'].get('lr_scheduler') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training'].get('min_lr', 0.0)
        )
    elif config['training'].get('lr_scheduler') == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training'].get('lr_decay_epochs', 100),
            gamma=config['training'].get('lr_decay_rate', 0.1)
        )
    # else: constant LR (as in original paper)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        if args.local_rank == 0:
            print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        assert_finite_state_dict(checkpoint['model_state_dict'])
        if args.distributed:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if args.local_rank == 0:
            print(f"Resumed from epoch {checkpoint['epoch']}")
        assert_finite_model(model)
    
    # Create data loader
    train_loader, train_sampler = get_vae_dataloader(
        data_path=args.data_path,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        img_size=config['data']['img_size'],
        augmentation_type=config['data'].get('augmentation_type', 'simple'),
        distributed=args.distributed,
        world_size=args.world_size,
        rank=args.local_rank
    )
    
    if args.local_rank == 0:
        print(f"Training dataset: {len(train_loader.dataset)} images")
        print(f"Number of batches per GPU: {len(train_loader)}")
    
    # Create tensorboard writer (only on rank 0)
    writer = None
    if args.local_rank == 0:
        log_dir = os.path.join(config['output']['checkpoint_dir'], 'logs')
        writer = SummaryWriter(log_dir=log_dir)
    
    # Training loop
    if args.local_rank == 0:
        print("\nStarting training...")
        print(f"Following original VAE paper settings:")
        print(f"  - Latent dim: {config['model']['latent_dim']} (paper used 2-50 for MNIST)")
        print(f"  - Optimizer: Adam (as in paper)")
        print(f"  - Learning rate: {config['training']['lr']}")
        print(f"  - Beta: {config['training']['beta']} (standard VAE)")
        print(f"  - No weight decay (as in paper)")
        print()
    
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        # Set epoch for distributed sampler
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train one epoch
        epoch_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, config, writer, args)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            if args.local_rank == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Learning rate: {current_lr:.6f}")
                writer.add_scalar('train/lr', current_lr, epoch)
        
        # Save a finite latest checkpoint every epoch during stabilization so
        # walltime/preemption does not discard all progress before epoch 100.
        if args.local_rank == 0:
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'config': config,
                'loss': epoch_loss
            }
            save_checkpoint(checkpoint_state, config['output']['checkpoint_dir'], 'checkpoint_latest.pth')
            if epoch % config['training']['save_freq'] == 0 or epoch == config['training']['epochs']:
                save_checkpoint(checkpoint_state, config['output']['checkpoint_dir'], epoch)
    
    if args.local_rank == 0:
        print("Training completed!")
        writer.close()
    
    # Cleanup distributed training
    cleanup_distributed()

    # Returned so a caller does not have to re-derive the result from the log.
    # `epoch_loss` is the mean loss over the last epoch, which is what the
    # training loop already reports.
    return {"epochs": epoch, "final_loss": float(epoch_loss)}


def main():
    run(parse_args())


if __name__ == '__main__':
    main()
