"""
SimSiam Step 1 training: ResNet-50 on ImageNet-1k.

Strictly follows Chen & He (2020) arXiv:2011.10566:
  Encoder    : ResNet-50 (zero_init_residual=True, weights=None)
  Projector  : 3-layer MLP (2048→2048→2048) BN+ReLU; output BN(affine=False)
  Predictor  : 2-layer bottleneck (2048→512→2048) BN+ReLU; no output BN
  Loss       : -0.5 * [cos(p1, sg(z2)) + cos(p2, sg(z1))]
  Dataset    : ImageNet-1k (1.28M images)
  Epochs     : 100  (checkpoints every 50 epochs)
  Batch      : 512 total (64 per GPU × 8 H200)
  Optimizer  : SGD  init_lr=0.1 (base_lr=0.05 × 512/256)
               momentum=0.9, weight_decay=1e-4
  LR sched   : cosine decay (NO warmup); predictor LR is FIXED at init_lr
  Augment    : RandomResizedCrop + ColorJitter(p=0.8) + Grayscale(p=0.2) +
               GaussianBlur(p=0.5) + HFlip  (SimCLR-style)

Collapse monitor: we track the std of normalized z vectors.
  - Healthy: std ≈ 1/sqrt(dim) ≈ 0.022 for dim=2048
  - Collapsed: std ≈ 0

Supports multi-GPU via DDP (torchrun --nproc_per_node=8).

Changed during the port, and recorded in provenance.json:

  - **the device is resolved instead of assumed.** The captured trainer sent
    its batches and its model to CUDA unconditionally, so it could not start
    on a machine without a GPU. `resolve_device` picks one; nothing else about
    the computation changed
  - **`main()` is split into `parse_args()` and `run(args, config)`,** so an
    adapter can call the training loop in-process and be handed the metrics.
    The captured `main()` computed the epoch loss and the collapse monitor and
    then discarded both
  - **the run is seeded.** The captured trainer seeded torch only, which is in
    fact enough here -- every augmentation is a stock torchvision transform
    and those draw from torch's generator, which was measured rather than
    assumed. `random` is seeded as well, so that a transform added later
    cannot break reproducibility in silence
"""

import os
import random
import sys
import time
import yaml
import argparse
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models import build_simsiam_resnet, simsiam_loss
from data import get_simsiam_dataloader


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def is_main() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def setup_dist():
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, local_rank, dist.get_world_size()


def adjust_learning_rate(
    optimizer, init_lr: float, epoch: int, total_epochs: int
) -> float:
    """
    Cosine LR decay for encoder+projector; FIXED LR for predictor.

    From facebookresearch/simsiam main_simsiam.py:
      cur_lr = init_lr * 0.5 * (1 + cos(π * epoch / epochs))
      predictor param group ('fix_lr'=True) keeps init_lr throughout.
    """
    cur_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))
    for pg in optimizer.param_groups:
        if pg.get("fix_lr", False):
            pg["lr"] = init_lr   # predictor: FIXED
        else:
            pg["lr"] = cur_lr    # encoder+projector: cosine decay
    return cur_lr


def resolve_device(spec: str, local_rank: int = 0) -> "torch.device":
    """Which device to run on, decided rather than assumed.

    The captured trainer called `.cuda(local_rank)` on the model and on every
    batch, so it raised on a machine with no GPU. `"cuda"` is honoured only
    when there is one: falling back quietly would let a run that was asked for
    a GPU report success from a CPU, and the two are not the same run.
    """
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device is 'cuda' but no CUDA device is visible. Ask for "
                "'auto' to accept a CPU; asking for a GPU and getting a CPU "
                "silently would misreport what ran")
        return torch.device(f"cuda:{local_rank}")
    if spec == "auto":
        return torch.device(f"cuda:{local_rank}"
                            if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unknown device {spec!r}; expected auto, cuda or cpu")


def make_deterministic(seed: int) -> None:
    """Seed everything the run draws from.

    Measured, not assumed: every augmentation in `data/simsiam_dataset.py` is
    a stock torchvision transform, and those draw from torch's generator --
    which also seeds each DataLoader worker deterministically. `random` is
    seeded too, as insurance against a transform added later that uses it.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_ckpt(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    if is_main():
        print(f"  [ckpt] Saved: {path}")


def train_epoch(
    model, loader, optimizer, epoch, cfg, writer, distributed, device
) -> tuple:
    model.train()
    losses = AverageMeter()
    t0 = time.time()
    print_freq = cfg["training"]["print_freq"]

    z_std_meter = AverageMeter()

    for i, (v1, v2, _) in enumerate(loader):
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)

        p1, p2, z1, z2 = model(v1, v2)
        loss = simsiam_loss(p1, p2, z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), v1.size(0))

        # Collapse monitor: std of unit-normalised z vectors (dim-averaged)
        with torch.no_grad():
            z_std = F.normalize(z1, dim=1).std(dim=0).mean().item()
        z_std_meter.update(z_std, v1.size(0))

        if is_main() and i % print_freq == 0:
            elapsed = time.time() - t0
            print(
                f"  [{epoch}][{i:5d}/{len(loader)}]  "
                f"loss={losses.avg:.4f}  z_std={z_std:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.6f}  t={elapsed:.1f}s"
            )

        if writer and is_main():
            step = epoch * len(loader) + i
            writer.add_scalar("train/loss",  losses.val, step)
            writer.add_scalar("train/z_std", z_std,      step)
            writer.add_scalar("train/lr",    optimizer.param_groups[0]["lr"], step)

    return losses.avg, z_std_meter.avg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SimSiam Step 1: ResNet-50")
    parser.add_argument("--config",    default="configs/step1_resnet.yaml")
    parser.add_argument("--data_path", default=None,
                        help="Override ImageNet root (parent of train/ and val/)")
    parser.add_argument("--resume",    default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device",    default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Added by the port; the captured trainer assumed "
                             "CUDA")
    return parser


def run(args, config: dict | None = None) -> dict:
    """The captured `main()`, with the config allowed to arrive in memory.

    Returning the metrics is the other change: the captured version computed
    the epoch loss and the collapse monitor and then dropped them on the
    floor, so there was nothing for an adapter to record.
    """
    if config is not None:
        cfg = config
    else:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    if args.data_path:
        cfg["data"]["train_path"] = os.path.join(args.data_path, "train")

    distributed, local_rank, world_size = setup_dist()
    device = resolve_device(getattr(args, "device", "auto"), local_rank)
    make_deterministic(int(cfg.get("seed", 42)) + local_rank)

    save_dir = cfg["checkpoint"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    writer = None
    if is_main():
        log_dir = os.path.join(save_dir, "logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
        writer = SummaryWriter(log_dir)
        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg, f)

    # Build model
    model = build_simsiam_resnet(
        dim=cfg["model"]["dim"],
        pred_dim=cfg["model"]["pred_dim"],
    ).to(device)

    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Compute init_lr: base_lr × batch_size / 256  (linear scaling rule)
    total_batch = cfg["training"]["batch_size"]
    init_lr = cfg["training"]["base_lr"] * total_batch / 256

    # TWO parameter groups:
    #   - encoder + projector: cosine LR decay
    #   - predictor           : FIXED LR (critical for SimSiam stability)
    raw = model.module if distributed else model
    optim_params = [
        {"params": list(raw.backbone.parameters()) + list(raw.projector.parameters()),
         "fix_lr": False},
        {"params": raw.predictor.parameters(),
         "fix_lr": True},
    ]
    optimizer = optim.SGD(
        optim_params,
        lr=init_lr,
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Data loader
    per_gpu_batch = total_batch // world_size
    train_loader, _ = get_simsiam_dataloader(
        data_path=cfg["data"]["train_path"],
        augmentation="step1",
        batch_size=per_gpu_batch,
        num_workers=cfg["data"]["num_workers"],
        img_size=cfg["data"]["img_size"],
        distributed=distributed,
    )

    # Optional resume
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        if not cfg.get("checkpoint", {}).get("allow_resume", False):
            raise RuntimeError(
                "Refusing to resume SimSiam unless checkpoint.allow_resume=true. "
                "Use a fresh SyncBN run for Step 1 acceptance."
            )
        ckpt = torch.load(args.resume, map_location="cpu")
        start_epoch = ckpt["epoch"] + 1
        raw = model.module if distributed else model
        raw.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if is_main():
            print(f"Resumed from epoch {ckpt['epoch']}")

    total_epochs = cfg["training"]["epochs"]
    save_freq    = cfg["training"]["save_freq"]

    if is_main():
        print("=" * 70)
        print("SimSiam  Step 1: ResNet-50  (Chen & He, arXiv:2011.10566)")
        print(f"  epochs={total_epochs}  batch={total_batch}")
        print(f"  base_lr={cfg['training']['base_lr']}  init_lr={init_lr:.4f}")
        print(f"  wd={cfg['training']['weight_decay']}  pred_dim={cfg['model']['pred_dim']}")
        print(f"  world_size={world_size}  per_gpu_batch={per_gpu_batch}")
        print(f"  save_freq={save_freq} epochs  (checkpoints at {list(range(save_freq, total_epochs+1, save_freq))})")
        print("=" * 70)

    for epoch in range(start_epoch, total_epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        cur_lr = adjust_learning_rate(optimizer, init_lr, epoch, total_epochs)

        if is_main():
            print(f"\n=== Epoch {epoch}/{total_epochs - 1}  encoder_lr={cur_lr:.6f}  "
                  f"predictor_lr={init_lr:.6f} (fixed) ===")

        avg_loss, avg_zstd = train_epoch(
            model, train_loader, optimizer, epoch, cfg, writer, distributed, device
        )

        if writer and is_main():
            writer.add_scalar("epoch/loss",  avg_loss,  epoch)
            writer.add_scalar("epoch/z_std", avg_zstd,  epoch)
            writer.add_scalar("epoch/lr",    cur_lr,    epoch)

        # Save checkpoint every save_freq epochs and at the final epoch
        if is_main() and ((epoch + 1) % save_freq == 0 or epoch == total_epochs - 1):
            raw = model.module if distributed else model
            save_ckpt(
                {
                    "epoch":      epoch,
                    "state_dict": raw.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "config":     cfg,
                },
                os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
            )

    if writer:
        writer.close()
    if distributed:
        dist.destroy_process_group()
    if is_main():
        print("\nSimSiam Step 1 training complete!")

    # An epoch that never ran leaves these unset. Reporting a loss of zero
    # would be a number where there was no measurement, so the absence is
    # passed on and the adapter counts it.
    return {
        "epochs": total_epochs - start_epoch,
        "final_loss": avg_loss if total_epochs > start_epoch else None,
        "final_z_std": avg_zstd if total_epochs > start_epoch else None,
    }


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
