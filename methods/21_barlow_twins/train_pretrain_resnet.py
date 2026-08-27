"""
Barlow Twins Step 1 training: ResNet-50 on ImageNet-1k.

Strictly follows Zbontar et al. (2021) arXiv:2103.03230:
  Backbone   : ResNet-50 (zero_init_residual=True)
  Projector  : 3-layer MLP  [8192-8192-8192]
  Dataset    : ImageNet-1k (1.28M images)
  Epochs     : 1000
  Batch      : 2048 total (256 per GPU x8 H200)
  Optimizer  : LARS  (weights lr=0.2*bs/256, biases lr=0.0048*bs/256)
  Weight dec : 1e-6
  LR sched   : linear warmup 10 epochs + cosine decay (per-step)
  Lambda     : 0.0051
  Augment    : asymmetric two-view (view1: GaussBlur p=1.0, Solarize p=0.0;
                                    view2: GaussBlur p=0.1, Solarize p=0.2)
  Checkpoint : every 50 epochs + final epoch

Supports multi-GPU via DDP (torchrun --nproc_per_node=8).
"""

import os
import random
import sys
import time
import math
import yaml
import argparse
import contextlib
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models import build_barlow_resnet
from data import get_barlow_dataloader
from optim import LARS, exclude_bias_and_norm


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def is_main() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def setup_dist():
    # Single-process runs (WORLD_SIZE<=1) skip the process group entirely: the
    # local backend exports WORLD_SIZE=1/RANK=0/LOCAL_RANK=0 but no MASTER_ADDR,
    # so keying off LOCAL_RANK's mere presence would call init_process_group and
    # fail. Gate on WORLD_SIZE, matching the CPU device invariant (docs/GPU.md).
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, local_rank, dist.get_world_size()


def adjust_learning_rate(
    optimizer,
    step: int,
    total_steps: int,
    warmup_steps: int,
    batch_size: int,
    lr_weights: float,
    lr_biases: float,
):
    """
    Per-step LR schedule matching the original Barlow Twins implementation.
    base_lr = batch_size / 256  (linear scaling rule)
    Warmup: linear ramp from 0 to base_lr over warmup_steps.
    After warmup: cosine decay from base_lr to base_lr * 0.001.
    Final LR = base_lr * lr_weights  (for weights)
           or = base_lr * lr_biases  (for biases/BN)
    """
    base_lr = batch_size / 256.0
    max_s   = total_steps - warmup_steps

    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        s  = step - warmup_steps
        q  = 0.5 * (1.0 + math.cos(math.pi * s / max_s))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1.0 - q)

    optimizer.param_groups[0]["lr"] = lr * lr_weights
    optimizer.param_groups[1]["lr"] = lr * lr_biases


def save_ckpt(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    if is_main():
        print(f"  [ckpt] Saved: {path}")


def resolve_device(spec: str, local_rank: int = 0) -> "torch.device":
    """Which device to run on, decided rather than assumed.

    The captured trainer sent the model and every batch to CUDA
    unconditionally, so it could not start without a GPU. `"cuda"` is honoured
    only when there is one: falling back quietly would let a run that was
    asked for a GPU report success from a CPU.
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

    **`random` matters here, and it did not in the port before this one.**
    The solarisation and the blur in `data/barlow_dataset.py` call
    `random.random()` directly rather than going through a torchvision
    transform, so seeding torch alone would not determine the augmentation in
    this process.

    Loader workers need nothing extra: torch's worker loop seeds `random`
    itself. That was measured -- `random.seed` appears in `_worker_loop`, and
    two runs with no `worker_init_fn` draw the same values -- after a first
    version of this port added a `seed_worker` on the assumption that it did
    not, and modified the captured loader to take one. Both were removed.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(precision: str, device_type: str = "cuda"):
    """The captured version wrote `device_type="cuda"` in, so it could not run
    anywhere else. The device is now passed in."""
    if precision == "amp_fp16":
        return torch.amp.autocast(device_type=device_type,
                                  dtype=torch.float16)
    if precision == "bf16":
        return torch.amp.autocast(device_type=device_type,
                                  dtype=torch.bfloat16)
    if precision == "fp32":
        return contextlib.nullcontext()
    raise ValueError(f"Unsupported training.precision: {precision!r}")


def train_epoch(
    model, loader, optimizer, epoch: int, total_epochs: int,
    total_steps: int, warmup_steps: int, global_step_start: int,
    cfg: dict, writer, distributed: bool, device, scaler,
    precision: str,
) -> tuple:
    model.train()
    losses = AverageMeter()
    t0     = time.time()
    batch_size  = cfg["training"]["batch_size"]
    lr_weights  = cfg["training"]["lr_weights"]
    lr_biases   = cfg["training"]["lr_biases"]
    print_freq  = cfg["training"]["print_freq"]

    for i, (y1, y2, _) in enumerate(loader):
        global_step = global_step_start + i
        adjust_learning_rate(
            optimizer, global_step, total_steps, warmup_steps,
            batch_size, lr_weights, lr_biases,
        )

        y1 = y1.to(device, non_blocking=True)
        y2 = y2.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast_context(precision, device.type):
            loss = model(y1, y2)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite Barlow Twins loss at epoch={epoch} "
                f"iter={i} global_step={global_step}: {loss.item()}"
            )
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), y1.size(0))

        if is_main() and i % print_freq == 0:
            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [{epoch}/{total_epochs-1}][{i:5d}/{len(loader)}]  "
                f"loss={losses.avg:.4f}  lr={current_lr:.6f}  t={elapsed:.1f}s"
            )

        if writer and is_main():
            writer.add_scalar("train/loss", losses.val, global_step)
            writer.add_scalar("train/lr_weights", optimizer.param_groups[0]["lr"], global_step)

    return losses.avg, global_step_start + len(loader)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Barlow Twins Step 1: ResNet-50")
    parser.add_argument("--config",    default="configs/pretrain_resnet.yaml")
    parser.add_argument("--data_path", default=None,
                        help="Override ImageNet root (contains train/ and val/)")
    parser.add_argument("--resume",    default=None)
    parser.add_argument(
        "--end_epoch",
        type=int,
        default=None,
        help=(
            "Exclusive stop epoch for short pilots. The LR schedule still uses "
            "training.epochs from the config, so this does not shorten the "
            "official cosine schedule."
        ),
    )
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Added by the port; the captured trainer "
                             "assumed CUDA")
    return parser


def run(args, config: dict | None = None) -> dict:
    """The captured `main()`, with the config allowed to arrive in memory and
    the metrics returned. The captured version computed the epoch loss and
    then dropped it."""

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
        # Imported here, not at module top: tensorboard is native-trainer-only
        # logging machinery, and importing it eagerly would drag it into the ViT
        # Step-2 path (which imports resolve_device/make_deterministic from this
        # module) under venvs that have timm but not tensorboard.
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(save_dir, "logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
        writer = SummaryWriter(log_dir)
        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg, f)

    # Model
    model = build_barlow_resnet(
        projector=cfg["model"]["projector"],
        lambd=cfg["barlow"]["lambd"],
    ).to(device)

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # LARS optimizer — separate weights vs. bias/BN params
    raw = model.module if distributed else model
    param_weights = []
    param_biases  = []
    for p in raw.parameters():
        if p.ndim == 1:
            param_biases.append(p)
        else:
            param_weights.append(p)

    optimizer = LARS(
        [{"params": param_weights}, {"params": param_biases}],
        lr=0,                                    # set per-step by schedule
        weight_decay=cfg["training"]["weight_decay"],
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    # Data loader (per-GPU batch size)
    total_batch  = cfg["training"]["batch_size"]
    per_gpu_batch = total_batch // world_size
    train_loader, _ = get_barlow_dataloader(
        data_path=cfg["data"]["train_path"],
        augmentation="step1",
        batch_size=per_gpu_batch,
        num_workers=cfg["data"]["num_workers"],
        img_size=cfg["data"]["img_size"],
        distributed=distributed,
    )

    total_epochs   = cfg["training"]["epochs"]
    warmup_epochs  = cfg["training"]["warmup_epochs"]
    steps_per_epoch = len(train_loader)
    total_steps    = total_epochs  * steps_per_epoch
    warmup_steps   = warmup_epochs * steps_per_epoch
    save_freq      = cfg["training"]["save_freq"]
    precision      = cfg["training"].get("precision", "amp_fp16")
    if precision not in {"amp_fp16", "bf16", "fp32"}:
        raise ValueError(
            "training.precision must be one of: amp_fp16, bf16, fp32; "
            f"got {precision!r}"
        )
    scaler = torch.amp.GradScaler(device.type,
                                  enabled=(precision == "amp_fp16"))
    loop_end_epoch = total_epochs if args.end_epoch is None else min(args.end_epoch, total_epochs)

    # Optionally resume from checkpoint
    start_epoch  = 0
    global_step  = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_epoch = ckpt["epoch"] + 1
        global_step = start_epoch * steps_per_epoch
        raw = model.module if distributed else model
        raw.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            if scaler.is_enabled():
                scaler.load_state_dict(ckpt["scaler"])
            elif is_main():
                print("Skipping AMP scaler restore because precision is not amp_fp16")
        if is_main():
            print(f"Resumed from epoch {ckpt['epoch']}")

    if loop_end_epoch <= start_epoch:
        raise ValueError(
            f"--end_epoch must be greater than start_epoch; got "
            f"end_epoch={loop_end_epoch}, start_epoch={start_epoch}"
        )

    if is_main():
        print("=" * 70)
        print("Barlow Twins  Step 1: ResNet-50  (Zbontar et al., arXiv:2103.03230)")
        print(f"  epochs={total_epochs}  batch={total_batch}")
        print(f"  lr_weights={cfg['training']['lr_weights']}  "
              f"lr_biases={cfg['training']['lr_biases']}  "
              f"wd={cfg['training']['weight_decay']}")
        print(f"  lambda={cfg['barlow']['lambd']}  projector={cfg['model']['projector']}")
        print(f"  world_size={world_size}  per_gpu_batch={per_gpu_batch}")
        print(f"  precision={precision}")
        print(f"  save_freq={save_freq} epochs")
        if args.end_epoch is not None:
            print(f"  pilot_stop_epoch={loop_end_epoch} (exclusive; LR schedule still uses {total_epochs})")
        print("=" * 70)

    for epoch in range(start_epoch, loop_end_epoch):
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        if is_main():
            print(f"\n=== Epoch {epoch}/{total_epochs - 1} ===")

        avg_loss, global_step = train_epoch(
            model, train_loader, optimizer,
            epoch, total_epochs,
            total_steps, warmup_steps,
            global_step_start=epoch * steps_per_epoch,
            cfg=cfg, writer=writer,
            distributed=distributed, device=device,
            scaler=scaler,
            precision=precision,
        )

        if writer and is_main():
            writer.add_scalar("epoch/loss", avg_loss, epoch)

        # Save checkpoint every save_freq epochs, at the full final epoch, and
        # at the pilot stop epoch so short numeric checks are resumable.
        if is_main() and (
            (epoch + 1) % save_freq == 0
            or epoch == total_epochs - 1
            or epoch == loop_end_epoch - 1
        ):
            raw = model.module if distributed else model
            save_ckpt(
                {
                    "epoch":      epoch,
                    "state_dict": raw.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "scaler":     scaler.state_dict(),
                    "config":     cfg,
                },
                os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
            )

    if is_main() and loop_end_epoch >= total_epochs:
        raw = model.module if distributed else model
        torch.save(raw.backbone.state_dict(), os.path.join(save_dir, "resnet50.pth"))

    if writer:
        writer.close()
    if distributed:
        dist.destroy_process_group()
    if is_main():
        if loop_end_epoch >= total_epochs:
            print("\nBarlow Twins Step 1 training complete!")
        else:
            print(f"\nBarlow Twins Step 1 pilot stopped cleanly at epoch {loop_end_epoch}.")

    # An epoch that never ran leaves the loss unset. Reporting zero would be a
    # number where there was no measurement.
    return {
        "epochs": loop_end_epoch - start_epoch,
        "final_loss": avg_loss if loop_end_epoch > start_epoch else None,
    }


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
