"""
SwAV - Step 1: As-is SSL Comparison
Training with ResNet-50 strictly following Caron et al. (2020) NeurIPS:
  Encoder     : ResNet-50
  Proj. head  : Linear(2048,2048) -> BN -> ReLU -> Linear(2048,128) -> L2-norm
  Prototypes  : K=3000  (L2-normalised weight matrix)
  Multi-crop  : 2x224 (scale [0.14,1.0]) + 6x96 (scale [0.05,0.14])
  Dataset     : ImageNet-1k (1.28M images)
  Epochs      : 200  (checkpoint every 100 epochs)
  Batch       : 4096 total (512 per GPU x 8 H200)
  Optimizer   : LARC-SGD  lr=4.8, final_lr=0.0048, momentum=0.9, wd=1e-6
  LR sched    : cosine decay with 10-epoch linear warmup
  Loss        : SwAV  tau=0.1, Sinkhorn-Knopp (3 iters, eps=0.05)
  Proto freeze: first epoch / official 200ep setting (313 steps)
Multi-GPU via torchrun --nproc_per_node=8  (DDP + SyncBatchNorm).
"""

import os, sys, time, math, random, yaml, argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.resnet_swav import build_resnet_swav
from data   import get_swav_dataloader


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count


def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def setup_dist():
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, local_rank, dist.get_world_size()


# --------------------------------------------------------------------------- #
# LARC wrapper
# --------------------------------------------------------------------------- #
class LARC:
    """Apex-compatible LARC wrapper around any base optimizer."""
    def __init__(self, optimizer, trust_coefficient=0.001, clip=True, eps=1e-8):
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.clip = clip
        self.eps = eps

    @property
    def param_groups(self): return self.optim.param_groups

    def state_dict(self): return self.optim.state_dict()
    def load_state_dict(self, sd): self.optim.load_state_dict(sd)
    def zero_grad(self): self.optim.zero_grad()

    @torch.no_grad()
    def step(self):
        weight_decays = []
        for group in self.optim.param_groups:
            wd = group.get("weight_decay", 0.0)
            weight_decays.append(wd)
            # Apex LARC absorbs weight decay into the adapted gradient and
            # temporarily disables the wrapped optimizer's own weight decay.
            group["weight_decay"] = 0.0
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                w_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)
                if w_norm > 0 and g_norm > 0:
                    adaptive_lr = self.trust_coefficient * w_norm / (
                        g_norm + w_norm * wd + self.eps
                    )
                    if self.clip:
                        adaptive_lr = min(adaptive_lr / lr, 1.0)
                    p.grad.data.add_(p.data, alpha=wd)
                    p.grad.data.mul_(adaptive_lr)
        self.optim.step()
        for group, wd in zip(self.optim.param_groups, weight_decays):
            group["weight_decay"] = wd


# --------------------------------------------------------------------------- #
# Sinkhorn-Knopp
# --------------------------------------------------------------------------- #
@torch.no_grad()
def distributed_sinkhorn(out, niters=3, eps=0.05, world_size=1):
    """Official SwAV Sinkhorn on local scores with global row normalization.

    ``out`` is the local-rank score tensor [B_local, K]. The full assignment
    matrix is never materialized; only scalar and row sums are all-reduced, as
    in facebookresearch/swav. Elementwise all-reducing ``Q`` would add columns
    from unrelated samples on different ranks and corrupt the assignments.
    """
    Q = torch.exp(out / eps).t().contiguous()   # [K, B_local]
    K, B_local = Q.shape
    B = B_local * world_size

    sum_Q = torch.sum(Q)
    if world_size > 1:
        dist.all_reduce(sum_Q)
    Q /= sum_Q.clamp_min(1e-12)

    for _ in range(niters):
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if world_size > 1:
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows.clamp_min(1e-12)
        Q /= K

        Q /= torch.sum(Q, dim=0, keepdim=True).clamp_min(1e-12)
        Q /= B

    Q *= B
    return Q.t().float()


# --------------------------------------------------------------------------- #
# SwAV loss
# --------------------------------------------------------------------------- #
def swav_loss(scores, cfg, world_size):
    tau       = cfg["loss"]["temperature"]
    nmb_crops = cfg["data"]["nmb_crops"]
    n_views   = sum(nmb_crops)
    n_global  = nmb_crops[0]
    sk_iters  = cfg["loss"].get("sinkhorn_iters", 3)
    sk_eps    = cfg["loss"].get("sinkhorn_eps", 0.05)

    qs = [distributed_sinkhorn(scores[i].detach(), sk_iters, sk_eps, world_size)
          for i in range(n_global)]

    loss    = torch.tensor(0.0, device=scores[0].device)
    n_terms = 0
    for i in range(n_views):
        for j in range(n_global):
            if i == j:
                continue
            p = F.softmax(scores[i] / tau, dim=1)
            loss -= (qs[j] * torch.log(p + 1e-8)).sum(dim=1).mean()
            n_terms += 1
    return loss / n_terms


# --------------------------------------------------------------------------- #
# LR helpers
# --------------------------------------------------------------------------- #
def build_lr_schedule(base_lr, final_lr, start_warmup, epochs, steps_per_epoch,
                      warmup_epochs):
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    schedule = []
    for step in range(total_steps):
        if step < warmup_steps:
            if warmup_steps <= 1:
                lr = base_lr
            else:
                lr = start_warmup + (base_lr - start_warmup) * step / (warmup_steps - 1)
        else:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            lr = final_lr + 0.5 * (base_lr - final_lr) * (1.0 + math.cos(math.pi * progress))
        schedule.append(lr)
    return schedule


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def save_ckpt(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    if is_main():
        print(f"  Checkpoint saved -> {path}")


# --------------------------------------------------------------------------- #
# Training epoch
# --------------------------------------------------------------------------- #
def train_epoch(model, loader, optimizer, epoch, cfg, writer,
                distributed, device, world_size, global_step, lr_schedule):
    model.train()
    losses = AverageMeter()
    t0     = time.time()
    freeze_steps = cfg["training"].get("freeze_prototypes_steps", 313)

    model_for_prototypes = model.module if hasattr(model, "module") else model

    for i, (crops, _) in enumerate(loader):
        schedule_step = epoch * len(loader) + i
        if lr_schedule is not None and schedule_step < len(lr_schedule):
            set_lr(optimizer, lr_schedule[schedule_step])

        crops = [c.to(device, non_blocking=True) for c in crops]
        model_for_prototypes.normalize_prototypes()
        _embeddings, output = model(crops)

        bs = crops[0].size(0)
        n_views = sum(cfg["data"]["nmb_crops"])
        scores = [output[bs * view: bs * (view + 1)] for view in range(n_views)]

        loss = swav_loss(scores, cfg, world_size)
        optimizer.zero_grad()
        loss.backward()

        if global_step < freeze_steps:
            for name, param in model.named_parameters():
                if "prototypes" in name and param.grad is not None:
                    param.grad = None

        optimizer.step()
        losses.update(loss.item(), crops[0].size(0))
        global_step += 1

        if is_main() and i % cfg["training"]["print_freq"] == 0:
            print(f"  [{epoch}][{i}/{len(loader)}]  loss={losses.avg:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}  t={time.time()-t0:.1f}s")
        if writer and is_main():
            writer.add_scalar("train/loss", losses.val, global_step)
            writer.add_scalar("train/lr",   optimizer.param_groups[0]["lr"], global_step)

    return losses.avg, global_step


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def resolve_device(spec: str, local_rank: int = 0) -> "torch.device":
    """Which device to run on, decided rather than assumed.

    The captured trainer sent the model and every crop to CUDA
    unconditionally. `"cuda"` is honoured only when there is one: falling back
    quietly would let a run asked for a GPU report success from a CPU.
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

    `random` is seeded because the multi-crop augmentation's blur reaches for
    it directly, as this method's loader does rather than going through a
    torchvision transform. Loader workers need nothing extra: torch's worker
    loop seeds `random` itself, which was measured for the port before this
    one after a version of it wrongly assumed otherwise.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/step1_resnet.yaml")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--resume",    default=None)
    parser.add_argument("--device",    default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Added by the port; the captured trainer "
                             "assumed CUDA")
    return parser


def run(args, config: dict | None = None) -> dict:
    """The captured `main()`, with the config allowed to arrive in memory and
    the metrics returned. The captured version computed the epoch loss and
    then discarded it."""
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
        writer  = SummaryWriter(log_dir)
        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg, f)
        print("=" * 70)
        print("SwAV Step 1 - ResNet-50 | ImageNet-1k")
        print(f"  GPUs={world_size}  batch={cfg['training']['batch_size']}  "
              f"epochs={cfg['training']['epochs']}")
        print("=" * 70)

    # Model
    model = build_resnet_swav(
        out_dim=cfg["model"]["out_dim"],
        hidden_mlp=cfg["model"]["hidden_mlp"],
        nmb_prototypes=cfg["model"]["nmb_prototypes"],
    )
    model = model.to(device)
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])

    # Data
    per_gpu_batch = cfg["training"]["batch_size"] // world_size
    loader, sampler = get_swav_dataloader(
        data_path=cfg["data"]["train_path"],
        size_crops=cfg["data"]["size_crops"],
        nmb_crops=cfg["data"]["nmb_crops"],
        min_scale_crops=cfg["data"]["min_scale_crops"],
        max_scale_crops=cfg["data"]["max_scale_crops"],
        batch_size=per_gpu_batch,
        num_workers=cfg["data"]["num_workers"],
        color_jitter_strength=cfg["data"].get("color_jitter_strength", 1.0),
        distributed=distributed,
    )

    # Optimizer: LARC-SGD
    base_lr  = cfg["training"]["lr"]
    final_lr = cfg["training"].get("final_lr", base_lr * 1e-3)
    sgd = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    optimizer = LARC(
        sgd,
        trust_coefficient=cfg["training"].get("eta", 0.001),
        clip=cfg["training"].get("larc_clip", False),
    )

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=f"cuda:{local_rank}")
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        if is_main():
            print(f"  Resumed from epoch {ckpt['epoch']}")

    total_epochs  = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"].get("warmup_epochs", 10)
    save_freq     = cfg["training"].get("save_freq", 100)
    lr_schedule = build_lr_schedule(
        base_lr=base_lr,
        final_lr=final_lr,
        start_warmup=cfg["training"].get("start_warmup", 0.0),
        epochs=total_epochs,
        steps_per_epoch=len(loader),
        warmup_epochs=warmup_epochs,
    )

    for epoch in range(start_epoch, total_epochs):
        if distributed:
            sampler.set_epoch(epoch)
        lr = lr_schedule[epoch * len(loader)]
        set_lr(optimizer, lr)

        if is_main():
            print(f"\nEpoch {epoch}/{total_epochs - 1}  lr={lr:.6f}")

        avg_loss, global_step = train_epoch(
            model, loader, optimizer, epoch, cfg, writer,
            distributed, device, world_size, global_step, lr_schedule
        )

        if is_main():
            print(f"  Epoch {epoch} done - avg_loss={avg_loss:.4f}")
            if writer:
                writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        if is_main() and ((epoch + 1) % save_freq == 0 or epoch == total_epochs - 1):
            raw = model.module if hasattr(model, "module") else model
            save_ckpt(
                {"epoch": epoch, "global_step": global_step,
                 "state_dict": raw.state_dict(), "optimizer": optimizer.state_dict(),
                 "config": cfg},
                os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            )

    if is_main():
        print("\nSwAV Step 1 training finished!")
        if writer:
            writer.close()
    if distributed:
        dist.destroy_process_group()

    # An epoch that never ran leaves the loss unset. Reporting zero would be
    # a number where there was no measurement.
    return {
        "epochs": total_epochs - start_epoch,
        "final_loss": avg_loss if total_epochs > start_epoch else None,
    }


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
