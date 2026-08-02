"""
iBOT Step 1 training: ViT-Small/16 on ImageNet-1k.

Strictly follows Zhou et al. (2021) arXiv:2111.07832:
  Backbone  : ViT-Small/16  (embed_dim=384, depth=12, heads=6)
  Dataset   : ImageNet-1k (1.28M images)
  Epochs    : 800  (checkpoints every 50 epochs)
  Batch     : 1024 total (128 per GPU × 8 H200)
  Optimizer : AdamW  base lr=5e-4 scaled by total batch / 256, beta=(0.9, 0.999)
  LR sched  : cosine decay with 10-epoch per-step linear warmup
  WD sched  : cosine from 0.04 to 0.4
  Teacher   : EMA momentum cosine 0.996 → 1.0
  Head      : DINOHead(384, 8192, hidden=2048, bottleneck=256)
  Crops     : 2×224 (global, with block masking) + 10×96 (local)
  Masking   : --pred_ratio 0 0.3 --pred_ratio_var 0 0.2 (block masking)
  Temps     : student=0.1, teacher 0.04→0.07 over 30 epochs
  Centering : MLP center EMA momentum=0.9
  freeze_last_layer: 1 epoch (prevents CLS collapse — required for DINO/iBOT)

Supports multi-GPU via DDP (torchrun --nproc_per_node=8).

Changed during the port, and recorded in provenance.json:

  - **the device is resolved instead of assumed.** The captured trainer called
    `.cuda(local_rank)` on the model, the loss and every crop and mask, so it
    could not start on a machine without a GPU. `resolve_device` picks one;
    asking for `cuda` where there is none is refused rather than served a CPU
    in silence, which would misreport what ran. Nothing else about the
    computation changed
  - **`main()` is split into `build_parser()` and `run(args, config)`,** so an
    adapter can call the training loop in process and be handed the metrics.
    The captured `main()` computed the epoch loss and its two components and
    then discarded them
  - **the run is seeded through `make_deterministic`.** The captured trainer
    seeded torch only; `random` is seeded as well, as insurance against a
    transform added later that draws from it
  - **the backbone is chosen from `model.arch`** rather than hard-coded to
    `vit_small`, so the resolved config says which architecture ran. Only
    `vit_small` is accepted here: `vit_base` belongs to step 2, which this port
    does not include
"""

import os
import random
import sys
import time
import copy
import yaml
import math
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models import vit_small, DINOHead, iBOT, iBOTLoss
from data   import get_ibot_dataloader


# ── Utilities ─────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def setup_dist():
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, local_rank, dist.get_world_size()


# Step 1 is ViT-Small/16. `vit_base` is step 2's backbone and step 2 is not
# part of this port, so it is refused by name rather than silently built.
_VIT_BUILDERS = {"vit_small": vit_small}


def resolve_device(spec: str, local_rank: int = 0) -> "torch.device":
    """Which device to run on, decided rather than assumed.

    The captured trainer called `.cuda(local_rank)` on the model, the loss and
    every crop, so it raised on a machine with no GPU. `"cuda"` is honoured
    only when there is one: falling back quietly would let a run that was asked
    for a GPU report success from a CPU, and the two are not the same run.
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

    The captured trainer seeded torch only. `random` is seeded too, as
    insurance against a transform added later that draws from it.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── LR / WD schedules ─────────────────────────────────────────────────────────

def cosine_scheduler_value(step, total_steps, base, final=0.0, warmup_steps=0, warmup_start=0.0):
    """Value from the official iBOT/DINO cosine scheduler at a global step."""
    if warmup_steps > 0 and step < warmup_steps:
        return warmup_start + (base - warmup_start) * step / max(warmup_steps - 1, 1)
    cosine_step = step - warmup_steps
    cosine_total = max(total_steps - warmup_steps, 1)
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * cosine_step / cosine_total))


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def set_wd(optimizer, wd):
    for pg in optimizer.param_groups:
        if pg.get("apply_wd", True):
            pg["weight_decay"] = wd


def clip_gradients(modules, clip):
    """Official iBOT-style per-parameter norm clipping."""
    if clip <= 0:
        return []
    if not isinstance(modules, (list, tuple)):
        modules = [modules]
    norms = []
    for module in modules:
        for p in module.parameters():
            if p.grad is None:
                continue
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def save_ckpt(state, path, mark_valid=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    if is_main():
        print(f"  [ckpt] Saved: {path}")
        if mark_valid:
            marker = os.path.join(os.path.dirname(path), "latest_valid_checkpoint.txt")
            with open(marker, "w") as f:
                f.write(path)
            print(f"  [ckpt] Marked as latest valid: {marker}")


def is_checkpoint_healthy(avg_loss, cls_loss, patch_loss, cfg):
    """Conservative guard against marking collapsed iBOT checkpoints as valid."""
    health = cfg["training"].get("checkpoint_health", {})
    min_total = float(health.get("min_total_loss", 0.1))
    min_component = float(health.get("min_component_loss", 0.01))
    max_total = float(health.get("max_total_loss", 50.0))
    vals = {
        "loss": float(avg_loss),
        "cls_loss": float(cls_loss),
        "patch_loss": float(patch_loss),
    }
    for value in vals.values():
        if not math.isfinite(value):
            return False, f"non-finite checkpoint metrics: {vals}"
    if vals["loss"] < min_total or vals["loss"] > max_total:
        return False, f"total loss outside healthy range: {vals}"
    if vals["cls_loss"] < min_component or vals["patch_loss"] < min_component:
        return False, f"component loss below collapse threshold: {vals}"
    return True, f"healthy checkpoint metrics: {vals}"


def assert_epoch_health(avg_loss, cls_loss, patch_loss, cfg, epoch):
    fail_after = int(cfg["training"].get("fail_fast_after_epoch", 50))
    if (epoch + 1) < fail_after:
        return
    healthy, health_msg = is_checkpoint_healthy(avg_loss, cls_loss, patch_loss, cfg)
    if not healthy:
        raise RuntimeError(
            "iBOT health check failed after the pilot window; "
            f"stopping before spending more compute. {health_msg}"
        )


# ── Training epoch ─────────────────────────────────────────────────────────────

def train_one_epoch(model, criterion, loader, optimizer, epoch, cfg,
                    writer, distributed, device):
    model.train()
    criterion.train()

    total_epochs      = cfg["training"]["epochs"]
    base_lr           = cfg["training"]["lr"] * cfg["training"]["batch_size"] / 256.0
    wd_start          = cfg["training"]["weight_decay_start"]
    wd_end            = cfg["training"]["weight_decay_end"]
    min_lr            = cfg["training"].get("min_lr", 1e-6)
    warmup_epochs     = cfg["training"]["warmup_epochs"]
    freeze_last_layer = cfg["training"].get("freeze_last_layer", 1)
    mom_base          = cfg["ibot"]["teacher_momentum_start"]
    mom_final         = cfg["ibot"]["teacher_momentum_end"]
    print_freq        = cfg["training"]["print_freq"]
    n_global          = cfg["data"]["n_global_crops"]
    max_steps_per_epoch = cfg["training"].get("max_steps_per_epoch")

    n_steps_per_epoch = len(loader)
    total_steps       = total_epochs * n_steps_per_epoch
    warmup_steps      = warmup_epochs * n_steps_per_epoch

    loss_meter       = AverageMeter()
    cls_loss_meter   = AverageMeter()
    patch_loss_meter = AverageMeter()
    mask_ratio_meter = AverageMeter()
    zero_mask_meter = AverageMeter()
    teacher_proto_meter = AverageMeter()
    teacher_std_meter = AverageMeter()
    t0 = time.time()

    raw_model = model.module if distributed else model

    for i, (crops, masks, _labels) in enumerate(loader):
        if max_steps_per_epoch is not None and i >= int(max_steps_per_epoch):
            if is_main():
                print(f"  [pilot] stopping epoch after {max_steps_per_epoch} steps")
            break

        # Per-step LR warmup / cosine decay (smoother than per-epoch update)
        global_step = epoch * n_steps_per_epoch + i
        lr = cosine_scheduler_value(global_step, total_steps, base_lr, final=min_lr, warmup_steps=warmup_steps)
        wd = cosine_scheduler_value(global_step, total_steps, wd_start, final=wd_end)
        teacher_mom = cosine_scheduler_value(global_step, total_steps, mom_base, final=mom_final)
        set_lr(optimizer, lr)
        set_wd(optimizer, wd)

        # Move to the resolved device
        crops = [c.to(device, non_blocking=True) for c in crops]
        masks = [m.to(device, non_blocking=True) for m in masks]

        # DDP-visible forward. Calling raw_model/module here bypasses gradient
        # all-reduce and was a collapse-grade bug in the previous run.
        student_cls_list, student_patch_list, teacher_cls_list, teacher_patch_list = model(
            crops, masks, n_global
        )

        with torch.no_grad():
            mask_counts = torch.cat([m.flatten(1).sum(dim=1).float() for m in masks])
            mask_ratio = (mask_counts / masks[0].flatten(1).shape[1]).mean().item()
            zero_mask_frac = (mask_counts == 0).float().mean().item()
            teacher_cls_cat = torch.cat(teacher_cls_list, dim=0)
            proto = teacher_cls_cat.argmax(dim=-1)
            proto_max_frac = torch.bincount(proto, minlength=teacher_cls_cat.shape[-1]).float().max().item()
            proto_max_frac /= max(proto.numel(), 1)
            teacher_std = teacher_cls_cat.std().item()

        # Loss
        loss, cls_loss, patch_loss = criterion(
            student_cls_list,
            teacher_cls_list,
            student_patch_list,
            teacher_patch_list,
            masks,
            epoch,
        )

        # Guard against NaN/Inf — skip optimizer step rather than corrupt the model
        if not math.isfinite(loss.item()):
            if is_main():
                print(f"  [WARN] Non-finite loss={loss.item():.6f} at "
                      f"epoch={epoch} step={i}, skipping batch.")
            continue

        optimizer.zero_grad()
        loss.backward()

        # Official iBOT clips each parameter tensor independently.
        clip_gradients([raw_model.student, raw_model.head], cfg["training"]["grad_clip"])

        # ── freeze_last_layer: zero gradients for the head's final linear layer ──
        # Prevents CLS-token representation collapse during the first N epochs.
        # This is critical for DINO/iBOT — without it the model collapses to
        # loss ≈ 0 within the first few warmup epochs.
        if epoch < freeze_last_layer:
            for name, param in raw_model.head.named_parameters():
                if "last_layer" in name and param.grad is not None:
                    param.grad = None

        optimizer.step()

        # EMA teacher update (once per step)
        raw_model.update_teacher(teacher_mom)

        bs = crops[0].size(0)
        loss_meter.update(loss.item(), bs)
        cls_loss_meter.update(cls_loss.item(), bs)
        patch_loss_meter.update(patch_loss.item(), bs)
        mask_ratio_meter.update(mask_ratio, bs)
        zero_mask_meter.update(zero_mask_frac, bs)
        teacher_proto_meter.update(proto_max_frac, bs)
        teacher_std_meter.update(teacher_std, bs)

        if is_main() and i % print_freq == 0:
            elapsed = time.time() - t0
            print(
                f"  [{epoch}][{i:5d}/{n_steps_per_epoch}]  "
                f"loss={loss_meter.avg:.4f} "
                f"(cls={cls_loss_meter.avg:.4f} patch={patch_loss_meter.avg:.4f})  "
                f"lr={lr:.2e}  wd={wd:.4f}  mom={teacher_mom:.5f}  t={elapsed:.1f}s"
            )
            print(
                f"        mask_ratio={mask_ratio_meter.avg:.3f} zero_mask={zero_mask_meter.avg:.3f} "
                f"teacher_std={teacher_std_meter.avg:.3f} teacher_proto_max={teacher_proto_meter.avg:.3f}"
            )

        if writer and is_main():
            step = global_step
            writer.add_scalar("train/loss",        loss_meter.val,  step)
            writer.add_scalar("train/cls_loss",    cls_loss_meter.val,   step)
            writer.add_scalar("train/patch_loss",  patch_loss_meter.val, step)
            writer.add_scalar("train/lr",          lr,                   step)
            writer.add_scalar("train/wd",          wd,                   step)
            writer.add_scalar("train/teacher_mom", teacher_mom,          step)

    return loss_meter.avg, cls_loss_meter.avg, patch_loss_meter.avg


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="iBOT Step 1: ViT-Small/16 original settings")
    parser.add_argument("--config",    default="configs/step1.yaml")
    parser.add_argument("--data_path", default=None,
                        help="Override ImageNet root (contains train/ and val/)")
    parser.add_argument("--resume",    default=None)
    parser.add_argument("--device",    default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Added by the port; the captured trainer assumed "
                             "CUDA")
    return parser


def run(args, config: dict | None = None) -> dict:
    """The captured `main()`, with the config allowed to arrive in memory.

    Returning the metrics is the other change: the captured version computed
    the epoch loss and its cls/patch components and then dropped them, so there
    was nothing for an adapter to record.
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
        writer  = SummaryWriter(log_dir)
        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg, f)

    # ── Build model ────────────────────────────────────────────────────────
    embed_dim    = cfg["model"]["embed_dim"]          # 384 for ViT-S
    out_dim      = cfg["ibot"]["out_dim"]              # 8192
    shared_head  = bool(cfg["ibot"].get("shared_head", True))
    if not shared_head:
        raise ValueError("This local iBOT implementation supports the official shared_head=true protocol only.")

    arch = cfg["model"]["arch"]
    if arch not in _VIT_BUILDERS:
        raise ValueError(
            f"model.arch is {arch!r}; step 1 supports "
            f"{', '.join(sorted(_VIT_BUILDERS))} only (vit_base belongs to "
            "step 2, which this port does not include)")
    builder = _VIT_BUILDERS[arch]
    student_vit  = builder(patch_size=cfg["model"]["patch_size"],
                           drop_path_rate=cfg["model"].get("drop_path_rate", 0.1),
                           use_mask_token=True)
    teacher_vit  = builder(patch_size=cfg["model"]["patch_size"],
                           drop_path_rate=0.0,
                           use_mask_token=False)   # teacher never needs mask token

    head = DINOHead(
        in_dim=embed_dim,
        out_dim=out_dim,
        hidden_dim=cfg["ibot"]["head_hidden_dim"],
        bottleneck_dim=cfg["ibot"]["head_bottleneck_dim"],
        nlayers=cfg["ibot"]["head_nlayers"],
        norm_last_layer=cfg["ibot"].get("norm_last_layer", True),
    )

    ibot_model = iBOT(student_vit, teacher_vit, head).to(device)

    if distributed:
        ibot_model = DDP(ibot_model, device_ids=[local_rank], find_unused_parameters=False)

    raw_model = ibot_model.module if distributed else ibot_model

    # ── Loss ───────────────────────────────────────────────────────────────
    criterion = iBOTLoss(
        out_dim=out_dim,
        patch_out_dim=out_dim,
        student_temp=cfg["ibot"]["student_temp"],
        teacher_temp=cfg["ibot"]["teacher_temp"],
        teacher_patch_temp=cfg["ibot"].get("teacher_patch_temp", cfg["ibot"]["teacher_temp"]),
        teacher_temp_warmup=cfg["ibot"]["teacher_temp_warmup"],
        teacher_patch_temp_warmup=cfg["ibot"].get(
            "teacher_patch_temp_warmup", cfg["ibot"]["teacher_temp_warmup"]
        ),
        teacher_temp_warmup_epochs=cfg["ibot"]["teacher_temp_warmup_epochs"],
        center_momentum=cfg["ibot"]["center_momentum"],
        center_momentum_patch=cfg["ibot"].get("center_momentum_patch", cfg["ibot"]["center_momentum"]),
        lambda_token=cfg["ibot"]["lambda_token"],
        n_global_crops=cfg["data"]["n_global_crops"],
        n_local_crops=cfg["data"]["n_local_crops"],
    ).to(device)

    # ── Optimizer: AdamW, separate param groups (no WD on bias/norm) ───────
    params_with_wd    = []
    params_without_wd = []
    for name, p in list(raw_model.student.named_parameters()) + list(raw_model.head.named_parameters()):
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)

    optimizer = optim.AdamW(
        [
            {"params": params_with_wd,    "apply_wd": True},
            {"params": params_without_wd, "apply_wd": False, "weight_decay": 0.0},
        ],
        lr=cfg["training"]["lr"],
        betas=(0.9, 0.999),
    )

    # ── Data loader ────────────────────────────────────────────────────────
    per_gpu_batch = cfg["training"]["batch_size"] // world_size
    train_loader, sampler = get_ibot_dataloader(
        data_path=cfg["data"]["train_path"],
        batch_size=per_gpu_batch,
        num_workers=cfg["data"]["num_workers"],
        n_local_crops=cfg["data"]["n_local_crops"],
        global_size=cfg["data"]["global_size"],
        local_size=cfg["data"]["local_size"],
        patch_size=cfg["model"]["patch_size"],
        global_crops_scale=cfg["data"].get("global_crops_scale", (0.25, 1.0)),
        local_crops_scale=cfg["data"].get("local_crops_scale", (0.05, 0.25)),
        pred_ratio=cfg["ibot"].get("pred_ratio", [cfg["ibot"].get("mask_ratio_min", 0.1),
                                                  cfg["ibot"].get("mask_ratio_max", 0.5)]),
        pred_ratio_var=cfg["ibot"].get("pred_ratio_var", 0.0),
        pred_shape=cfg["ibot"].get("pred_shape", "block"),
        pred_start_epoch=cfg["ibot"].get("pred_start_epoch", 0),
        step="step1",
        distributed=distributed,
    )

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        start_epoch = ckpt["epoch"] + 1
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        criterion.load_state_dict(ckpt["criterion"])
        if is_main():
            print(f"Resumed from epoch {ckpt['epoch']}")

    total_epochs = cfg["training"]["epochs"]
    save_freq    = cfg["training"]["save_freq"]
    stop_after_epochs = cfg["training"].get("stop_after_epochs")

    if is_main():
        print("=" * 75)
        print("iBOT  Step 1: ViT-Small/16 on ImageNet-1k  (arXiv:2111.07832)")
        print(f"  epochs={total_epochs}  batch={cfg['training']['batch_size']}  "
              f"per_gpu={per_gpu_batch}  world_size={world_size}")
        effective_lr = cfg["training"]["lr"] * cfg["training"]["batch_size"] / 256.0
        print(f"  lr_base={cfg['training']['lr']:.2e}  lr_effective={effective_lr:.2e}  "
              f"wd={cfg['training']['weight_decay_start']:.3f}→{cfg['training']['weight_decay_end']:.3f}")
        print(f"  K={out_dim}  crops=2×{cfg['data']['global_size']}+{cfg['data']['n_local_crops']}×{cfg['data']['local_size']}")
        print(f"  crop_scales global={cfg['data'].get('global_crops_scale', (0.25, 1.0))} "
              f"local={cfg['data'].get('local_crops_scale', (0.05, 0.25))}")
        print(f"  pred_ratio={cfg['ibot'].get('pred_ratio')}  "
              f"pred_ratio_var={cfg['ibot'].get('pred_ratio_var')}  "
              f"pred_shape={cfg['ibot'].get('pred_shape', 'block')}  "
              f"lambda_token={cfg['ibot']['lambda_token']}")
        print(f"  shared_head={shared_head}  norm_last_layer={cfg['ibot'].get('norm_last_layer', True)}")
        print(f"  teacher_mom={cfg['ibot']['teacher_momentum_start']}→{cfg['ibot']['teacher_momentum_end']}")
        print(f"  t_teacher={cfg['ibot']['teacher_temp_warmup']}→{cfg['ibot']['teacher_temp']} "
              f"({cfg['ibot']['teacher_temp_warmup_epochs']} ep warmup)  t_student={cfg['ibot']['student_temp']}")
        print("=" * 75)

    # An epoch that never runs leaves these unset. Reporting a loss of zero
    # would be a number where there was no measurement, so the absence is
    # passed on and the adapter counts it.
    avg_loss = cls_loss = patch_loss = None

    for epoch in range(start_epoch, total_epochs):
        if stop_after_epochs is not None and (epoch - start_epoch) >= int(stop_after_epochs):
            if is_main():
                print(f"\n[pilot] stop_after_epochs={stop_after_epochs}; stopping without changing schedule horizon.")
            break
        if distributed and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        if is_main():
            print(f"\n=== Epoch {epoch}/{total_epochs - 1} ===")

        avg_loss, cls_loss, patch_loss = train_one_epoch(
            ibot_model, criterion, train_loader, optimizer,
            epoch, cfg, writer, distributed, device,
        )

        if writer and is_main():
            writer.add_scalar("epoch/loss",       avg_loss,   epoch)
            writer.add_scalar("epoch/cls_loss",   cls_loss,   epoch)
            writer.add_scalar("epoch/patch_loss", patch_loss, epoch)

        if is_main():
            assert_epoch_health(avg_loss, cls_loss, patch_loss, cfg, epoch)

        # Checkpoint every save_freq epochs and at the final epoch
        if is_main() and ((epoch + 1) % save_freq == 0 or epoch == total_epochs - 1):
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            healthy, health_msg = is_checkpoint_healthy(avg_loss, cls_loss, patch_loss, cfg)
            print(f"  [ckpt] Health check: {health_msg}")
            save_ckpt(
                {
                    "epoch":     epoch,
                    "model":     raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "criterion": criterion.state_dict(),
                    "health": {
                        "is_valid": healthy,
                        "message": health_msg,
                        "loss": float(avg_loss),
                        "cls_loss": float(cls_loss),
                        "patch_loss": float(patch_loss),
                    },
                    "config":    cfg,
                },
                ckpt_path,
                mark_valid=healthy,
            )

    main_proc = is_main()
    if writer:
        writer.close()
    if distributed:
        dist.destroy_process_group()
    if main_proc:
        print("\niBOT Step 1 training complete!")

    ran = total_epochs > start_epoch
    return {
        "epochs": total_epochs - start_epoch,
        "final_loss": avg_loss if ran else None,
        "final_cls_loss": cls_loss if ran else None,
        "final_patch_loss": patch_loss if ran else None,
    }


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
