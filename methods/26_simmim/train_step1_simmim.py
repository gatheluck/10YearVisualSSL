"""SimMIM step 1 (Xie et al., 2022; arXiv:2111.09886), the Swin-B path.

A self-contained re-implementation, ported from the lab's own SimMIM code. A
random block of Swin patch tokens is replaced by a learned mask token, the grid is
encoded, a Conv + PixelShuffle decoder reconstructs pixels, and an L1 loss is
taken only on the masked pixels. AdamW under a per-iteration schedule: linear
warmup then a multistep decay, with the base LR optionally scaled by the global
batch size.

The lab wrapper trains under DistributedDataParallel with AMP autocast and logs to
TensorBoard; none is needed for a single-process run, so the loop here is
single-process fp32, the device is resolved rather than assumed CUDA, and AMP /
TensorBoard / tqdm are dropped. `encoder.pt` is the bare Swin encoder; the learned
mask token and the decoder are excluded.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_simmim_swinb           # noqa: E402
from data import get_simmim_dataloader          # noqa: E402


def resolve_device(spec: str, local_rank: int = 0) -> "torch.device":
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device is 'cuda' but no CUDA device is visible. Ask for "
                "'auto' to accept a CPU; getting a CPU silently would misreport "
                "what ran")
        return torch.device(f"cuda:{local_rank}")
    if spec == "auto":
        return torch.device(f"cuda:{local_rank}"
                            if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unknown device {spec!r}; expected auto, cuda or cpu")


def make_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)


def get_lr_scale(cfg) -> float:
    training = cfg["training"]
    if not training.get("scale_lr_by_global_batch", False):
        return 1.0
    ref_batch = float(training.get("lr_reference_batch_size", 512))
    return float(training["batch_size"]) / ref_batch


def get_lr_at_update(update: int, updates_per_epoch: int, cfg) -> float:
    """SimMIM per-iteration LR: linear warmup then a multistep decay, with the
    base LR optionally scaled by the global batch size."""
    training = cfg["training"]
    lr_scale = get_lr_scale(cfg)
    base_lr = training["lr"] * lr_scale
    warmup_updates = int(training["warmup_epochs"] * updates_per_epoch)
    warmup_lr = training.get("warmup_lr", 0.0) * lr_scale

    if warmup_updates > 0 and update < warmup_updates:
        return warmup_lr + update * (base_lr - warmup_lr) / warmup_updates
    milestones = [int(e) * updates_per_epoch
                  for e in training.get("lr_multisteps", [])]
    gamma = float(training.get("lr_gamma", 0.1))
    return base_lr * (gamma ** bisect_right(milestones, update))


def set_lr_at_update(optimizer, update, updates_per_epoch, cfg) -> float:
    lr = get_lr_at_update(update, updates_per_epoch, cfg)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def get_optimizer_param_groups(model, weight_decay):
    skip = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name in skip:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SimMIM step 1 (Swin-B)")
    parser.add_argument("--config", default="configs/step1.yaml")
    parser.add_argument("--data_path", default=None,
                        help="Override the ImageNet root (must contain train/)")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Added by the port; the lab wrapper assumed CUDA")
    return parser


def run(args, config: "dict | None" = None) -> dict:
    if config is not None:
        cfg = config
    else:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    if getattr(args, "data_path", None):
        cfg["data"]["data_root"] = args.data_path

    device = resolve_device(getattr(args, "device", "auto"))
    seed = int(cfg.get("seed", 0))
    make_deterministic(seed)

    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    m = cfg["model"]
    d = cfg["data"]
    t = cfg["training"]

    model = build_simmim_swinb(
        img_size=int(m["img_size"]), patch_size=int(m["patch_size"]),
        window_size=int(m["window_size"]), embed_dim=int(m["embed_dim"]),
        depths=tuple(m["depths"]), num_heads=tuple(m["num_heads"]),
        mask_patch_size=int(m["mask_patch_size"]),
        drop_path_rate=float(m["drop_path_rate"])).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        get_optimizer_param_groups(model, float(t["weight_decay"])),
        lr=float(t["lr"]), betas=tuple(float(b) for b in t["betas"]),
        weight_decay=0.0)

    loader, dataset = get_simmim_dataloader(
        d["data_root"], img_size=int(m["img_size"]),
        mask_patch_size=int(m["mask_patch_size"]),
        mask_ratio=float(d["mask_ratio"]), batch_size=int(t["batch_size"]),
        num_workers=int(d["num_workers"]), model_patch_size=int(m["patch_size"]),
        return_pixel_mask=False, seed=seed)

    total_epochs = int(t["epochs"])
    clip_grad = float(t["clip_grad"])
    updates_per_epoch = max(1, len(loader))

    print("=" * 72)
    print("SimMIM  Step 1: Swin-B + masked image modeling  (arXiv:2111.09886)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"img_size={m['img_size']}  mask_ratio={d['mask_ratio']}  "
          f"encoder_dim={model.encoder_dim}")
    print("=" * 72)

    final_loss = None
    for epoch in range(total_epochs):
        running, count = 0.0, 0
        for i, (imgs, masks, _) in enumerate(loader):
            set_lr_at_update(optimizer, epoch * updates_per_epoch + i,
                             updates_per_epoch, cfg)
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            loss, _ = model(imgs, masks)
            if not math.isfinite(loss.item()):
                raise FloatingPointError(
                    f"SimMIM loss became non-finite: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            running += loss.item() * imgs.size(0)
            count += imgs.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] simmim_l1_loss={final_loss}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": final_loss, "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nSimMIM Step 1 (Swin-B) training complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
