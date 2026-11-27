"""SimMIM Step 2 (unified): ViT-B/16 masked image modelling, single process.

The additive unified path (recipe: unified). It plugs the same SimMIM objective
into a ViT-B/16 encoder (vs the native Swin-B), following the capture's
``train_step2_vit.py``. The lab wrapper trains under DistributedDataParallel with
AMP and logs to TensorBoard; none is needed for a single-process run, so this loop
is single-process fp32, the device is resolved rather than assumed CUDA, and
DDP / AMP / TensorBoard are dropped -- the same treatment ``train_pretrain_simmim.py``
gave the native Swin path.

What is faithful to the capture's step-2 recipe: the lr is used directly (the
capture baked ``lr = 1.5e-4 x 1024/256 = 6e-4`` into the config, so it is not
rescaled), AdamW with the config betas, a linear warmup then cosine decay to
``min_lr``, gradient-norm clipping, and the L1 loss on masked patches (computed by
the model). Masking is pixel-space: the dataloader is asked for a pixel-resolution
mask (``return_pixel_mask=True``, one mask unit = one ViT patch). Milestone
checkpoints ``checkpoint_epoch_{N}.pth`` are written at each ``save_at_epochs``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_simmim_vit                        # noqa: E402
from data import get_simmim_dataloader                     # noqa: E402
# One implementation per method: the device resolution and the seeding are the
# native trainer's, imported here rather than copied.
from train_pretrain_simmim import resolve_device, make_deterministic  # noqa: E402


def warmup_cosine_lr(optimizer, epoch, warmup_epochs, total_epochs,
                     base_lr, min_lr, warmup_lr):
    if epoch < warmup_epochs:
        lr = warmup_lr + (base_lr - warmup_lr) * epoch / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SimMIM Step 2: ViT-B/16, single process")
    parser.add_argument("--config", default="configs/pretrain_vit.yaml")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    return parser


def run(args, config: "dict | None" = None) -> dict:
    if config is not None:
        cfg = config
    else:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    m, d, t = cfg["model"], cfg["data"], cfg["training"]
    data_root = getattr(args, "data_path", None) or d["data_root"]

    device = resolve_device(getattr(args, "device", "auto"))
    seed = int(cfg.get("seed", 0))
    make_deterministic(seed)

    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    model = build_simmim_vit(
        img_size=int(m["img_size"]), patch_size=int(m["patch_size"]),
        mask_patch_size=int(m["mask_patch_size"]), embed_dim=int(m["embed_dim"]),
        depth=int(m["depth"]), num_heads=int(m["num_heads"]),
        mlp_ratio=float(m["mlp_ratio"]),
        drop_path_rate=float(m["drop_path_rate"])).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(t["lr"]),
        betas=tuple(float(b) for b in t["betas"]),
        weight_decay=float(t["weight_decay"]))

    loader, dataset = get_simmim_dataloader(
        data_path=data_root, img_size=int(m["img_size"]),
        mask_patch_size=int(m["mask_patch_size"]), mask_ratio=float(d["mask_ratio"]),
        batch_size=int(t["batch_size"]), num_workers=int(d["num_workers"]),
        model_patch_size=int(m["mask_patch_size"]), return_pixel_mask=True,
        seed=seed)

    total_epochs = int(t["epochs"])
    warmup_epochs = int(t["warmup_epochs"])
    base_lr, min_lr, warmup_lr = float(t["lr"]), float(t["min_lr"]), float(t["warmup_lr"])
    clip_grad = float(t["clip_grad"])
    # Milestone checkpoints for the 100/200/300 frozen-backbone probe sweep.
    save_at = {int(n) for n in t.get("save_at_epochs", [])}

    print("=" * 72)
    print("SimMIM  Step 2: ViT-B/16 masked image modelling  (arXiv:2111.09886)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"embed_dim={m['embed_dim']}  lr={base_lr:.2e}  "
          f"mask_ratio={d['mask_ratio']}  save_at_epochs={sorted(save_at)}")
    print("=" * 72)

    final_loss = None
    for epoch in range(total_epochs):
        lr = warmup_cosine_lr(optimizer, epoch, warmup_epochs, total_epochs,
                              base_lr, min_lr, warmup_lr)
        running, count = 0.0, 0
        for imgs, masks, _labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            loss, _ = model(imgs, masks)
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"SimMIM loss became non-finite: {loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            running += loss.item() * imgs.size(0)
            count += imgs.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] simmim_loss={final_loss}  lr={lr:.3g}")

        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(),
                "loss": final_loss, "config": cfg}
        torch.save(ckpt, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(ckpt, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nSimMIM Step 2 training complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
