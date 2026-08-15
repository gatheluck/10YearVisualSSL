"""AIM unified Step-2 pretraining: ViT-B/16 from scratch, single process.

A port of the capture's `methods/30_aim/train_step2_vit.py`. AIM is autoregressive:
a prefix-LM ViT trunk sees a random prefix of patches bidirectionally and predicts
the remaining patches in raster order (per-patch normalised pixels, MSE). There is
no teacher, no multi-crop and no masking collator -- the prefix is sampled inside
the model's forward. The capture ran it under DDP with bf16 autocast, a checkpoint
protocol and TensorBoard; this port owns a thin single-process fp32 loop, resolves
the device instead of assuming CUDA, and drops DDP / autocast / TensorBoard / the
protocol machinery. The recipe -- AdamW (betas 0.9/0.95), a per-epoch warmup->cosine
LR, gradient-norm clipping, the next-patch MSE objective -- is kept faithfully.

`encoder.pt` is the AIM **trunk** (patch embed, positional buffer, transformer
blocks and norm); the MLP prediction head (`predictor.*`) is training machinery and
is excluded. Milestone `checkpoint_epoch_{N}.pth` is written at each
`training.save_at_epochs`.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import AIMViT                        # noqa: E402
from data import get_pretrain_loader             # noqa: E402


def resolve_device(spec: str) -> "torch.device":
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device is 'cuda' but no CUDA device is visible. Ask for 'auto' "
                "to accept a CPU; getting a CPU silently would misreport what ran")
        return torch.device("cuda")
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def cosine_lr(optimizer, epoch, total_epochs, base_lr, min_lr=0.0,
              warmup_epochs=10) -> float:
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def build_aim_model(m: dict) -> AIMViT:
    return AIMViT(
        img_size=int(m["img_size"]), patch_size=int(m["patch_size"]),
        embed_dim=int(m["embed_dim"]), depth=int(m["depth"]),
        num_heads=int(m["num_heads"]), mlp_ratio=float(m["mlp_ratio"]),
        head_depth=int(m["head_depth"]), head_dim=int(m["head_dim"]),
        prefix_fraction_range=(float(m["prefix_fraction_min"]),
                               float(m["prefix_fraction_max"])))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AIM Step 2: ViT-B/16, single process")
    parser.add_argument("--config", default="configs/pretrain_vit.yaml")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser


def run(args, config: "dict | None" = None) -> dict:
    if config is not None:
        cfg = config
    else:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    train_path = getattr(args, "data_path", None) or cfg["data"]["train_path"]
    device = resolve_device(getattr(args, "device", "auto"))
    seed = int(cfg.get("seed", 0))
    make_deterministic(seed)

    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    m, d, t = cfg["model"], cfg["data"], cfg["training"]
    total_epochs = int(t["epochs"])
    batch_size = int(t["batch_size"])
    save_at = {int(n) for n in t.get("save_at_epochs", [])}

    model = build_aim_model(m).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(t["lr"]),
                            betas=(float(t["beta1"]), float(t["beta2"])),
                            weight_decay=float(t["weight_decay"]))
    loader, _ = get_pretrain_loader(
        train_path, batch_size=batch_size, img_size=int(m["img_size"]),
        num_workers=int(d["num_workers"]), distributed=False, step=2,
        persistent_workers=False)

    clip_grad = float(t["grad_clip"])
    warmup_epochs = int(t["warmup_epochs"])
    min_lr = float(t["min_lr"])

    print("=" * 72)
    print("AIM  Step 2: ViT-B/16 from scratch  (prefix-LM next-patch MSE)")
    print(f"  device={device}  epochs={total_epochs}  batch={batch_size}  "
          f"lr={t['lr']}  embed_dim={m['embed_dim']}  save_at_epochs={sorted(save_at)}")
    print("=" * 72)

    final_loss = None
    for epoch in range(total_epochs):
        lr = cosine_lr(optimizer, epoch, total_epochs, float(t["lr"]),
                       min_lr=min_lr, warmup_epochs=warmup_epochs)
        model.train()
        running, count = 0.0, 0
        for imgs, _labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            loss, _, _ = model(imgs)
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"AIM loss became non-finite: {loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            running += loss.item() * imgs.size(0)
            count += imgs.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] aim_loss={final_loss}  lr={lr:.3g}")

        ckpt = {"epoch": epoch, "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(), "loss": final_loss,
                "config": cfg}
        torch.save(ckpt, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(ckpt, os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nAIM Step 2 training complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
