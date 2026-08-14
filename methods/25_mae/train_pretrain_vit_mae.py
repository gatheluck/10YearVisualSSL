"""Step-2 unified ViT-B/16 MAE pretraining, in one process.

The capture's Step 2 is the *same* masked-autoencoder objective as Step 1, but on
the unified **ViT-B/16** encoder (`arch: vit_base_patch16`, which `build_mae`
supports) under the unified recipe: AdamW (betas 0.9/0.95, fixed weight decay
0.05) with a **cosine LR schedule + 10-epoch warmup to min_lr** -- which the
native Step-1 trainer does NOT have (it uses a fixed LR) -- and milestone
checkpoints at `save_at_epochs` (100/200/300). Selected by `recipe: unified`
(absent = the native ViT-L/16 recipe, byte-for-byte unchanged). Reuses `build_mae`,
`_build_loader`, `model_kwargs` and the Step-1 helpers; the ViT is the port's own
(no timm), so no lock change.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_mae                                        # noqa: E402
from train_pretrain_mae import (_build_loader, make_deterministic,  # noqa: E402
                                model_kwargs, resolve_device)


def lr_at(epoch: int, base_lr: float, min_lr: float, warmup: int,
          total: int) -> float:
    """Linear warmup then cosine decay to min_lr (per-epoch)."""
    if warmup > 0 and epoch < warmup:
        return base_lr * float(epoch + 1) / float(warmup)
    span = max(1, total - warmup)
    progress = min(1.0, max(0.0, (epoch - warmup) / span))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MAE Step-2 unified ViT-B/16")
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

    if getattr(args, "data_path", None):
        cfg["data"]["data_root"] = args.data_path

    device = resolve_device(getattr(args, "device", "auto"))
    seed = int(cfg.get("seed", 42))
    make_deterministic(seed)

    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    mk = model_kwargs(cfg["model"])
    model = build_mae(cfg["model"]["arch"], **mk).to(device)
    model.train()

    t = cfg["training"]
    base_lr = float(t["lr"])
    min_lr = float(t["min_lr"])
    warmup = int(t["warmup_epochs"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=float(t["weight_decay"]),
        betas=(0.9, 0.95))

    dataset, loader = _build_loader(
        cfg["data"]["data_root"], mk["img_size"], int(t["batch_size"]),
        int(t["num_workers"]), seed)

    total_epochs = int(t["epochs"])
    save_at = {int(e) for e in t["save_at_epochs"]}

    print("=" * 70)
    print("MAE  pretrain: unified ViT-B/16 (Step 2 protocol, from scratch)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"arch={cfg['model']['arch']}  mask_ratio={mk['mask_ratio']}  "
          f"save_at={sorted(save_at)}")
    print("=" * 70)

    final_loss = None
    for epoch in range(total_epochs):
        lr = lr_at(epoch, base_lr, min_lr, warmup, total_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        running, count = 0.0, 0
        for imgs, _labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            loss, _pred, _mask = model(imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
            count += imgs.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] lr={lr:.2e} mse_loss={final_loss}")

        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(state, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nMAE Step-2 ViT pretraining complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
