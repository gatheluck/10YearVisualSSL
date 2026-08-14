"""Step-2 unified ViT-B/16 MoCo v3 pretraining, in one process.

The capture's Step 2 is the *same* MoCo v3 objective (symmetric InfoNCE, momentum
key encoder), ViT-B/16 backbone, projector and predictor as Step 1 -- only the
recipe changes to the unified one: direct `lr` 6e-4 (Step 1 uses `learning_rate`,
the base LR scaled by batch/256), a **fixed** EMA momentum (Step 1 cosine-anneals
it to 1.0), gradient clipping, and the unified schedule (epochs 300, batch 1024,
wd 0.05, 10-epoch warmup). Milestone checkpoints at `save_at_epochs` (100/200/300).
Selected by `recipe: unified` (absent = the native paper recipe, byte-for-byte
unchanged). Reuses `build_mocov3_vit`, `get_mocov3_dataloader`,
`adjust_learning_rate` and the Step-1 helpers; timm is already a dependency (no
lock change).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_mocov3_vit                                 # noqa: E402
from data import get_mocov3_dataloader                             # noqa: E402
from train_pretrain_mocov3 import (adjust_learning_rate,            # noqa: E402
                                   make_deterministic, resolve_device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MoCo v3 Step-2 unified ViT-B/16")
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

    m = cfg["model"]
    mv = cfg["mocov3"]
    d = cfg["data"]
    t = cfg["training"]

    model = build_mocov3_vit(
        arch=str(m["arch"]), proj_dim=int(m["proj_dim"]),
        mlp_dim=int(m["mlp_dim"]), temperature=float(mv["temperature"]),
        momentum=float(mv["momentum"]),
        stop_grad_conv1=bool(m["stop_grad_conv1"]),
        img_size=int(m["img_size"])).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(t["lr"]),
        betas=tuple(float(b) for b in t["betas"]),
        weight_decay=float(t["weight_decay"]))

    loader, dataset = get_mocov3_dataloader(
        d["data_root"], img_size=int(m["img_size"]),
        batch_size=int(t["batch_size"]), num_workers=int(t["num_workers"]),
        crop_min=float(d["crop_min"]), seed=seed)

    total_epochs = int(t["epochs"])
    base_lr = float(t["lr"])
    min_lr = float(t["min_lr"])
    warmup_epochs = int(t["warmup_epochs"])
    # The unified recipe uses a FIXED EMA momentum (no cosine schedule).
    base_m = float(mv["momentum"])
    clip_grad = float(t["clip_grad"])
    save_at = {int(e) for e in t["save_at_epochs"]}
    steps = max(1, len(loader))

    print("=" * 70)
    print("MoCo v3  pretrain: unified ViT-B/16 (Step 2 protocol, from scratch)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"arch={m['arch']}  fixed_m={base_m}  save_at={sorted(save_at)}")
    print("=" * 70)

    final_loss = None
    for epoch in range(total_epochs):
        running, count = 0.0, 0
        for i, (view1, view2, _) in enumerate(loader):
            progress = epoch + i / steps
            adjust_learning_rate(optimizer, progress, total_epochs, base_lr,
                                 warmup_epochs=warmup_epochs, min_lr=min_lr)
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            loss = model(view1, view2, momentum=base_m)
            optimizer.zero_grad()
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            running += loss.item() * view1.size(0)
            count += view1.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] infonce_loss={final_loss}")

        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(state, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nMoCo v3 Step-2 ViT pretraining complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
