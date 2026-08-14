"""Step-2 unified ViT-B/16 Instance Discrimination pretraining, in one process.

A faithful port of the capture's `train_step2_vit.py`: a ViT-B/16 (timm, from
scratch) + a 128-d L2-normalised head is trained with the non-parametric NCE loss
over a momentum memory bank (one row per training instance), reusing the port's
`ImageFolderWithIndex` (which carries the dataset index the bank is keyed on) and
`NCELoss`. Optimiser AdamW (default betas); linear warmup then cosine decay to
`min_lr`. Checkpoints at each `save_at_epochs` milestone (100/200/300) plus
`checkpoint_latest.pth`; the memory bank rides in the checkpoint under `memory`
and a resume restores it, exactly as the native path does. DDP/torchrun and
TensorBoard are dropped; the device is resolved, not sniffed. Matching the
capture's ViT NCE loop, there is no AMP and no gradient clipping.
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

from data import ImageFolderWithIndex, get_instdisc_transforms    # noqa: E402
from nce import NCELoss                                           # noqa: E402
from models.vit_instdisc import build_vit_instdisc               # noqa: E402
from train_pretrain_instdisc import make_deterministic, resolve_device  # noqa: E402


def model_kwargs(m: dict) -> dict:
    return {"feature_dim": int(m["feature_dim"]),
            "image_size": int(m["img_size"]),
            "patch_size": int(m["patch_size"]),
            "embed_dim": int(m["embed_dim"]),
            "depth": int(m["depth"]),
            "num_heads": int(m["num_heads"]),
            "mlp_ratio": float(m["mlp_ratio"]),
            "drop_rate": float(m["drop_rate"]),
            "attn_drop_rate": float(m["attn_drop_rate"])}


def lr_at(epoch: int, base_lr: float, min_lr: float, warmup: int,
          total: int) -> float:
    if warmup > 0 and epoch < warmup:
        return base_lr * float(epoch + 1) / float(warmup)
    span = max(1, total - warmup)
    progress = min(1.0, max(0.0, (epoch - warmup) / span))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="InstDisc Step-2 ViT-B/16")
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
    model = build_vit_instdisc(**model_kwargs(m)).to(device)
    model.train()

    dataset = ImageFolderWithIndex(
        cfg["data"]["data_root"],
        transform=get_instdisc_transforms("train", int(cfg["data"]["img_size"])))
    tr = cfg["training"]
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(tr["batch_size"]), shuffle=True,
        num_workers=int(tr["num_workers"]), drop_last=True,
        generator=torch.Generator().manual_seed(seed))

    nce_fn = NCELoss(
        num_samples=len(dataset), feature_dim=int(m["feature_dim"]),
        temperature=float(cfg["nce"]["temperature"]),
        momentum=float(cfg["nce"]["momentum"]),
        num_negatives=int(cfg["nce"]["num_negatives"])).to(device)

    base_lr = float(tr["lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=float(tr["weight_decay"]))
    total_epochs = int(tr["epochs"])
    warmup = int(tr["warmup_epochs"])
    min_lr = float(tr["min_lr"])
    save_at = {int(e) for e in tr["save_at_epochs"]}

    start_epoch = 0
    if getattr(args, "resume", None) and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_epoch = state["epoch"] + 1
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        nce_fn.memory.copy_(state["memory"])
        print(f"Resumed from epoch {state['epoch']}")

    print("=" * 70)
    print("InstDisc  pretrain: unified ViT-B/16 (Step 2 protocol, from scratch)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"feature_dim={m['feature_dim']}  negatives={cfg['nce']['num_negatives']}"
          f"  save_at={sorted(save_at)}")
    print("=" * 70)

    final_loss = None
    for epoch in range(start_epoch, total_epochs):
        lr = lr_at(epoch, base_lr, min_lr, warmup, total_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        running, count = 0.0, 0
        for imgs, idx, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)
            feats = model(imgs)
            loss = nce_fn(feats, idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nce_fn.update_memory(feats, idx)
            running += loss.item() * idx.size(0)
            count += idx.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] lr={lr:.2e} nce_loss={final_loss}")

        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "memory": nce_fn.memory.cpu(), "loss": final_loss,
                 "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(state, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nInstDisc Step-2 ViT pretraining complete!")
    ran = total_epochs > start_epoch and final_loss is not None
    return {"epochs": total_epochs - start_epoch,
            "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
