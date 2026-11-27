"""Step-2 unified ViT-B/16 MoCo v2 pretraining, in one process.

A faithful port of the capture's `train_step2_vit.py`: a ViT-B/16 (timm, from
scratch) query/key encoder with a momentum queue and a 2-layer MLP head is
trained with InfoNCE on two augmented views (reusing the port's `MoCoV2Dataset`,
which adds MoCo v2's Gaussian blur). Optimiser AdamW with betas (0.9, 0.95);
linear warmup then cosine decay to `min_lr`. Checkpoints at each `save_at_epochs`
milestone (100/200/300) plus `checkpoint_latest.pth`. The capture's DDP/torchrun
launch and TensorBoard are dropped, as in every port; the device is resolved, not
sniffed. Matching the capture's ViT MoCo loop, there is no AMP and no gradient
clipping.
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

from data import MoCoV2Dataset                                     # noqa: E402
from models.vit_mocov2 import build_mocov2_vit                    # noqa: E402
from train_pretrain_mocov2 import make_deterministic, resolve_device  # noqa: E402


def model_kwargs(m: dict) -> dict:
    """Backbone-shaping build args (from a flat train dict), for load_encoder.
    The queue/momentum/temperature do not shape any saved weight, so they take
    build defaults when rebuilding for a load."""
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
    parser = argparse.ArgumentParser(description="MoCo v2 Step-2 ViT-B/16")
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
    mo = cfg["moco"]
    model = build_mocov2_vit(queue_size=int(mo["queue_size"]),
                             momentum=float(mo["key_momentum"]),
                             temperature=float(mo["temperature"]),
                             **model_kwargs(m)).to(device)
    model.train()

    d = cfg["data"]
    dataset = MoCoV2Dataset(d["data_root"], image_size=int(d["img_size"]))
    tr = cfg["training"]
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(tr["batch_size"]), shuffle=True,
        num_workers=int(tr["num_workers"]), drop_last=True,
        generator=torch.Generator().manual_seed(seed))

    base_lr = float(tr["lr"])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=base_lr,
        weight_decay=float(tr["weight_decay"]), betas=(0.9, 0.95))

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
        print(f"Resumed from epoch {state['epoch']}")

    print("=" * 70)
    print("MoCo v2  pretrain: unified ViT-B/16 (Step 2 protocol, from scratch)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"feature_dim={m['feature_dim']}  K={mo['queue_size']}  "
          f"save_at={sorted(save_at)}")
    print("=" * 70)

    final_loss = None
    for epoch in range(start_epoch, total_epochs):
        lr = lr_at(epoch, base_lr, min_lr, warmup, total_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        running, count = 0.0, 0
        for im_q, im_k, _ in loader:
            im_q = im_q.to(device, non_blocking=True)
            im_k = im_k.to(device, non_blocking=True)
            loss, _, _ = model(im_q, im_k)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * im_q.size(0)
            count += im_q.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] lr={lr:.2e} infonce_loss={final_loss}")

        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(state, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nMoCo v2 Step-2 ViT pretraining complete!")
    ran = total_epochs > start_epoch and final_loss is not None
    return {"epochs": total_epochs - start_epoch,
            "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
