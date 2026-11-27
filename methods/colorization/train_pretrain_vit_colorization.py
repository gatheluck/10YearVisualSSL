"""Step-2 unified ViT-B/16 colorization pretraining, in one process.

A faithful port of the capture's `train_step2_vit.py`: a from-scratch ViT-B/16
(hand-written, no timm) + a CNN decoder predicts the per-pixel 313-bin ab
classification from the grayscale L channel -- the same colorization pretext the
native CNN path uses, with the unified Step-2 recipe: AdamW (betas 0.9/0.999),
linear warmup then cosine decay to `min_lr`, mixed precision on CUDA, and
gradient clipping. Reuses the port's `ColorizationDataset`/`get_class_weights`
and `resolve_device`/`make_deterministic`. Checkpoints at each `save_at_epochs`
milestone (100/200/300) plus `checkpoint_latest.pth`. DDP/torchrun and
TensorBoard are dropped; the device is resolved, not sniffed.
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

from data import ColorizationDataset, get_class_weights            # noqa: E402
from models.vit_colorization import build_vit_colorization          # noqa: E402
from train_pretrain_colorization import (make_deterministic,        # noqa: E402
                                         resolve_device)

_ENCODER_INT = ("patch_size", "embed_dim", "depth", "num_heads")
_FLOAT = ("mlp_ratio", "drop_rate", "attn_drop_rate")


def model_kwargs(m: dict) -> dict:
    """Args for the ViT colorization model. The trunk's position embedding is
    sized to the **crop** the ViT actually sees (not the resize `img_size`), so
    `img_size` here is `crop_size`. `num_bins` shapes only the decoder (not the
    loaded encoder.*), so it defaults when absent -- the linear_eval config,
    which rebuilds only to read the backbone, omits it."""
    out = {k: int(m[k]) for k in _ENCODER_INT}
    out["img_size"] = int(m["crop_size"])
    out.update({k: float(m[k]) for k in _FLOAT})
    if "num_bins" in m:
        out["num_bins"] = int(m["num_bins"])
    return out


def lr_at(epoch: int, base_lr: float, min_lr: float, warmup: int,
          total: int) -> float:
    if warmup > 0 and epoch < warmup:
        return min_lr + (base_lr - min_lr) * float(epoch) / float(warmup)
    span = max(1, total - warmup)
    progress = min(1.0, max(0.0, (epoch - warmup) / span))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colorization Step-2 ViT-B/16")
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
    d = cfg["data"]
    t = cfg["training"]

    model = build_vit_colorization(**model_kwargs({**m, **d})).to(device)
    model.train()

    dataset = ColorizationDataset(d["data_root"], mode="train",
                                  image_size=int(d["img_size"]),
                                  crop_size=int(d["crop_size"]))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(t["batch_size"]), shuffle=True,
        num_workers=int(t["num_workers"]), drop_last=True,
        generator=torch.Generator().manual_seed(seed))

    if bool(t["use_class_rebalancing"]):
        weights = get_class_weights(
            d["data_root"], num_bins=int(m["num_bins"]),
            sample_size=int(t["rebalance_sample_size"]),
            lambda_smooth=float(t["rebalance_lambda"])).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    base_lr = float(t["lr"])
    min_lr = float(t["min_lr"])
    warmup = int(t["warmup_epochs"])
    clip_grad = float(t["clip_grad"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=float(t["weight_decay"]),
                                  betas=(0.9, 0.999))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_epochs = int(t["epochs"])
    save_at = {int(e) for e in t["save_at_epochs"]}

    print("=" * 70)
    print("Colorization  pretrain: unified ViT-B/16 (Step 2 protocol, scratch)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"num_bins={m['num_bins']}  save_at={sorted(save_at)}")
    print("=" * 70)

    final_loss = None
    for epoch in range(total_epochs):
        lr = lr_at(epoch, base_lr, min_lr, warmup, total_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        running, count = 0.0, 0
        for l_channel, ab_target in loader:
            l_channel = l_channel.to(device, non_blocking=True)
            ab_target = ab_target.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(l_channel)
                loss = criterion(logits, ab_target)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * l_channel.size(0)
            count += l_channel.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] lr={lr:.2e} colorization_loss={final_loss}")

        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(state, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nColorization Step-2 ViT pretraining complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
