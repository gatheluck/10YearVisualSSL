"""Step-2 unified ViT-B/16 split-brain pretraining, in one process.

A faithful port of the capture's ViT Step-2 path: the split-brain autoencoder's
two cross-channel branches each use a from-scratch half-width ViT-B/16
(`embed_dim=384`, `num_heads=6`) + a conv decoder. ``net1`` predicts the ab bins
from L, ``net2`` predicts the L bins from ab; the loss is the sum of the two
per-pixel cross-entropies (targets downsampled to the decoder resolution), which
reaches every parameter of both branches. Recipe: AdamW (betas 0.9/0.999),
linear warmup then cosine decay to `min_lr`, mixed precision on CUDA. Reuses the
port's `SplitBrainDataset`, `_downsample_target`, `resolve_device` and
`make_deterministic`. Checkpoints at each `save_at_epochs` milestone (100/200/300)
plus `checkpoint_latest.pth`. DDP/torchrun and TensorBoard are dropped; the
device is resolved, not sniffed.
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

from data import SplitBrainDataset                                 # noqa: E402
from models.vit_split_brain import build_split_brain_vit           # noqa: E402
from train_pretrain_split_brain import (_downsample_target,        # noqa: E402
                                        make_deterministic, resolve_device)

_ENCODER_INT = ("patch_size", "embed_dim", "depth", "num_heads")


def model_kwargs(m: dict) -> dict:
    """Args for the ViT split-brain model. The trunks' position embeddings are
    sized to the crop the ViT sees, so `img_size` here is `crop_size`. The two
    heads are fixed at 313 (ab) and 50 (L) bins -- not knobs."""
    out = {k: int(m[k]) for k in _ENCODER_INT}
    out["img_size"] = int(m["crop_size"])
    out["mlp_ratio"] = float(m["mlp_ratio"])
    return out


def lr_at(epoch: int, base_lr: float, min_lr: float, warmup: int,
          total: int) -> float:
    if warmup > 0 and epoch < warmup:
        return base_lr * float(epoch + 1) / float(warmup)
    span = max(1, total - warmup)
    progress = min(1.0, max(0.0, (epoch - warmup) / span))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split-Brain Step-2 ViT-B/16")
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

    model = build_split_brain_vit(**model_kwargs({**m, **d})).to(device)
    model.train()

    dataset = SplitBrainDataset(d["data_root"], crop_size=int(d["crop_size"]),
                                train=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(t["batch_size"]), shuffle=True,
        num_workers=int(t["num_workers"]), drop_last=True,
        generator=torch.Generator().manual_seed(seed))

    criterion = nn.CrossEntropyLoss()
    base_lr = float(t["lr"])
    min_lr = float(t["min_lr"])
    warmup = int(t["warmup_epochs"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=float(t["weight_decay"]),
                                  betas=(0.9, 0.999))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_epochs = int(t["epochs"])
    save_at = {int(e) for e in t["save_at_epochs"]}

    print("=" * 70)
    print("Split-Brain  pretrain: dual half-ViT-B/16 (Step 2, from scratch)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"embed_dim={m['embed_dim']}  save_at={sorted(save_at)}")
    print("=" * 70)

    final_loss = None
    for epoch in range(total_epochs):
        lr = lr_at(epoch, base_lr, min_lr, warmup, total_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        running, count = 0.0, 0
        for l_input, ab_input, l_target, ab_target, _ in loader:
            l_input = l_input.to(device, non_blocking=True)
            ab_input = ab_input.to(device, non_blocking=True)
            l_target = l_target.to(device, non_blocking=True)
            ab_target = ab_target.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                ab_pred, l_pred = model(l_input, ab_input)
                loss = (criterion(ab_pred,
                                  _downsample_target(ab_target, ab_pred.shape[2:]))
                        + criterion(l_pred,
                                    _downsample_target(l_target, l_pred.shape[2:])))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * l_input.size(0)
            count += l_input.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] lr={lr:.2e} cross_channel_loss={final_loss}")

        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(state, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nSplit-Brain Step-2 ViT pretraining complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
