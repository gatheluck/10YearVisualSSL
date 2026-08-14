"""Step-2 unified ViT-B/16 BYOL pretraining, in one process.

A faithful port of the capture's `train_step2_vit.py`: a ViT-B/16 (timm, from
scratch) online encoder + projector + predictor is trained so its prediction of
one view matches an EMA target network's projection of the other, under the
symmetric negative-cosine BYOL loss (reusing the port's two-view `byol` dataset
and `BYOLLoss`). Optimiser AdamW (default betas 0.9/0.999); the batch-linear LR
scaling and warmup->cosine schedule are reused from the native trainer; the EMA
tau follows its cosine schedule per step. Unlike the native path (and the MoCo /
SimSiam ViT trainers), the capture's BYOL ViT loop uses **AMP on CUDA and gradient
clipping**, so both are kept here. Checkpoints at each `save_at_epochs` milestone
(100/200/300) plus `checkpoint_latest.pth`, under the `model_state_dict` key the
native path uses. DDP/torchrun and TensorBoard are dropped; the device is
resolved, not sniffed; the target BN stays in training mode (batch stats) with
target parameters frozen, which is all BYOL's target-network semantics need in a
single process.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import get_byol_dataloader                                # noqa: E402
from models import BYOLLoss, build_byol_vit, compute_ema_tau        # noqa: E402
from train_pretrain_byol import (cosine_lr_with_warmup,             # noqa: E402
                                 make_deterministic, resolve_device)

_MODEL_KEYS = ("encoder_dim", "proj_hidden_dim", "proj_output_dim",
               "pred_hidden_dim", "pred_output_dim", "image_size", "patch_size",
               "depth", "num_heads", "mlp_ratio", "drop_rate", "attn_drop_rate")
_FLOATS = ("mlp_ratio", "drop_rate", "attn_drop_rate")


def model_kwargs(m: dict) -> dict:
    """Full build args, for the trainer (the pretrain model section has all)."""
    return {k: (float(m[k]) if k in _FLOATS else int(m[k])) for k in _MODEL_KEYS}


def encoder_kwargs(m: dict) -> dict:
    """The subset that shapes online_encoder, for load_encoder. The
    projector/predictor dims do not shape any saved online_encoder.* weight, so
    they take build defaults when rebuilding for a load."""
    keys = ("encoder_dim", "image_size", "patch_size", "depth", "num_heads",
            "mlp_ratio", "drop_rate", "attn_drop_rate")
    return {k: (float(m[k]) if k in _FLOATS else int(m[k])) for k in keys}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BYOL Step-2 ViT-B/16")
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

    d = cfg["data"]
    t = cfg["training"]
    model = build_byol_vit(**model_kwargs(cfg["model"])).to(device)
    model.train()
    criterion = BYOLLoss()

    loader, dataset = get_byol_dataloader(
        d["data_root"], batch_size=int(t["batch_size"]),
        num_workers=int(t["num_workers"]), img_size=int(d["image_size"]),
        augmentation=d.get("augmentation", "byol"), seed=seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0,
                                  weight_decay=float(t["weight_decay"]))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    clip_grad = float(t["clip_grad"])

    total_epochs = int(t["epochs"])
    tau_base = float(t.get("ema_tau_base", 0.996))
    tau_final = float(t.get("ema_tau_final", 1.0))
    save_at = {int(e) for e in t["save_at_epochs"]}
    steps = max(1, len(loader))

    print("=" * 70)
    print("BYOL  pretrain: unified ViT-B/16 (Step 2 protocol, from scratch)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"proj_out={cfg['model']['proj_output_dim']}  tau0={tau_base}  "
          f"save_at={sorted(save_at)}")
    print("=" * 70)

    final_loss = None
    for epoch in range(total_epochs):
        lr = cosine_lr_with_warmup(optimizer, epoch, cfg)
        running, count = 0.0, 0
        for i, ((x1, x2), _) in enumerate(loader):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                p1, p2, z1, z2 = model(x1, x2)
                loss = criterion(p1, p2, z1, z2)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
            tau = compute_ema_tau(epoch + i / steps, total_epochs, tau_base,
                                  tau_final)
            model.update_target_network(tau)
            running += loss.item() * x1.size(0)
            count += x1.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] lr={lr:.6f} byol_loss={final_loss}")

        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(state, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nBYOL Step-2 ViT pretraining complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
