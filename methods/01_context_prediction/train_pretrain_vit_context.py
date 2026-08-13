"""Step-2 unified ViT-B/16 context-prediction pretraining, in one process.

A faithful port of the capture's `train_step2_vit.py`: two patches (center and a
neighbour) share a ViT-B/16 (timm, from scratch); their CLS tokens are
concatenated and an 8-way head predicts the relative position (CrossEntropy).
AdamW; linear warmup then cosine decay to `min_lr`; gradient clipping; mixed
precision on CUDA. Checkpoints at each `save_at_epochs` milestone (100/200/300)
plus a training `final.pth` (the shape 01's adapter reads). Reuses the pinned
`OfficialContextPredictionDataset` at the ViT input size. Config-based (unlike the
native step-based, arg-based trainer); the device is resolved, not sniffed; AMP is
CUDA-only.
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

from data.context_dataset_official import (                          # noqa: E402
    OfficialContextPredictionDataset)
from models.vit_context import build_vit_context_model              # noqa: E402
from train_pretrain_alexnet_official import (make_deterministic,     # noqa: E402
                                             resolve_device)


def model_kwargs(m: dict) -> dict:
    return {"num_classes": int(m["num_classes"]),
            "image_size": int(m["input_size"]),
            "patch_size": int(m["patch_size"]),
            "embed_dim": int(m["embed_dim"]),
            "depth": int(m["depth"]),
            "num_heads": int(m["num_heads"]),
            "mlp_ratio": float(m["mlp_ratio"]),
            "hidden_dim": int(m["hidden_dim"]),
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
    parser = argparse.ArgumentParser(description="Context Step-2 ViT-B/16")
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

    device = resolve_device(getattr(args, "device", "auto"), 0)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    make_deterministic()

    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    m = cfg["model"]
    model = build_vit_context_model(**model_kwargs(m)).to(device)
    model.train()

    dataset = OfficialContextPredictionDataset(
        cfg["data"]["data_root"], patch_size=int(m["input_size"]),
        patch_gap=int(m["patch_gap"]), jitter=int(m["jitter"]), mode="train")
    tr = cfg["training"]
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(tr["batch_size"]), shuffle=True,
        num_workers=int(tr["num_workers"]), drop_last=True,
        generator=torch.Generator().manual_seed(seed))

    criterion = nn.CrossEntropyLoss()
    base_lr = float(tr["lr"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr,
        betas=tuple(float(b) for b in tr["betas"]),
        weight_decay=float(tr["weight_decay"]))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_epochs = int(tr["epochs"])
    warmup = int(tr["warmup_epochs"])
    min_lr = float(tr["min_lr"])
    clip_grad = float(tr["clip_grad"])
    save_at = {int(e) for e in tr["save_at_epochs"]}

    start_epoch = 0
    if getattr(args, "resume", None) and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_epoch = state["epoch"] + 1
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        print(f"Resumed from epoch {state['epoch']}")

    print("=" * 70)
    print("Context  pretrain: unified ViT-B/16 (Step 2 protocol, from scratch)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"embed_dim={m['embed_dim']}  save_at={sorted(save_at)}")
    print("=" * 70)

    final_loss = final_acc = None
    for epoch in range(start_epoch, total_epochs):
        lr = lr_at(epoch, base_lr, min_lr, warmup, total_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        running, correct, count = 0.0, 0, 0
        for first, second, labels in loader:
            first = first.to(device, non_blocking=True)
            second = second.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(first, second)
                loss = criterion(logits, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
        final_loss = running / count if count else None
        final_acc = 100.0 * correct / count if count else None
        print(f"  [{epoch}] lr={lr:.2e} ce_loss={final_loss} acc={final_acc}")

        state = {"epoch": epoch, "state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        torch.save(state, os.path.join(save_dir, "final.pth"))
        if (epoch + 1) in save_at:
            torch.save(state, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nContext Step-2 ViT pretraining complete!")
    ran = total_epochs > start_epoch and final_loss is not None
    # 01's pretrain vocabulary: val_loss/val_acc1 -> final_pretext_*; here they are
    # the pretext loss/accuracy over the last epoch (the port's pretext number).
    return {"global_step": (total_epochs - start_epoch) * max(1, len(loader)),
            "val_loss": final_loss if ran else None,
            "val_acc1": final_acc if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
