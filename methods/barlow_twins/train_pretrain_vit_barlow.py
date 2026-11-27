"""Step-2 unified ViT-B/16 Barlow Twins pretraining, in one process.

A faithful port of the capture's `train_step2_vit.py`: a ViT-B/16 (timm, from
scratch) + 3-layer projector is trained with the Barlow Twins cross-correlation
loss on two augmented views (reusing the port's `get_barlow_dataloader`,
`augmentation="step2"`, and the in-model loss). Optimiser AdamW (default betas);
linear warmup then cosine decay to `min_lr`. Checkpoints at each `save_at_epochs`
milestone (100/200/300) plus `checkpoint_latest.pth`, under the `state_dict` key
the native path uses. The capture's DDP/torchrun launch and TensorBoard are
dropped, as in every port; the device is resolved, not sniffed. Matching the
capture's ViT Barlow loop, there is no AMP and no gradient clipping.
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

from data import get_barlow_dataloader                             # noqa: E402
from models import build_barlow_vit                                # noqa: E402
from train_pretrain_resnet import make_deterministic, resolve_device  # noqa: E402

_MODEL_KEYS = ("projector", "image_size", "patch_size", "embed_dim", "depth",
               "num_heads", "mlp_ratio", "drop_rate", "attn_drop_rate")
_FLOATS = ("mlp_ratio", "drop_rate", "attn_drop_rate")


def model_kwargs(m: dict) -> dict:
    """Build args for the ViT Barlow model, from a flat train/model dict. The
    key `img_size` maps to the builder's `image_size`; `lambd` is passed
    separately by the trainer and defaults on a load."""
    out = {}
    for k in _MODEL_KEYS:
        src = "img_size" if k == "image_size" else k
        v = m[src]
        out[k] = str(v) if k == "projector" else (
            float(v) if k in _FLOATS else int(v))
    return out


def lr_at(epoch: int, base_lr: float, min_lr: float, warmup: int,
          total: int) -> float:
    if warmup > 0 and epoch < warmup:
        return base_lr * float(epoch + 1) / float(warmup)
    span = max(1, total - warmup)
    progress = min(1.0, max(0.0, (epoch - warmup) / span))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Barlow Twins Step-2 ViT-B/16")
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
        cfg["data"]["train_path"] = str(Path(args.data_path) / "train")

    device = resolve_device(getattr(args, "device", "auto"))
    seed = int(cfg.get("seed", 42))
    make_deterministic(seed)

    save_dir = cfg["checkpoint"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    m = cfg["model"]
    model = build_barlow_vit(lambd=float(cfg["barlow"]["lambd"]),
                             **model_kwargs(m)).to(device)
    model.train()

    d = cfg["data"]
    t = cfg["training"]
    loader, dataset = get_barlow_dataloader(
        d["train_path"], augmentation="step2", batch_size=int(t["batch_size"]),
        num_workers=int(d["num_workers"]), img_size=int(d["img_size"]))

    base_lr = float(t["lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=float(t["weight_decay"]))
    total_epochs = int(t["epochs"])
    warmup = int(t["warmup_epochs"])
    min_lr = float(t["min_lr"])
    save_at = {int(e) for e in t["save_at_epochs"]}

    print("=" * 70)
    print("Barlow Twins  pretrain: unified ViT-B/16 (Step 2 protocol, scratch)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"projector={m['projector']}  lambd={cfg['barlow']['lambd']}  "
          f"save_at={sorted(save_at)}")
    print("=" * 70)

    final_loss = None
    for epoch in range(total_epochs):
        lr = lr_at(epoch, base_lr, min_lr, warmup, total_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        running, count = 0.0, 0
        for y1, y2, _ in loader:
            y1 = y1.to(device, non_blocking=True)
            y2 = y2.to(device, non_blocking=True)
            loss = model(y1, y2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * y1.size(0)
            count += y1.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] lr={lr:.2e} barlow_loss={final_loss}")

        state = {"epoch": epoch, "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(state, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nBarlow Twins Step-2 ViT pretraining complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
