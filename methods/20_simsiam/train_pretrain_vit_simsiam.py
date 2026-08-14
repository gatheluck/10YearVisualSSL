"""Step-2 unified ViT-B/16 SimSiam pretraining, in one process.

A faithful port of the capture's `train_step2_vit.py`: a ViT-B/16 (timm, from
scratch) with a 3-layer projector and 2-layer predictor is trained with the
negative-cosine SimSiam objective (stop-gradient on the projector outputs) on two
augmented views (reusing the port's `get_simsiam_dataloader`, `augmentation=
"step2"`, and the shared `simsiam_loss`). Optimiser AdamW with betas (0.9, 0.95);
linear warmup then cosine decay to `min_lr`, applied to **every** parameter group
including the predictor (the capture's scheduler fix). Checkpoints at each
`save_at_epochs` milestone (100/200/300) plus `checkpoint_latest.pth`, under the
`state_dict` key the native path uses. The capture's DDP/torchrun launch and
TensorBoard are dropped, as in every port; the device is resolved, not sniffed.
Matching the capture's ViT SimSiam loop, there is no AMP and no gradient clipping.
`z_std` (the L2-normalised-embedding std, SimSiam's collapse monitor) is reported.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import get_simsiam_dataloader                            # noqa: E402
from models import simsiam_loss                                    # noqa: E402
from models.vit_simsiam import build_simsiam_vit                   # noqa: E402
from train_pretrain_resnet import make_deterministic, resolve_device  # noqa: E402


def model_kwargs(m: dict) -> dict:
    """The build args for the ViT SimSiam model, from a flat train dict."""
    return {"dim": int(m["dim"]),
            "pred_dim": int(m["pred_dim"]),
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
    parser = argparse.ArgumentParser(description="SimSiam Step-2 ViT-B/16")
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
    model = build_simsiam_vit(**model_kwargs(m)).to(device)
    model.train()

    d = cfg["data"]
    tr = cfg["training"]
    loader, dataset = get_simsiam_dataloader(
        data_path=d["train_path"], augmentation="step2",
        batch_size=int(tr["batch_size"]), num_workers=int(d["num_workers"]),
        img_size=int(d["img_size"]), distributed=False)

    base_lr = float(tr["lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=float(tr["weight_decay"]),
                                  betas=(0.9, 0.95))

    total_epochs = int(tr["epochs"])
    warmup = int(tr["warmup_epochs"])
    min_lr = float(tr["min_lr"])
    save_at = {int(e) for e in tr["save_at_epochs"]}

    start_epoch = 0
    if getattr(args, "resume", None) and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_epoch = state["epoch"] + 1
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        print(f"Resumed from epoch {state['epoch']}")

    print("=" * 70)
    print("SimSiam  pretrain: unified ViT-B/16 (Step 2 protocol, from scratch)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"dim={m['dim']}  pred_dim={m['pred_dim']}  save_at={sorted(save_at)}")
    print("=" * 70)

    final_loss = final_z_std = None
    for epoch in range(start_epoch, total_epochs):
        lr = lr_at(epoch, base_lr, min_lr, warmup, total_epochs)
        for group in optimizer.param_groups:      # predictor included (fix)
            group["lr"] = lr
        running_loss = running_zstd = count = 0.0
        for v1, v2, _ in loader:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            p1, p2, z1, z2 = model(v1, v2)
            loss = simsiam_loss(p1, p2, z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                z_std = F.normalize(z1, dim=1).std(dim=0).mean().item()
            n = v1.size(0)
            running_loss += loss.item() * n
            running_zstd += z_std * n
            count += n
        final_loss = running_loss / count if count else None
        final_z_std = running_zstd / count if count else None
        print(f"  [{epoch}] lr={lr:.2e} loss={final_loss} z_std={final_z_std}")

        state = {"epoch": epoch, "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(state, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nSimSiam Step-2 ViT pretraining complete!")
    ran = total_epochs > start_epoch and final_loss is not None
    return {"epochs": total_epochs - start_epoch,
            "final_loss": final_loss if ran else None,
            "final_z_std": final_z_std if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
