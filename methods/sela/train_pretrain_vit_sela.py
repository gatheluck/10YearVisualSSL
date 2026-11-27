"""Step-2 unified ViT-B/16 SeLa pretraining, in one process.

A faithful port of the capture's `train_step2_vit.py`: a ViT-B/16 (timm, from
scratch) + a single linear prototype head is trained by SeLa self-labelling --
each epoch, Sinkhorn-Knopp optimal transport turns the prototype logits over the
dataset into balanced hard pseudo-labels, and the network is trained with
cross-entropy on them (reusing the port's `compute_hard_sinkhorn_assignments` and
`create_indexed_train_loader`). Optimiser AdamW (betas 0.9/0.999); linear warmup
then cosine decay to `min_lr`; mixed precision on CUDA (as the capture's ViT SeLa
loop uses). Checkpoints at each `save_at_epochs` milestone (100/200/300) plus
`checkpoint_latest.pth`. DDP/torchrun is dropped; the device is resolved, not
sniffed.
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

from data import create_indexed_train_loader                      # noqa: E402
from utils import compute_hard_sinkhorn_assignments               # noqa: E402
from models.vit_sela import build_vit_sela                        # noqa: E402
from train_pretrain_sela import make_deterministic, resolve_device  # noqa: E402

_ENCODER_INT = ("image_size", "patch_size", "embed_dim", "depth", "num_heads")
_FLOAT = ("mlp_ratio", "drop_rate", "attn_drop_rate")


def model_kwargs(m: dict) -> dict:
    """Build args for the ViT SeLa model. `k` shapes only the prototype head
    (not the loaded backbone), so it defaults when absent -- the linear_eval
    config, which rebuilds only to read the backbone, omits it."""
    out = {k: int(m[k]) for k in _ENCODER_INT}
    out.update({k: float(m[k]) for k in _FLOAT})
    if "k" in m:
        out["num_classes"] = int(m["k"])
    return out


def lr_at(epoch: int, base_lr: float, min_lr: float, warmup: int,
          total: int) -> float:
    if warmup > 0 and epoch < warmup:
        return base_lr * float(epoch + 1) / float(warmup)
    span = max(1, total - warmup)
    progress = min(1.0, max(0.0, (epoch - warmup) / span))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SeLa Step-2 ViT-B/16")
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
    cl = cfg["clustering"]
    d = cfg["data"]
    t = cfg["training"]

    model = build_vit_sela(
        **model_kwargs({**m, "k": cl["k"]})).to(device)
    model.train()

    loader, dataset = create_indexed_train_loader(
        d["data_root"], image_size=int(d["image_size"]),
        batch_size=int(t["batch_size"]), num_workers=int(t["num_workers"]),
        seed=seed)

    base_lr = float(t["lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=float(t["weight_decay"]),
                                  betas=(0.9, 0.999))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_epochs = int(t["epochs"])
    warmup = int(t["warmup_epochs"])
    min_lr = float(t["min_lr"])
    tau = float(cl["temperature"])
    save_at = {int(e) for e in t["save_at_epochs"]}

    print("=" * 70)
    print("SeLa  pretrain: unified ViT-B/16 (Step 2 protocol, from scratch)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"K={cl['k']}  save_at={sorted(save_at)}")
    print("=" * 70)

    final_loss = None
    for epoch in range(total_epochs):
        lr = lr_at(epoch, base_lr, min_lr, warmup, total_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        # Sinkhorn self-labelling over the whole dataset (single prototype head).
        labels = compute_hard_sinkhorn_assignments(
            model, loader, device, num_heads=1,
            n_iters=int(cl["sinkhorn_iters"]), temperature=tau,
            epsilon=float(cl["epsilon"]), lamb=float(cl["lambda"]),
            tol=float(cl["sinkhorn_tol"]), verbose=False).to(device)
        model.train()
        running, count = 0.0, 0
        for images, _lbl, indices in loader:
            images = images.to(device, non_blocking=True)
            targets = labels[indices.to(device, non_blocking=True)]
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = F.cross_entropy(logits / tau, targets)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * images.size(0)
            count += images.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] lr={lr:.2e} sela_loss={final_loss}")

        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(state, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nSeLa Step-2 ViT pretraining complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
