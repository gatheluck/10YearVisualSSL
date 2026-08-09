"""MoCo v3 step 1 (Chen et al., 2021), the ViT path.

A self-contained re-implementation, ported from the lab's own MoCo v3 code. Two
augmented views feed a base encoder (ViT + projector + predictor) and a momentum
encoder (an EMA copy, no gradient); a symmetric InfoNCE loss contrasts the
predicted query against the momentum key. AdamW under a per-iteration cosine LR
schedule with warmup; the EMA momentum follows a half-cycle cosine from its base
to 1.0.

The lab wrapper trains under DistributedDataParallel with AMP autocast and logs
to TensorBoard; none is needed for a single-process run, so the loop here is
single-process fp32, the device is resolved rather than assumed CUDA, and AMP /
TensorBoard / tqdm are dropped. `encoder.pt` is the base ViT trunk; the
projector, predictor and momentum encoder are excluded.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_mocov3_vit               # noqa: E402
from data import get_mocov3_dataloader            # noqa: E402

def model_config(train: dict) -> dict:
    """The kwargs build_mocov3_vit needs to rebuild the model for loading. Only
    arch + img_size shape the ViT trunk (all that encoder.pt carries); the
    projector/predictor dims shape the excluded heads, so load_encoder rebuilds
    with the build defaults for those."""
    return {"arch": str(train["arch"]), "img_size": int(train["img_size"])}


def resolve_device(spec: str, local_rank: int = 0) -> "torch.device":
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device is 'cuda' but no CUDA device is visible. Ask for "
                "'auto' to accept a CPU; getting a CPU silently would misreport "
                "what ran")
        return torch.device(f"cuda:{local_rank}")
    if spec == "auto":
        return torch.device(f"cuda:{local_rank}"
                            if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unknown device {spec!r}; expected auto, cuda or cpu")


def make_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)


def adjust_learning_rate(optimizer, progress: float, total_epochs: int,
                         base_lr: float, warmup_epochs: int = 40,
                         min_lr: float = 0.0) -> float:
    """Per-iteration linear warmup then cosine annealing (progress is the
    fractional epoch)."""
    if progress < warmup_epochs:
        lr = base_lr * progress / max(warmup_epochs, 1)
    else:
        p = (progress - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * p))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def adjust_moco_momentum(progress: float, total_epochs: int,
                         base_momentum: float) -> float:
    """Half-cycle cosine EMA momentum schedule (--moco-m-cos): base -> 1.0."""
    p = progress / total_epochs
    return 1.0 - 0.5 * (1.0 + math.cos(math.pi * p)) * (1.0 - base_momentum)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MoCo v3 step 1 (ViT)")
    parser.add_argument("--config", default="configs/step1.yaml")
    parser.add_argument("--data_path", default=None,
                        help="Override the ImageFolder root of training images")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Added by the port; the lab wrapper assumed CUDA")
    return parser


def run(args, config: dict | None = None) -> dict:
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
        model.parameters(), lr=float(t["learning_rate"]),
        betas=tuple(float(b) for b in t["betas"]),
        weight_decay=float(t["weight_decay"]))

    loader, dataset = get_mocov3_dataloader(
        d["data_root"], img_size=int(m["img_size"]),
        batch_size=int(t["batch_size"]), num_workers=int(t["num_workers"]),
        crop_min=float(d["crop_min"]), seed=seed)

    total_epochs = int(t["epochs"])
    base_lr = float(t["learning_rate"])
    min_lr = float(t.get("min_lr", 0.0))
    warmup_epochs = int(t.get("warmup_epochs", 40))
    base_m = float(mv["momentum"])
    momentum_cosine = bool(mv.get("momentum_cosine", True))
    steps = max(1, len(loader))

    print("=" * 70)
    print("MoCo v3  Step 1: ViT + momentum encoder + InfoNCE  (arXiv:2104.02057)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"arch={m['arch']}  proj_dim={m['proj_dim']}  tau={mv['temperature']}")
    print("=" * 70)

    final_loss = None
    for epoch in range(total_epochs):
        running, count = 0.0, 0
        for i, (view1, view2, _) in enumerate(loader):
            progress = epoch + i / steps
            adjust_learning_rate(optimizer, progress, total_epochs, base_lr,
                                 warmup_epochs=warmup_epochs, min_lr=min_lr)
            mom = adjust_moco_momentum(progress, total_epochs, base_m) \
                if momentum_cosine else base_m
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            loss = model(view1, view2, momentum=mom)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * view1.size(0)
            count += view1.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] infonce_loss={final_loss}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": final_loss, "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nMoCo v3 Step 1 training complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
