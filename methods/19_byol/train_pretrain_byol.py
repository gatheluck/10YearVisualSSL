"""BYOL step 1 (Grill et al., 2020), the ResNet-50 path.

A self-contained re-implementation, ported from the lab's own BYOL code. The
online network (backbone + projector + predictor) is trained so its prediction of
one view matches the target network's projection of the other; the target is an
EMA copy of the online backbone + projector (no predictor, no gradient), updated
each step with a cosine-scheduled momentum tau. The loss is a symmetric negative
cosine similarity -- no negatives, no queue. Optimised with LARS under a cosine LR
schedule with warmup.

The lab wrapper trains under DistributedDataParallel with AMP autocast and logs to
TensorBoard; none is needed for a single-process run, so the loop here is
single-process fp32, the device is resolved rather than assumed CUDA, and AMP /
TensorBoard / tqdm are dropped. `encoder.pt` is the online ResNet-50 backbone; the
projector, predictor and target network are excluded.
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

from models import (build_byol_resnet50, build_lars_optimizer, BYOLLoss,  # noqa: E402
                    compute_ema_tau)
from data import get_byol_dataloader                                      # noqa: E402


def model_config(model: dict) -> dict:
    """The online backbone (all that encoder.pt carries) is a fixed ResNet-50,
    independent of the projector/predictor dims, so load_encoder can rebuild with
    the build defaults."""
    return {}


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


def _base_lr(config: dict) -> float:
    lr = config["training"]["learning_rate"]
    if config["training"].get("lr_scale_by_batch", False):
        base = config["training"].get("lr_scale_base", 256)
        lr = lr * config["training"]["batch_size"] / base
    return lr


def cosine_lr_with_warmup(optimizer, epoch: int, config: dict) -> float:
    """Cosine LR decay with linear warmup, on the batch-scaled base LR."""
    warmup_epochs = config["training"].get("warmup_epochs", 10)
    total_epochs = config["training"]["epochs"]
    min_lr = config["training"].get("min_lr", 0.0)
    base_lr = _base_lr(config)
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BYOL step 1 (ResNet-50)")
    parser.add_argument("--config", default="configs/pretrain.yaml")
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

    d = cfg["data"]
    t = cfg["training"]

    model = build_byol_resnet50(cfg["model"]).to(device)
    model.train()
    optimizer = build_lars_optimizer(model, cfg)
    criterion = BYOLLoss()

    loader, dataset = get_byol_dataloader(
        d["data_root"], batch_size=int(t["batch_size"]),
        num_workers=int(t["num_workers"]), img_size=int(d["image_size"]),
        augmentation=d.get("augmentation", "byol"), seed=seed)

    total_epochs = int(t["epochs"])
    tau_base = float(t.get("ema_tau_base", 0.996))
    tau_final = float(t.get("ema_tau_final", 1.0))
    print("=" * 70)
    print("BYOL  Step 1: ResNet-50 + EMA target + symmetric cosine  "
          "(arXiv:2006.07733)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"proj_out={cfg['model']['proj_output_dim']}  tau0={tau_base}")
    print("=" * 70)

    final_loss = None
    steps = max(1, len(loader))
    for epoch in range(total_epochs):
        lr = cosine_lr_with_warmup(optimizer, epoch, cfg)
        running, count = 0.0, 0
        for i, ((x1, x2), _) in enumerate(loader):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            p1, p2, z1, z2 = model(x1, x2)
            loss = criterion(p1, p2, z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_frac = epoch + i / steps
            tau = compute_ema_tau(step_frac, total_epochs, tau_base, tau_final)
            model.update_target_network(tau)
            running += loss.item() * x1.size(0)
            count += x1.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] lr={lr:.6f} byol_loss={final_loss}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": final_loss, "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nBYOL Step 1 training complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
