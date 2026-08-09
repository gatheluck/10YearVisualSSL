"""SimCLR v1 step 1 (Chen et al., 2020), the paper-faithful ResNet-50 path.

A self-contained re-implementation, ported from the lab's own SimCLR v1 code. Two
independently-augmented views feed a shared ResNet-50 + projection head; the
NT-Xent loss contrasts the matching pair against every other view in the batch,
optimised with LARS under a cosine schedule with linear warmup.

The lab wrapper trains under DistributedDataParallel with SyncBatchNorm and logs
to TensorBoard; none is needed for a single-process run, so the loop here is
single-process fp32, the device is resolved rather than assumed CUDA, and
TensorBoard is dropped (the NT-Xent all-gather path is kept but inert
single-process). `encoder.pt` is the ResNet-50 backbone; the projection head is
excluded.
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

from models import build_resnet_simclr        # noqa: E402
from loss import NTXentLoss                    # noqa: E402
from data import SimCLRDataset                 # noqa: E402
from optim import LARS                         # noqa: E402

MODEL_KEYS = ("out_dim",)


def model_config(model: dict) -> dict:
    """The kwargs build_resnet_simclr needs to rebuild the model for loading.
    Only out_dim shapes the head; the backbone (all that encoder.pt carries) is
    fixed ResNet-50, so load_encoder can rebuild with the build default."""
    return {k: int(model[k]) for k in MODEL_KEYS}


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


def cosine_lr_with_warmup(base_lr: float, epoch: int, total_epochs: int,
                          warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(warmup_epochs, 1)
    p = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * p))


def set_lr(optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SimCLR v1 step 1 (ResNet-50)")
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
    lo = cfg["loss"]
    d = cfg["data"]
    t = cfg["training"]

    model = build_resnet_simclr(out_dim=int(m["out_dim"])).to(device)
    model.train()

    dataset = SimCLRDataset(
        d["data_root"], image_size=int(m["img_size"]),
        color_jitter_strength=float(d["color_jitter_strength"]))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(t["batch_size"]), shuffle=True,
        num_workers=int(t["num_workers"]), drop_last=True,
        generator=torch.Generator().manual_seed(seed))

    criterion = NTXentLoss(temperature=float(lo["temperature"]))
    optimizer = LARS(
        model.parameters(), lr=float(t["lr"]), momentum=float(t["momentum"]),
        weight_decay=float(t["weight_decay"]), eta=float(t["eta"]))

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_epoch = state["epoch"] + 1
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        print(f"Resumed from epoch {state['epoch']}")

    total_epochs = int(t["epochs"])
    warmup_epochs = int(t["warmup_epochs"])
    base_lr = float(t["lr"])
    print("=" * 70)
    print("SimCLR v1  Step 1: ResNet-50 + NT-Xent + LARS  (arXiv:2002.05709)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"out_dim={m['out_dim']}  tau={lo['temperature']}")
    print("=" * 70)

    final_loss = None
    for epoch in range(start_epoch, total_epochs):
        lr = cosine_lr_with_warmup(base_lr, epoch, total_epochs, warmup_epochs)
        set_lr(optimizer, lr)
        running, count = 0.0, 0
        for view1, view2, _ in loader:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            z1 = model(view1)
            z2 = model(view2)
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * view1.size(0)
            count += view1.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] lr={lr:.6f} ntxent_loss={final_loss}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": final_loss, "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nSimCLR v1 Step 1 training complete!")
    ran = total_epochs > start_epoch and final_loss is not None
    return {"epochs": total_epochs - start_epoch,
            "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
