"""Visual CPC 2018 step 1 (van den Oord et al., 2018), the paper-faithful path.

A self-contained re-implementation, ported from the lab's own visual_cpc2018
code. Each image becomes a patch grid; the encoder maps every patch to a
z-vector, a PixelCNN context autoregresses over the grid, and an InfoNCE loss
predicts future rows.

The lab wrapper trains under DistributedDataParallel with AMP and logs to
TensorBoard; none is needed for a single-process run, so the loop here is
single-process fp32, the device is resolved rather than assumed CUDA, TensorBoard
is dropped, and InfoNCE negatives come from within the batch. `encoder.pt` is the
patch encoder; the context and predictors are excluded.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_visual_cpc2018_from_config     # noqa: E402
from data import VisualCPC2018Dataset                   # noqa: E402

MODEL_KEYS = ("z_dim", "c_dim", "pred_steps", "context_layers",
              "encoder_width_mult")


def model_config(model: dict) -> dict:
    """The model sub-config build_visual_cpc2018_from_config reads. load_encoder
    rebuilds with the same set, so it lives here once."""
    return {"model": {k: model[k] for k in MODEL_KEYS}}


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visual CPC 2018 step 1")
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

    m = cfg["model"]
    d = cfg["data"]
    model = build_visual_cpc2018_from_config(model_config(m)).to(device)
    model.train()

    dataset = VisualCPC2018Dataset(
        d["data_root"], mode="train", image_size=int(d["img_size"]),
        source_size=int(d["source_size"]), patch_size=int(d["patch_size"]),
        patch_crop_size=int(d["patch_crop_size"]), stride=int(d["stride"]))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]), drop_last=True,
        generator=torch.Generator().manual_seed(seed))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg["training"]["lr"]),
        betas=(float(cfg["training"]["beta1"]), float(cfg["training"]["beta2"])),
        weight_decay=float(cfg["training"]["weight_decay"]))

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_epoch = state["epoch"] + 1
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        print(f"Resumed from epoch {state['epoch']}")

    total_epochs = int(cfg["training"]["epochs"])
    print("=" * 70)
    print("Visual CPC 2018  Step 1: patch encoder + PixelCNN + InfoNCE  "
          "(arXiv:1807.03748)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"grid={dataset.grid_size}x{dataset.grid_size}  z_dim={m['z_dim']}")
    print("=" * 70)

    final_loss = None
    for epoch in range(start_epoch, total_epochs):
        running, count = 0.0, 0
        for patches, _ in loader:
            patches = patches.to(device, non_blocking=True)
            z_grid, c_grid = model(patches)
            loss = model.cpc_loss(z_grid, c_grid, use_ddp_negatives=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * patches.size(0)
            count += patches.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] infonce_loss={final_loss}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": final_loss, "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nVisual CPC 2018 Step 1 training complete!")
    ran = total_epochs > start_epoch and final_loss is not None
    return {"epochs": total_epochs - start_epoch,
            "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
