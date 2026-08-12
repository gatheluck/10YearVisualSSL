"""Split-Brain Autoencoder step 1 (Zhang et al., CVPR 2017), the AlexNet path.

A self-contained re-implementation, ported from the lab's own code. Two
cross-channel AlexNet branches predict one Lab channel from the other: net1 maps
L -> quantised ab (313 bins), net2 maps ab -> quantised L (50 bins). The loss is
the sum of the two per-pixel cross-entropies (targets downsampled to the decoder
output resolution, as in the lab's loop).

The lab's captured train.py trains the ViT step 2 under DistributedDataParallel
with AdamW and a canonical-contract scheduler; none of that applies to the
single-process AlexNet step 1, so this port owns a thin fp32 loop with a plain
Adam optimiser (the capture ships no AlexNet step-1 recipe; the optimiser knobs
are the port's, exposed in the config). The device is resolved rather than
assumed CUDA. `encoder.pt` is the two branch encoders; the decoders are excluded.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_split_brain_from_config      # noqa: E402
from data import SplitBrainDataset                     # noqa: E402


def model_config(model: dict) -> dict:
    """The split-brain AlexNet has no configurable model params (the ab=313 and
    L=50 bin counts are fixed); load_encoder rebuilds from an empty config."""
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


def adjust_lr(optimizer, epoch: int, base_lr: float, decay_epochs, decay_rate):
    steps = sum(epoch >= m for m in decay_epochs)
    lr = base_lr * (decay_rate ** steps)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def _downsample_target(target, size):
    return F.interpolate(target.unsqueeze(1).float(), size=size,
                         mode="nearest").squeeze(1).long()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split-Brain step 1 (AlexNet)")
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

    model = build_split_brain_from_config(model_config({})).to(device)
    model.train()

    dataset = SplitBrainDataset(d["data_root"], crop_size=int(d["crop_size"]),
                                train=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(t["batch_size"]), shuffle=True,
        num_workers=int(t["num_workers"]), drop_last=True,
        generator=torch.Generator().manual_seed(seed))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(t["lr"]),
        betas=(float(t["beta1"]), float(t["beta2"])),
        weight_decay=float(t["weight_decay"]))
    decay_epochs = [int(e) for e in t["lr_decay_epochs"]]
    decay_rate = float(t["lr_decay_rate"])

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_epoch = state["epoch"] + 1
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        print(f"Resumed from epoch {state['epoch']}")

    total_epochs = int(t["epochs"])
    print("=" * 70)
    print("Split-Brain  Step 1: two cross-channel AlexNets (L<->ab)  "
          "(Zhang et al., CVPR 2017)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}")
    print("=" * 70)

    final_loss = None
    for epoch in range(start_epoch, total_epochs):
        adjust_lr(optimizer, epoch, float(t["lr"]), decay_epochs, decay_rate)
        running, count = 0.0, 0
        for l_input, ab_input, l_target, ab_target, _ in loader:
            l_input = l_input.to(device, non_blocking=True)
            ab_input = ab_input.to(device, non_blocking=True)
            l_target = l_target.to(device, non_blocking=True)
            ab_target = ab_target.to(device, non_blocking=True)
            ab_pred, l_pred = model(l_input, ab_input)
            loss = (criterion(ab_pred, _downsample_target(ab_target,
                                                          ab_pred.shape[2:]))
                    + criterion(l_pred, _downsample_target(l_target,
                                                           l_pred.shape[2:])))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * l_input.size(0)
            count += l_input.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] cross_channel_loss={final_loss}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": final_loss, "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nSplit-Brain Step 1 training complete!")
    ran = total_epochs > start_epoch and final_loss is not None
    return {"epochs": total_epochs - start_epoch,
            "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
