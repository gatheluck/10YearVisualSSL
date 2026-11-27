"""Colorization step 1 (Zhang et al., ECCV 2016), the paper-faithful CNN path.

A self-contained re-implementation, ported from the lab's own colorization code.
The L channel of a Lab image is the input; a VGG-style CNN predicts the ab
channels quantised to 313 bins, trained with a (optionally class-rebalanced)
per-pixel cross-entropy.

The lab wrapper trains under DistributedDataParallel with AMP and logs to
TensorBoard; none is needed for a single-process run, so the loop here is
single-process fp32, the device is resolved rather than assumed CUDA, and
TensorBoard is dropped. `encoder.pt` is the CNN encoder trunk; the decoder and
the 313-bin head are pretext machinery and are excluded.
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

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_colorization_cnn              # noqa: E402
from data import ColorizationDataset, get_class_weights  # noqa: E402

MODEL_KEYS = ("num_bins",)


def model_config(model: dict) -> dict:
    """The model sub-config build_colorization_cnn reads. load_encoder rebuilds
    with the same set, so it lives here once."""
    return {"model": {k: int(model[k]) for k in MODEL_KEYS}}


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colorization step 1 (CNN)")
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
    t = cfg["training"]

    model = build_colorization_cnn(model_config(m)).to(device)
    model.train()

    dataset = ColorizationDataset(d["data_root"], mode="train",
                                  image_size=int(d["img_size"]),
                                  crop_size=int(d["crop_size"]))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(t["batch_size"]), shuffle=True,
        num_workers=int(t["num_workers"]), drop_last=True,
        generator=torch.Generator().manual_seed(seed))

    if bool(t["use_class_rebalancing"]):
        weights = get_class_weights(
            d["data_root"], num_bins=int(m["num_bins"]),
            sample_size=int(t["rebalance_sample_size"]),
            lambda_smooth=float(t["rebalance_lambda"])).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
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
    print("Colorization  Step 1: VGG-style CNN + 313-bin ab classification  "
          "(Zhang et al., 2016)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"num_bins={m['num_bins']}  rebalance={bool(t['use_class_rebalancing'])}")
    print("=" * 70)

    final_loss = None
    for epoch in range(start_epoch, total_epochs):
        adjust_lr(optimizer, epoch, float(t["lr"]), decay_epochs, decay_rate)
        running, count = 0.0, 0
        for l_channel, ab_target in loader:
            l_channel = l_channel.to(device, non_blocking=True)
            ab_target = ab_target.to(device, non_blocking=True)
            outputs = model(l_channel)
            loss = criterion(outputs, ab_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * l_channel.size(0)
            count += l_channel.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] ce_loss={final_loss}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": final_loss, "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nColorization Step 1 training complete!")
    ran = total_epochs > start_epoch and final_loss is not None
    return {"epochs": total_epochs - start_epoch,
            "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
