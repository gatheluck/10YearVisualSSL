"""MoCo v1 step 1 (He et al., 2019), the paper-faithful ResNet-50 path.

A self-contained re-implementation, ported from the lab's own MoCo v1 code. Two
augmented views feed a query encoder and a momentum key encoder; an InfoNCE loss
contrasts the query against the matching key and a FIFO queue of K past keys.

The lab wrapper trains under DistributedDataParallel and logs to TensorBoard;
none is needed for a single-process run, so the loop here is single-process fp32,
the device is resolved rather than assumed CUDA, TensorBoard is dropped, and the
queue is filled from within the batch (the shuffle-BN / all-gather paths are kept
but inert single-process). `encoder.pt` is the query ResNet-50 backbone; the
projection head, the key encoder and the queue are excluded.
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

from models import build_moco_resnet          # noqa: E402
from data import MoCoDataset                   # noqa: E402

MODEL_KEYS = ("feature_dim",)


def model_config(model: dict) -> dict:
    """The kwargs build_moco_resnet needs to rebuild the encoder for loading.
    Only feature_dim shapes the backbone/head; the queue params do not, so
    load_encoder can rebuild with the build defaults."""
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


def adjust_lr(optimizer, epoch: int, base_lr: float, decay_epochs, decay_rate):
    steps = sum(epoch >= m for m in decay_epochs)
    lr = base_lr * (decay_rate ** steps)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MoCo v1 step 1 (ResNet-50)")
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
    mo = cfg["moco"]
    d = cfg["data"]
    t = cfg["training"]

    model = build_moco_resnet(
        feature_dim=int(m["feature_dim"]), queue_size=int(mo["queue_size"]),
        momentum=float(mo["key_momentum"]),
        temperature=float(mo["temperature"])).to(device)
    model.train()

    dataset = MoCoDataset(d["data_root"], mode="step1",
                          image_size=int(d["img_size"]))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(t["batch_size"]), shuffle=True,
        num_workers=int(t["num_workers"]), drop_last=True,
        generator=torch.Generator().manual_seed(seed))

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(t["lr"]), momentum=float(t["momentum"]),
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
    print("MoCo v1  Step 1: ResNet-50 + momentum queue  (arXiv:1911.05722)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"feature_dim={m['feature_dim']}  K={mo['queue_size']}")
    print("=" * 70)

    final_loss = None
    for epoch in range(start_epoch, total_epochs):
        adjust_lr(optimizer, epoch, float(t["lr"]), decay_epochs, decay_rate)
        running, count = 0.0, 0
        for im_q, im_k, _ in loader:
            im_q = im_q.to(device, non_blocking=True)
            im_k = im_k.to(device, non_blocking=True)
            loss, _, _ = model(im_q, im_k)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * im_q.size(0)
            count += im_q.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] infonce_loss={final_loss}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": final_loss, "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nMoCo v1 Step 1 training complete!")
    ran = total_epochs > start_epoch and final_loss is not None
    return {"epochs": total_epochs - start_epoch,
            "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
