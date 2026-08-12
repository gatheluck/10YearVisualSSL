"""Instance Discrimination step 1: ResNet-50 + NCE memory bank (Wu et al., 2018).

A self-contained re-implementation, ported from the lab's own code. A ResNet-50
maps each image to a 128-d L2-normalised embedding; an NCE loss over a momentum
memory bank (one row per training instance) treats every image as its own class.

The lab wrapper trains under DistributedDataParallel and logs to TensorBoard;
neither is needed for a single-process run, so the loop here is single-process,
the device is resolved rather than assumed CUDA, and TensorBoard is dropped.
`encoder.pt` is the ResNet-50 backbone; the projection head and the memory bank
are training machinery and are excluded.
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

from models import build_resnet_instdisc              # noqa: E402
from nce import NCELoss                               # noqa: E402
from data import ImageFolderWithIndex, get_instdisc_transforms   # noqa: E402

MODEL_KEYS = ("feature_dim", "img_size")


def model_kwargs(model: dict) -> dict:
    """The arguments build_resnet_instdisc takes. load_encoder rebuilds with the
    same set, so they live here once."""
    return {"feature_dim": int(model["feature_dim"])}


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
    parser = argparse.ArgumentParser(description="InstDisc step 1 (ResNet + NCE)")
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
    model = build_resnet_instdisc(**model_kwargs(m)).to(device)
    model.train()

    dataset = ImageFolderWithIndex(
        cfg["data"]["data_root"],
        transform=get_instdisc_transforms("train", int(cfg["data"]["img_size"])))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]), drop_last=True,
        generator=torch.Generator().manual_seed(seed))

    nce_fn = NCELoss(
        num_samples=len(dataset), feature_dim=int(m["feature_dim"]),
        temperature=float(cfg["nce"]["temperature"]),
        momentum=float(cfg["nce"]["momentum"]),
        num_negatives=int(cfg["nce"]["num_negatives"])).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(cfg["training"]["lr"]),
        momentum=float(cfg["training"]["momentum"]),
        weight_decay=float(cfg["training"]["weight_decay"]))

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_epoch = state["epoch"] + 1
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        nce_fn.memory.copy_(state["memory"])
        print(f"Resumed from epoch {state['epoch']}")

    total_epochs = int(cfg["training"]["epochs"])
    print("=" * 70)
    print("InstDisc  Step 1: ResNet-50 + NCE memory bank  (arXiv:1805.01978)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"feature_dim={m['feature_dim']}  negatives={cfg['nce']['num_negatives']}")
    print("=" * 70)

    final_loss = None
    for epoch in range(start_epoch, total_epochs):
        running, count = 0.0, 0
        for imgs, idx, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)
            feats = model(imgs)
            loss = nce_fn(feats, idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nce_fn.update_memory(feats, idx)
            running += loss.item() * idx.size(0)
            count += idx.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] nce_loss={final_loss}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "memory": nce_fn.memory.cpu(), "loss": final_loss,
                    "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nInstDisc Step 1 training complete!")
    ran = total_epochs > start_epoch and final_loss is not None
    return {"epochs": total_epochs - start_epoch,
            "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
