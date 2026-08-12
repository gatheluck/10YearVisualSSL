"""Jigsaw step 1: permutation-classification pretext (Noroozi & Favaro, 2016).

A self-contained re-implementation, ported from the lab's own code. Tiles are
permuted and the CFN predicts which permutation was applied (cross-entropy over
the permutation set).

The lab wrapper trains under DistributedDataParallel and logs to TensorBoard;
neither is needed for a single-process run, so the loop here is single-process,
the device is resolved rather than assumed CUDA, and TensorBoard is dropped.
`encoder.pt` is the shared CFN encoder; the permutation classifier is excluded.
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

from models import build_alexnet_jigsaw_model         # noqa: E402
from data import JigsawPuzzleDataset                  # noqa: E402

MODEL_KEYS = ("num_permutations", "dropout", "tile_size", "tile_gap",
              "image_size")


def model_kwargs(model: dict) -> dict:
    """The arguments build_alexnet_jigsaw_model takes. load_encoder rebuilds with
    the same set, so they live here once."""
    return {"num_classes": int(model["num_permutations"]),
            "dropout": float(model["dropout"])}


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


def _tile_aug():
    import torchvision.transforms as T
    return T.Compose([
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.1),
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jigsaw step 1 (permutation)")
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
    model = build_alexnet_jigsaw_model(**model_kwargs(m)).to(device)
    model.train()

    dataset = JigsawPuzzleDataset(
        cfg["data"]["data_root"], num_permutations=int(m["num_permutations"]),
        tile_size=int(m["tile_size"]), tile_gap=int(m["tile_gap"]),
        image_size=int(m["image_size"]), mode="train", transform=_tile_aug())
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]), drop_last=True,
        generator=torch.Generator().manual_seed(seed))

    criterion = nn.CrossEntropyLoss()
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
        print(f"Resumed from epoch {state['epoch']}")

    total_epochs = int(cfg["training"]["epochs"])
    print("=" * 70)
    print("Jigsaw  Step 1: permutation-classification pretext  (arXiv:1603.09246)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"permutations={m['num_permutations']}")
    print("=" * 70)

    final_loss = final_acc = None
    for epoch in range(start_epoch, total_epochs):
        running, correct, count = 0.0, 0, 0
        for tiles, labels in loader:
            tiles = tiles.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(tiles)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
        final_loss = running / count if count else None
        final_acc = 100.0 * correct / count if count else None
        print(f"  [{epoch}] ce_loss={final_loss} acc={final_acc}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": final_loss, "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nJigsaw Step 1 training complete!")
    ran = total_epochs > start_epoch and final_loss is not None
    return {"epochs": total_epochs - start_epoch,
            "final_loss": final_loss if ran else None,
            "final_acc": final_acc if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
