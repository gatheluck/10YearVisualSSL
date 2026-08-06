"""Rotation step 1: rotation-classification pretext (Gidaris et al., ICLR 2018).

A self-contained re-implementation, ported from the lab's own code. Each image
is shown as its four right-angle rotations and the AlexNet-BN predicts which
rotation was applied (cross-entropy over the four classes).

The lab wrapper trains under DistributedDataParallel and logs to TensorBoard;
neither is needed for a single-process run, so the loop here is single-process,
the device is resolved rather than assumed CUDA, and TensorBoard is dropped.
`encoder.pt` is the shared AlexNet-BN encoder; the rotation head is excluded.
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

from models import build_alexnet_rotation_model        # noqa: E402
from data import RotationDataset, rotation_collate      # noqa: E402

MODEL_KEYS = ("num_classes", "image_size")


def model_kwargs(model: dict) -> dict:
    """The arguments build_alexnet_rotation_model takes. load_encoder rebuilds
    with the same set, so they live here once."""
    return {"num_classes": int(model["num_classes"])}


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


def _train_transform(image_size: int):
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(256),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rotation step 1 (rotation)")
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
    model = build_alexnet_rotation_model(**model_kwargs(m)).to(device)
    model.train()

    dataset = RotationDataset(
        cfg["data"]["data_root"],
        transform=_train_transform(int(m["image_size"])), normalize=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]), drop_last=True,
        collate_fn=rotation_collate,
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
    print("Rotation  Step 1: rotation-classification pretext  (arXiv:1803.07728)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"classes={m['num_classes']}")
    print("=" * 70)

    final_loss = final_acc = None
    for epoch in range(start_epoch, total_epochs):
        running, correct, count = 0.0, 0, 0
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
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

    print("\nRotation Step 1 training complete!")
    ran = total_epochs > start_epoch and final_loss is not None
    return {"epochs": total_epochs - start_epoch,
            "final_loss": final_loss if ran else None,
            "final_acc": final_acc if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
