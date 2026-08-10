"""PIRL step 1 (Misra & van der Maaten, CVPR 2020), the ResNet-50 path.

A self-contained re-implementation, ported from the lab's own PIRL code. A
ResNet-50 trunk encodes an image and a jigsaw-shuffled view of the same image;
both are contrasted against a momentum-updated memory bank (one row per training
image) with an NCE cross-entropy; the loss is a convex combination of the
image-NCE and the jigsaw-NCE. The bank is updated with the image representation
each step.

The lab wrapper trains under DistributedDataParallel and logs to TensorBoard;
none is needed for a single-process run, so the loop here is single-process fp32,
the device is resolved rather than assumed CUDA, and TensorBoard / tqdm are
dropped. `encoder.pt` is the ResNet-50 trunk; the projection heads are excluded,
and the memory bank lives in the loss module, not the model.
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

from models import build_resnet_pirl           # noqa: E402
from data import build_pirl_loader             # noqa: E402
from loss import PIRLMemoryBankNCE             # noqa: E402


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


def adjust_lr(optimizer, epoch: int, training: dict) -> float:
    """Linear warmup then a step (milestone) decay."""
    base_lr = float(training["lr"])
    warmup = int(training["warmup_epochs"])
    if warmup > 0 and epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        milestones = list(training["lr_milestones"])
        gamma = float(training["lr_gamma"])
        passed = sum(1 for mstone in milestones if epoch >= int(mstone))
        lr = base_lr * (gamma ** passed)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


@torch.no_grad()
def initialize_memory_bank(model, memory_bank, loader, device) -> None:
    """Seed the memory bank with the model's image features (one pass)."""
    model.eval()
    for images, _patches, indices, _labels in loader:
        images = images.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)
        memory_bank.update_memory(model.forward_original(images), indices)
    model.train()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PIRL step 1 (ResNet-50)")
    parser.add_argument("--config", default="configs/step1.yaml")
    parser.add_argument("--data_path", default=None,
                        help="Override the ImageFolder root of training images")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Added by the port; the lab wrapper assumed CUDA")
    return parser


def run(args, config: "dict | None" = None) -> dict:
    if config is not None:
        cfg = config
    else:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    if getattr(args, "data_path", None):
        cfg["data"]["data_root"] = args.data_path

    device = resolve_device(getattr(args, "device", "auto"))
    seed = int(cfg.get("seed", 0))
    make_deterministic(seed)

    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    m = cfg["model"]
    d = cfg["data"]
    n = cfg["nce"]
    t = cfg["training"]
    jigsaw_weight = float(cfg["loss"]["jigsaw_weight"])

    model = build_resnet_pirl(feature_dim=int(m["feature_dim"]),
                              num_patches=int(m["num_patches"])).to(device)
    model.train()

    loader, dataset = build_pirl_loader(
        d["data_root"], d, train=True, batch_size=int(t["batch_size"]),
        num_workers=int(d["num_workers"]), seed=seed)

    memory_bank = PIRLMemoryBankNCE(
        num_samples=len(dataset), feature_dim=int(m["feature_dim"]),
        temperature=float(n["temperature"]), momentum=float(n["momentum"]),
        num_negatives=int(n["num_negatives"])).to(device)

    if bool(cfg["memory"]["initialize_from_model"]):
        initialize_memory_bank(model, memory_bank, loader, device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(t["lr"]), momentum=float(t["momentum"]),
        weight_decay=float(t["weight_decay"]))

    total_epochs = int(t["epochs"])

    print("=" * 72)
    print("PIRL  Step 1: ResNet-50 + jigsaw + memory-bank NCE  (Misra & vdMaaten)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"feature_dim={m['feature_dim']}  jigsaw_weight={jigsaw_weight}  "
          f"num_negatives={n['num_negatives']}")
    print("=" * 72)

    final_loss = None
    for epoch in range(total_epochs):
        lr = adjust_lr(optimizer, epoch, t)
        running, count = 0.0, 0
        for images, patches, indices, _labels in loader:
            images = images.to(device, non_blocking=True)
            patches = patches.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)

            image_features, jigsaw_features = model(images, patches)
            loss_image = memory_bank(image_features, indices)
            loss_jigsaw = memory_bank(jigsaw_features, indices)
            loss = (1.0 - jigsaw_weight) * loss_image + jigsaw_weight * loss_jigsaw

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            memory_bank.update_memory(image_features.detach(), indices)

            running += loss.item() * images.size(0)
            count += images.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] pirl_loss={final_loss}  lr={lr:.6f}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": final_loss, "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nPIRL Step 1 training complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
