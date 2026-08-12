"""DeepCluster step 1 (Caron et al., ECCV 2018), the AlexNet-BN path.

A self-contained re-implementation, ported from the lab's own code. Each epoch:
extract fc7 features for the whole training set, PCA-whiten and k-means them
(faiss) into k pseudo-labels, reset the classification head, and train the
backbone + head to predict the pseudo-labels (cross-entropy).

The lab wrapper trains under DistributedDataParallel, gathering features to rank 0
and exchanging cluster assignments through atomic files, and logs to TensorBoard;
none is needed for a single-process run, so this port clusters the whole feature
matrix in-process, the device is resolved rather than assumed CUDA, and
TensorBoard is dropped. `encoder.pt` is the backbone (features + fc6/fc7); the
reset-each-epoch top_layer and the fixed Sobel front-end are excluded.
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

import adapterlib

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_alexnet_deepcluster       # noqa: E402
from data import build_base_dataset, DeepClusterDataset   # noqa: E402

MODEL_KEYS = ("sobel",)


def model_config(model: dict) -> dict:
    """The kwargs build_alexnet_deepcluster needs to rebuild the backbone for
    loading; only ``sobel`` shapes the backbone (top_layer is excluded from
    encoder.pt), so load_encoder can use the build default for num_classes."""
    return {"sobel": bool(model["sobel"])}


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
    parser = argparse.ArgumentParser(description="DeepCluster step 1 (AlexNet-BN)")
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

    # faiss is imported lazily so the module (and resolve_device) load without it.
    from utils.clustering import extract_features_for_clustering, run_kmeans

    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    m = cfg["model"]
    cl = cfg["clustering"]
    d = cfg["data"]
    t = cfg["training"]
    k = int(cl["k"])
    pca_dim = int(cl["pca_dim"])

    model = build_alexnet_deepcluster(sobel=bool(m["sobel"]),
                                      num_classes=k).to(device)

    # DATA_ROOT is the dataset root; step-1 reads its train/ subdirectory. Both
    # the training-transform view and the deterministic feature-extraction view
    # are built over the training images, so both resolve the train/ split here.
    # (build_base_dataset is left root-generic: the linear-eval loader passes an
    # already-resolved split directory and must not be double-joined.)
    train_root = adapterlib.dataset_split_dir(d["data_root"], "train")
    base_train = build_base_dataset(train_root, crop_size=int(d["crop_size"]),
                                    train=True)
    base_feat = build_base_dataset(train_root, crop_size=int(d["crop_size"]),
                                   train=False)
    feat_loader = torch.utils.data.DataLoader(
        base_feat, batch_size=int(t["feat_batch_size"]), shuffle=False,
        num_workers=int(t["num_workers"]))

    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(t["lr"]), momentum=float(t["momentum"]),
        weight_decay=float(t["weight_decay"]))
    decay_epochs = [int(e) for e in t["lr_decay_epochs"]]
    decay_rate = float(t["lr_decay_rate"])
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_epoch = state["epoch"] + 1
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        print(f"Resumed from epoch {state['epoch']}")

    total_epochs = int(t["epochs"])
    print("=" * 70)
    print("DeepCluster  Step 1: AlexNet-BN + PCA/k-means pseudo-labels  "
          "(Caron et al., 2018)")
    print(f"  device={device}  epochs={total_epochs}  images={len(base_train)}  "
          f"k={k}  pca_dim={pca_dim}")
    print("=" * 70)

    final_loss = None
    for epoch in range(start_epoch, total_epochs):
        adjust_lr(optimizer, epoch, float(t["lr"]), decay_epochs, decay_rate)

        feats = extract_features_for_clustering(model, feat_loader, device)
        assignments, _ = run_kmeans(feats, k=k, pca_dim=pca_dim,
                                    use_gpu=(device.type == "cuda"), seed=seed,
                                    verbose=False)
        model.reset_top_layer(k, device, seed=seed)
        model.train()

        train_ds = DeepClusterDataset(base_train, assignments)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=int(t["batch_size"]), shuffle=True,
            num_workers=int(t["num_workers"]), drop_last=True,
            generator=torch.Generator().manual_seed(seed))

        running, count = 0.0, 0
        for imgs, pseudo in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            pseudo = pseudo.to(device, non_blocking=True)
            out = model(imgs)
            loss = criterion(out, pseudo)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
            count += imgs.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] pseudo_label_loss={final_loss}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": final_loss, "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nDeepCluster Step 1 training complete!")
    ran = total_epochs > start_epoch and final_loss is not None
    return {"epochs": total_epochs - start_epoch,
            "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
