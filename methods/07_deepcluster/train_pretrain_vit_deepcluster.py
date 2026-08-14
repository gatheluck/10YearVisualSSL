"""Step-2 unified ViT-B/16 DeepCluster pretraining, in one process.

A faithful port of the capture's `train_step2_vit.py`: a ViT-B/16 (timm, from
scratch) + a reset-each-epoch linear `top_layer` is trained by the DeepCluster
algorithm. Each epoch, CLS features for the whole training set are extracted (no
grad), PCA-whitened and k-means clustered (faiss) into pseudo-labels, the
`top_layer` is reset, and the network is trained with cross-entropy to predict
them -- exactly the native AlexNet loop, with the ViT backbone and the unified
Step-2 recipe: AdamW (betas 0.9/0.999), linear warmup then cosine decay to
`min_lr`, and mixed precision on CUDA. Checkpoints at each `save_at_epochs`
milestone (100/200/300) plus `checkpoint_latest.pth`. DDP/torchrun is dropped;
the device is resolved, not sniffed; TensorBoard is off.

The clustering reuses the port's `utils.clustering` (faiss), so this path is
GPU / x86_64-linux only, the same as the native path.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

import adapterlib

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import build_base_dataset, DeepClusterDataset            # noqa: E402
from models.vit_deepcluster import build_vit_deepcluster           # noqa: E402
from train_pretrain_deepcluster import (make_deterministic,        # noqa: E402
                                        resolve_device)

_ENCODER_INT = ("image_size", "patch_size", "embed_dim", "depth", "num_heads")
_FLOAT = ("mlp_ratio", "drop_rate", "attn_drop_rate")


def model_kwargs(m: dict) -> dict:
    """Build args for the ViT DeepCluster model. `k` shapes only the
    reset-each-epoch top_layer (not the loaded backbone), so it defaults when
    absent -- the linear_eval config, which rebuilds only to read the backbone,
    omits it."""
    out = {k: int(m[k]) for k in _ENCODER_INT}
    out.update({k: float(m[k]) for k in _FLOAT})
    if "k" in m:
        out["num_classes"] = int(m["k"])
    return out


def lr_at(epoch: int, base_lr: float, min_lr: float, warmup: int,
          total: int) -> float:
    if warmup > 0 and epoch < warmup:
        return base_lr * float(epoch + 1) / float(warmup)
    span = max(1, total - warmup)
    progress = min(1.0, max(0.0, (epoch - warmup) / span))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepCluster Step-2 ViT-B/16")
    parser.add_argument("--config", default="configs/pretrain_vit.yaml")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
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
    image_size = int(d["image_size"])

    model = build_vit_deepcluster(**model_kwargs({**m, "k": k})).to(device)

    # DATA_ROOT is the dataset root; the pretrain stage reads its train/
    # subdirectory. Both the training-transform view and the deterministic
    # feature-extraction view are built over the training images.
    train_root = adapterlib.dataset_split_dir(d["data_root"], "train")
    base_train = build_base_dataset(train_root, crop_size=image_size, train=True)
    base_feat = build_base_dataset(train_root, crop_size=image_size, train=False)
    feat_loader = torch.utils.data.DataLoader(
        base_feat, batch_size=int(t["feat_batch_size"]), shuffle=False,
        num_workers=int(t["num_workers"]))

    base_lr = float(t["lr"])
    min_lr = float(t["min_lr"])
    warmup = int(t["warmup_epochs"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=float(t["weight_decay"]),
                                  betas=(0.9, 0.999))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss()
    total_epochs = int(t["epochs"])
    save_at = {int(e) for e in t["save_at_epochs"]}

    print("=" * 70)
    print("DeepCluster  pretrain: unified ViT-B/16 (Step 2 protocol, from scratch)")
    print(f"  device={device}  epochs={total_epochs}  images={len(base_train)}  "
          f"k={k}  pca_dim={pca_dim}  save_at={sorted(save_at)}")
    print("=" * 70)

    final_loss = None
    for epoch in range(total_epochs):
        lr = lr_at(epoch, base_lr, min_lr, warmup, total_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr

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
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(imgs)
                loss = criterion(out, pseudo)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * imgs.size(0)
            count += imgs.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] lr={lr:.2e} pseudo_label_loss={final_loss}")

        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(state, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nDeepCluster Step-2 ViT pretraining complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
