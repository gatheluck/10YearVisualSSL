"""Jigsaw++ knowledge-transfer stage (Noroozi et al., CVPR 2018, "Boosting SSL
via Knowledge Transfer"), the AlexNet cluster-classification path.

A self-contained re-implementation, ported from the lab's own code. Given a
trained VGG16 pretext encoder: extract its conv4 features for the whole dataset,
L2-normalise and k-means them (faiss) once into k pseudo-labels, then train a
standard AlexNet to classify the pseudo-labels. The AlexNet is the
knowledge-transfer output the linear probe reads.

The lab wrapper runs the clustering and the AlexNet training as two DDP scripts;
this port does both in one single-process fp32 stage, the device is resolved
rather than assumed CUDA, and TensorBoard is dropped. `encoder.pt` is the AlexNet
conv trunk. The clustering uses faiss (GPU / x86_64-linux only), imported lazily
so this module (and resolve_device) load without it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_step1_jigsaw_pp import resolve_device, make_deterministic   # noqa: E402
from models import (build_vgg16_jigsaw_pp_model,                       # noqa: E402
                    build_alexnet_cluster_cls_model)
from data import build_kt_dataset, KTPseudoLabelDataset               # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Jigsaw++ knowledge transfer (cluster + AlexNet)")
    parser.add_argument("--config", default="configs/knowledge_transfer.yaml")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--encoder", default=None,
                        help="VGG16 encoder.pt from a step-1 run")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    return parser


def run(args, config: dict | None = None, vgg_model=None) -> dict:
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
    from utils.clustering import extract_conv4_features, run_kmeans

    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    kt = cfg["kt"]
    t = cfg["training"]
    d = cfg["data"]
    dropout = float(kt["dropout"])
    img_size = int(kt["image_size"])
    num_clusters = int(kt["num_clusters"])

    if vgg_model is None:
        # Standalone: load the VGG16 encoder.pt named on the CLI/config.
        state = torch.load(args.encoder or cfg["encoder"], map_location="cpu",
                           weights_only=True)
        vgg_model = build_vgg16_jigsaw_pp_model(num_classes=701, dropout=dropout)
        vgg_model.load_state_dict(state, strict=False)
    vgg_encoder = vgg_model.encoder.to(device)
    vgg_encoder.eval()
    for p in vgg_encoder.parameters():
        p.requires_grad = False

    # --- cluster once: VGG16 conv4 features -> k-means -> pseudo-labels -------
    feat_ds = build_kt_dataset(d["data_root"], image_size=img_size, train=False)
    feat_loader = torch.utils.data.DataLoader(
        feat_ds, batch_size=int(t["batch_size"]), shuffle=False,
        num_workers=int(t["num_workers"]))
    features = extract_conv4_features(vgg_encoder, feat_loader, device)
    assignments, _ = run_kmeans(features, num_clusters=num_clusters, seed=seed,
                                use_gpu=(device.type == "cuda"), verbose=True)
    unique = np.unique(assignments)
    remap = {int(old): new for new, old in enumerate(unique)}
    pseudo = np.array([remap[int(a)] for a in assignments], dtype=np.int64)
    n_active = len(unique)

    # --- train the AlexNet to classify the pseudo-labels ---------------------
    alexnet = build_alexnet_cluster_cls_model(num_classes=n_active,
                                              dropout=dropout).to(device)
    alexnet.train()
    train_ds = KTPseudoLabelDataset(
        build_kt_dataset(d["data_root"], image_size=img_size, train=True), pseudo)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=int(t["batch_size"]), shuffle=True,
        num_workers=int(t["num_workers"]), drop_last=True,
        generator=torch.Generator().manual_seed(seed))

    optimizer = torch.optim.SGD(alexnet.parameters(), lr=float(t["lr"]),
                                momentum=float(t["momentum"]),
                                weight_decay=float(t["weight_decay"]))
    criterion = nn.CrossEntropyLoss()

    total_epochs = int(t["epochs"])
    print("=" * 70)
    print("Jigsaw++ Knowledge Transfer: VGG16 conv4 -> k-means -> AlexNet  "
          "(Noroozi et al., 2018)")
    print(f"  device={device}  epochs={total_epochs}  images={len(feat_ds)}  "
          f"k={num_clusters}  active_clusters={n_active}")
    print("=" * 70)

    final_loss = None
    for epoch in range(total_epochs):
        running, count = 0.0, 0
        for imgs, pl in loader:
            imgs = imgs.to(device, non_blocking=True)
            pl = pl.to(device, non_blocking=True)
            out = alexnet(imgs)
            loss = criterion(out, pl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
            count += imgs.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] cluster_cls_loss={final_loss}")
        torch.save({"epoch": epoch, "model_state_dict": alexnet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": final_loss, "active_clusters": n_active,
                    "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print(f"\nJigsaw++ knowledge transfer complete! (active clusters: "
          f"{n_active})")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    a = build_parser().parse_args()
    import yaml
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    run(a, cfg)


if __name__ == "__main__":
    main()
