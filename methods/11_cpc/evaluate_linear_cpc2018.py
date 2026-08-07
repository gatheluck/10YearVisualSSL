"""Linear evaluation for visual CPC 2018: a probe on the frozen patch encoder.

The representation is the model this port trains -- the patch encoder read
through `get_encoder()`, which maps a patch grid to the grid-averaged z
(`avg_z`, z_dim-d). Patch grids are built with the deterministic val pipeline
(centre crop, no per-patch augmentation). The probe follows the lab's ARSSL
protocol: features are extracted once and cached, mean-centred and L2-normalised,
and a single linear layer is trained with SGD (momentum) under a cosine schedule;
top-1 and top-5 are reported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_step1_cpc2018 import make_deterministic, resolve_device   # noqa: E402
from data import VisualCPC2018Dataset                               # noqa: E402


def _build_loader(data_root: str, split: str, d: dict, batch_size: int,
                  num_workers: int):
    dataset = VisualCPC2018Dataset(
        str(Path(data_root) / split), mode="val",
        image_size=int(d["img_size"]), source_size=int(d["source_size"]),
        patch_size=int(d["patch_size"]), patch_crop_size=int(d["patch_crop_size"]),
        stride=int(d["stride"]))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=False)
    return dataset, loader


@torch.no_grad()
def extract_features(encoder, loader, device):
    feats, labels = [], []
    for patches, lbs in loader:
        feats.append(encoder(patches.to(device, non_blocking=True)).float().cpu())
        labels.append(lbs)
    return torch.cat(feats), torch.cat(labels)


def normalize_features(train_feats, val_feats, center: bool = True):
    if center:
        mean = train_feats.mean(dim=0, keepdim=True)
        train_feats = train_feats - mean
        val_feats = val_feats - mean
    train_feats = train_feats / (train_feats.norm(dim=1, keepdim=True) + 1e-8)
    val_feats = val_feats / (val_feats.norm(dim=1, keepdim=True) + 1e-8)
    return train_feats, val_feats


def _topk(outputs, labels, k):
    topk = outputs.topk(min(k, outputs.size(1)), dim=1).indices
    return topk.eq(labels.view(-1, 1)).any(dim=1).float().mean().item()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visual CPC 2018 linear eval")
    parser.add_argument("--config", default="configs/linear_eval.yaml")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    return parser


def run(args, config: "dict | None" = None, model=None) -> dict:
    if config is not None:
        cfg = config
    else:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    train = cfg["train"]
    data_root = args.data_path or cfg["data_root"]
    device = resolve_device(getattr(args, "device", "auto"))
    seed = int(cfg.get("seed", 42))
    make_deterministic(seed)

    if model is None:
        from models import build_visual_cpc2018_from_config
        from train_step1_cpc2018 import model_config
        model = build_visual_cpc2018_from_config(model_config(train))
    encoder = model.get_encoder().to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"Visual CPC 2018 linear eval  device={device}  "
          f"img_size={train['img_size']}")

    bs = int(train["batch_size"])
    nw = int(train["num_workers"])
    tr_ds, tr_loader = _build_loader(data_root, "train", train, bs, nw)
    va_ds, va_loader = _build_loader(data_root, "val", train, bs, nw)
    if tr_ds.classes != va_ds.classes:
        raise RuntimeError(
            f"train and val hold different classes: {tr_ds.classes} vs "
            f"{va_ds.classes}")
    num_classes = len(tr_ds.classes)

    train_feats, train_labels = extract_features(encoder, tr_loader, device)
    val_feats, val_labels = extract_features(encoder, va_loader, device)
    train_feats, val_feats = normalize_features(train_feats, val_feats)
    in_dim = train_feats.shape[1]

    classifier = nn.Linear(in_dim, num_classes).to(device)
    nn.init.normal_(classifier.weight, std=0.01)
    nn.init.zeros_(classifier.bias)
    optimizer = torch.optim.SGD(
        classifier.parameters(), lr=float(train["lr"]),
        momentum=float(train["momentum"]),
        weight_decay=float(train["weight_decay"]))
    epochs = int(train["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=epochs)

    tr = torch.utils.data.TensorDataset(train_feats, train_labels)
    va = torch.utils.data.TensorDataset(val_feats, val_labels)
    tr_dl = torch.utils.data.DataLoader(tr, batch_size=bs, shuffle=True)
    va_dl = torch.utils.data.DataLoader(va, batch_size=bs, shuffle=False)

    best_top1 = best_top5_at_best = 0.0
    final_top1 = final_top5 = 0.0
    for epoch in range(epochs):
        classifier.train()
        for feats, labels in tr_dl:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss = F.cross_entropy(classifier(feats), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        classifier.eval()
        top1_sum = top5_sum = n = 0.0
        with torch.no_grad():
            for feats, labels in va_dl:
                feats = feats.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                out = classifier(feats)
                b = feats.size(0)
                top1_sum += _topk(out, labels, 1) * b
                top5_sum += _topk(out, labels, 5) * b
                n += b
        final_top1 = (top1_sum / n) * 100.0 if n else 0.0
        final_top5 = (top5_sum / n) * 100.0 if n else 0.0
        if final_top1 > best_top1:
            best_top1 = final_top1
            best_top5_at_best = final_top5
        print(f"[{epoch + 1:3d}/{epochs}] "
              f"val_top1={final_top1:.2f}% val_top5={final_top5:.2f}%")

    return {
        "best_top1_acc": float(best_top1),
        "best_top5_acc_at_best_top1": float(best_top5_at_best),
        "final_top1_acc": float(final_top1),
        "final_top5_acc": float(final_top5),
        "epochs": epochs,
    }


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
