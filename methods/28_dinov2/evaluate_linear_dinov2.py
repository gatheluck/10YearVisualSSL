"""Linear evaluation for DINOv2: a probe on the frozen pretrained backbone.

Ported from the lab's `methods/28_dinov2/evaluate_linear.py`. The representation
is DINOv2's official pretrained ViT CLS token
(`forward_features(x)["x_norm_clstoken"]`), frozen; a single linear layer is
fitted on it. This is a genuine SSL representation, so the number is comparable
(the capture's "As-is comparison" reuses the official DINOv2 backbone because the
from-scratch data, LVD-142M, is not public).

The model is the pinned upstream under `third_party/dinov2`, imported not copied.
A real run loads the official checkpoint (a hash-pinned download, passed as
`ckpt`); the hermetic smoke leaves `ckpt` empty and builds a **random** backbone
(`pretrained=False`), so nothing is downloaded.

Changed during the port, and recorded in `provenance.json`:
  - **the device is resolved** (`resolve_device`) rather than assumed CUDA (the
    capture called torch.hub.load and `.cuda()`);
  - **the xformers path is disabled** (`XFORMERS_DISABLED=1`) so the forward is
    torch-only and reproducible on any machine -- the giant's SwiGLU falls back
    to a torch implementation with the same `w12`/`w3` weight keys, so the
    official weights still load strict; and
  - **features are extracted in fp32** (the capture used a bfloat16 autocast, a
    GPU speed path with no meaningful effect on a frozen-feature linear probe).
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
import torch.nn.functional as F

# The dinov2 model uses xformers when present; disable it before the backbone is
# imported so the forward is the torch fallback -- reproducible on CPU / any GPU.
os.environ.setdefault("XFORMERS_DISABLED", "1")

# The checkpoint is native to a 518px input (a 37x37 position-embedding grid);
# the model is built at that size and the input is interpolated to `resolution`.
_CHECKPOINT_IMG_SIZE = 518


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


def build_model(official_dir: Path, train: dict, device: "torch.device"):
    """Build the DINOv2 backbone from the pinned upstream and freeze it.

    `ckpt` empty -> `pretrained=False`, a random backbone (the hermetic smoke).
    A path -> build the same architecture and load the official state_dict strict
    (0 missing, 0 unexpected: the giant's torch SwiGLU keys match the fused
    checkpoint)."""
    os.environ["XFORMERS_DISABLED"] = "1"
    official_dir = str(official_dir)
    if official_dir not in sys.path:
        sys.path.insert(0, official_dir)
    from dinov2.hub.backbones import (dinov2_vits14, dinov2_vitb14,
                                      dinov2_vitl14, dinov2_vitg14)
    builders = {"dinov2_vits14": dinov2_vits14, "dinov2_vitb14": dinov2_vitb14,
                "dinov2_vitl14": dinov2_vitl14, "dinov2_vitg14": dinov2_vitg14}
    name = train["name"]
    if name not in builders:
        raise ValueError(f"unsupported DINOv2 model: {name}")
    model = builders[name](pretrained=False, img_size=_CHECKPOINT_IMG_SIZE)
    ckpt = train.get("ckpt") or ""
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"{name} checkpoint does not match the model: "
                f"missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_cls(model, imgs, train: dict, device) -> "torch.Tensor":
    """The probed representation.

    Step 2 (from-scratch): the encoder is this port's own ``DINOv2Backbone``,
    whose ``get_cls_token`` returns the CLS ([B, D]) -- the capture's own choice.
    Step 1 (as-is): the official DINOv2 backbone returns a dict; the ``feature_key``
    output (a CLS vector) is taken as is, a token grid ([B, N, D]) mean-pooled."""
    imgs = imgs.to(device)
    if hasattr(model, "get_cls_token"):        # the from-scratch DINOv2Backbone
        return model.get_cls_token(imgs)
    feature_key = train.get("feature_key", "x_norm_clstoken")
    out = model.forward_features(imgs)
    feat = out[feature_key]
    if feat.ndim == 3:
        feat = feat.mean(dim=1)
    return feat


def _build_loader(data_root: str, split: str, resolution: int, batch_size: int,
                  num_workers: int):
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    normalize = T.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
    resize = int(round(resolution / 0.875))
    transform = T.Compose([
        T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(resolution),
        T.ToTensor(),
        normalize,
    ])
    dataset = ImageFolder(str(Path(data_root) / split), transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=False)
    return dataset, loader


@torch.no_grad()
def extract_features(model, loader, train, device):
    feats, labels = [], []
    for imgs, lbs in loader:
        feats.append(extract_cls(model, imgs, train, device).float().cpu())
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
    parser = argparse.ArgumentParser(description="DINOv2 linear eval (frozen backbone)")
    parser.add_argument("--config", default="configs/linear_eval.yaml")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    return parser


def run(args, config: "dict | None" = None, model=None,
        official_dir: "Path | None" = None) -> dict:
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
        if official_dir is None:
            official_dir = Path(__file__).resolve().parent.parent.parent \
                / "third_party" / "dinov2"
        model = build_model(Path(official_dir), train, device)
    if train.get("recipe") == "unified":
        print(f"DINOv2 linear eval (Step 2)  device={device}  "
              f"arch={train.get('arch')}  backbone=trained encoder.pt (CLS probe)")
    else:
        print(f"DINOv2 linear eval  device={device}  arch={train['name']}  "
              f"backbone={'pretrained' if train.get('ckpt') else 'random (smoke)'}")

    res = int(train["resolution"])
    bs = int(train["batch_size"])
    nw = int(train["num_workers"])
    tr_ds, tr_loader = _build_loader(data_root, "train", res, bs, nw)
    va_ds, va_loader = _build_loader(data_root, "val", res, bs, nw)
    if tr_ds.classes != va_ds.classes:
        raise RuntimeError(
            f"train and val hold different classes: {tr_ds.classes} vs "
            f"{va_ds.classes}")
    num_classes = len(tr_ds.classes)

    train_feats, train_labels = extract_features(model, tr_loader, train, device)
    val_feats, val_labels = extract_features(model, va_loader, train, device)
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
