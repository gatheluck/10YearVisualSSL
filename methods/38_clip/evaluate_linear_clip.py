"""Linear evaluation for CLIP: a probe on the frozen CLIP image tower.

Two backbones share this probe:
- **Step 1 (as-is)** -- the official OpenAI **ViT-B/32**, loaded through the pinned
  `clip.load` from the sha256-pinned download; its image tower is frozen and its
  pooled image embedding (`visual(x)`, i.e. `encode_image`) is probed. A real run
  needs the download (passed as `ckpt`); the hermetic smoke leaves `ckpt` empty and
  builds a **random** tiny image tower, so nothing is downloaded.
- **Step 2 (unified)** -- the from-scratch ViT-B/16 image tower, rebuilt from the
  trained `encoder.pt` (the adapter passes it as `model`), same pooled embedding.

Changed during the port, and recorded in `provenance.json`:
  - the device is **resolved** (`resolve_device`) instead of assumed CUDA;
  - features are extracted in **fp32** (the capture used a bf16 autocast, a GPU
    speed path with no effect on a frozen-feature probe);
  - the probe follows this repository's shared linear protocol -- **mean-centre +
    L2-normalise, a single Linear layer, SGD momentum + cosine, top1/top5** -- so
    the number is comparable across the ported methods. (The capture's own CLIP
    probe used no feature normalisation; the deviation is deliberate and is why the
    Step-2 number stays labelled a supervised-adaptation reference, not a
    comparable VSSL row.)
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_device(spec: str, local_rank: int = 0) -> "torch.device":
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device is 'cuda' but no CUDA device is visible. Ask for 'auto' to "
                "accept a CPU; getting a CPU silently would misreport what ran")
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
    """Build the frozen CLIP image tower for the Step-1 (as-is) probe.

    `ckpt` empty -> a random tiny `VisionTransformer` (the hermetic smoke). A path
    -> the official ViT-B/32 through `clip.load` (checksum-pinned), its `.visual`
    tower. The tower is forced to fp32 and frozen either way."""
    ROOT = Path(__file__).resolve().parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from models import build_clip_visual, load_official_vit_b32

    ckpt = train.get("ckpt") or ""
    if ckpt:
        clip_model = load_official_vit_b32(ckpt, device, verify_checksum=True)
        visual = clip_model.visual
    else:
        visual = build_clip_visual({
            "resolution": int(train["resolution"]),
            "patch_size": int(train["patch_size"]),
            "width": int(train["width"]),
            "layers": int(train["layers"]),
            "heads": int(train["heads"]),
            "output_dim": int(train["output_dim"]),
        })
    visual = visual.float().to(device)
    visual.eval()
    for p in visual.parameters():
        p.requires_grad = False
    return visual


@torch.no_grad()
def extract_cls(model, imgs, train: dict, device) -> "torch.Tensor":
    """The probed representation: the CLIP image tower's pooled embedding [B, D]."""
    imgs = imgs.to(device).float()
    feat = model(imgs)
    if feat.ndim == 3:
        feat = feat.mean(dim=1)
    return feat.float()


def _build_loader(data_root: str, split: str, resolution: int, batch_size: int,
                  num_workers: int):
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder

    # The official CLIP deterministic evaluation transform (clip/clip.py).
    from models import CLIP_MEAN, CLIP_STD
    transform = T.Compose([
        T.Resize(resolution, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(resolution),
        T.Lambda(lambda image: image.convert("RGB")),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
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
    parser = argparse.ArgumentParser(description="CLIP linear eval (frozen backbone)")
    parser.add_argument("--config", default="configs/linear_eval.yaml")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
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
                / "third_party" / "CLIP"
        model = build_model(Path(official_dir), train, device)
    if train.get("recipe") == "unified":
        print(f"CLIP linear eval (Step 2)  device={device}  arch=ViT-B/16  "
              f"backbone=trained encoder.pt (image-tower probe)")
    else:
        print(f"CLIP linear eval (Step 1 as-is)  device={device}  arch=ViT-B/32  "
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
