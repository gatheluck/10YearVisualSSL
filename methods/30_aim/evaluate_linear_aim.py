"""Linear evaluation for AIM: a probe on the frozen pretrained backbone.

Ported from the lab's `methods/30_aim/evaluate_linear_probe.py`. The representation
is AIM's official pretrained ViT features: the outputs of the last
`num_feature_layers` transformer blocks are averaged and mean-pooled over patches,
one vector per image, and a single linear layer is fitted on frozen features. This
is a genuine SSL representation, so the number is comparable (the capture's
"As-is SSL comparison" reuses the official AIM-600M backbone because the
from-scratch data, DFN-2B+, is not public).

The model is the pinned upstream under `third_party/ml-aim`, imported not copied.
A real run builds AIM-600M (ViT-H/14) and loads the official checkpoint (a
hash-pinned download, passed as `ckpt`); the hermetic smoke leaves `ckpt` empty
and builds a **random tiny** AIM (a few small blocks), so nothing is downloaded.

Changed during the port, and recorded in `provenance.json`:
  - **the device is resolved** (`resolve_device`) rather than assumed CUDA (the
    capture called `.cuda()`);
  - **the backbone is built from explicit dims** via ml-aim's `_aim` (AIM-600M's
    dims for a real run) instead of ml-aim's `load_pretrained`, which downloads
    from HuggingFace at call time -- this port pins the code (submodule) and the
    weights (a hash-pinned download) instead; and
  - **features are extracted in fp32** (the capture used a float16 autocast, a
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


def _find_blocks(aim_model) -> list:
    """The transformer blocks, however ml-aim names the trunk."""
    for attr in ("trunk", "model", "encoder"):
        obj = getattr(aim_model, attr, None)
        if obj is not None:
            for blocks_attr in ("blocks", "layers"):
                blocks = getattr(obj, blocks_attr, None)
                if blocks is not None and hasattr(blocks, "__len__"):
                    return list(blocks)
    for module in aim_model.modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 2:
            return list(module)
    raise RuntimeError("could not locate AIM transformer blocks")


def build_model(official_dir: Path, train: dict, device: "torch.device"):
    """Build the AIM backbone from the pinned ml-aim upstream and freeze it.

    Built from explicit dims via ml-aim's `_aim`; for a real run the dims are
    AIM-600M's (ViT-H/14: embed_dim 1536, 24 blocks, 12 heads, patch 14) and the
    official checkpoint is loaded strict (only the AIM head, absent from the
    backbone checkpoint, is missing). `ckpt` empty -> a random tiny AIM (the
    hermetic smoke)."""
    official_dir = str(official_dir)
    if official_dir not in sys.path:
        sys.path.insert(0, official_dir)
    from aim.v1.torch.models import AIM, _aim

    num_blocks = int(train["num_blocks"])
    num_feature_layers = int(train["num_feature_layers"])
    probe_layers = tuple(range(max(0, num_blocks - num_feature_layers), num_blocks))
    preprocessor, trunk, head = _aim(
        img_size=int(train["img_size"]), patch_size=int(train["patch_size"]),
        embed_dim=int(train["embed_dim"]), num_blocks=num_blocks,
        num_heads=int(train["num_heads"]), probe_layers=probe_layers)
    model = AIM(preprocessor, trunk, head)

    ckpt = train.get("ckpt") or ""
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(
                f"AIM checkpoint carries keys the model does not have: "
                f"{unexpected[:5]}")
        stray = [k for k in missing if not k.startswith("head.")]
        if stray:
            raise RuntimeError(
                f"AIM checkpoint is missing backbone weights: {stray[:5]}. "
                "Only the AIM head is expected to be absent from a backbone "
                "checkpoint; the trunk and preprocessor are not")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_feature(model, imgs, train: dict, device) -> "torch.Tensor":
    """The probed representation: average the last `num_feature_layers` block
    outputs, then mean-pool the patch tokens -> [B, D]."""
    num_feature_layers = int(train["num_feature_layers"])
    blocks = _find_blocks(model)[-num_feature_layers:]
    captured: list = []
    hooks = [b.register_forward_hook(
        lambda _m, _i, out: captured.append(
            out[0] if isinstance(out, (tuple, list)) else out))
        for b in blocks]
    try:
        _ = model(imgs.to(device))
    finally:
        for h in hooks:
            h.remove()
    feats = [f if f.ndim == 3 else f.unsqueeze(1) for f in captured]
    if not feats:
        raise RuntimeError("no AIM features were captured from the hooks")
    stacked = torch.stack(feats, dim=0).mean(dim=0)   # (B, N, D)
    return stacked.mean(dim=1)                         # (B, D), patch mean-pool


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
        feats.append(extract_feature(model, imgs, train, device).float().cpu())
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
    parser = argparse.ArgumentParser(description="AIM linear eval (frozen backbone)")
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
                / "third_party" / "ml-aim" / "aim-v1"
        model = build_model(Path(official_dir), train, device)
    print(f"AIM linear eval  device={device}  name={train.get('name')}  "
          f"backbone={'pretrained' if train.get('ckpt') else 'random (smoke)'}")

    res = int(train["img_size"])
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
