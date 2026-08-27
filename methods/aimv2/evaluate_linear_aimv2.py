"""Linear evaluation for AIMv2: a probe on the frozen pretrained backbone.

AIMv2's as-is comparison freezes the official multimodal-autoregressively
pretrained ViT (`aimv2_large_patch14_224.apple_pt`) and fits a single linear layer
on its pooled features. The representation is a genuine learned feature, so the
number is comparable (the "pretrained-backbone reuse" row, analogous to eva02 /
DINOv2 / Franca).

The model class is timm's -- a pinned pip dependency (`timm==1.0.28`), not a git
submodule. A real run loads the official checkpoint (a hash-pinned download, passed
as `ckpt`) into the named timm architecture; the hermetic smoke leaves `ckpt`
empty and builds a **random tiny** AIMv2-style ViT from the config's architecture
keys, so nothing is downloaded and the pipeline runs on a CPU.

AIMv2's timm architecture is a `VisionTransformer` with RMSNorm, a SwiGLU MLP,
SiLU activation, no class token and average pooling over patch tokens, no qkv/proj
bias -- so the tiny smoke model is built in that style, only smaller.

The probe protocol matches this port's other frozen-backbone evals: extract the
pooled feature once, mean-centre with the train mean then L2-normalise, and fit a
linear layer with SGD (momentum 0.9) on a cosine schedule, reporting top-1/top-5.

Changed during the port, and recorded in `provenance.json`:
  - **the device is resolved** (`resolve_device`) rather than assumed CUDA.
  - **features are extracted in fp32** (no autocast), so the frozen-feature probe
    runs identically on a CPU or a pre-Ampere GPU.
"""

from __future__ import annotations

import argparse
import random
import sys
from functools import partial
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


def build_model(train: dict, device: "torch.device"):
    """Build the AIMv2 backbone from timm and freeze it, `num_classes=0` so it
    yields the pooled pre-classifier feature.

    `ckpt` empty -> a random tiny AIMv2-style `VisionTransformer` built directly
    from the config's architecture keys (the hermetic smoke); nothing is
    downloaded. A path -> the named timm architecture (`train["name"]`, e.g.
    `aimv2_large_patch14_224`) built with `pretrained=False` and the downloaded
    checkpoint loaded into it."""
    import timm
    from timm.layers import RmsNorm, SwiGLU
    from timm.models.vision_transformer import VisionTransformer

    ckpt = train.get("ckpt") or ""
    if ckpt:
        model = timm.create_model(str(train["name"]), pretrained=False,
                                  num_classes=0, img_size=int(train["img_size"]))
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        state = state.get("model", state) if isinstance(state, dict) else state
        missing, unexpected = model.load_state_dict(state, strict=False)
        # The classification head is dropped by num_classes=0; any other missing
        # backbone weight means the checkpoint does not match the architecture.
        backbone_missing = [k for k in missing if not k.startswith("head")]
        if backbone_missing:
            raise RuntimeError(
                f"checkpoint is missing backbone weights: {backbone_missing[:5]}")
    else:
        # AIMv2's timm style: RMSNorm, SwiGLU MLP, SiLU, no class token, average
        # pooling over patch tokens, no qkv/proj bias -- smaller for the smoke.
        model = VisionTransformer(
            img_size=int(train["img_size"]), patch_size=int(train["patch_size"]),
            in_chans=3, num_classes=0, embed_dim=int(train["embed_dim"]),
            depth=int(train["depth"]), num_heads=int(train["num_heads"]),
            class_token=False, fc_norm=False, mlp_ratio=2.75, global_pool="avg",
            qkv_bias=False, proj_bias=False, act_layer="silu",
            norm_layer=partial(RmsNorm, eps=1e-5),
            embed_norm_layer=partial(RmsNorm, eps=1e-5), mlp_layer=SwiGLU)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_feature(model, imgs, device) -> "torch.Tensor":
    """The probed representation: the pooled pre-classifier feature ([B, D])."""
    imgs = imgs.to(device)
    return model(imgs).float()


def _build_loader(data_root: str, split: str, resolution: int, batch_size: int,
                  num_workers: int, mean, std, crop_pct: float,
                  interpolation: str):
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    modes = {"bicubic": T.InterpolationMode.BICUBIC,
             "bilinear": T.InterpolationMode.BILINEAR}
    normalize = T.Normalize(mean=tuple(mean), std=tuple(std))
    resize = int(round(resolution / crop_pct))
    transform = T.Compose([
        T.Resize(resize, interpolation=modes.get(interpolation,
                                                 T.InterpolationMode.BICUBIC)),
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
def extract_features(model, loader, device):
    feats, labels = [], []
    for imgs, lbs in loader:
        feats.append(extract_feature(model, imgs, device).cpu())
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
    parser = argparse.ArgumentParser(
        description="AIMv2 linear eval (frozen backbone)")
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
        model = build_model(train, device)
    print(f"AIMv2 linear eval  device={device}  "
          f"backbone={'pretrained' if train.get('ckpt') else 'random (smoke)'}")

    # The normalisation follows the backbone's own data config (AIMv2 uses a
    # CLIP-style mean/std, not ImageNet's); the crop resolution is taken from the
    # config so the tiny smoke can shrink it.
    import timm.data
    dc = timm.data.resolve_model_data_config(model)
    res = int(train["img_size"])
    bs = int(train["batch_size"])
    nw = int(train["num_workers"])
    loader_kw = dict(mean=dc["mean"], std=dc["std"], crop_pct=dc["crop_pct"],
                     interpolation=dc["interpolation"])
    tr_ds, tr_loader = _build_loader(data_root, "train", res, bs, nw, **loader_kw)
    va_ds, va_loader = _build_loader(data_root, "val", res, bs, nw, **loader_kw)
    if tr_ds.classes != va_ds.classes:
        raise RuntimeError(
            f"train and val hold different classes: {tr_ds.classes} vs "
            f"{va_ds.classes}")
    num_classes = len(tr_ds.classes)

    train_feats, train_labels = extract_features(model, tr_loader, device)
    val_feats, val_labels = extract_features(model, va_loader, device)
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
