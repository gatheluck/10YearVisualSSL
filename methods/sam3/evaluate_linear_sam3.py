"""Linear evaluation for sam3: a probe on the frozen SAM 3 vision encoder.

Meta SAM 3's as-is comparison freezes the official promptable-segmentation model's
vision encoder and fits a single linear layer on its **mean-pooled patch tokens**
(no text/box prompts), matching the capture's Step-3 CompEval read. The
representation is a genuine learned feature, so the number is comparable (the
multimodal "pretrained-backbone reuse" row, the `transformers`-sourced sibling of
`data2vec2`).

The model class is `transformers`' `Sam3ViTModel` -- a pinned pip dependency, not a
git submodule. A real run loads the official checkpoint (a hash-pinned download,
passed as `ckpt`) into a `Sam3ViTModel` built from the config's architecture keys,
via `sam3_trunk.load_official_trunk` (the official `sam3.pt` uses ViTDet-style
trunk keys -- fused qkv, a CLS in `pos_embed` -- that a plain `load_state_dict`
would leave entirely unmatched). The hermetic smoke leaves `ckpt` empty and builds
a **random tiny** `Sam3ViTModel` from those same keys, so nothing is downloaded and
the pipeline runs on a CPU.

The probe protocol matches this port's other frozen-backbone evals: extract the
feature once, mean-centre with the train mean then L2-normalise, and fit a linear
layer with SGD (momentum 0.9) on a cosine schedule, reporting top-1/top-5.

Changed during the port, and recorded in `provenance.json`:
  - **the device is resolved** (`resolve_device`) rather than assumed CUDA.
  - **features are extracted in fp32** (no autocast / no bf16), so the frozen
    feature probe runs identically on a CPU or a pre-Ampere GPU.
  - the feature is the vision encoder's patch tokens, **mean-pooled** over the
    sequence (SAM 3's ViT has no CLS token); `feature_dim = hidden_size`.
  - the input resolution is `pretrain_image_size=336` (the config's
    `pretrain_image_size`), not the native 1008, so the probe is executable; the
    resolution is read from the config so the hermetic smoke can shrink it.
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

# SAM 3's vision encoder is probed with ImageNet normalisation (the capture's
# SAM3 adapter, methods_step3/SegFM/SAM3/adapter.py).
SAM3_MEAN = (0.485, 0.456, 0.406)
SAM3_STD = (0.229, 0.224, 0.225)

# SAM 3 ViT architecture flags that are not free config knobs: they define the
# variant the released checkpoint was pretrained as (the Sam3ViTConfig defaults for
# facebook/sam3). The smoke and a real run build the same variant; only the size
# keys differ.
SAM3_ARCH = dict(
    hidden_act="gelu",
    layer_norm_eps=1e-6,
    window_size=24,
    global_attn_indexes=[7, 15, 23, 31],
    rope_theta=10000.0,
    hidden_dropout=0.0,
    attention_dropout=0.0,
)


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
    """Build the SAM 3 vision encoder from transformers and freeze it.

    A `Sam3ViTModel` is built from the config's architecture keys (there is no
    create-model-by-name in transformers), yielding the patch tokens as
    `last_hidden_state`. `ckpt` empty -> a random tiny model (the hermetic smoke);
    nothing is downloaded. A path -> the same architecture with the sha256-pinned
    official checkpoint converted onto it by `sam3_trunk.load_official_trunk` (a
    checkpoint that is not this trunk is refused, not half-loaded)."""
    from transformers import Sam3ViTConfig, Sam3ViTModel

    img_size = int(train["img_size"])
    ckpt = train.get("ckpt") or ""
    if ckpt:
        # A real run: the official ViT-L trunk. The architecture (including the
        # non-4x intermediate_size 4736) is inferred from the checkpoint itself,
        # so the size keys below are not consulted -- they only shape the random
        # tiny model the hermetic smoke builds.
        if str(METHOD_DIR := Path(__file__).resolve().parent) not in sys.path:
            sys.path.insert(0, str(METHOD_DIR))
        from sam3_trunk import load_official_trunk
        model = load_official_trunk(ckpt, img_size=img_size)
    else:
        embed_dim = int(train["embed_dim"])
        config = Sam3ViTConfig(
            hidden_size=embed_dim,
            num_hidden_layers=int(train["depth"]),
            num_attention_heads=int(train["num_heads"]),
            intermediate_size=4 * embed_dim,
            image_size=img_size,
            pretrain_image_size=img_size,
            patch_size=int(train["patch_size"]),
            num_channels=3,
            **SAM3_ARCH)
        model = Sam3ViTModel(config)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_feature(model, imgs, device) -> "torch.Tensor":
    """The probed representation: the patch tokens, mean-pooled ([B, D])."""
    imgs = imgs.to(device)
    out = model(pixel_values=imgs)
    tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return tokens.mean(dim=1).float()


def _build_loader(data_root: str, split: str, resolution: int, batch_size: int,
                  num_workers: int):
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    normalize = T.Normalize(mean=SAM3_MEAN, std=SAM3_STD)
    transform = T.Compose([
        T.Resize(resolution, interpolation=T.InterpolationMode.BICUBIC),
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
        description="sam3 linear eval (frozen vision encoder)")
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
    print(f"sam3 linear eval  device={device}  "
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
