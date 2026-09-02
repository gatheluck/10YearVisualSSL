"""Linear evaluation for cosmos3_super: a probe on the frozen Cosmos3-Super tower.

NVIDIA Cosmos3-Super's as-is comparison freezes the released world-foundation
model's Qwen3-VL **vision encoder** and fits a single linear layer on its
**mean-pooled patch tokens**, matching the capture's Step-3 VideoGen read (the 64B
MoT / DiT / VAE are never loaded). The representation is a genuine learned feature,
so the number is comparable (the multimodal "pretrained-backbone reuse" row, the
`transformers`-sourced sibling of `data2vec2` and `sam3`).

The model class is `transformers`' `Qwen3VLVisionModel` -- a pinned pip dependency,
not a git submodule. Unlike `sam3`, the released checkpoint is **directly loadable**
by the HF class: a real run calls `Qwen3VLVisionModel.from_pretrained` on the
`vision_encoder/` directory (config.json + model.safetensors, a hash-pinned
download passed as `ckpt`); no trunk conversion is needed and a `save_pretrained`
-> `from_pretrained` round-trip is exact. The hermetic smoke leaves `ckpt` empty and
builds a **random tiny** `Qwen3VLVisionModel` from the config's architecture keys,
so nothing is downloaded and the pipeline runs on a CPU.

The vision tower takes flattened patches, not a pixel grid: each image is unfolded
into `patch_size`x`patch_size` patches, each patch repeated `temporal_patch_size`
times (a single frame presented as the tower's minimal temporal window), and the
per-image grid is described by `grid_thw = [[1, H_p, W_p]]`. The feature is the
returned `last_hidden_state` patch tokens, mean-pooled per image (`feature_dim =
hidden_size = 1152`, the frozen-probe width the capture uses -- not the merger's
`out_hidden_size` 5120).

The probe protocol matches this port's other frozen-backbone evals: extract the
feature once, mean-centre with the train mean then L2-normalise, and fit a linear
layer with SGD (momentum 0.9) on a cosine schedule, reporting top-1/top-5.

Changed during the port, and recorded in `provenance.json`:
  - **the device is resolved** (`resolve_device`) rather than assumed CUDA.
  - **features are extracted in fp32** (no autocast / no bf16), so the frozen
    feature probe runs identically on a CPU or a pre-Ampere GPU, rather than the
    checkpoint's bf16.
  - the feature is the vision tower's patch tokens, **mean-pooled** over the
    sequence; `feature_dim = hidden_size`.
  - the input resolution is read from the config (`img_size`, the native 448 in the
    shipped recipe), so the hermetic smoke can shrink it and build a random tiny
    tower.
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

# Cosmos3-Super's vision encoder is probed with ImageNet normalisation (the
# capture's adapter, methods_step3/VideoGen/Cosmos3-Super/adapter.py).
COSMOS3_MEAN = (0.485, 0.456, 0.406)
COSMOS3_STD = (0.229, 0.224, 0.225)

# Qwen3-VL vision-tower flags that are not free config knobs: they define the
# variant the released checkpoint was trained as (the Qwen3VLVisionConfig fixed
# fields for nvidia/Cosmos3-Super). The smoke and a real run build the same
# variant; only the size keys differ.
COSMOS3_ARCH = dict(
    hidden_act="gelu_pytorch_tanh",
    spatial_merge_size=2,
    temporal_patch_size=2,
    initializer_range=0.02,
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
    """Build the Cosmos3-Super vision tower from transformers and freeze it.

    `ckpt` empty -> a random tiny `Qwen3VLVisionModel` built from the config's
    architecture keys (the hermetic smoke); nothing is downloaded. A path -> the
    released tower loaded directly with `from_pretrained` on the `vision_encoder/`
    directory (config.json + model.safetensors); the architecture is read from the
    checkpoint's own config, so the size keys below are not consulted. A path that
    is not a loadable model directory raises rather than falling back to a random
    init."""
    from transformers import Qwen3VLVisionConfig, Qwen3VLVisionModel

    ckpt = train.get("ckpt") or ""
    if ckpt:
        # A real run: the released vision tower loads directly (no conversion).
        model = Qwen3VLVisionModel.from_pretrained(
            str(ckpt), dtype=torch.float32, local_files_only=True)
    else:
        embed_dim = int(train["embed_dim"])
        img_size = int(train["img_size"])
        patch_size = int(train["patch_size"])
        config = Qwen3VLVisionConfig(
            depth=int(train["depth"]),
            hidden_size=embed_dim,
            num_heads=int(train["num_heads"]),
            intermediate_size=4 * embed_dim,
            in_channels=3,
            patch_size=patch_size,
            out_hidden_size=embed_dim,
            num_position_embeddings=max(64, (img_size // patch_size) ** 2),
            deepstack_visual_indexes=[],
            **COSMOS3_ARCH)
        model = Qwen3VLVisionModel(config)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _extract_patches(model, imgs):
    """Unfold images into the flattened patches the Qwen3-VL tower consumes.

    Each image becomes `H_p*W_p` patches of `C*tps*ps*ps` values (a single frame
    presented over the tower's minimal temporal window `tps`), and its grid is
    described by `grid_thw = [[1, H_p, W_p]]` per image."""
    cfg = model.config
    ps = int(getattr(cfg, "patch_size", 16))
    tps = int(getattr(cfg, "temporal_patch_size", 2))
    B, C, H, W = imgs.shape
    H_p, W_p = H // ps, W // ps
    x = imgs.unfold(2, ps, ps).unfold(3, ps, ps)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
    x = x.reshape(B * H_p * W_p, C, ps, ps)
    x = x.unsqueeze(2).expand(-1, -1, tps, -1, -1).contiguous()
    patches = x.reshape(B * H_p * W_p, C * tps * ps * ps)
    grid_thw = torch.tensor([[1, H_p, W_p]], dtype=torch.long,
                            device=imgs.device).expand(B, 3).contiguous()
    return patches, grid_thw


@torch.no_grad()
def extract_feature(model, imgs, device) -> "torch.Tensor":
    """The probed representation: the patch tokens, mean-pooled ([B, D])."""
    imgs = imgs.to(device=device, dtype=torch.float32)
    patches, grid_thw = _extract_patches(model, imgs)
    out = model(hidden_states=patches, grid_thw=grid_thw)
    tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    B = imgs.shape[0]
    # last_hidden_state is (total_patches, hidden); every image contributes an
    # equal number of patch tokens, so a per-image mean is a reshape then a mean.
    return tokens.reshape(B, tokens.shape[0] // B, -1).mean(dim=1).float()


def _build_loader(data_root: str, split: str, resolution: int, batch_size: int,
                  num_workers: int):
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    normalize = T.Normalize(mean=COSMOS3_MEAN, std=COSMOS3_STD)
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
        description="cosmos3_super linear eval (frozen vision encoder)")
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
    print(f"cosmos3_super linear eval  device={device}  "
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
