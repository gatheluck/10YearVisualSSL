"""Linear evaluation for data2vec2: a probe on the frozen pretrained backbone.

data2vec 2.0's as-is comparison freezes the official pretrained vision backbone
(`facebook/data2vec-vision-base`, a BEiT-architecture ViT self-distilled on
IN-1k) and fits a single linear layer on its CLS token. The representation is a
genuine SSL feature, so the number is comparable (the "pretrained-backbone reuse"
row, analogous to eva02 / DINOv2 / Franca).

The model class is `transformers`' `Data2VecVisionModel` -- a pinned pip
dependency, not a git submodule. A real run loads the official checkpoint (a
hash-pinned download, passed as `ckpt`) into a `Data2VecVisionModel` built from
the config's architecture keys; the hermetic smoke leaves `ckpt` empty and builds
a **random tiny** Data2VecVisionModel from those same keys, so nothing is
downloaded and the pipeline runs on a CPU.

The probe protocol matches this port's other frozen-backbone evals: extract the
CLS feature once, mean-centre with the train mean then L2-normalise, and fit a
linear layer with SGD (momentum 0.9) on a cosine schedule, reporting top-1/top-5.

Changed during the port, and recorded in `provenance.json`:
  - **the device is resolved** (`resolve_device`) rather than assumed CUDA.
  - **features are extracted in fp32** (no autocast), so the frozen-feature probe
    runs identically on a CPU or a pre-Ampere GPU.
  - the model is built with `add_pooling_layer=False` and the CLS token of
    `last_hidden_state` is probed (data2vec-vision-base is a pretraining
    checkpoint: `use_mean_pooling=False`, it carries no trained pooler head).
  - the input normalisation follows the backbone's own preprocessor config
    (mean/std 0.5, bicubic resize to a square, no centre crop), not ImageNet's.
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

# The official data2vec-vision preprocessor (BEiTFeatureExtractor config on the
# HuggingFace model card): symmetric mean/std, a bicubic square resize, no centre
# crop. Read on 2026-08-30 from facebook/data2vec-vision-base/preprocessor_config
# (image_mean/std [0.5,0.5,0.5], resample 3 = BICUBIC, do_center_crop false).
D2V_MEAN = (0.5, 0.5, 0.5)
D2V_STD = (0.5, 0.5, 0.5)

# data2vec-vision architecture flags that are not free config knobs: they define
# the BEiT variant the released checkpoint was pretrained as. Read on 2026-08-30
# from facebook/data2vec-vision-base/config.json (use_mean_pooling false, a shared
# relative-position bias, no absolute position embeddings, layer-scale 0.1). The
# smoke and a real run build the same variant; only the size keys differ.
D2V_ARCH = dict(
    use_mean_pooling=False,
    use_shared_relative_position_bias=True,
    use_relative_position_bias=False,
    use_absolute_position_embeddings=False,
    layer_scale_init_value=0.1,
    use_mask_token=False,
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
    """Build the data2vec-vision backbone from transformers and freeze it.

    A `Data2VecVisionModel` is built from the config's architecture keys (there
    is no create-model-by-name in transformers), with `add_pooling_layer=False`
    so it yields the CLS token of `last_hidden_state`. `ckpt` empty -> a random
    tiny model (the hermetic smoke); nothing is downloaded. A path -> the same
    architecture with the sha256-pinned official checkpoint loaded into it with
    `strict=False` (the checkpoint carries a derived `relative_position_index`
    buffer transformers rebuilds; any *missing* backbone weight means the
    checkpoint does not match the architecture)."""
    from transformers import Data2VecVisionConfig, Data2VecVisionModel

    embed_dim = int(train["embed_dim"])
    config = Data2VecVisionConfig(
        hidden_size=embed_dim,
        num_hidden_layers=int(train["depth"]),
        num_attention_heads=int(train["num_heads"]),
        intermediate_size=4 * embed_dim,
        image_size=int(train["img_size"]),
        patch_size=int(train["patch_size"]),
        num_channels=3,
        **D2V_ARCH)
    model = Data2VecVisionModel(config, add_pooling_layer=False)

    ckpt = train.get("ckpt") or ""
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        state = state.get("model", state) if isinstance(state, dict) else state
        result = model.load_state_dict(state, strict=False)
        # add_pooling_layer=False drops the pooler; any other missing weight means
        # the checkpoint does not match the architecture. Unexpected keys (the
        # derived relative_position_index buffer) are harmless and ignored.
        backbone_missing = [k for k in result.missing_keys
                            if not k.startswith("pooler")]
        if backbone_missing:
            raise RuntimeError(
                f"checkpoint is missing backbone weights: {backbone_missing[:5]}")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_feature(model, imgs, device) -> "torch.Tensor":
    """The probed representation: the CLS token of last_hidden_state ([B, D])."""
    imgs = imgs.to(device)
    out = model(imgs)
    return out.last_hidden_state[:, 0].float()


def _build_loader(data_root: str, split: str, resolution: int, batch_size: int,
                  num_workers: int):
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    normalize = T.Normalize(mean=D2V_MEAN, std=D2V_STD)
    transform = T.Compose([
        T.Resize((resolution, resolution),
                 interpolation=T.InterpolationMode.BICUBIC),
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
        description="data2vec2 linear eval (frozen backbone)")
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
    print(f"data2vec2 linear eval  device={device}  "
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
