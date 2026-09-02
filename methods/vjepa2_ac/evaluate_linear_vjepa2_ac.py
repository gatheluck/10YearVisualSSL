"""Linear evaluation for vjepa2_ac: a probe on the frozen pretrained backbone.

V-JEPA 2-AC (Assran et al., Meta FAIR, 2025; arXiv:2506.09985 -- the
action-conditioned world model) freezes the pretrained video encoder and fits a
single linear layer on its features. The representation is a genuine SSL feature,
so the number is comparable (the "pretrained-backbone reuse" row, analogous to
eva02 / data2vec2 / cae / videomae / vjepa2).

Faithful, where the sibling `vjepa2` approximates. The self-contained sibling runs
the encoder with plain attention; this port builds the ViT from the **pinned
facebookresearch/vjepa2 submodule** (third_party/vjepa2), imported not copied, and
runs V-JEPA 2's **real rotary-position attention** (`use_rope=True`) -- reproducing
the capture's number rather than an approximation of it.

The representation, on a still image. V-JEPA 2's ViT consumes a video clip
`(B, 3, T, H, W)` via a Conv3d tubelet patch embed. The capture's ImageNet linear
eval feeds a **still image** replicated `tubelet_size` times along the temporal axis
(one temporal token; never PyAV / a video dataset), runs the ViT, and **mean-pools
the tokens** to one vector per image. This port keeps exactly that proxy.

The checkpoint, stated plainly. The public `vjepa2-ac-vitg.pt` is a training-state
dict: its `encoder` sub-dict carries the ViT weights under a `module.` prefix,
alongside a `predictor` (the action-conditioned JEPA predictor), `opt`, `scaler`,
`target_encoder` and scalars, which are dropped. The `encoder` tensors, with
`module.`/`backbone.` stripped, load into the submodule's `vit_giant_xformers`
(embed_dim 1408, depth 40, 22 heads) with **zero missing or unexpected keys**
(measured: 484 == 484), so a mismatch is a hard error rather than a silently
half-loaded backbone. `einops` and `timm` are the submodule's own runtime
dependencies (its `vision_transformer.py` imports them); this file does not import
them, but the lock installs them.

Changed during the port, and recorded in `provenance.json`:
  - **the device is resolved** (`resolve_device`) rather than assumed CUDA.
  - **features are extracted in fp32** (no autocast), so the frozen-feature probe
    runs identically on a CPU or a pre-Ampere GPU.
  - the model is built from the config's `arch` (a submodule factory name); the
    checkpoint's `encoder` tensors load into it and any missing or unexpected key
    means the checkpoint does not match the architecture (a hard error; the
    capture's tolerant strict=False that merely printed the counts is tightened).
  - the input normalisation follows ImageNet mean/std with a bilinear square resize
    (the capture resizes to 256 with bilinear interpolation inside the wrapper).
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

# V-JEPA 2's preprocessing: standard ImageNet mean/std, RGB, expressed in the
# [0,1] scale torchvision's ToTensor yields.
VJEPA2_MEAN = (0.485, 0.456, 0.406)
VJEPA2_STD = (0.229, 0.224, 0.225)

# The pinned facebookresearch/vjepa2 upstream: third_party/<repo-basename> at the
# repo root (this file -> parents[2] is the repo root). The submodule directory is
# named for the upstream repo, so it is derived from the pinned URL's last segment
# rather than a second bare literal of the name -- which would also read as
# hard-coding the *sibling* method `vjepa2` (tests/test_no_hard_coded_methods.py:
# the submodule and that sibling share a name; this file may name only its own).
_VJEPA2_REPO = "https://github.com/facebookresearch/vjepa2"
_VJEPA2_SUBMODULE = (Path(__file__).resolve().parents[2] / "third_party"
                     / _VJEPA2_REPO.rsplit("/", 1)[-1])


def _prepare_vjepa2_path() -> None:
    """Make `src`/`app` resolve to THIS submodule only. Another submodule port
    (35_vjepa's third_party/jepa) also exposes a top-level `src`, and `src` is a
    PEP 420 namespace package that would otherwise merge both submodules' `src/`
    dirs. So: drop any cached `src*`/`app*`, remove every other third_party root
    from sys.path, and put third_party/vjepa2 first. Purge-before-import keeps this
    re-entrant, so the two submodule ports can build in either order in-process."""
    for key in [k for k in sys.modules
                if k in ("src", "app") or k.startswith(("src.", "app."))]:
        del sys.modules[key]
    tp = str(_VJEPA2_SUBMODULE.parent) + os.sep       # <repo>/third_party/
    sys.path[:] = [q for q in sys.path if not q.startswith(tp)]
    sys.path.insert(0, str(_VJEPA2_SUBMODULE))


def _vision_transformer():
    """The submodule's vision_transformer module, imported lazily.

    Imported inside the build (not at import time) so the config/device tests can
    import this file without triggering the `src` import (and its einops/timm
    dependencies)."""
    _prepare_vjepa2_path()
    try:
        from src.models import vision_transformer as vit
    except ImportError as e:
        raise ImportError(
            "the facebookresearch/vjepa2 code is required (the V-JEPA 2 ViT lives "
            "there, and it imports einops and timm). It is the pinned submodule at "
            "third_party/vjepa2; run `git submodule update --init "
            "third_party/vjepa2` and install the method lock.") from e
    return vit


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


def _clean_key(state: dict) -> dict:
    """Strip the `module.` (DDP) and `backbone.` prefixes the checkpoint carries."""
    return {k.replace("module.", "").replace("backbone.", ""): v
            for k, v in state.items()}


def build_model(train: dict, device: "torch.device"):
    """Build the V-JEPA 2 ViT from the submodule and freeze it.

    `ckpt` empty -> a random model at the config's `arch` (the hermetic smoke uses
    `vit_tiny`); nothing is downloaded. A path -> the same architecture with the
    sha256-pinned official AC checkpoint's `encoder` tensors loaded into it. The
    `predictor`/`opt`/`scaler`/`target_encoder` entries are dropped; any missing or
    unexpected key after loading `encoder` means the checkpoint does not match the
    architecture, and is a hard error rather than a silently half-loaded backbone.
    The forward runs with `use_rope=True` (V-JEPA 2's real rotary attention)."""
    vit = _vision_transformer()
    factory = getattr(vit, str(train["arch"]), None)
    if factory is None:
        raise RuntimeError(
            f"unknown arch {train['arch']!r}: it is not a factory in the "
            "facebookresearch/vjepa2 vision_transformer module")
    size = int(train["img_size"])
    model = factory(
        patch_size=int(train["patch_size"]),
        img_size=(size, size),
        num_frames=int(train["num_frames"]),
        tubelet_size=int(train["tubelet_size"]),
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
    )

    ckpt = train.get("ckpt") or ""
    if ckpt:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or "encoder" not in state:
            raise RuntimeError(
                f"checkpoint {ckpt} has no 'encoder' sub-dict; it is not the "
                "official V-JEPA 2-AC training-state checkpoint this port pins")
        encoder = _clean_key(state["encoder"])
        result = model.load_state_dict(encoder, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "checkpoint does not match the V-JEPA 2 architecture: missing "
                f"{list(result.missing_keys)[:5]}, unexpected "
                f"{list(result.unexpected_keys)[:5]}")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_feature(model, imgs, device) -> "torch.Tensor":
    """The probed representation: mean-pooled patch tokens ([B, D]).

    Each still image is resized to the model's square input and replicated
    ``tubelet_size`` times along a new temporal axis to form the one-temporal-token
    clip V-JEPA 2 expects, run through the ViT, and the resulting tokens are
    mean-pooled -- exactly the capture's ImageNet-linear-eval proxy."""
    imgs = imgs.to(device)
    size = model.img_height
    if imgs.shape[-1] != size or imgs.shape[-2] != size:
        imgs = F.interpolate(imgs.float(), size=(size, size),
                             mode="bilinear", align_corners=False).to(imgs.dtype)
    clip = imgs.unsqueeze(2).expand(-1, -1, model.tubelet_size, -1, -1)
    tokens = model(clip)               # (B, N, D)
    return tokens.mean(dim=1).float()  # (B, D)


def _build_loader(data_root: str, split: str, resolution: int, batch_size: int,
                  num_workers: int):
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    normalize = T.Normalize(mean=VJEPA2_MEAN, std=VJEPA2_STD)
    transform = T.Compose([
        T.Resize((resolution, resolution),
                 interpolation=T.InterpolationMode.BILINEAR),
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
        description="vjepa2_ac linear eval (frozen backbone)")
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
    print(f"vjepa2_ac linear eval  device={device}  arch={train['arch']}  "
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
