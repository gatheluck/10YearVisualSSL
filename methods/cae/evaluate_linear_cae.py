"""Linear evaluation for cae: a probe on the frozen pretrained backbone.

CAE (Context Autoencoder)'s as-is comparison freezes the pretrained vision
backbone (a BEiT-architecture ViT-B/16) and fits a single linear layer on its
CLS token. The representation is a genuine SSL feature, so the number is
comparable (the "pretrained-backbone reuse" row, analogous to eva02 / data2vec2
/ 36_franca).

The checkpoint, stated plainly. Neither timm nor transformers carries a CAE model
class, and the official weights are not publicly downloadable (the capture logged
this as deficiency DEF-02 and fell back to a BEiT-v2 proxy). This port ships no
proxy: it pins the OpenMMLab mmselfsup **reproduction** of CAE ViT-B (a real,
hash-pinned public download, Apache-2.0), a faithful reproduction rather than the
authors' released weights (see provenance.json). Because the checkpoint is in
mmselfsup format, the model is a **small self-contained BEiT-style ViT** here (no
mmcv/mmpretrain dependency); the checkpoint's `backbone.*` tensors load into it
directly, and the mmengine bookkeeping the checkpoint pickles is read with a
tolerant unpickler so no mmengine is needed.

The CAE encoder, read from the checkpoint's own keys (2026-08-30): a ViT-B/16 with
a class token and an **absolute** position embedding (no relative-position bias),
per-block LayerScale (`gamma_1`/`gamma_2`), and BEiT-style attention (a packed
`qkv` with separate q/v bias and no k-bias), followed by a final LayerNorm. The
probed feature is the final-norm'd CLS token.

The probe protocol matches this port's other frozen-backbone evals: extract the
CLS feature once, mean-centre with the train mean then L2-normalise, and fit a
linear layer with SGD (momentum 0.9) on a cosine schedule, reporting top-1/top-5.

Changed during the port, and recorded in `provenance.json`:
  - **the device is resolved** (`resolve_device`) rather than assumed CUDA.
  - **features are extracted in fp32** (no autocast), so the frozen-feature probe
    runs identically on a CPU or a pre-Ampere GPU.
  - the model is a self-contained BEiT-style ViT built from the config's
    architecture keys; the checkpoint's `backbone.*` tensors load into it and any
    missing/unexpected key means the checkpoint does not match the architecture.
  - the input normalisation follows the backbone's own preprocessor config
    (ImageNet mean/std, a bicubic square resize, no centre crop).
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The OpenMMLab CAE data preprocessor (its primary normalisation): standard
# ImageNet mean/std, RGB. Read on 2026-08-30 from mmpretrain's config
# configs/cae/cae_beit-base-p16_8xb256-amp-coslr-300e_in1k.py (data_preprocessor
# mean [123.675, 116.28, 103.53], std [58.395, 57.12, 57.375], to_rgb=True; the
# `second_mean`/`second_std` there belong to the BEiT target tokenizer branch, not
# the encoder input). Expressed in the [0,1] scale torchvision's ToTensor yields.
CAE_MEAN = (0.485, 0.456, 0.406)
CAE_STD = (0.229, 0.224, 0.225)


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


# ── the CAE (mmselfsup) ViT encoder, self-contained ───────────────────────────
# The module names mirror the checkpoint's `backbone.*` keys exactly, so its
# tensors load with no remapping: `patch_embed.projection`, `layers.N.gamma_1`,
# `layers.N.attn.qkv`/`q_bias`/`v_bias`/`proj`, `layers.N.ffn.layers.0.0`/`.1`,
# `ln1` (per-block pre-norms and the final norm), `cls_token`, `pos_embed`.

class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # BEiT-style: a packed qkv with no bias of its own; q and v carry a
        # separate learned bias and k carries none.
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        bias = torch.cat([self.q_bias, torch.zeros_like(self.v_bias),
                          self.v_bias])
        qkv = (x @ self.qkv.weight.t() + bias).reshape(
            B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _FFN(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        # mmselfsup FFN: layers[0] is Sequential(Linear, GELU); layers[1] is the
        # output Linear -- hence the checkpoint keys `ffn.layers.0.0` and
        # `ffn.layers.1`.
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden), nn.GELU()),
            nn.Linear(hidden, dim),
        ])

    def forward(self, x):
        return self.layers[1](self.layers[0](x))


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_hidden: int):
        super().__init__()
        self.gamma_1 = nn.Parameter(torch.ones(dim))
        self.gamma_2 = nn.Parameter(torch.ones(dim))
        self.ln1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = _FFN(dim, mlp_hidden)

    def forward(self, x):
        x = x + self.gamma_1 * self.attn(self.ln1(x))
        x = x + self.gamma_2 * self.ffn(self.ln2(x))
        return x


class _PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, dim: int):
        super().__init__()
        self.projection = nn.Conv2d(3, dim, kernel_size=patch_size,
                                    stride=patch_size)

    def forward(self, x):
        return self.projection(x).flatten(2).transpose(1, 2)


class CAEViT(nn.Module):
    """The CAE encoder. Its ``forward`` returns the final-norm'd CLS token."""

    def __init__(self, img_size: int, patch_size: int, embed_dim: int,
                 depth: int, num_heads: int):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.patch_embed = _PatchEmbed(patch_size, embed_dim)
        self.layers = nn.ModuleList([
            _Block(embed_dim, num_heads, 4 * embed_dim) for _ in range(depth)])
        self.ln1 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for block in self.layers:
            x = block(x)
        x = self.ln1(x)
        return x[:, 0]


class _DiscardedObject:
    """A placeholder for an mmengine/mmcv object we do not need (see below)."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass

    def __reduce__(self):
        return (_DiscardedObject, ())


class _TolerantUnpickler(pickle.Unpickler):
    """Reads an mmselfsup checkpoint without importing mmengine/mmcv.

    The OpenMMLab checkpoint pickles mmengine bookkeeping (a HistoryBuffer in its
    `meta`/`message_hub`) alongside the tensors. Only the tensors are needed, so
    any mmengine/mmcv class is mapped to a harmless placeholder rather than
    requiring those heavy, build-fragile packages in the fleet.
    """

    def find_class(self, module, name):
        if module.split(".")[0] in ("mmengine", "mmcv"):
            return _DiscardedObject
        return super().find_class(module, name)


def _load_checkpoint_state_dict(ckpt: str) -> dict:
    shim = types.ModuleType("cae_tolerant_pickle")
    shim.Unpickler = _TolerantUnpickler
    shim.load = pickle.load
    obj = torch.load(ckpt, map_location="cpu", weights_only=False,
                     pickle_module=shim)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def build_model(train: dict, device: "torch.device"):
    """Build the CAE ViT and freeze it.

    `ckpt` empty -> a random tiny model (the hermetic smoke); nothing is
    downloaded. A path -> the same architecture with the sha256-pinned OpenMMLab
    checkpoint's `backbone.*` tensors loaded into it. Any missing or unexpected
    backbone key means the checkpoint does not match the architecture, and is a
    hard error rather than a silently half-loaded backbone."""
    model = CAEViT(
        img_size=int(train["img_size"]),
        patch_size=int(train["patch_size"]),
        embed_dim=int(train["embed_dim"]),
        depth=int(train["depth"]),
        num_heads=int(train["num_heads"]))

    ckpt = train.get("ckpt") or ""
    if ckpt:
        state = _load_checkpoint_state_dict(ckpt)
        backbone = {k[len("backbone."):]: v for k, v in state.items()
                    if k.startswith("backbone.")}
        if not backbone:
            raise RuntimeError(
                f"checkpoint {ckpt} has no backbone.* weights; it is not the "
                "mmselfsup CAE checkpoint this port pins")
        result = model.load_state_dict(backbone, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "checkpoint does not match the CAE architecture: missing "
                f"{result.missing_keys[:5]}, unexpected "
                f"{result.unexpected_keys[:5]}")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_feature(model, imgs, device) -> "torch.Tensor":
    """The probed representation: the final-norm'd CLS token ([B, D])."""
    imgs = imgs.to(device)
    return model(imgs).float()


def _build_loader(data_root: str, split: str, resolution: int, batch_size: int,
                  num_workers: int):
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    normalize = T.Normalize(mean=CAE_MEAN, std=CAE_STD)
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
        description="cae linear eval (frozen backbone)")
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
    print(f"cae linear eval  device={device}  "
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
