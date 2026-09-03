"""Linear evaluation for vjepa2: a probe on the frozen pretrained backbone.

V-JEPA 2 (Assran et al., Meta FAIR, 2025; arXiv:2506.09985)'s as-is comparison
freezes the pretrained video encoder and fits a single linear layer on its
features. The representation is a genuine SSL feature, so the number is comparable
(the "pretrained-backbone reuse" row, analogous to eva02 / data2vec2 / cae /
videomae).

The representation, on a still image. V-JEPA 2's ViT consumes a video clip
`(B, 3, T, H, W)` via a Conv3d tubelet patch embed. The capture's ImageNet linear
eval feeds a **still image** replicated `num_frames` times along the temporal axis
(never PyAV / a video dataset), runs the ViT, and **mean-pools the tokens** to one
vector per image. This port keeps exactly that proxy.

The checkpoint, stated plainly. Neither timm nor transformers is a dependency: the
model is a **small self-contained V-JEPA 2 ViT** here, whose module names mirror
the official checkpoint's `encoder.*` key hierarchy so its tensors load with no
remapping. The official `facebook/vjepa2-vitl-fpc64-256` weights (public, MIT) are
a `VJEPA2Model` safetensors carrying `encoder.*` (the encoder) and `predictor.*`
(the JEPA predictor) keys; only the `encoder.*` encoder is loaded (the `predictor.*`
keys are training machinery and are dropped). The backbone omits position
embeddings, as the capture's reimplementation does; the official checkpoint stores
none either (V-JEPA 2 uses rotary position embeddings applied at run time, holding
no learned position parameters), so the stripped state loads exactly with no
missing or unexpected key.

A faithfulness note. The capture's own backbone wrapper runs the encoder with
**plain** (non-rotary) attention -- it loads the weights and mean-pools, without
re-deriving V-JEPA 2's rotary positional mechanism. This port mirrors the capture's
forward exactly, so the probe number reproduces what the capture's eval produced;
it is an approximation of the full rotary forward, and that is stated plainly here
and in `provenance.json`.

The probe protocol matches this port's other frozen-backbone evals: extract the
feature once, mean-centre with the train mean then L2-normalise, and fit a linear
layer with SGD (momentum 0.9) on a cosine schedule, reporting top-1/top-5.

Changed during the port, and recorded in `provenance.json`:
  - **the device is resolved** (`resolve_device`) rather than assumed CUDA.
  - **features are extracted in fp32** (no autocast), so the frozen-feature probe
    runs identically on a CPU or a pre-Ampere GPU.
  - the model is built from the config's architecture keys; the checkpoint's
    `encoder.*` tensors load into it and any missing or unexpected encoder key
    means the checkpoint does not match the architecture (a hard error, not a
    silently half-loaded backbone; the capture's silent random-weight fallback on
    a failed download is removed).
  - the input normalisation follows ImageNet mean/std with a bicubic square resize.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# V-JEPA 2's preprocessing: standard ImageNet mean/std, RGB, expressed in the
# [0,1] scale torchvision's ToTensor yields.
VJEPA2_MEAN = (0.485, 0.456, 0.406)
VJEPA2_STD = (0.229, 0.224, 0.225)


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


# ── the V-JEPA 2 ViT encoder, self-contained ──────────────────────────────────
# The module names mirror the official checkpoint's `encoder.*` keys exactly, so
# its tensors load with no remapping: `embeddings.patch_embeddings.proj` (a Conv3d
# tubelet embed), `layer.N.attention.{query,key,value,proj}` (separate q/k/v, each
# biased), `layer.N.mlp.{fc1,fc2}`, `layer.N.{norm1,norm2}`, and the final
# `layernorm`. There is no `cls` token and no learned position embedding.

class _PatchEmbed(nn.Module):
    """encoder.embeddings.patch_embeddings"""
    def __init__(self, patch_size: int, tubelet_size: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv3d(
            3, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size))

    def forward(self, x):
        """x: (B, C, T, H, W) -> (B, N, D)"""
        return self.proj(x).flatten(2).transpose(1, 2)


class _Embeddings(nn.Module):
    """encoder.embeddings"""
    def __init__(self, patch_size: int, tubelet_size: int, embed_dim: int):
        super().__init__()
        self.patch_embeddings = _PatchEmbed(patch_size, tubelet_size, embed_dim)

    def forward(self, x):
        return self.patch_embeddings(x)


class _SelfAttention(nn.Module):
    """encoder.layer.N.attention -- separate query/key/value/proj, each biased."""
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.query(x).view(B, N, H, D).transpose(1, 2)
        k = self.key(x).view(B, N, H, D).transpose(1, 2)
        v = self.value(x).view(B, N, H, D).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class _MLP(nn.Module):
    """encoder.layer.N.mlp"""
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        h = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, embed_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    """encoder.layer.N"""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = _SelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = _MLP(embed_dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VJEPA2ViT(nn.Module):
    """The V-JEPA 2 encoder (the checkpoint's `encoder.*` sub-tree).

    Its ``forward`` maps a clip ``(B, 3, T, H, W)`` to per-token features
    ``(B, N, D)`` after the final LayerNorm. The probed representation is the
    mean over the N tokens (see ``extract_feature``). The `layer` ModuleList sits
    directly on this module, so keys strip only the `encoder.` prefix.
    """

    def __init__(self, patch_size: int, tubelet_size: int, embed_dim: int,
                 depth: int, num_heads: int, num_frames: int, img_size: int):
        super().__init__()
        self.embeddings = _Embeddings(patch_size, tubelet_size, embed_dim)
        self.layer = nn.ModuleList(
            [_Block(embed_dim, num_heads) for _ in range(depth)])
        self.layernorm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self._img_size = img_size

    def forward(self, clip):
        """clip: (B, C, T, H, W) -> (B, N, embed_dim)"""
        h = self.embeddings(clip)
        for blk in self.layer:
            h = blk(h)
        return self.layernorm(h)


def build_model(train: dict, device: "torch.device"):
    """Build the V-JEPA 2 ViT and freeze it.

    `ckpt` empty -> a random tiny model (the hermetic smoke); nothing is
    downloaded. A path -> the same architecture with the sha256-pinned official
    checkpoint's `encoder.*` tensors loaded into it. The `predictor.*` predictor
    keys are dropped; any missing or unexpected encoder key means the checkpoint
    does not match the architecture, and is a hard error rather than a silently
    half-loaded backbone."""
    model = VJEPA2ViT(
        patch_size=int(train["patch_size"]),
        tubelet_size=int(train["tubelet_size"]),
        embed_dim=int(train["embed_dim"]),
        depth=int(train["depth"]),
        num_heads=int(train["num_heads"]),
        num_frames=int(train["num_frames"]),
        img_size=int(train["img_size"]))

    ckpt = train.get("ckpt") or ""
    if ckpt:
        from safetensors.torch import load_file
        state = load_file(str(ckpt), device="cpu")
        # Keys: encoder.* (the encoder) and predictor.* (the JEPA predictor).
        # Keep only the encoder, strip the 'encoder.' prefix; drop predictor.* .
        encoder = {k[len("encoder."):]: v for k, v in state.items()
                   if k.startswith("encoder.")}
        if not encoder:
            raise RuntimeError(
                f"checkpoint {ckpt} has no encoder.* weights; it is not the "
                "official V-JEPA 2 checkpoint this port pins")
        result = model.load_state_dict(encoder, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "checkpoint does not match the V-JEPA 2 architecture: missing "
                f"{result.missing_keys[:5]}, unexpected "
                f"{result.unexpected_keys[:5]}")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_feature(model, imgs, device) -> "torch.Tensor":
    """The probed representation: mean-pooled patch tokens ([B, D]).

    Each still image is replicated ``num_frames`` times along a new temporal axis
    to form the clip V-JEPA 2 expects, run through the ViT, and the resulting
    tokens are mean-pooled -- exactly the capture's ImageNet-linear-eval proxy."""
    imgs = imgs.to(device)
    clip = imgs.unsqueeze(2).expand(-1, -1, model.num_frames, -1, -1)
    tokens = model(clip)               # (B, N, D)
    return tokens.mean(dim=1).float()  # (B, D)


def _build_loader(data_root: str, split: str, resolution: int, batch_size: int,
                  num_workers: int):
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    normalize = T.Normalize(mean=VJEPA2_MEAN, std=VJEPA2_STD)
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
        description="vjepa2 linear eval (frozen backbone)")
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
    print(f"vjepa2 linear eval  device={device}  "
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
