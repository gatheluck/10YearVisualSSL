"""Linear evaluation for videomae: a probe on the frozen pretrained backbone.

VideoMAE (Tong et al., NeurIPS 2022; arXiv:2203.12602)'s as-is comparison freezes
the pretrained video encoder and fits a single linear layer on its features. The
representation is a genuine SSL feature, so the number is comparable (the
"pretrained-backbone reuse" row, analogous to eva02 / data2vec2 / cae).

The representation, on a still image. VideoMAE's ViT consumes a video clip
`(B, 3, T, H, W)` via a Conv3d tubelet patch embed. The capture's ImageNet linear
eval feeds a **still image** replicated `num_frames` times along the temporal axis
(never PyAV / a video dataset), runs the ViT, and **mean-pools the tokens** to one
vector per image. This port keeps exactly that proxy.

The checkpoint, stated plainly. Neither timm nor transformers is a dependency: the
model is a **small self-contained VideoMAE ViT** here, whose module names mirror
the official checkpoint's `videomae.*` key hierarchy so its tensors load with no
remapping. The official `MCG-NJU/videomae-base` weights (public, CC-BY-NC-4.0) are
a `VideoMAEForPreTraining` safetensors carrying `videomae.*` (encoder) and
`decoder.*` keys; only the `videomae.*` encoder is loaded (the `decoder.*` keys are
the pretext decoder and are dropped). The backbone omits position embeddings, as
the capture's reimplementation does; the official checkpoint stores none either
(VideoMAE uses fixed sin-cos embeddings computed at run time in the HF class, held
as a non-persistent buffer), so the stripped state loads exactly with no missing
or unexpected key.

The probe protocol matches this port's other frozen-backbone evals: extract the
feature once, mean-centre with the train mean then L2-normalise, and fit a linear
layer with SGD (momentum 0.9) on a cosine schedule, reporting top-1/top-5.

Changed during the port, and recorded in `provenance.json`:
  - **the device is resolved** (`resolve_device`) rather than assumed CUDA.
  - **features are extracted in fp32** (no autocast), so the frozen-feature probe
    runs identically on a CPU or a pre-Ampere GPU.
  - the model is built from the config's architecture keys; the checkpoint's
    `videomae.*` tensors load into it and any missing or unexpected encoder key
    means the checkpoint does not match the architecture (a hard error, not a
    silently half-loaded backbone; the capture's silent random-weight fallback on
    a failed download is removed).
  - the input normalisation follows the backbone's own preprocessor config
    (ImageNet mean/std, a bicubic square resize, no centre crop).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# VideoMAE's preprocessing (its primary normalisation): standard ImageNet mean/std,
# RGB, expressed in the [0,1] scale torchvision's ToTensor yields (the HF
# preprocessor_config.json for MCG-NJU/videomae-base carries the same values).
VIDEOMAE_MEAN = (0.485, 0.456, 0.406)
VIDEOMAE_STD = (0.229, 0.224, 0.225)


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


# ── the VideoMAE ViT encoder, self-contained ──────────────────────────────────
# The module names mirror the official checkpoint's `videomae.*` keys exactly, so
# its tensors load with no remapping: `embeddings.patch_embeddings.projection`
# (a Conv3d tubelet embed), `encoder.layer.N.attention.attention.{query,key,
# value}`/`q_bias`/`v_bias` (q and v carry a separate bias, k none),
# `.attention.output.dense`, `.intermediate.dense`, `.output.dense`,
# `.layernorm_before`/`.layernorm_after`, and the final `layernorm`.

class _PatchEmbed(nn.Module):
    """videomae.embeddings.patch_embeddings"""
    def __init__(self, patch_size: int, tubelet_size: int, embed_dim: int):
        super().__init__()
        self.projection = nn.Conv3d(
            3, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size))

    def forward(self, x):
        """x: (B, C, T, H, W) -> (B, N, D)"""
        return self.projection(x).flatten(2).transpose(1, 2)


class _Embeddings(nn.Module):
    """videomae.embeddings"""
    def __init__(self, patch_size: int, tubelet_size: int, embed_dim: int):
        super().__init__()
        self.patch_embeddings = _PatchEmbed(patch_size, tubelet_size, embed_dim)

    def forward(self, x):
        return self.patch_embeddings(x)


class _SelfAttention(nn.Module):
    """videomae.encoder.layer.N.attention.attention
    q and v carry a separate learned bias; k carries none."""
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(embed_dim))
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim
        q = (self.query(x) + self.q_bias).view(B, N, H, D).transpose(1, 2)
        k = self.key(x).view(B, N, H, D).transpose(1, 2)
        v = (self.value(x) + self.v_bias).view(B, N, H, D).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return (attn @ v).transpose(1, 2).reshape(B, N, C)


class _AttentionOutput(nn.Module):
    """videomae.encoder.layer.N.attention.output"""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        return self.dense(x)


class _Attention(nn.Module):
    """videomae.encoder.layer.N.attention"""
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = _SelfAttention(embed_dim, num_heads)
        self.output = _AttentionOutput(embed_dim)

    def forward(self, x):
        return self.output(self.attention(x))


class _Intermediate(nn.Module):
    """videomae.encoder.layer.N.intermediate"""
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dense = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.dense(x))


class _Output(nn.Module):
    """videomae.encoder.layer.N.output"""
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dense = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)

    def forward(self, x):
        return self.dense(x)


class _EncoderLayer(nn.Module):
    """videomae.encoder.layer.N"""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(embed_dim)
        self.attention = _Attention(embed_dim, num_heads)
        self.layernorm_after = nn.LayerNorm(embed_dim)
        self.intermediate = _Intermediate(embed_dim, mlp_ratio)
        self.output = _Output(embed_dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attention(self.layernorm_before(x))
        x = x + self.output(self.intermediate(self.layernorm_after(x)))
        return x


class _Encoder(nn.Module):
    """Holds encoder.layer to match 'encoder.layer.N.*' key paths."""
    def __init__(self, depth: int, embed_dim: int, num_heads: int):
        super().__init__()
        self.layer = nn.ModuleList(
            [_EncoderLayer(embed_dim, num_heads) for _ in range(depth)])

    def forward(self, x):
        for blk in self.layer:
            x = blk(x)
        return x


class VideoMAEViT(nn.Module):
    """The VideoMAE encoder (the checkpoint's `videomae.*` sub-tree).

    Its ``forward`` maps a clip ``(B, 3, T, H, W)`` to per-token features
    ``(B, N, D)`` after the final LayerNorm. The probed representation is the
    mean over the N tokens (see ``extract_feature``).
    """

    def __init__(self, patch_size: int, tubelet_size: int, embed_dim: int,
                 depth: int, num_heads: int, num_frames: int, img_size: int):
        super().__init__()
        self.embeddings = _Embeddings(patch_size, tubelet_size, embed_dim)
        self.encoder = _Encoder(depth, embed_dim, num_heads)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self._img_size = img_size

    def forward(self, clip):
        """clip: (B, C, T, H, W) -> (B, N, embed_dim)"""
        h = self.embeddings(clip)
        h = self.encoder(h)
        return self.layernorm(h)


def build_model(train: dict, device: "torch.device"):
    """Build the VideoMAE ViT and freeze it.

    `ckpt` empty -> a random tiny model (the hermetic smoke); nothing is
    downloaded. A path -> the same architecture with the sha256-pinned official
    checkpoint's `videomae.*` tensors loaded into it. Any missing or unexpected
    encoder key means the checkpoint does not match the architecture, and is a
    hard error rather than a silently half-loaded backbone."""
    model = VideoMAEViT(
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
        # Keys: videomae.* (encoder) and decoder.* (the pretext decoder). Strip
        # the 'videomae.' prefix and load only the encoder; drop decoder.* .
        encoder = {k[len("videomae."):]: v for k, v in state.items()
                   if k.startswith("videomae.")}
        if not encoder:
            raise RuntimeError(
                f"checkpoint {ckpt} has no videomae.* weights; it is not the "
                "official VideoMAE checkpoint this port pins")
        result = model.load_state_dict(encoder, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "checkpoint does not match the VideoMAE architecture: missing "
                f"{result.missing_keys[:5]}, unexpected "
                f"{result.unexpected_keys[:5]}")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_feature(model, imgs, device) -> "torch.Tensor":
    """The probed representation: temporally-averaged patch tokens ([B, D]).

    Each still image is replicated ``num_frames`` times along a new temporal axis
    to form the clip VideoMAE expects, run through the ViT, and the resulting
    tokens are mean-pooled -- exactly the capture's ImageNet-linear-eval proxy."""
    imgs = imgs.to(device)
    clip = imgs.unsqueeze(2).expand(-1, -1, model.num_frames, -1, -1)
    tokens = model(clip)               # (B, N, D)
    return tokens.mean(dim=1).float()  # (B, D)


def _build_loader(data_root: str, split: str, resolution: int, batch_size: int,
                  num_workers: int):
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    normalize = T.Normalize(mean=VIDEOMAE_MEAN, std=VIDEOMAE_STD)
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
        description="videomae linear eval (frozen backbone)")
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
    print(f"videomae linear eval  device={device}  "
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
