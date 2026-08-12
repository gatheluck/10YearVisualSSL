"""Linear evaluation for VAR: a probe on the VQVAE tokeniser's features.

**What this measures, stated plainly.** Faithful to the lab's ARSSL protocol
(`src/features/extract.py: extract_var_features` and
`src/probing/linear_probe.py`), the representation probed here is the **VQVAE
encoder's** continuous feature map, global-average-pooled -- *not* the VAR
transformer this port trains in step 1. So `encoder.pt` (the VAR representation
side) is deliberately **not read** here, and the number this stage produces is a
property of the fixed, pretrained VQVAE tokeniser rather than of VAR's learned
representation. This is the answer this port gives to the open question in
CONTRACT section 7 for a generative model; see `docs/EVAL_DOWNLOAD.md`.

A real run needs the pretrained tokeniser (`vqvae_ckpt`, a download, obtained by
`bin/fetch-weights.py`). The hermetic smoke builds a **random** VQVAE (as step 1
does), so no download happens and its accuracy is meaningless -- only the
pipeline is exercised.

The probe itself follows ARSSL: features are extracted once and cached, then
mean-centred and L2-normalised, and a single linear layer is trained with SGD
(momentum) under a cosine schedule. Top-1 and top-5 are reported.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reused so the tokeniser is built and seeded in exactly one place, and the
# device is resolved the same way as in training.
from train_pretrain_var import (                    # noqa: E402
    build_vqvae, make_deterministic, model_kwargs, resolve_device,
    VAE_DOWNSAMPLE,
)


def encode(vae, imgs: "torch.Tensor") -> "torch.Tensor":
    """The VAR representation the lab probes: the VQVAE encoder's continuous
    feature map, global-average-pooled to `Cvae` dims. This is the tokeniser's
    encoder, before quantisation -- not the VAR transformer."""
    with torch.no_grad():
        z = vae.encoder(imgs)          # [B, Cvae, H, W]
        return z.mean(dim=[2, 3])      # [B, Cvae]


def _build_loader(data_root: str, split: str, img_size: int, batch_size: int,
                  num_workers: int):
    """An ImageFolder over `data_root/<split>`, normalised to [-1, 1] as the
    VQVAE expects. Order is fixed (no shuffle): features are cached, so the
    order does not matter and a fixed one keeps the run reproducible."""
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = ImageFolder(str(Path(data_root) / split), transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=False)
    return dataset, loader


@torch.no_grad()
def extract_features(vae, loader, device):
    feats, labels = [], []
    for imgs, lbs in loader:
        feats.append(encode(vae, imgs.to(device, non_blocking=True)).cpu())
        labels.append(lbs)
    return torch.cat(feats), torch.cat(labels)


def normalize_features(train_feats, val_feats, center: bool = True):
    """Mean-centre (with the train-set mean) then L2-normalise, as ARSSL does:
    removing the large common component makes the features more linearly
    separable."""
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
    parser = argparse.ArgumentParser(description="VAR linear eval (VQVAE probe)")
    parser.add_argument("--config", default="configs/linear_eval.yaml")
    parser.add_argument("--data_path", default=None,
                        help="Override the ImageFolder root (reads <root>/train "
                             "and <root>/val)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    return parser


def run(args, config: "dict | None" = None, vae=None) -> dict:
    """Fit a linear probe on the frozen VQVAE features, returning its numbers.

    The adapter hands in the `vae` it built (one place knows how a tokeniser is
    built and loaded); the stand-alone CLI builds it here from the config.
    """
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

    if vae is None:
        vae, _var = build_vqvae(train, train.get("vqvae_ckpt"), device)
    vae.eval()
    print(f"VAR linear eval (VQVAE probe)  device={device}  "
          f"vqvae={'pretrained' if train.get('vqvae_ckpt') else 'random (smoke)'}")

    img_size = int(train["img_size"])
    bs = int(train["batch_size"])
    nw = int(train["num_workers"])
    tr_ds, tr_loader = _build_loader(data_root, "train", img_size, bs, nw)
    va_ds, va_loader = _build_loader(data_root, "val", img_size, bs, nw)
    if tr_ds.classes != va_ds.classes:
        raise RuntimeError(
            "train and val hold different classes; the probe would score "
            f"against a different label set: {tr_ds.classes} vs {va_ds.classes}")
    num_classes = len(tr_ds.classes)

    train_feats, train_labels = extract_features(vae, tr_loader, device)
    val_feats, val_labels = extract_features(vae, va_loader, device)
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
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
