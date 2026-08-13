"""Linear evaluation for the VAE: a probe on the frozen encoder's latent mean.

The representation is the model this port trains -- the VAE encoder read through
`VAE_CNN.get_features(x)`, which returns the latent mean `mu` (the conv encoder,
flattened, then `fc_mu`), one `latent_dim`-d vector per image. The encoder is
frozen; a single linear layer is trained on top.

**Faithful to the capture's VAE eval, not the shared ARSSL probe.** The capture
(`methods/2_vae/evaluate_linear.py`) keeps inputs in `[0,1]` (no ImageNet
mean/std) to match the VAE's reconstruction training, feeds `mu` to the linear
layer **without** mean-centring or L2-normalising, and trains with SGD
(momentum) under a cosine schedule with cross-entropy. Top-1 and top-5 are
reported.

**Dataset-agnostic**, as the capture's own loader is: it auto-detects
`torchvision.datasets.MNIST` (the shipped MNIST pretrain) versus an
`ImageFolder` (ImageNet), infers the class count from the dataset, and resizes
to the encoder's `img_size`. Probing an MNIST-trained encoder on ImageNet is the
capture's stated MNIST->ImageNet cross-domain transfer; the number is comparable
in name (`*_linear_probe_top1/5_accuracy`) but its scale depends on the dataset,
as recorded in docs/EVALUATION.md.

Features are extracted once with a deterministic transform (no train-time
augmentation) and reused across epochs -- the repo's cached-probe convention.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_pretrain_cnn import make_deterministic, resolve_device  # noqa: E402


def _is_mnist(data_root: str) -> bool:
    """The same test the port's training loader uses (data/vae_dataset.py):
    MNIST is chosen by inspecting the path, and nothing is ever downloaded."""
    return "MNIST" in data_root or os.path.exists(
        os.path.join(data_root, "MNIST"))


def _mnist_transform(img_size: int):
    ops = []
    if img_size != 28:
        ops.append(transforms.Resize(img_size))
    ops.append(transforms.ToTensor())
    # MNIST is one channel; the VAE encoder expects three. Keep [0,1].
    ops.append(transforms.Lambda(
        lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x))
    return transforms.Compose(ops)


def _imagefolder_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),          # [0,1], no normalisation (VAE convention)
    ])


def _loaders(data_root: str, img_size: int, batch_size: int, num_workers: int):
    """Train and val loaders, plus the class count, auto-detecting the dataset.

    Returns `(train_loader, val_loader, num_classes)`. Both splits carry labels;
    the frozen encoder never sees them, the linear probe does.
    """
    if _is_mnist(data_root):
        tf = _mnist_transform(img_size)
        tr = datasets.MNIST(data_root, train=True, download=False, transform=tf)
        va = datasets.MNIST(data_root, train=False, download=False, transform=tf)
        num_classes = len(tr.classes)
    else:
        tf = _imagefolder_transform(img_size)
        tr = datasets.ImageFolder(os.path.join(data_root, "train"), transform=tf)
        va = datasets.ImageFolder(os.path.join(data_root, "val"), transform=tf)
        if tr.classes != va.classes:
            raise RuntimeError(
                f"train and val hold different classes: {tr.classes} vs "
                f"{va.classes}")
        num_classes = len(tr.classes)

    def _dl(ds):
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            drop_last=False)

    return _dl(tr), _dl(va), num_classes


@torch.no_grad()
def _extract(model, loader, device):
    """The frozen `mu` for every image, and its label."""
    feats, labels = [], []
    for imgs, lbs in loader:
        feats.append(model.get_features(
            imgs.to(device, non_blocking=True)).float().cpu())
        labels.append(lbs)
    return torch.cat(feats), torch.cat(labels)


def _topk(outputs, labels, k):
    topk = outputs.topk(min(k, outputs.size(1)), dim=1).indices
    return topk.eq(labels.view(-1, 1)).any(dim=1).float().mean().item()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VAE linear eval")
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
    torch.manual_seed(seed)
    make_deterministic()

    if model is None:
        from models.vae_cnn import VAE_CNN
        model = VAE_CNN(latent_dim=int(train["latent_dim"]),
                        image_size=int(train["img_size"]))
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    img_size = int(train["img_size"])
    bs = int(train["batch_size"])
    nw = int(train["num_workers"])
    tr_loader, va_loader, num_classes = _loaders(data_root, img_size, bs, nw)
    print(f"VAE linear eval  device={device}  img_size={img_size}  "
          f"classes={num_classes}")

    train_feats, train_labels = _extract(model, tr_loader, device)
    val_feats, val_labels = _extract(model, va_loader, device)
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
