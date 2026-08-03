"""
Linear evaluation for the official-style Context Prediction PyTorch port.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data.context_dataset_official import seed_worker

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.alexnet_context_official import build_official_context_alexnet
from train_step1_alexnet_official import make_deterministic, resolve_device


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int = 4096, num_classes: int = 1000) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        out = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            out.append(correct_k.mul_(100.0 / target.size(0)).item())
        return out


def make_loaders(data_path: str, batch_size: int, num_workers: int, img_size: int, seed: int = 42):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=train_transform)
    val = datasets.ImageFolder(os.path.join(data_path, "val"), transform=val_transform)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                   generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )


@torch.no_grad()
def extract_features(encoder: nn.Module, loader, device):
    encoder.eval()
    features = []
    labels = []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        feat = encoder(images).float().cpu()
        features.append(feat)
        labels.append(target)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def run_epoch(classifier, features, labels, criterion, optimizer, batch_size, device):
    classifier.train()
    order = torch.randperm(features.size(0))
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    count = 0
    for start in range(0, features.size(0), batch_size):
        idx = order[start : start + batch_size]
        x = features[idx].to(device, non_blocking=True)
        y = labels[idx].to(device, non_blocking=True)
        logits = classifier(x)
        loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        top1, top5 = accuracy(logits, y)
        n = y.numel()
        loss_sum += float(loss.item()) * n
        top1_sum += top1 * n
        top5_sum += top5 * n
        count += n
    return loss_sum / count, top1_sum / count, top5_sum / count


@torch.no_grad()
def validate(classifier, features, labels, criterion, batch_size, device):
    classifier.eval()
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    count = 0
    for start in range(0, features.size(0), batch_size):
        x = features[start : start + batch_size].to(device, non_blocking=True)
        y = labels[start : start + batch_size].to(device, non_blocking=True)
        logits = classifier(x)
        loss = criterion(logits, y)
        top1, top5 = accuracy(logits, y)
        n = y.numel()
        loss_sum += float(loss.item()) * n
        top1_sum += top1 * n
        top5_sum += top5 * n
        count += n
    return loss_sum / count, top1_sum / count, top5_sum / count


def read_encoder_state(obj) -> tuple[str, dict]:
    """The encoder's weights, from whichever shape was handed over.

    Two things legitimately arrive here and they are not the same shape:

    - **`encoder.pt`**, which the contract produces: the encoder alone, keys
      unprefixed. Refusing it would make the contract decorative -- nothing
      could consume what the previous stage was required to emit
    - **a training checkpoint**, `{"state_dict": <whole model>}`, which is
      what the cluster's own runs write. Refusing it would break a path that
      works today

    Told apart by their keys, never guessed. Loading the wrong one would
    evaluate an encoder full of default initialisation and report a number
    that looks like a result.
    """
    reference = build_official_context_alexnet(num_classes=8)
    encoder_keys = set(reference.get_encoder().state_dict())

    if isinstance(obj, dict) and "state_dict" in obj:
        full = obj["state_dict"]
        prefix = "encoder."
        state = {k[len(prefix):]: v for k, v in full.items()
                 if k.startswith(prefix)}
        if set(state) == encoder_keys:
            return "checkpoint", state
        raise RuntimeError(
            "the checkpoint has a state_dict, but nothing under 'encoder.' "
            "matches this model's encoder; it is for a different model")

    if isinstance(obj, dict) and set(obj) == encoder_keys:
        return "encoder", dict(obj)

    raise RuntimeError(
        "this is neither an encoder state dict nor a training checkpoint. "
        f"Expected the {len(encoder_keys)} keys of the encoder, or a "
        "'state_dict' entry holding the whole model")


def build_parser() -> argparse.ArgumentParser:
    """The original command line, unchanged, plus one addition.

    Every original flag stays: the cluster's job scripts call this file
    directly. `--encoder` is added so the contract's own artifact can be the
    input; exactly one of it and `--checkpoint` is required.
    """
    parser = argparse.ArgumentParser(description="Official-style Context Prediction linear eval")
    parser.add_argument("--checkpoint")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--feature_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--encoder",
                        help="an encoder.pt as the contract defines it; "
                             "an alternative to --checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto",
                        choices=("auto", "cuda", "cpu"))
    return parser


def run(args) -> dict:
    """The original body, unchanged apart from the input, seeding and return."""
    source = args.encoder or args.checkpoint
    if bool(args.encoder) == bool(args.checkpoint):
        raise RuntimeError("give exactly one of --encoder and --checkpoint")

    make_deterministic()
    device = resolve_device(getattr(args, "device", "auto"), args.gpu)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    kind, encoder_state = read_encoder_state(ckpt)
    print(f"loaded the encoder from a {kind}: {source}", flush=True)
    model = build_official_context_alexnet(num_classes=8)
    model.to(device)
    encoder = model.get_encoder()
    encoder.load_state_dict(encoder_state, strict=True)
    for p in encoder.parameters():
        p.requires_grad = False

    train_loader, val_loader = make_loaders(args.data_path, args.feature_batch_size, args.num_workers, args.img_size, seed=args.seed)
    print("extract train features", flush=True)
    train_features, train_labels = extract_features(encoder, train_loader, device)
    print("extract val features", flush=True)
    val_features, val_labels = extract_features(encoder, val_loader, device)
    print(f"train_features={tuple(train_features.shape)} val_features={tuple(val_features.shape)}", flush=True)

    classifier = LinearClassifier(4096, 1000).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_top1 = 0.0
    best_top5 = 0.0
    final_top1 = 0.0
    final_top5 = 0.0
    for epoch in range(args.epochs):
        train_loss, train_top1, train_top5 = run_epoch(
            classifier, train_features, train_labels, criterion, optimizer, args.batch_size, device
        )
        val_loss, final_top1, final_top5 = validate(
            classifier, val_features, val_labels, criterion, args.batch_size, device
        )
        scheduler.step()
        if final_top1 > best_top1:
            best_top1 = final_top1
            best_top5 = final_top5
            torch.save({"epoch": epoch, "classifier": classifier.state_dict(), "best_top1": best_top1}, save_dir / "best_linear_classifier.pth")
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_top1={train_top1:.3f} "
            f"val_loss={val_loss:.4f} val_top1={final_top1:.3f} val_top5={final_top5:.3f}",
            flush=True,
        )

    results = {
        "checkpoint": source,
        "checkpoint_global_step": ckpt.get("global_step"),
        "model_type": "official_style_alexnet_context",
        "feature_dim": 4096,
        "img_size": args.img_size,
        "linear_eval_preprocess": "ImageNet RandomResizedCrop/CenterCrop + ImageNet mean/std",
        "feature_protocol": "fc6 with adaptive 2x2 pool for 224x224 ImageNet images",
        "best_top1_acc": best_top1,
        "best_top5_acc": best_top5,
        "final_top1_acc": final_top1,
        "final_top5_acc": final_top5,
        "provenance": ckpt.get("protocol", {}),
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
