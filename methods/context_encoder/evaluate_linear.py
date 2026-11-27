"""
Linear evaluation for the Context Encoder pretrain (AlexNet) representation.

Freezes the pretrained encoder, extracts its 4096-d bottleneck features once,
and trains a single linear classifier on ImageNet labels.

Changed during the port, and recorded in provenance.json:

- **the device is resolved rather than assumed.** The captured code called
  `torch.cuda.set_device` and `.cuda(args.gpu)` unconditionally, so it could not
  start without a GPU.
- **the encoder may be handed in.** The captured code rebuilds the model from a
  training checkpoint with `strict=True`; the contract's artifact is
  `encoder.pt`, the encoder + bottleneck alone. The caller passes the encoder it
  already built, so there is one place that knows how an encoder is loaded.
- **`main()` is split into `build_parser()` and `run(args, encoder, in_dim)`,**
  which returns its metrics; the captured version wrote results.json and
  returned nothing.
- **`model_type` other than 'alexnet' is refused by name.** The ViT (step 2) and
  the official Caffe feature paths were not brought across; the step-2 protocol
  imports from `utils` were dropped.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.context_encoder import create_model
from datasets import create_dataloader
from train_pretrain import make_deterministic, resolve_device


class AverageMeter:
    def __init__(self, name=""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


class LinearClassifier(nn.Linear):
    """A plain single Linear probe head (no normalization or hidden layer)."""
    def __init__(self, in_features, num_classes=1000):
        super().__init__(in_features, num_classes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Context Encoder Step 1 linear eval')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='alexnet',
                        choices=['alexnet', 'vit', 'official_alexnet'])
    parser.add_argument('--img_size', type=int, default=227)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Added by the port; the captured eval assumed CUDA')
    parser.add_argument('--seed', type=int, default=42)
    return parser


def load_encoder(checkpoint, model_type):
    """Rebuild the model from a training checkpoint and return it.

    Used only on the stand-alone CLI path; the adapter hands in an encoder it
    built from `encoder.pt` instead. Only 'alexnet' is supported: the ViT and
    the official Caffe feature paths belong to step 2 and were not brought
    across, so they are refused by name.
    """
    if model_type != 'alexnet':
        raise NotImplementedError(
            f"model_type={model_type!r} belongs to step 2 / the official Caffe "
            "feature path, which this port does not include; only 'alexnet' is "
            "available")
    model = create_model('alexnet', channels=3)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    return model, None, ckpt


@torch.no_grad()
def extract_features(model, data_loader, device, model_type="alexnet"):
    """The frozen representation, per arch: the AlexNet 4096-d bottleneck
    (`model(x) -> (_, features)`), or the ViT mean patch-token feature
    (`model.get_features(x)`, embed_dim)."""
    model.eval()
    feats, labels = [], []
    for images, lbs in data_loader:
        images = images.to(device, non_blocking=True)
        if model_type == "vit":
            features = model.get_features(images)
        else:
            _, features = model(images)
        feats.append(features.cpu())
        labels.append(lbs)
    return torch.cat(feats), torch.cat(labels)


def _topk(outputs, labels, k):
    topk = outputs.topk(min(k, outputs.size(1)), dim=1).indices
    return topk.eq(labels.view(-1, 1)).any(dim=1).float().mean().item()


def run(args, encoder=None, in_dim=None) -> dict:
    """The captured evaluation, callable in process and returning its numbers.

    The adapter hands in a rebuilt `encoder`; on that path both 'alexnet' and
    'vit' are supported. The stand-alone CLI path (encoder is None) can only
    rebuild the fixed AlexNet architecture -- the ViT needs its config dims and
    the official Caffe features are step 2 -- so it still refuses non-alexnet.
    """
    if encoder is None and args.model_type != 'alexnet':
        raise NotImplementedError(
            f"model_type={args.model_type!r} cannot be rebuilt stand-alone (the "
            "ViT needs its config dimensions; the official Caffe features are "
            "step 2). Hand in an encoder built from the config, or use 'alexnet'")

    device = resolve_device(getattr(args, 'device', 'auto'), 0)
    make_deterministic(int(getattr(args, 'seed', 42)))
    os.makedirs(args.save_dir, exist_ok=True)

    if encoder is None:
        encoder, in_dim, _ = load_encoder(args.checkpoint, args.model_type)
    model = encoder.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    train_loader = create_dataloader(
        'linear_probe', args.data_path, split='train',
        batch_size=args.batch_size, num_workers=args.num_workers,
        img_size=args.img_size, preprocess='torch')
    val_loader = create_dataloader(
        'linear_probe', args.data_path, split='val',
        batch_size=args.batch_size, num_workers=args.num_workers,
        img_size=args.img_size, preprocess='torch')

    # Features are extracted once and cached, as in the captured evaluation.
    train_features, train_labels = extract_features(
        model, train_loader, device, args.model_type)
    val_features, val_labels = extract_features(
        model, val_loader, device, args.model_type)
    if in_dim is None:
        in_dim = train_features.shape[1]

    classifier = LinearClassifier(in_dim, num_classes=1000).to(device)
    optimizer = optim.SGD(classifier.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    def adjust_lr(epoch):
        if epoch < 60:
            return 1.0
        if epoch < 80:
            return 0.1
        return 0.01

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=adjust_lr)
    criterion = nn.CrossEntropyLoss()

    tr_ds = torch.utils.data.TensorDataset(train_features, train_labels)
    va_ds = torch.utils.data.TensorDataset(val_features, val_labels)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    best_top1 = 0.0
    best_top5_at_best_top1 = 0.0
    final_top1 = final_top5 = 0.0
    for epoch in range(args.epochs):
        classifier.train()
        for features, labels in tr_dl:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss = criterion(classifier(features), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        classifier.eval()
        va_top1 = AverageMeter()
        va_top5 = AverageMeter()
        with torch.no_grad():
            for features, labels in va_dl:
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = classifier(features)
                n = features.size(0)
                va_top1.update(_topk(outputs, labels, 1), n)
                va_top5.update(_topk(outputs, labels, 5), n)
        scheduler.step()

        final_top1 = va_top1.avg * 100.0
        final_top5 = va_top5.avg * 100.0
        if final_top1 > best_top1:
            best_top1 = final_top1
            best_top5_at_best_top1 = final_top5
        print(f"[{epoch + 1:3d}/{args.epochs}] "
              f"val_top1={final_top1:.2f}% val_top5={final_top5:.2f}%")

    results = {
        'checkpoint': args.checkpoint,
        'model_type': args.model_type,
        'best_top1_acc': round(float(best_top1), 4),
        'best_top5_acc_at_best_top1': round(float(best_top5_at_best_top1), 4),
        'final_top1_acc': round(float(final_top1), 4),
        'final_top5_acc': round(float(final_top5), 4),
    }
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Best Top-1: {best_top1:.2f}%")

    return {
        'best_top1_acc': float(best_top1),
        'best_top5_acc_at_best_top1': float(best_top5_at_best_top1),
        'final_top1_acc': float(final_top1),
        'final_top5_acc': float(final_top5),
        'epochs': args.epochs,
    }


def main():
    run(build_parser().parse_args())


if __name__ == '__main__':
    main()
