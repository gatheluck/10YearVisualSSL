"""
Linear evaluation for Barlow Twins.

Freezes the SSL-pretrained encoder and trains a linear classifier on ImageNet-1k.
  Epochs    : 100
  Batch     : 256
  Optimizer : SGD  lr=<arg>, momentum=0.9, weight_decay=0
  LR sched  : cosine annealing

Supports:
  --model_type resnet  : ResNet-50 backbone (Step 1), feature dim=2048
  --model_type vit     : ViT-Base/16  backbone (Step 2), feature dim=768

Features are extracted once and cached in memory before training the linear head,
which is significantly faster than re-extracting per-epoch.

Changed during the port, and recorded in provenance.json:

- **the device is resolved rather than assumed.** The captured code built
  `torch.device("cuda:...")` and called `torch.cuda.set_device` unconditionally,
  so it could not start without a GPU.
- **the encoder may be handed in.** The captured code rebuilds the whole model
  from a training checkpoint with `strict=True` and takes its backbone; the
  contract's artifact is `encoder.pt`, the backbone alone. The caller passes the
  encoder it already built, so there is one place that knows how an encoder is
  loaded.
- **`main()` is split into `build_parser()` and `run(args, encoder, in_dim)`,**
  and `run` returns its metrics; the captured version wrote results.json and
  returned nothing.
- **`model_type='vit'` is refused by name.** Step 2 was not brought across, so
  `build_barlow_vit` is absent; importing it would have failed at import time.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models import build_barlow_resnet
from train_step1_resnet import make_deterministic, resolve_device


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def topk_acc(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        bs   = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        correct = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
        return [correct[:k].reshape(-1).float().sum().mul_(100.0 / bs) for k in topk]


def get_dataloaders(data_path, batch_size, num_workers=8, img_size=224):
    norm = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        norm,
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_path, "train"), train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_path, "val"),   val_tf)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl


@torch.no_grad()
def extract_features(encoder, loader, device):
    encoder.eval()
    feats, labels = [], []
    for imgs, lbs in loader:
        feats.append(encoder(imgs.to(device)).cpu())
        labels.append(lbs)
    return torch.cat(feats), torch.cat(labels)


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, num_classes=1000):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x.flatten(1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Barlow Twins Linear Evaluation")
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--model_type",  choices=["resnet", "vit"], required=True)
    parser.add_argument("--data_path",   required=True)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--lr",          type=float, default=0.3,
                        help="LR for linear head (0.3 for ResNet, 1.0 for ViT)")
    parser.add_argument("--num_workers", type=int,   default=8)
    parser.add_argument("--img_size",    type=int,   default=224)
    parser.add_argument("--save_dir",    default="./results/linear_eval")
    parser.add_argument("--gpu",         type=int,   default=0)
    parser.add_argument("--device",      default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Added by the port; the captured evaluation "
                             "assumed CUDA")
    parser.add_argument("--seed",        type=int,   default=42)
    return parser


def load_encoder(checkpoint, model_type):
    """Rebuild the model from a training checkpoint and return its backbone.

    Used only on the stand-alone CLI path; the adapter hands in an encoder it
    built from `encoder.pt` instead. `model_type='vit'` is refused by name:
    step 2 was not brought across, so `build_barlow_vit` is absent and importing
    it would fail at import time.
    """
    if model_type != "resnet":
        raise NotImplementedError(
            "model_type='vit' belongs to step 2, which this port does not "
            "include: build_barlow_vit was not brought across because the "
            "capture has no official-style step 2")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    projector = ckpt.get("config", {}).get("model", {}).get(
        "projector", "8192-8192-8192")
    model = build_barlow_resnet(projector=str(projector))
    state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    return model.get_encoder(), 2048, ckpt


def run(args, encoder=None, in_dim=None) -> dict:
    """The captured `main()`, callable in process and returning its numbers.

    The encoder may be handed in (the contract's `encoder.pt` is the backbone
    alone); otherwise it is rebuilt from a training checkpoint via
    `load_encoder`. The device is resolved rather than assumed.
    """
    device = resolve_device(getattr(args, "device", "auto"), 0)
    make_deterministic(int(getattr(args, "seed", 42)))
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 70)
    print(f"Barlow Twins Linear Evaluation  [{args.model_type.upper()}]")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Data       : {args.data_path}")
    print(f"  LR         : {args.lr}  epochs={args.epochs}")
    print("=" * 70)

    if encoder is None:
        encoder, in_dim, _ = load_encoder(args.checkpoint, args.model_type)
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    train_dl, val_dl = get_dataloaders(args.data_path, args.batch_size,
                                       args.num_workers, args.img_size)
    print(f"Train: {len(train_dl.dataset):,}  Val: {len(val_dl.dataset):,}")

    print("Extracting training features ...")
    t0 = time.time()
    tr_f, tr_l = extract_features(encoder, train_dl, device)
    print(f"  train {tr_f.shape}  ({time.time()-t0:.1f}s)")

    print("Extracting validation features ...")
    t0 = time.time()
    va_f, va_l = extract_features(encoder, val_dl, device)
    print(f"  val   {va_f.shape}  ({time.time()-t0:.1f}s)")

    cls       = LinearClassifier(in_dim).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(cls.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_dir = os.path.join(args.save_dir, "logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer  = SummaryWriter(log_dir)
    best_acc1 = 0.0
    # A run with no epochs leaves these unset; a zero-avg meter is the honest
    # floor for an accuracy that was never measured.
    va_acc1 = AverageMeter()
    va_acc5 = AverageMeter()

    for epoch in range(args.epochs):
        # Train
        cls.train()
        tr_loss = AverageMeter()
        tr_acc1 = AverageMeter()
        tr_acc5 = AverageMeter()
        perm = torch.randperm(tr_f.size(0))

        for i in range(0, tr_f.size(0), args.batch_size):
            xb = tr_f[perm[i : i + args.batch_size]].to(device)
            yb = tr_l[perm[i : i + args.batch_size]].to(device)
            logits = cls(xb)
            loss   = criterion(logits, yb)
            a1, a5 = topk_acc(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n = xb.size(0)
            tr_loss.update(loss.item(), n)
            tr_acc1.update(a1.item(), n)
            tr_acc5.update(a5.item(), n)

        # Validate
        cls.eval()
        va_loss = AverageMeter()
        va_acc1 = AverageMeter()
        va_acc5 = AverageMeter()

        with torch.no_grad():
            for i in range(0, va_f.size(0), args.batch_size):
                xb = va_f[i : i + args.batch_size].to(device)
                yb = va_l[i : i + args.batch_size].to(device)
                logits = cls(xb)
                loss   = criterion(logits, yb)
                a1, a5 = topk_acc(logits, yb)
                n = xb.size(0)
                va_loss.update(loss.item(), n)
                va_acc1.update(a1.item(), n)
                va_acc5.update(a5.item(), n)

        scheduler.step()

        print(
            f"[{epoch:3d}/{args.epochs}]  "
            f"Tr loss={tr_loss.avg:.4f} acc@1={tr_acc1.avg:.2f}%  |  "
            f"Va loss={va_loss.avg:.4f} acc@1={va_acc1.avg:.2f}% acc@5={va_acc5.avg:.2f}%"
        )

        writer.add_scalars("acc1", {"train": tr_acc1.avg, "val": va_acc1.avg}, epoch)
        writer.add_scalar("val/acc1", va_acc1.avg, epoch)
        writer.add_scalar("val/acc5", va_acc5.avg, epoch)

        if va_acc1.avg > best_acc1:
            best_acc1 = va_acc1.avg
            torch.save(
                {"epoch": epoch, "classifier": cls.state_dict(), "best_acc1": best_acc1},
                os.path.join(args.save_dir, "best_linear_classifier.pth"),
            )

    print(f"\nBest Top-1 Accuracy: {best_acc1:.2f}%")

    results = {
        "checkpoint":     args.checkpoint,
        "model_type":     args.model_type,
        "best_top1_acc":  best_acc1,
        "final_top1_acc": va_acc1.avg,
        "final_top5_acc": va_acc5.avg,
    }
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    writer.close()
    print("Linear evaluation done.")

    return {
        "best_top1_acc": float(best_acc1),
        "final_top1_acc": float(va_acc1.avg),
        "final_top5_acc": float(va_acc5.avg),
        "epochs": args.epochs,
    }


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
