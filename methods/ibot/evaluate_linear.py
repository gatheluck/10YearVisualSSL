"""
Linear evaluation for iBOT (ImageNet-1k).

Freezes the SSL-pretrained ViT encoder and trains a linear classifier.

Protocol:
  Epochs    : 100
  Batch     : 256
  Optimizer : SGD  lr=<arg>, momentum=0.9, weight_decay=0
  LR sched  : cosine annealing
  Feature   : official ViT-S default is teacher checkpoint, last 4 CLS tokens

Supports:
  --model_type vit_small : ViT-S/16 backbone (Step 1)
  --model_type vit_base  : ViT-B/16 backbone (Step 2)
"""

import os
import sys
import time
import json
import math
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models import vit_small, vit_base, iBOT, DINOHead
from train_pretrain import make_deterministic, resolve_device


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_EMBED_DIMS = {
    "vit_small": 384,
    "vit_base":  768,
}

_VIT_BUILDERS = {
    "vit_small": vit_small,
    "vit_base":  vit_base,
}


# ── Utilities ─────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

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
        correct  = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
        return [correct[:k].reshape(-1).float().sum().mul_(100.0 / bs) for k in topk]


# ── Feature extraction ────────────────────────────────────────────────────────

def get_dataloaders(data_path, batch_size, num_workers=8, img_size=224, shuffle_train=True):
    norm = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        norm,
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_path, "train"), train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_path, "val"),   val_tf)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train,
                          num_workers=num_workers, pin_memory=True, drop_last=False)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_dl, val_dl


def feature_dim(embed_dim, n_last_blocks, avgpool_patchtokens):
    if avgpool_patchtokens == 0:
        return embed_dim * n_last_blocks
    if avgpool_patchtokens == 1:
        return embed_dim
    if avgpool_patchtokens == 2:
        return embed_dim * (n_last_blocks + 1)
    raise ValueError(f"Unsupported avgpool_patchtokens={avgpool_patchtokens}")


@torch.no_grad()
def forward_features(encoder, imgs, n_last_blocks=4, avgpool_patchtokens=0):
    intermediate = encoder.get_intermediate_layers(imgs, n_last_blocks)
    if avgpool_patchtokens == 0:
        output = [x[:, 0] for x in intermediate]
    elif avgpool_patchtokens == 1:
        output = [torch.mean(intermediate[-1][:, 1:], dim=1)]
    elif avgpool_patchtokens == 2:
        output = [x[:, 0] for x in intermediate]
        output.append(torch.mean(intermediate[-1][:, 1:], dim=1))
    else:
        raise ValueError(f"Unsupported avgpool_patchtokens={avgpool_patchtokens}")
    return torch.cat(output, dim=-1)


@torch.no_grad()
def extract_features(encoder, loader, device, n_last_blocks=4, avgpool_patchtokens=0):
    """Extract frozen features for the optional cached-feature mode."""
    encoder.eval()
    feats, labels = [], []
    for imgs, lbs in loader:
        feat = forward_features(
            encoder,
            imgs.to(device),
            n_last_blocks=n_last_blocks,
            avgpool_patchtokens=avgpool_patchtokens,
        )
        feats.append(feat.cpu())
        labels.append(lbs)
    return torch.cat(feats), torch.cat(labels)


def run_one_train_epoch(encoder, classifier, criterion, optimizer, loader, device,
                        n_last_blocks, avgpool_patchtokens):
    classifier.train()
    loss_m = AverageMeter()
    acc1_m = AverageMeter()
    acc5_m = AverageMeter()
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            feats = forward_features(encoder, imgs, n_last_blocks, avgpool_patchtokens)
        logits = classifier(feats)
        loss = criterion(logits, labels)
        acc1, acc5 = topk_acc(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n = imgs.size(0)
        loss_m.update(loss.item(), n)
        acc1_m.update(acc1.item(), n)
        acc5_m.update(acc5.item(), n)
    return loss_m.avg, acc1_m.avg, acc5_m.avg


@torch.no_grad()
def validate_online(encoder, classifier, criterion, loader, device,
                    n_last_blocks, avgpool_patchtokens):
    classifier.eval()
    loss_m = AverageMeter()
    acc1_m = AverageMeter()
    acc5_m = AverageMeter()
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        feats = forward_features(encoder, imgs, n_last_blocks, avgpool_patchtokens)
        logits = classifier(feats)
        loss = criterion(logits, labels)
        acc1, acc5 = topk_acc(logits, labels)
        n = imgs.size(0)
        loss_m.update(loss.item(), n)
        acc1_m.update(acc1.item(), n)
        acc5_m.update(acc5.item(), n)
    return loss_m.avg, acc1_m.avg, acc5_m.avg


# ── Linear classifier ─────────────────────────────────────────────────────────

class LinearClassifier(nn.Module):
    def __init__(self, in_dim, num_classes=1000):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="iBOT Linear Evaluation on ImageNet-1k")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to iBOT checkpoint (.pth)")
    parser.add_argument("--model_type",  choices=["vit_small", "vit_base"], required=True)
    parser.add_argument("--data_path",   required=True,
                        help="ImageNet root (contains train/ and val/)")
    parser.add_argument("--patch_size",  type=int, default=16)
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--epochs",      type=int, default=100)
    parser.add_argument("--lr",          type=float, default=0.001,
                        help="LR for linear head (default 1e-3; iBOT paper uses 1e-3 for ViT-S, 2e-4 for ViT-B)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_dir",    default="./results/linear_eval")
    parser.add_argument("--resume_linear", default="",
                        help="Path to a full linear-probe checkpoint to resume")
    parser.add_argument("--gpu",         type=int, default=0)
    parser.add_argument("--device",      default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Added by the port; the captured evaluation "
                             "assumed CUDA")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--checkpoint_key", choices=["student", "teacher"], default="teacher",
                        help="Backbone to evaluate. Official iBOT reports generally use teacher.")
    parser.add_argument("--n_last_blocks", type=int, default=4,
                        help="Official ViT-S linear eval concatenates CLS from the last 4 blocks.")
    parser.add_argument("--avgpool_patchtokens", type=int, choices=[0, 1, 2], default=0,
                        help="0: last-block CLS concat, 1: patch avgpool, 2: CLS concat + patch avgpool.")
    parser.add_argument("--eval_mode", choices=["online", "cached"], default="online",
                        help="online recomputes augmented features each epoch like official eval; cached is diagnostic only.")
    parser.add_argument("--allow_unverified_checkpoint", action="store_true",
                        help="Allow evaluation of checkpoints without health metadata.")
    return parser


def run(args, encoder=None, in_dim=None) -> dict:
    """The captured `main()`, callable in process and returning its numbers.

    Changed during the port, and recorded in provenance.json:

    - **the device is resolved rather than assumed.** The captured code built
      `torch.device("cuda:...")` and called `torch.cuda.set_device`
      unconditionally, so it could not start without a GPU
    - **the encoder may be handed in.** The captured code rebuilds the whole
      iBOT model from a training checkpoint with `strict=True` and takes its
      teacher; the contract's artifact is `encoder.pt`, the teacher backbone
      alone. The caller passes the encoder it already built, so there is one
      place that knows how an encoder is loaded, and the checkpoint health
      gate is reached only on the stand-alone path
    - **it returns its metrics.** The captured version wrote results.json and
      returned nothing, so an adapter had nothing to record
    """
    device = resolve_device(getattr(args, "device", "auto"), 0)
    make_deterministic(int(getattr(args, "seed", 42)))
    os.makedirs(args.save_dir, exist_ok=True)

    embed_dim = _EMBED_DIMS[args.model_type]
    if in_dim is None:
        in_dim = feature_dim(embed_dim, args.n_last_blocks, args.avgpool_patchtokens)

    print("=" * 75)
    print(f"iBOT Linear Evaluation  [{args.model_type}]")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Data       : {args.data_path}")
    print(f"  Feature dim: {in_dim}  LR={args.lr}  epochs={args.epochs}  mode={args.eval_mode}")
    print(f"  checkpoint_key={args.checkpoint_key}  n_last_blocks={args.n_last_blocks}  "
          f"avgpool_patchtokens={args.avgpool_patchtokens}")
    print("=" * 75)

    # ── Load checkpoint (only when the caller did not hand in an encoder) ────
    ckpt = {}
    health = None
    ckpt_epoch = "?"
    if encoder is None:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        ckpt_epoch = ckpt.get("epoch", "?")
        print(f"Loaded checkpoint from epoch {ckpt_epoch}")
        health = ckpt.get("health")
        if health is None:
            if not args.allow_unverified_checkpoint:
                raise RuntimeError(
                    "Refusing to evaluate iBOT checkpoint without health metadata. "
                    "Pass --allow_unverified_checkpoint only for documented legacy audits."
                )
        else:
            values = [health.get(key) for key in ("loss", "cls_loss", "patch_loss")]
            finite = all(value is not None and math.isfinite(float(value)) for value in values)
            healthy_losses = (
                finite
                and float(values[0]) >= 0.1
                and float(values[1]) >= 0.01
                and float(values[2]) >= 0.01
            )
            if not bool(health.get("is_valid", False)) or not healthy_losses:
                raise RuntimeError(f"Refusing collapsed/health-invalid iBOT checkpoint: {health}")

        # Build full iBOT model to restore state_dict, then extract backbone.
        # The pretraining head always consumes the ViT embedding dimension; the
        # linear classifier below consumes the concatenated eval feature dimension.
        builder     = _VIT_BUILDERS[args.model_type]
        student_vit = builder(patch_size=args.patch_size, use_mask_token=True)
        teacher_vit = builder(patch_size=args.patch_size, use_mask_token=False)
        ckpt_cfg    = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}
        ibot_cfg    = ckpt_cfg.get("ibot", {}) if isinstance(ckpt_cfg.get("ibot", {}), dict) else {}
        head        = DINOHead(
            in_dim=embed_dim,
            out_dim=ibot_cfg.get("out_dim", 8192),
            hidden_dim=ibot_cfg.get("head_hidden_dim", 2048),
            bottleneck_dim=ibot_cfg.get("head_bottleneck_dim", 256),
            nlayers=ibot_cfg.get("head_nlayers", 3),
            norm_last_layer=ibot_cfg.get("norm_last_layer", True),
        )

        full_model = iBOT(student_vit, teacher_vit, head)

        # Strip DDP prefix if present
        state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
        full_model.load_state_dict(state)

        # Official iBOT evaluation generally reports the teacher backbone.
        encoder = full_model.teacher if args.checkpoint_key == "teacher" else full_model.student

    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # ── Data / classifier ──────────────────────────────────────────────────
    train_dl, val_dl = get_dataloaders(
        args.data_path,
        args.batch_size,
        args.num_workers,
        shuffle_train=(args.eval_mode == "online"),
    )
    print(f"Train: {len(train_dl.dataset):,}  Val: {len(val_dl.dataset):,}")

    cls       = LinearClassifier(in_dim).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(cls.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_dir = os.path.join(args.save_dir, "logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer  = SummaryWriter(log_dir)
    best_acc1 = 0.0
    best_acc5_at_best_acc1 = 0.0
    best_epoch = -1
    start_epoch = 0

    if args.resume_linear and os.path.isfile(args.resume_linear):
        resume = torch.load(args.resume_linear, map_location="cpu", weights_only=False)
        expected = {
            "pretrain_checkpoint": os.path.realpath(args.checkpoint),
            "checkpoint_key": args.checkpoint_key,
            "model_type": args.model_type,
            "n_last_blocks": args.n_last_blocks,
            "avgpool_patchtokens": args.avgpool_patchtokens,
            "eval_mode": args.eval_mode,
            "linear_epochs": args.epochs,
        }
        actual = {key: resume.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"iBOT linear-resume protocol mismatch: expected={expected}, actual={actual}"
            )
        cls.load_state_dict(resume["classifier"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        start_epoch = int(resume["epoch"]) + 1
        best_acc1 = float(resume["best_acc1"])
        best_acc5_at_best_acc1 = float(resume["best_acc5_at_best_acc1"])
        best_epoch = int(resume["best_epoch"])
        print(
            f"Resumed linear probe at epoch {start_epoch}/{args.epochs}; "
            f"best Top-1={best_acc1:.2f}%"
        )
    elif args.resume_linear:
        print(f"No linear resume checkpoint at {args.resume_linear}; starting fresh")

    if args.eval_mode == "cached":
        print("Extracting training features ...")
        t0 = time.time()
        tr_f, tr_l = extract_features(
            encoder, train_dl, device, args.n_last_blocks, args.avgpool_patchtokens
        )
        print(f"  train {tr_f.shape}  ({time.time()-t0:.1f}s)")

        print("Extracting validation features ...")
        t0 = time.time()
        va_f, va_l = extract_features(
            encoder, val_dl, device, args.n_last_blocks, args.avgpool_patchtokens
        )
        print(f"  val   {va_f.shape}  ({time.time()-t0:.1f}s)")

    # A run with no epochs leaves these unset; 0.0 is the honest floor for an
    # accuracy that was never measured, and the adapter counts a missing best.
    va_acc1 = va_acc5 = 0.0
    for epoch in range(start_epoch, args.epochs):
        if args.eval_mode == "online":
            tr_loss, tr_acc1, tr_acc5 = run_one_train_epoch(
                encoder, cls, criterion, optimizer, train_dl, device,
                args.n_last_blocks, args.avgpool_patchtokens,
            )
            va_loss, va_acc1, va_acc5 = validate_online(
                encoder, cls, criterion, val_dl, device,
                args.n_last_blocks, args.avgpool_patchtokens,
            )
        else:
            cls.train()
            tr_loss_m = AverageMeter()
            tr_acc1_m = AverageMeter()
            tr_acc5_m = AverageMeter()
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
                tr_loss_m.update(loss.item(), n)
                tr_acc1_m.update(a1.item(), n)
                tr_acc5_m.update(a5.item(), n)

            cls.eval()
            va_loss_m = AverageMeter()
            va_acc1_m = AverageMeter()
            va_acc5_m = AverageMeter()

            with torch.no_grad():
                for i in range(0, va_f.size(0), args.batch_size):
                    xb = va_f[i : i + args.batch_size].to(device)
                    yb = va_l[i : i + args.batch_size].to(device)
                    logits = cls(xb)
                    loss   = criterion(logits, yb)
                    a1, a5 = topk_acc(logits, yb)
                    n = xb.size(0)
                    va_loss_m.update(loss.item(), n)
                    va_acc1_m.update(a1.item(), n)
                    va_acc5_m.update(a5.item(), n)
            tr_loss, tr_acc1, tr_acc5 = tr_loss_m.avg, tr_acc1_m.avg, tr_acc5_m.avg
            va_loss, va_acc1, va_acc5 = va_loss_m.avg, va_acc1_m.avg, va_acc5_m.avg

        scheduler.step()

        print(
            f"[{epoch:3d}/{args.epochs}]  "
            f"Tr loss={tr_loss:.4f} acc@1={tr_acc1:.2f}%  |  "
            f"Va loss={va_loss:.4f} acc@1={va_acc1:.2f}% acc@5={va_acc5:.2f}%"
        )

        writer.add_scalars("acc1", {"train": tr_acc1, "val": va_acc1}, epoch)
        writer.add_scalar("val/acc1", va_acc1, epoch)
        writer.add_scalar("val/acc5", va_acc5, epoch)

        if va_acc1 > best_acc1:
            best_acc1 = va_acc1
            best_acc5_at_best_acc1 = va_acc5
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "classifier": cls.state_dict(),
                    "best_acc1": best_acc1,
                    "best_acc5_at_best_acc1": best_acc5_at_best_acc1,
                },
                os.path.join(args.save_dir, "best_linear_classifier.pth"),
            )

        torch.save(
            {
                "epoch": epoch,
                "classifier": cls.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc1": best_acc1,
                "best_acc5_at_best_acc1": best_acc5_at_best_acc1,
                "best_epoch": best_epoch,
                "pretrain_checkpoint": os.path.realpath(args.checkpoint),
                "checkpoint_key": args.checkpoint_key,
                "model_type": args.model_type,
                "n_last_blocks": args.n_last_blocks,
                "avgpool_patchtokens": args.avgpool_patchtokens,
                "eval_mode": args.eval_mode,
                "linear_epochs": args.epochs,
            },
            args.resume_linear
            or os.path.join(args.save_dir, "last_linear_checkpoint.pth"),
        )

    print(
        f"\nBest Accuracy at linear epoch {best_epoch + 1}: "
        f"Top-1={best_acc1:.2f}% Top-5={best_acc5_at_best_acc1:.2f}%"
    )

    health_epoch = health.get("epoch") if isinstance(health, dict) else None
    if isinstance(health_epoch, int):
        pretrain_epoch = health_epoch
    elif isinstance(ckpt_epoch, int):
        pretrain_epoch = ckpt_epoch + 1
    else:
        pretrain_epoch = ckpt_epoch
    if args.checkpoint_key == "teacher" and args.n_last_blocks == 1 and args.avgpool_patchtokens == 2:
        feature_protocol = "teacher last-1 CLS + last-layer patch average"
    elif args.checkpoint_key == "teacher" and args.n_last_blocks == 4 and args.avgpool_patchtokens == 0:
        feature_protocol = "teacher last-4 CLS"
    else:
        feature_protocol = "custom"

    results = {
        "checkpoint":         args.checkpoint,
        "checkpoint_key":     args.checkpoint_key,
        "eval_mode":          args.eval_mode,
        "feature_protocol":   feature_protocol,
        "n_last_blocks":      args.n_last_blocks,
        "avgpool_patchtokens": args.avgpool_patchtokens,
        "model_type":         args.model_type,
        "pretrain_epoch":     pretrain_epoch,
        "checkpoint_epoch_zero_indexed": ckpt_epoch,
        "linear_epochs":      args.epochs,
        "best_linear_epoch":  best_epoch + 1,
        "best_top1_acc":      best_acc1,
        "best_top5_acc_at_best_top1": best_acc5_at_best_acc1,
        "final_top1_acc":     va_acc1,
        "final_top5_acc":     va_acc5,
    }
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    writer.close()
    print("Linear evaluation done.")

    return {
        "best_top1_acc": float(best_acc1),
        "best_top5_acc_at_best_top1": float(best_acc5_at_best_acc1),
        "final_top1_acc": float(va_acc1),
        "final_top5_acc": float(va_acc5),
        "epochs": args.epochs - start_epoch,
    }


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
