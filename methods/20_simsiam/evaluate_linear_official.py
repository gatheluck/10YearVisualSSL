"""
Official-style SimSiam linear evaluation.

Matches facebookresearch/simsiam main_lincls.py defaults closely:
frozen ResNet-50 backbone, global batch 4096, 90 epochs, base LR 0.1
scaled by batch/256, LARS optimizer, cosine LR, ImageNet normalization.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models import build_simsiam_resnet
from train_pretrain_resnet import make_deterministic, resolve_device

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LARS(optim.Optimizer):
    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0, eta=0.001):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                dp = p.grad
                if p.ndim > 1:
                    if group["weight_decay"] > 0:
                        dp = dp.add(p, alpha=group["weight_decay"])
                    p_norm = torch.norm(p)
                    dp_norm = torch.norm(dp)
                    if p_norm > 0 and dp_norm > 0:
                        dp = dp.mul(group["eta"] * p_norm / dp_norm)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(dp)
                p.add_(buf, alpha=-group["lr"])
        return loss


class FrozenBackboneLinear(nn.Module):
    def __init__(self, encoder, in_dim, num_classes=1000):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(in_dim, num_classes)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc.bias, 0.0)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x).flatten(1)
        return self.fc(feats)


def setup_dist():
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, local_rank, dist.get_world_size()


def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def reduce_meter(meter, device):
    if not (dist.is_available() and dist.is_initialized()):
        return meter.avg
    t = torch.tensor([meter.sum, meter.count], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t[0] / t[1]).item()


def topk_acc(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        bs = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        correct = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
        return [
            correct[:k].reshape(-1).float().sum().mul_(100.0 / bs).item()
            for k in topk
        ]


def get_dataloaders(data_path, batch_size, num_workers, img_size, distributed):
    normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_path, "train"), train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_path, "val"), val_tf)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_dl, val_dl, train_sampler


def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def load_encoder(checkpoint, model_type):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_model = ckpt.get("config", {}).get("model", {})
    if model_type == "resnet":
        ssl_model = build_simsiam_resnet(
            dim=cfg_model.get("dim", 2048),
            pred_dim=cfg_model.get("pred_dim", 512),
        )
        in_dim = 2048
    else:
        # Step 2 (ViT) has no official-style variant in the capture and was
        # not brought across, so the model it would need is not here. Refused
        # by name rather than left to fail as an ImportError three frames
        # away, which is how it first showed up.
        raise NotImplementedError(
            "model_type='vit' belongs to step 2, which this port does not "
            "include: models/simsiam_vit.py was not brought across because "
            "the capture has no official-style step 2")
    state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    ssl_model.load_state_dict(state, strict=True)
    return ssl_model.get_encoder(), in_dim, ckpt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official-style SimSiam linear eval")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", choices=["resnet", "vit"], required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--batch_size", type=int, default=4096, help="Global batch size")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--lr", type=float, default=0.1, help="Base LR before batch scaling")
    parser.add_argument("--optimizer", choices=["lars", "sgd"], default="lars")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_dir", default="./results/linear_eval")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume_linear", default="")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Added by the port; the captured evaluation "
                             "assumed CUDA")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run(args, encoder=None, in_dim=None) -> dict:
    """The captured `main()`, callable in process and returning its numbers.

    Changed during the port, and recorded in provenance.json:

    - **the device is resolved rather than assumed.** The captured code built
      `torch.device("cuda:...")` unconditionally and called
      `torch.cuda.set_device`, so it could not start without a GPU
    - **the encoder may be handed in.** The captured loader rebuilds the whole
      SimSiam model from a training checkpoint with `strict=True`; the
      contract's artifact is `encoder.pt`, which holds the backbone alone.
      Rather than teach this file a second way to recognise a file, the caller
      passes the encoder it already built -- so there is one place that knows
      how an encoder is loaded
    - **it returns its metrics.** The captured version wrote results.json and
      returned nothing, so an adapter had nothing to record
    """
    distributed, local_rank, world_size = setup_dist()
    device = resolve_device(getattr(args, "device", "auto"), local_rank)
    make_deterministic(int(getattr(args, "seed", 42)) + local_rank)
    os.makedirs(args.save_dir, exist_ok=True)

    ckpt: dict = {}
    if encoder is None:
        encoder, in_dim, ckpt = load_encoder(args.checkpoint, args.model_type)
    model = FrozenBackboneLinear(encoder, in_dim).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    per_rank_batch = max(args.batch_size // world_size, 1)
    train_dl, val_dl, train_sampler = get_dataloaders(
        args.data_path, per_rank_batch, args.num_workers,
        int(getattr(args, "img_size", 224)), distributed
    )

    raw = model.module if distributed else model
    init_lr = args.lr * args.batch_size / 256.0
    if args.optimizer == "lars":
        optimizer = LARS(raw.fc.parameters(), lr=init_lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(raw.fc.parameters(), lr=init_lr, momentum=0.9, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc1 = 0.0
    start_epoch = 0
    resume_path = args.resume_linear or os.path.join(args.save_dir, "linear_resume.pth")
    if os.path.isfile(resume_path):
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        raw.fc.load_state_dict(resume["classifier"])
        optimizer.load_state_dict(resume["optimizer"])
        best_acc1 = resume.get("best_acc1", 0.0)
        start_epoch = resume["epoch"] + 1
        if is_main():
            print(f"[resume] {resume_path} epoch={start_epoch} best={best_acc1:.3f}")

    writer = None
    if is_main():
        writer = SummaryWriter(os.path.join(args.save_dir, "logs", datetime.now().strftime("%Y%m%d_%H%M%S")))
        print("=" * 72)
        print("SimSiam official-style linear evaluation")
        print(f"checkpoint={args.checkpoint}")
        print(f"ckpt_epoch={ckpt.get('epoch', '?')} world_size={world_size}")
        print(f"epochs={args.epochs} global_batch={args.batch_size} base_lr={args.lr} init_lr={init_lr}")
        print(f"optimizer={args.optimizer} weight_decay={args.weight_decay}")
        print("=" * 72)

    final_acc1 = final_acc5 = 0.0
    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        lr = adjust_learning_rate(optimizer, init_lr, epoch, args.epochs)

        model.eval()
        train_loss, train_acc1, train_acc5 = AverageMeter(), AverageMeter(), AverageMeter()
        t0 = time.time()
        for imgs, labels in train_dl:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            a1, a5 = topk_acc(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n = imgs.size(0)
            train_loss.update(loss.item(), n)
            train_acc1.update(a1, n)
            train_acc5.update(a5, n)

        val_loss, val_acc1, val_acc5 = AverageMeter(), AverageMeter(), AverageMeter()
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(imgs)
                loss = criterion(logits, labels)
                a1, a5 = topk_acc(logits, labels)
                n = imgs.size(0)
                val_loss.update(loss.item(), n)
                val_acc1.update(a1, n)
                val_acc5.update(a5, n)

        train_acc1_g = reduce_meter(train_acc1, device)
        val_loss_g = reduce_meter(val_loss, device)
        val_acc1_g = reduce_meter(val_acc1, device)
        val_acc5_g = reduce_meter(val_acc5, device)
        final_acc1, final_acc5 = val_acc1_g, val_acc5_g

        if is_main():
            elapsed = time.time() - t0
            print(
                f"[{epoch + 1:3d}/{args.epochs}] "
                f"train_acc1={train_acc1_g:.2f} val_loss={val_loss_g:.4f} "
                f"val_acc1={val_acc1_g:.3f} val_acc5={val_acc5_g:.3f} "
                f"lr={lr:.5f} time={elapsed:.0f}s"
            )
            if writer:
                writer.add_scalar("val/acc1", val_acc1_g, epoch)
                writer.add_scalar("val/acc5", val_acc5_g, epoch)

            if val_acc1_g > best_acc1:
                best_acc1 = val_acc1_g
                torch.save(
                    {"epoch": epoch, "classifier": raw.fc.state_dict(), "best_acc1": best_acc1},
                    os.path.join(args.save_dir, "best_linear_classifier.pth"),
                )
            torch.save(
                {
                    "epoch": epoch,
                    "classifier": raw.fc.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_acc1": best_acc1,
                },
                os.path.join(args.save_dir, "linear_resume.pth"),
            )

    if is_main():
        results = {
            "checkpoint": args.checkpoint,
            "model_type": args.model_type,
            "best_top1_acc": round(float(best_acc1), 4),
            "final_top1_acc": round(float(final_acc1), 4),
            "final_top5_acc": round(float(final_acc5), 4),
            "protocol": {
                "source": "facebookresearch/simsiam main_lincls.py style",
                "epochs": args.epochs,
                "global_batch_size": args.batch_size,
                "base_lr": args.lr,
                "actual_initial_lr": init_lr,
                "optimizer": args.optimizer,
                "weight_decay": args.weight_decay,
                "preprocessing": "ImageNet RandomResizedCrop/HFlip + mean/std normalization",
            },
        }
        with open(os.path.join(args.save_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
        print(f"Best Top-1: {best_acc1:.3f}%")
        if writer:
            writer.close()

    if distributed:
        dist.destroy_process_group()

    return {
        "best_top1_acc": float(best_acc1),
        "final_top1_acc": float(final_acc1),
        "final_top5_acc": float(final_acc5),
        "epochs": args.epochs - start_epoch,
    }


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
