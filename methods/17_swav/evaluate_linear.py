"""
Linear evaluation for SwAV.
Freezes the SSL-pretrained encoder; trains a linear classifier on ImageNet-1k.
  Epochs: 100  Optimizer: SGD lr=0.3, wd=1e-6, cosine decay
  Supports ResNet-50 (Step 1) and ViT-Base (Step 2) backbones.

Changed during the port, and recorded in provenance.json:

- **the device is resolved rather than assumed.** The captured code built
  `torch.device("cuda:...")` and called `torch.cuda.set_device` unconditionally,
  so it could not start without a GPU.
- **the encoder may be handed in.** The captured code rebuilds the whole model
  from a training checkpoint and takes its backbone; the contract's artifact is
  `encoder.pt`, the backbone alone. The caller passes the encoder it already
  built, so there is one place that knows how an encoder is loaded.
- **`main()` is split into `build_parser()` and `run(args, encoder, in_dim)`,**
  and `run` returns its metrics; the captured version wrote results.json and
  returned nothing.
- **`model_type='vit'` is refused by name.** Step 2 was not brought across, so
  `build_vit_swav` is absent; the branch that imports it is unreachable here.
"""
import os, sys, time, json, argparse
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
from models.resnet_swav import build_resnet_swav
from train_pretrain_resnet import make_deterministic, resolve_device


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count


def topk_acc(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk); bs = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        correct = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
        return [correct[:k].reshape(-1).float().sum().mul_(100.0 / bs).item()
                for k in topk]


def get_dataloaders(data_path, batch_size, num_workers, img_size=224):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
    tr_t = transforms.Compose([transforms.RandomResizedCrop(img_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), norm])
    va_t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(img_size),
                                transforms.ToTensor(), norm])
    tr_ds = datasets.ImageFolder(os.path.join(data_path, "train"), tr_t)
    va_ds = datasets.ImageFolder(os.path.join(data_path, "val"),   va_t)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)
    return tr_dl, va_dl


class LinearCls(nn.Module):
    def __init__(self, in_dim, num_cls=1000):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_cls)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0)
    def forward(self, x): return self.fc(x)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--model_type",     choices=["resnet", "vit"], required=True)
    p.add_argument("--data_path",      required=True)
    p.add_argument("--batch_size",     type=int,   default=256)
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--lr",             type=float, default=0.3)
    p.add_argument("--weight_decay",   type=float, default=1e-6)
    p.add_argument("--num_workers",    type=int,   default=8)
    p.add_argument("--img_size",       type=int,   default=224)
    p.add_argument("--save_dir",       default="./results/linear_eval")
    p.add_argument("--gpu",            type=int,   default=0)
    p.add_argument("--device",         default="auto",
                   choices=["auto", "cuda", "cpu"],
                   help="Added by the port; the captured evaluation assumed CUDA")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--resume_linear",  default="",
                   help="Path to a saved linear checkpoint (last_linear_checkpoint.pth) to resume from")
    return p


def load_encoder(checkpoint, model_type):
    """Rebuild the model from a training checkpoint and return its backbone.

    Used only on the stand-alone CLI path; the adapter hands in an encoder it
    built from `encoder.pt` instead. `model_type='vit'` is refused by name:
    step 2 was not brought across, so `build_vit_swav` is absent.
    """
    if model_type != "resnet":
        raise NotImplementedError(
            "model_type='vit' belongs to step 2, which this port does not "
            "include: build_vit_swav was not brought across because the "
            "capture has no official-style step 2")
    ckpt       = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg        = ckpt.get("config", {})
    out_dim    = cfg.get("model", {}).get("out_dim", 128)
    hidden_mlp = cfg.get("model", {}).get("hidden_mlp", 2048)
    nmb_protos = cfg.get("model", {}).get("nmb_prototypes", 3000)
    ssl_model  = build_resnet_swav(out_dim, hidden_mlp, nmb_protos)
    state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    ssl_model.load_state_dict(state)
    return ssl_model.get_encoder(), 2048, ckpt


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
    print(f"SwAV Linear Evaluation [{args.model_type.upper()}]")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Data       : {args.data_path}")
    print("=" * 70)

    if encoder is None:
        encoder, in_dim, _ = load_encoder(args.checkpoint, args.model_type)
    encoder = encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    train_dl, val_dl = get_dataloaders(args.data_path, args.batch_size,
                                       args.num_workers, args.img_size)
    print(f"Train {len(train_dl.dataset):,}  Val {len(val_dl.dataset):,}")

    # Determine start epoch and best_acc (may be overwritten when resuming)
    start_epoch = 0
    best_acc    = 0.0

    cls       = LinearCls(in_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        cls.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )

    # Resume from a previous linear checkpoint if provided
    lin_ckpt = None
    if args.resume_linear and os.path.isfile(args.resume_linear):
        lin_ckpt    = torch.load(args.resume_linear, map_location="cpu")
        cls.load_state_dict(lin_ckpt["classifier"])
        start_epoch = lin_ckpt.get("epoch", 0) + 1
        best_acc    = lin_ckpt.get("best_acc1", 0.0)
        if "optimizer" in lin_ckpt:
            optimizer.load_state_dict(lin_ckpt["optimizer"])
        print(f"Resumed linear classifier from epoch {lin_ckpt.get('epoch', '?')}, "
              f"best_acc1={best_acc:.2f}%, continuing from epoch {start_epoch + 1}")
    elif args.resume_linear:
        print(f"WARNING: --resume_linear file not found: {args.resume_linear}. Starting from scratch.")

    # When resuming without a saved optimizer state, CosineAnnealingLR requires
    # 'initial_lr' to be present in param_groups (last_epoch != -1 path).
    # Set it explicitly so the scheduler can reconstruct the LR curve correctly.
    if start_epoch > 0:
        for group in optimizer.param_groups:
            if "initial_lr" not in group:
                group["initial_lr"] = args.lr

    # Build scheduler; last_epoch position is restored so LR continues from the right point
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, last_epoch=start_epoch - 1)
    if lin_ckpt is not None and "scheduler" in lin_ckpt:
        scheduler.load_state_dict(lin_ckpt["scheduler"])

    writer    = SummaryWriter(os.path.join(args.save_dir, "logs",
                              datetime.now().strftime("%Y%m%d_%H%M%S")))

    # A run with no epochs leaves these unset; a zero-avg meter is the honest
    # floor for an accuracy that was never measured.
    va_acc1 = AverageMeter(); va_acc5 = AverageMeter()

    for epoch in range(start_epoch, args.epochs):
        cls.train()
        tr_loss = AverageMeter(); tr_acc1 = AverageMeter(); tr_acc5 = AverageMeter()
        t0 = time.time()
        for imgs, labels in train_dl:
            imgs = imgs.to(device); labels = labels.to(device)
            with torch.no_grad():
                feats = encoder(imgs)
                if feats.dim() > 2:
                    feats = feats.view(feats.size(0), -1)
            logits = cls(feats); loss = criterion(logits, labels)
            a1, a5 = topk_acc(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            n = imgs.size(0)
            tr_loss.update(loss.item(), n); tr_acc1.update(a1, n); tr_acc5.update(a5, n)

        cls.eval()
        va_loss = AverageMeter(); va_acc1 = AverageMeter(); va_acc5 = AverageMeter()
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs = imgs.to(device); labels = labels.to(device)
                feats = encoder(imgs)
                if feats.dim() > 2:
                    feats = feats.view(feats.size(0), -1)
                logits = cls(feats); loss = criterion(logits, labels)
                a1, a5 = topk_acc(logits, labels)
                n = imgs.size(0)
                va_loss.update(loss.item(), n); va_acc1.update(a1, n); va_acc5.update(a5, n)

        scheduler.step()
        elapsed = time.time() - t0
        print(f"[{epoch+1:3d}/{args.epochs}]  "
              f"Tr loss={tr_loss.avg:.4f} acc@1={tr_acc1.avg:.2f}%  |  "
              f"Va loss={va_loss.avg:.4f} acc@1={va_acc1.avg:.2f}% acc@5={va_acc5.avg:.2f}%  "
              f"({elapsed:.0f}s)")
        writer.add_scalars("acc1", {"train": tr_acc1.avg, "val": va_acc1.avg}, epoch)
        writer.add_scalar("val/acc5", va_acc5.avg, epoch)

        if va_acc1.avg > best_acc:
            best_acc = va_acc1.avg
            torch.save(
                {"epoch": epoch, "classifier": cls.state_dict(), "best_acc1": best_acc},
                os.path.join(args.save_dir, "best_linear_classifier.pth"))

        # Save full state for potential resume
        torch.save(
            {"epoch":      epoch,
             "classifier": cls.state_dict(),
             "optimizer":  optimizer.state_dict(),
             "scheduler":  scheduler.state_dict(),
             "best_acc1":  best_acc},
            os.path.join(args.save_dir, "last_linear_checkpoint.pth"))

    print(f"\nBest Top-1: {best_acc:.2f}%")
    results = {
        "checkpoint":     args.checkpoint,
        "model_type":     args.model_type,
        "epochs":         args.epochs,
        "lr":             args.lr,
        "weight_decay":   args.weight_decay,
        "best_top1_acc":  round(best_acc,    4),
        "final_top1_acc": round(va_acc1.avg, 4),
        "final_top5_acc": round(va_acc5.avg, 4),
    }
    with open(os.path.join(args.save_dir, "results.json"), "w") as fout:
        json.dump(results, fout, indent=4)
    print(f"Results saved -> {args.save_dir}/results.json")
    writer.close()

    return {
        "best_top1_acc": float(best_acc),
        "final_top1_acc": float(va_acc1.avg),
        "final_top5_acc": float(va_acc5.avg),
        "epochs": args.epochs - start_epoch,
    }


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
