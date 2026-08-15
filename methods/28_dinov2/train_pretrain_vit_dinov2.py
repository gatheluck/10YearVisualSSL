"""DINOv2 unified Step-2 pretraining: ViT-B/16 from scratch, single process.

A port of the capture's `methods/28_dinov2/train_step2_vit.py`. The student and
its EMA teacher are ViT-B/16; the objective is DINO (cross-view) + iBOT (masked
patches) + KoLeo (spread), trained from scratch on ImageNet-1k. The capture ran
it under DistributedDataParallel with a health/collapse gate and TensorBoard; this
port owns a thin single-process fp32 loop, resolves the device instead of assuming
CUDA, and drops DDP / TensorBoard / the distributed health gate. The recipe
itself -- the per-epoch warmup->cosine LR (peak = base_lr x batch/1024), the
per-step teacher-momentum cosine, the per-epoch teacher-temp warmup, the shared
DINO/iBOT prototype head, freeze_last_layer for the first epoch and gradient-norm
clipping -- is kept faithfully.

`encoder.pt` is the EMA **teacher backbone** (`teacher_bb.*`, the prefix stripped
so it loads into a plain DINOv2Backbone); the heads, the student and the centering
buffers are training machinery. Milestone `checkpoint_epoch_{N}.pth` is written at
each `training.save_at_epochs`.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_dinov2_model, DINOLoss, iBOTLoss, KoLeoLoss   # noqa: E402
from data import get_dinov2_dataloader                                  # noqa: E402


def resolve_device(spec: str) -> "torch.device":
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device is 'cuda' but no CUDA device is visible. Ask for 'auto' "
                "to accept a CPU; getting a CPU silently would misreport what ran")
        return torch.device("cuda")
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def cosine_schedule(base: float, final: float, step: int, total: int) -> float:
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * step / max(total, 1)))


def linear_warmup_cosine_lr(optimizer, epoch, total_epochs, peak_lr, min_lr,
                            warmup_epochs) -> float:
    if epoch < warmup_epochs:
        lr = peak_lr * (epoch + 1) / max(warmup_epochs, 1)
    else:
        lr = cosine_schedule(peak_lr, min_lr, epoch - warmup_epochs,
                             total_epochs - warmup_epochs)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def get_teacher_temp(epoch, cfg) -> float:
    t_min = cfg["dino"]["teacher_temp_min"]
    t_max = cfg["dino"]["teacher_temp_max"]
    t_warmup = cfg["dino"]["teacher_temp_warmup_epochs"]
    if epoch < t_warmup:
        return t_min + (t_max - t_min) * epoch / max(t_warmup, 1)
    return t_max


def teacher_momentum_for_step(epoch, step, steps_per_epoch, total_epochs, cfg) -> float:
    global_step = epoch * steps_per_epoch + step
    total_steps = total_epochs * steps_per_epoch
    return cosine_schedule(cfg["training"]["momentum_teacher_min"],
                           cfg["training"]["momentum_teacher_max"],
                           global_step, total_steps)


def cancel_gradients_last_layer(model, epoch, freeze_epochs) -> None:
    if epoch >= freeze_epochs:
        return
    for head in model.student_projection_heads():
        for parameter in head.last_layer.parameters():
            parameter.grad = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DINOv2 Step 2: ViT-B/16, single process")
    parser.add_argument("--config", default="configs/pretrain_vit.yaml")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser


def run(args, config: "dict | None" = None) -> dict:
    if config is not None:
        cfg = config
    else:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    train_path = getattr(args, "data_path", None) or cfg["data"]["train_path"]
    device = resolve_device(getattr(args, "device", "auto"))
    seed = int(cfg.get("seed", 0))
    make_deterministic(seed)

    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    t = cfg["training"]
    total_epochs = int(t["epochs"])
    batch_size = int(t["batch_size"])
    peak_lr = float(t["base_lr"]) * batch_size / 1024.0
    save_at = {int(n) for n in t.get("save_at_epochs", [])}

    loader, _ = get_dinov2_dataloader(train_path, cfg, batch_size,
                                      distributed=False, seed=seed)
    model = build_dinov2_model(cfg).to(device)
    optimizer = optim.AdamW(list(model.student_parameters()), lr=peak_lr,
                            betas=(float(t["beta1"]), float(t["beta2"])),
                            weight_decay=float(t["weight_decay"]))
    dino_loss_fn = DINOLoss(out_dim=cfg["dino"]["out_dim"],
                            student_temp=cfg["dino"]["student_temp"],
                            teacher_temp=cfg["dino"]["teacher_temp_min"],
                            center_momentum=cfg["dino"]["center_momentum"]).to(device)
    ibot_loss_fn = iBOTLoss(out_dim=cfg["ibot"]["out_dim"],
                            student_temp=cfg["dino"]["student_temp"],
                            teacher_temp=cfg["dino"]["teacher_temp_min"],
                            center_momentum=cfg["dino"]["center_momentum"]).to(device)
    koleo_loss_fn = KoLeoLoss().to(device)

    dino_w = cfg["dino_loss_weight"]
    ibot_w = cfg["ibot"]["loss_weight"]
    koleo_w = cfg["koleo"]["loss_weight"]
    freeze_epochs = int(t["freeze_last_layer_epochs"])
    clip_grad = float(t["clip_grad"])
    steps_per_epoch = max(1, len(loader))

    print("=" * 72)
    print("DINOv2  Step 2: ViT-B/16 from scratch  (DINO + iBOT + KoLeo)")
    print(f"  device={device}  epochs={total_epochs}  batch={batch_size}  "
          f"peak_lr={peak_lr:.2e}  embed_dim={cfg['model']['embed_dim']}  "
          f"save_at_epochs={sorted(save_at)}")
    print("=" * 72)

    final_loss = None
    for epoch in range(total_epochs):
        lr = linear_warmup_cosine_lr(optimizer, epoch, total_epochs, peak_lr,
                                     float(t["min_lr"]), int(t["warmup_epochs"]))
        t_temp = get_teacher_temp(epoch, cfg)
        dino_loss_fn.teacher_temp = t_temp
        ibot_loss_fn.teacher_temp = t_temp
        model.train()

        running, count = 0.0, 0
        for step, (global_crops, local_crops, masks, _labels) in enumerate(loader):
            global_crops = [c.to(device, non_blocking=True) for c in global_crops]
            local_crops = [c.to(device, non_blocking=True) for c in local_crops]
            masks = [m.to(device, non_blocking=True) for m in masks]
            B = global_crops[0].shape[0]

            s_cls_g, s_patch_g, s_cls_l, _cls0, global_backbone_cls = model(
                global_crops, local_crops, masks)
            t_cls_g, t_patch_g = model.forward_teacher(global_crops, masks)

            loss_dino = dino_loss_fn(s_cls_g + s_cls_l, t_cls_g)
            loss_ibot = ibot_loss_fn(s_patch_g, t_patch_g, masks)
            loss_koleo = sum(koleo_loss_fn(cls) for cls in global_backbone_cls)
            loss = dino_w * loss_dino + ibot_w * loss_ibot + koleo_w * loss_koleo
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"DINOv2 loss became non-finite: {loss.item()}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            cancel_gradients_last_layer(model, epoch, freeze_epochs)
            nn.utils.clip_grad_norm_(list(model.student_parameters()), clip_grad)
            optimizer.step()
            momentum = teacher_momentum_for_step(epoch, step, steps_per_epoch,
                                                 total_epochs, cfg)
            model.update_teacher(momentum)

            running += loss.item() * B
            count += B
        final_loss = running / count if count else None
        print(f"  [{epoch}] dinov2_loss={final_loss}  lr={lr:.3g}  t_temp={t_temp:.4f}")

        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "loss": final_loss,
                "config": cfg}
        torch.save(ckpt, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(ckpt, os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nDINOv2 Step 2 training complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
