"""Franca unified Step-2 pretraining: ViT-B/16 from scratch, single process.

A port of the capture's `methods/36_franca/train_step2_vit.py`. A student and its
EMA teacher (a DINOv2-style ViT-B/16 backbone) are trained with Franca's nested
Matryoshka DINO/iBOT heads and Sinkhorn-Knopp losses (no centering), plus KoLeo,
from scratch on ImageNet-1k. The capture ran it under DDP with bf16 autocast and a
distributed health gate; this port owns a thin single-process fp32 loop, resolves
the device instead of assuming CUDA, and drops DDP / autocast / TensorBoard / the
health gate. The recipe -- per-step warmup->cosine LR, per-step teacher-momentum
and teacher-temp schedules, the Sinkhorn losses, gradient-norm clipping -- is kept.

`encoder.pt` is the EMA **teacher backbone** (`teacher_bb.*`, prefix stripped so it
loads into a plain DINOv2Backbone). Milestone `checkpoint_epoch_{N}.pth` is written
at each `training.save_at_epochs`.
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

from models import (build_franca_model, KoLeoLoss, FrancaDinoLoss,   # noqa: E402
                    FrancaIBOTLoss)
from data import get_franca_dataloader                                # noqa: E402


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


def cosine_value(base: float, final: float, step: int, total: int) -> float:
    if total <= 1:
        return final
    progress = min(max(step / total, 0.0), 1.0)
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * progress))


def warmup_cosine_value(base, final, step, total_steps, warmup_steps,
                        start_warmup_value: float = 0.0) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        progress = step / max(warmup_steps - 1, 1)
        return start_warmup_value + (base - start_warmup_value) * progress
    cosine_step = max(step - warmup_steps, 0)
    return cosine_value(base, final, cosine_step, max(total_steps - warmup_steps, 1))


def teacher_temp(step: int, steps_per_epoch: int, cfg: dict) -> float:
    warmup = int(cfg["dino"]["teacher_temp_warmup_epochs"]) * steps_per_epoch
    lo = float(cfg["dino"]["teacher_temp_min"])
    hi = float(cfg["dino"]["teacher_temp_max"])
    if step < warmup:
        return lo + (hi - lo) * step / max(warmup - 1, 1)
    return hi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Franca Step 2: ViT-B/16, single process")
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
    nesting_dims = [int(x) for x in cfg["model"]["nesting_dims"]]
    sk_iters = int(cfg["ibot"]["sinkhorn_iterations"])
    student_temp = float(cfg["dino"]["student_temp"])
    save_at = {int(n) for n in t.get("save_at_epochs", [])}

    loader, _ = get_franca_dataloader(train_path, cfg, batch_size,
                                      distributed=False, seed=seed)
    model = build_franca_model(cfg).to(device)
    optimizer = optim.AdamW(list(model.student_parameters()), lr=float(t["lr"]),
                            betas=(float(t["beta1"]), float(t["beta2"])),
                            weight_decay=float(t["weight_decay"]))
    dino_loss = FrancaDinoLoss(nesting_dims, student_temp, sk_iters)
    ibot_loss = FrancaIBOTLoss(nesting_dims, student_temp, sk_iters)
    koleo_loss = KoLeoLoss()

    dino_w = float(cfg["dino"]["loss_weight"])
    ibot_w = float(cfg["ibot"]["loss_weight"])
    koleo_w = float(cfg["koleo"]["loss_weight"])
    clip_grad = float(t["clip_grad"])
    steps_per_epoch = max(1, len(loader))
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = int(t["warmup_epochs"]) * steps_per_epoch

    print("=" * 72)
    print("Franca  Step 2: ViT-B/16 from scratch  (Matryoshka + Sinkhorn DINO/iBOT)")
    print(f"  device={device}  epochs={total_epochs}  batch={batch_size}  "
          f"lr={t['lr']}  nesting={nesting_dims}  save_at_epochs={sorted(save_at)}")
    print("=" * 72)

    final_loss = None
    for epoch in range(total_epochs):
        model.train()
        running, count = 0.0, 0
        for step, (global_crops, local_crops, masks, _labels) in enumerate(loader):
            global_crops = [c.to(device, non_blocking=True) for c in global_crops]
            local_crops = [c.to(device, non_blocking=True) for c in local_crops]
            masks = [m.to(device, non_blocking=True) for m in masks]
            B = global_crops[0].shape[0]
            global_step = epoch * steps_per_epoch + step

            lr = warmup_cosine_value(float(t["lr"]), float(t["min_lr"]),
                                     global_step, total_steps, warmup_steps)
            wd = cosine_value(float(t["weight_decay"]),
                              float(t["final_weight_decay"]), global_step, total_steps)
            momentum = cosine_value(float(t["momentum_teacher_min"]),
                                    float(t["momentum_teacher_max"]),
                                    global_step, total_steps)
            temp = teacher_temp(global_step, steps_per_epoch, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
                group["weight_decay"] = wd

            s_cls_g, s_patch_g, s_cls_l, global_backbone_cls = model(
                global_crops, local_crops, masks)
            t_cls_g, t_patch_g = model.forward_teacher(global_crops, masks)
            loss_dino = dino_loss(s_cls_g + s_cls_l, t_cls_g, temp)
            loss_ibot = ibot_loss(s_patch_g, t_patch_g, masks, temp)
            loss_koleo = sum(koleo_loss(cls) for cls in global_backbone_cls)
            loss = dino_w * loss_dino + ibot_w * loss_ibot + koleo_w * loss_koleo
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"Franca loss became non-finite: {loss.item()}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.student_parameters()), clip_grad)
            optimizer.step()
            model.update_teacher(momentum)

            running += loss.item() * B
            count += B
        final_loss = running / count if count else None
        print(f"  [{epoch}] franca_loss={final_loss}  lr={lr:.3g}  temp={temp:.4f}")

        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "loss": final_loss,
                "config": cfg}
        torch.save(ckpt, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(ckpt, os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nFranca Step 2 training complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
