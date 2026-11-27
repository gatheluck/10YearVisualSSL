"""Step-2 unified ViT-B/16 DINO pretraining, in one process.

A faithful port of the capture's `train_step2_vit_b.py`: the *same* DINO
self-distillation objective, head and multi-crop as Step 1, but on the unified
**ViT-B/16** backbone (`arch: vit_base`) under the unified Step-2 recipe -- AdamW
lr 6e-4, a **fixed** weight decay (Step 1 cosine-schedules it), betas (0.9, 0.95),
per-iteration cosine LR with a 10-epoch warmup, 300 epochs, milestone checkpoints
at `save_at_epochs` (100/200/300). Reuses `build_dino` (DINO's own ViT supports
`vit_base`, so this stays torch-only -- no timm), `get_dino_dataloader`, and the
Step-1 helpers (`resolve_device`, `make_deterministic`, `cosine_schedule`,
`lr_schedule_value`, `set_lr`, `clip_gradients`); the teacher EMA momentum keeps
its cosine schedule to 1.0. The capture's DDP / step2_protocol resume machinery
is dropped; the device is resolved, not sniffed.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_dino                                      # noqa: E402
from data import get_dino_dataloader                              # noqa: E402
from train_pretrain_dino import (clip_gradients, cosine_schedule,  # noqa: E402
                                 lr_schedule_value, make_deterministic,
                                 resolve_device, set_lr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DINO Step-2 unified ViT-B/16")
    parser.add_argument("--config", default="configs/pretrain_vit.yaml")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    return parser


def run(args, config: "dict | None" = None) -> dict:
    if config is not None:
        cfg = config
    else:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    if getattr(args, "data_path", None):
        cfg["data"]["data_root"] = args.data_path

    device = resolve_device(getattr(args, "device", "auto"))
    seed = int(cfg.get("seed", 0))
    make_deterministic(seed)

    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    m = cfg["model"]
    dn = cfg["dino"]
    d = cfg["data"]
    t = cfg["training"]
    img_size = int(m["img_size"])

    model = build_dino(
        arch=str(m["arch"]), out_dim=int(dn["out_dim"]),
        n_local_crops=int(dn["n_local_crops"]),
        student_temp=float(dn["student_temp"]),
        teacher_temp_init=float(dn["teacher_temp_init"]),
        teacher_temp_final=float(dn["teacher_temp_final"]),
        teacher_temp_warmup_epochs=int(dn["teacher_temp_warmup_epochs"]),
        hidden_dim=int(dn["hidden_dim"]), bottleneck_dim=int(dn["bottleneck_dim"]),
        use_bn_in_head=bool(dn["use_bn_in_head"]),
        norm_last_layer=bool(dn["norm_last_layer"]),
        drop_path_rate=float(t["drop_path_rate"]), img_size=img_size).to(device)
    model.train()

    # AdamW with no weight decay on biases and 1-D (norm) parameters, per DINO.
    # Unlike Step 1, the weight decay is FIXED (not cosine-scheduled), so the
    # decay group is not tagged for per-step updates.
    fixed_wd = float(t["weight_decay"])
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": fixed_wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=float(t["lr"]), betas=(0.9, 0.95), eps=1e-8)

    loader, dataset = get_dino_dataloader(
        d["data_root"], n_local_crops=int(dn["n_local_crops"]),
        batch_size=int(t["batch_size"]), num_workers=int(d["num_workers"]),
        global_size=img_size, local_size=int(dn["local_size"]),
        global_scale=tuple(float(s) for s in dn["global_crops_scale"]),
        local_scale=tuple(float(s) for s in dn["local_crops_scale"]), seed=seed)

    total_epochs = int(t["epochs"])
    base_lr = float(t["lr"])
    min_lr = float(t["min_lr"])
    mom_start = float(dn["momentum_teacher"])
    clip_grad = float(t["clip_grad"])
    freeze_last = int(t["freeze_last_layer"])
    save_at = {int(e) for e in t["save_at_epochs"]}
    steps = max(1, len(loader))
    total_steps = total_epochs * steps
    warmup_steps = int(t["warmup_epochs"]) * steps

    print("=" * 70)
    print("DINO  pretrain: unified ViT-B/16 (Step 2 protocol, from scratch)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"arch={m['arch']}  wd={fixed_wd}  save_at={sorted(save_at)}")
    print("=" * 70)

    global_step = 0
    final_loss = None
    for epoch in range(total_epochs):
        running, count = 0.0, 0
        for crops, _ in loader:
            lr = lr_schedule_value(global_step, total_steps, base_lr, min_lr,
                                   warmup_steps)
            set_lr(optimizer, lr)
            teacher_mom = cosine_schedule(global_step, total_steps, mom_start, 1.0)
            epoch_progress = global_step / steps
            crops = [c.to(device, non_blocking=True) for c in crops]
            loss = model(crops, epoch=epoch_progress)
            if not math.isfinite(loss.item()):
                raise FloatingPointError(
                    f"DINO loss became non-finite: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            if clip_grad > 0:
                clip_gradients(model, clip_grad)
            if epoch < freeze_last:
                model.cancel_last_layer_gradients()
            optimizer.step()
            model.update_teacher(teacher_mom)
            bsz = crops[0].size(0)
            running += loss.item() * bsz
            count += bsz
            global_step += 1
        final_loss = running / count if count else None
        print(f"  [{epoch}] dino_loss={final_loss}")

        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(state, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nDINO Step-2 ViT pretraining complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
