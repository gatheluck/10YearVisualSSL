"""DINO step 1 (Caron et al., 2021; arXiv:2104.14294), the ViT-S/16 path.

A self-contained re-implementation, ported from the lab's own DINO code. A student
(ViT + DINO head) sees all crops; a teacher (an EMA copy of the student) sees the
two global crops; a centred+sharpened cross-entropy distils teacher into student.
AdamW under a per-iteration cosine LR schedule with warmup and a cosine
weight-decay schedule; the teacher EMA momentum follows a cosine schedule to 1.0;
the student head's last layer is frozen for the first freeze_last_layer epochs.

The lab wrapper trains under DistributedDataParallel with AMP autocast and logs to
TensorBoard; none is needed for a single-process run, so the loop here is
single-process fp32, the device is resolved rather than assumed CUDA, and AMP /
TensorBoard / tqdm are dropped. `encoder.pt` is the teacher ViT backbone; the DINO
head, the centre and the whole student are excluded.
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

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_dino               # noqa: E402
from data import get_dino_dataloader        # noqa: E402


def resolve_device(spec: str, local_rank: int = 0) -> "torch.device":
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device is 'cuda' but no CUDA device is visible. Ask for "
                "'auto' to accept a CPU; getting a CPU silently would misreport "
                "what ran")
        return torch.device(f"cuda:{local_rank}")
    if spec == "auto":
        return torch.device(f"cuda:{local_rank}"
                            if torch.cuda.is_available() else "cpu")
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


def cosine_schedule(step: int, total_steps: int, base_value: float,
                    final_value: float) -> float:
    """Cosine annealing from base_value to final_value over total_steps."""
    if total_steps <= 0:
        return final_value
    ratio = min(max(step, 0), total_steps) / total_steps
    return final_value + 0.5 * (base_value - final_value) * (
        1.0 + math.cos(math.pi * ratio))


def lr_schedule_value(step: int, total_steps: int, base_lr: float,
                      min_lr: float, warmup_steps: int) -> float:
    """DINO per-iteration LR schedule: linear warmup from 0, then cosine decay."""
    if warmup_steps > 0 and step < warmup_steps:
        if warmup_steps == 1:
            return base_lr
        return base_lr * step / (warmup_steps - 1)
    return cosine_schedule(step - warmup_steps,
                           max(1, total_steps - warmup_steps), base_lr, min_lr)


def set_lr(optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def set_weight_decay(optimizer, weight_decay: float) -> None:
    for pg in optimizer.param_groups:
        if pg.get("weight_decay_schedule", False):
            pg["weight_decay"] = weight_decay


def clip_gradients(model, clip: float) -> None:
    """Official DINO per-parameter gradient clipping."""
    for _, p in model.named_parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        clip_coef = clip / (param_norm + 1e-6)
        if clip_coef < 1:
            p.grad.data.mul_(clip_coef)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DINO step 1 (ViT-S/16)")
    parser.add_argument("--config", default="configs/step1.yaml")
    parser.add_argument("--data_path", default=None,
                        help="Override the ImageFolder root of training images")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Added by the port; the lab wrapper assumed CUDA")
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
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(t["weight_decay_start"]),
          "weight_decay_schedule": True},
         {"params": no_decay, "weight_decay": 0.0,
          "weight_decay_schedule": False}],
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
    wd_start = float(t["weight_decay_start"])
    wd_end = float(t["weight_decay_end"])
    mom_start = float(dn["momentum_teacher"])
    clip_grad = float(t["clip_grad"])
    freeze_last = int(t["freeze_last_layer"])
    steps = max(1, len(loader))
    total_steps = total_epochs * steps
    warmup_steps = int(t["warmup_epochs"]) * steps

    print("=" * 70)
    print("DINO  Step 1: ViT + EMA teacher + self-distillation  (arXiv:2104.14294)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"arch={m['arch']}  out_dim={dn['out_dim']}  n_crops=2+{dn['n_local_crops']}")
    print("=" * 70)

    global_step = 0
    final_loss = None
    for epoch in range(total_epochs):
        running, count = 0.0, 0
        for crops, _ in loader:
            lr = lr_schedule_value(global_step, total_steps, base_lr, min_lr,
                                   warmup_steps)
            set_lr(optimizer, lr)
            set_weight_decay(optimizer,
                             cosine_schedule(global_step, total_steps,
                                             wd_start, wd_end))
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
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": final_loss, "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nDINO Step 1 training complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
