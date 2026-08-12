"""LeJEPA step 1 (Balestriero & LeCun, 2025; arXiv:2511.08544).

A self-contained re-implementation, ported from the lab's own LeJEPA code. Each
image is seen as several augmented views; a ViT backbone + projection MLP maps
each view to a projected feature. The loss is a convex combination of SIGReg (an
Epps-Pulley Gaussian regularizer over random 1-D slices of the batch, see
models/sigreg.py) and a cross-view invariance loss:

    loss = SIGReg(proj) * lambda + invariance(proj) * (1 - lambda)

AdamW under a cosine LR schedule with linear warmup, weight decay split off the
norms and biases. `encoder.pt` is the backbone; the projection MLP is training
machinery and is excluded.

The lab wrapper trains under DistributedDataParallel with a bfloat16 autocast, logs
to TensorBoard, and trains an online linear probe on *detached* features for
monitoring. None affects the backbone (the probe reads detached features, so it
never back-propagates into it), so this single-process port drops all of it: the
loop is single-process fp32, the device is resolved rather than assumed CUDA, and
AMP / TensorBoard / tqdm / the online probe are dropped. The elaborate collapse
guards are reduced to a finiteness check.
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

from models import build_lejepa, SIGReg          # noqa: E402
from data import get_lejepa_dataloader           # noqa: E402

MODEL_ARGS = ("name", "img_size", "drop_path_rate", "proj_hidden_dim",
              "proj_dim", "proj_layers", "final_bn")
AUG_ARGS = ("crop_scale", "color_jitter", "color_jitter_p", "grayscale_p",
            "blur_p", "blur_kernel", "solarize_p", "hflip_p")


def _model_kwargs(model_cfg: dict) -> dict:
    return {"model_name": str(model_cfg["name"]),
            "img_size": int(model_cfg["img_size"]),
            "drop_path_rate": float(model_cfg["drop_path_rate"]),
            "proj_hidden_dim": int(model_cfg["proj_hidden_dim"]),
            "proj_dim": int(model_cfg["proj_dim"]),
            "proj_layers": int(model_cfg["proj_layers"]),
            "final_bn": bool(model_cfg["final_bn"])}


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


def build_param_groups(model, model_lr: float, model_min_lr: float,
                       weight_decay: float) -> list:
    """Weight decay on the matrices, none on norms and biases (the lab split)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (param.ndim <= 1 or name.endswith(".bias")
                or "norm" in name.lower() or "bn" in name.lower()):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay,
                       "base_lr": model_lr, "min_lr": model_min_lr,
                       "lr": model_lr, "name": "model_decay"})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0,
                       "base_lr": model_lr, "min_lr": model_min_lr,
                       "lr": model_lr, "name": "model_no_decay"})
    return groups


def set_cosine_lr(optimizer, step: int, total_steps: int,
                  warmup_steps: int) -> float:
    """Linear warmup then cosine decay to each group's min_lr. Returns the first
    group's LR (they share the schedule scale)."""
    if step < warmup_steps:
        scale = float(step + 1) / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        scale = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    lr0 = None
    for group in optimizer.param_groups:
        base_lr = float(group.get("base_lr", group["lr"]))
        min_lr = float(group.get("min_lr", 0.0))
        group["lr"] = min_lr + (base_lr - min_lr) * scale
        if lr0 is None:
            lr0 = group["lr"]
    return lr0 if lr0 is not None else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LeJEPA step 1")
    parser.add_argument("--config", default="configs/pretrain.yaml")
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
    d = cfg["data"]
    aug = cfg["augmentation"]
    lj = cfg["lejepa"]
    t = cfg["training"]

    model = build_lejepa(**_model_kwargs(m)).to(device)
    model.train()

    sg = lj["sigreg"]
    sigreg = SIGReg(t_max=float(sg["t_max"]), knots=int(sg["knots"]),
                    num_slices=int(sg["num_slices"]),
                    seed=int(sg["seed"])).to(device)

    views = int(aug["views"])
    loader, dataset = get_lejepa_dataloader(
        d["data_root"], batch_size=int(t["batch_size"]), views=views,
        num_workers=int(d["num_workers"]), img_size=int(m["img_size"]),
        seed=seed, crop_scale=tuple(aug["crop_scale"]),
        color_jitter=tuple(aug["color_jitter"]),
        color_jitter_p=float(aug["color_jitter_p"]),
        grayscale_p=float(aug["grayscale_p"]), blur_p=float(aug["blur_p"]),
        blur_kernel=int(aug["blur_kernel"]), solarize_p=float(aug["solarize_p"]),
        hflip_p=float(aug["hflip_p"]))

    groups = build_param_groups(model, model_lr=float(t["lr"]),
                                model_min_lr=float(t["min_lr"]),
                                weight_decay=float(t["weight_decay"]))
    optimizer = torch.optim.AdamW(
        groups, lr=float(t["lr"]),
        betas=(float(t["beta1"]), float(t["beta2"])), eps=float(t["eps"]))

    lamb = float(lj["lambda"])
    clip_grad = float(t["clip_grad"])
    total_epochs = int(t["epochs"])
    steps_per_epoch = max(1, len(loader))
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = int(t["warmup_epochs"]) * steps_per_epoch

    print("=" * 72)
    print("LeJEPA  Step 1: ViT + projector + SIGReg + invariance  (arXiv:2511.08544)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"views={views}  backbone={m['name']}  lambda={lamb}")
    print("=" * 72)

    global_step = 0
    final_loss = None
    for epoch in range(total_epochs):
        running, count = 0.0, 0
        for batch_views, _target in loader:
            lr = set_cosine_lr(optimizer, global_step, total_steps, warmup_steps)
            batch_views = batch_views.to(device, non_blocking=True)
            features, proj = model(batch_views)
            inv_loss = (proj.mean(dim=0, keepdim=True) - proj).square().mean()
            sigreg_loss = sigreg(proj)
            loss = sigreg_loss * lamb + inv_loss * (1.0 - lamb)
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"LeJEPA loss became non-finite: {loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            bn = batch_views.size(0)
            running += loss.item() * bn
            count += bn
            global_step += 1
        final_loss = running / count if count else None
        print(f"  [{epoch}] lejepa_loss={final_loss}  sigreg={sigreg_loss.item():.4f}"
              f"  inv={inv_loss.item():.4f}  lr={lr:.3g}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": final_loss, "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nLeJEPA Step 1 training complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
