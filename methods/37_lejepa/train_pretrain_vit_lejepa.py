"""Step-2 unified ViT-B/16 LeJEPA pretraining, in one process.

The capture's Step 2 is the *same* LeJEPA objective, projector, augmentation and
timm ViT-B/16 backbone as Step 1 -- only the schedule changes (epochs 100->300,
lr 5e-4->6e-4, min_lr 5e-7->1e-6) and milestone checkpoints are written at
`save_at_epochs` (100/200/300). Selected by `recipe: unified` (absent = the native
paper recipe, byte-for-byte unchanged). This reuses the native `build_lejepa`,
`SIGReg`, `get_lejepa_dataloader` and the Step-1 helpers (`resolve_device`,
`make_deterministic`, `build_param_groups`, `set_cosine_lr`) -- so it is the native
loop plus milestone saving; timm is already a dependency (no lock change).
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

from models import build_lejepa, SIGReg                            # noqa: E402
from data import get_lejepa_dataloader                            # noqa: E402
from train_pretrain_lejepa import (_model_kwargs, build_param_groups,  # noqa: E402
                                   make_deterministic, resolve_device,
                                   set_cosine_lr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LeJEPA Step-2 unified ViT-B/16")
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
    save_at = {int(e) for e in t["save_at_epochs"]}
    steps_per_epoch = max(1, len(loader))
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = int(t["warmup_epochs"]) * steps_per_epoch

    print("=" * 72)
    print("LeJEPA  pretrain: unified ViT-B/16 (Step 2 protocol, from scratch)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"views={views}  backbone={m['name']}  save_at={sorted(save_at)}")
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
                raise FloatingPointError(
                    f"LeJEPA loss became non-finite: {loss.item()}")
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
        print(f"  [{epoch}] lejepa_loss={final_loss}  lr={lr:.3g}")

        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(state, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nLeJEPA Step-2 ViT pretraining complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
