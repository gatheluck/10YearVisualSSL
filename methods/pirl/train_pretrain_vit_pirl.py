"""Step-2 unified ViT-B/16 PIRL pretraining, in one process.

A faithful port of the capture's `train_step2_vit.py`: a ViT-B/16 (timm, from
scratch) encodes an image and its jigsaw view (the nine shuffled patches
reassembled into one image), both contrasted against a momentum-updated memory
bank; the loss is the weighted sum of the image-NCE and the jigsaw-NCE, reusing
the port's `build_pirl_loader`, `PIRLMemoryBankNCE` and `initialize_memory_bank`.
Optimiser AdamW with betas (0.9, 0.95); linear warmup then cosine decay to
`min_lr`; gradient clipping (the capture's ViT PIRL loop uses it). Checkpoints at
each `save_at_epochs` milestone (100/200/300) plus `checkpoint_latest.pth`. The
memory bank is re-seeded from the model each run (initialize_from_model), as the
native path does, so it is not carried in the checkpoint. DDP/torchrun and
TensorBoard are dropped; the device is resolved, not sniffed.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import build_pirl_loader                                # noqa: E402
from loss import PIRLMemoryBankNCE                                # noqa: E402
from models.vit_pirl import build_vit_pirl                        # noqa: E402
from train_pretrain_pirl import (initialize_memory_bank,          # noqa: E402
                                 make_deterministic, resolve_device)

# The ViT encoder is shaped by these; feature_dim/num_patches shape only the
# projector (not the loaded encoder.*), so they default when absent -- the
# linear_eval config, which rebuilds only to read the backbone, omits them.
_ENCODER_INT = ("image_size", "patch_size", "embed_dim", "depth", "num_heads")
_FLOAT = ("mlp_ratio", "drop_rate", "attn_drop_rate")
_PROJ_INT = ("feature_dim", "num_patches")


def model_kwargs(m: dict) -> dict:
    """Build args for the ViT PIRL model, from a flat train dict (which carries
    both the model knobs and image_size)."""
    out = {k: int(m[k]) for k in _ENCODER_INT}
    out.update({k: float(m[k]) for k in _FLOAT})
    out.update({k: int(m[k]) for k in _PROJ_INT if k in m})
    return out


def lr_at(epoch: int, base_lr: float, min_lr: float, warmup: int,
          total: int) -> float:
    if warmup > 0 and epoch < warmup:
        return base_lr * float(epoch + 1) / float(warmup)
    span = max(1, total - warmup)
    progress = min(1.0, max(0.0, (epoch - warmup) / span))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PIRL Step-2 ViT-B/16")
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
    seed = int(cfg.get("seed", 42))
    make_deterministic(seed)

    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    m = cfg["model"]
    d = cfg["data"]
    n = cfg["nce"]
    t = cfg["training"]
    jigsaw_weight = float(cfg["loss"]["jigsaw_weight"])

    model = build_vit_pirl(
        **model_kwargs({**m, "image_size": d["image_size"]})).to(device)
    model.train()

    loader, dataset = build_pirl_loader(
        d["data_root"], d, train=True, batch_size=int(t["batch_size"]),
        num_workers=int(d["num_workers"]), seed=seed)

    memory_bank = PIRLMemoryBankNCE(
        num_samples=len(dataset), feature_dim=int(m["feature_dim"]),
        temperature=float(n["temperature"]), momentum=float(n["momentum"]),
        num_negatives=int(n["num_negatives"])).to(device)
    if bool(cfg["memory"]["initialize_from_model"]):
        initialize_memory_bank(model, memory_bank, loader, device)

    base_lr = float(t["lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=float(t["weight_decay"]),
                                  betas=(0.9, 0.95))
    total_epochs = int(t["epochs"])
    warmup = int(t["warmup_epochs"])
    min_lr = float(t["min_lr"])
    clip_grad = float(t["clip_grad"])
    save_at = {int(e) for e in t["save_at_epochs"]}

    print("=" * 70)
    print("PIRL  pretrain: unified ViT-B/16 (Step 2 protocol, from scratch)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"feature_dim={m['feature_dim']}  jigsaw_weight={jigsaw_weight}  "
          f"save_at={sorted(save_at)}")
    print("=" * 70)

    final_loss = None
    for epoch in range(total_epochs):
        lr = lr_at(epoch, base_lr, min_lr, warmup, total_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        running, count = 0.0, 0
        for images, patches, indices, _labels in loader:
            images = images.to(device, non_blocking=True)
            patches = patches.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)
            image_features, jigsaw_features = model(images, patches)
            loss_image = memory_bank(image_features, indices)
            loss_jigsaw = memory_bank(jigsaw_features, indices)
            loss = (1.0 - jigsaw_weight) * loss_image + jigsaw_weight * loss_jigsaw
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            memory_bank.update_memory(image_features.detach(), indices)
            running += loss.item() * images.size(0)
            count += images.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] lr={lr:.2e} pirl_loss={final_loss}")

        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(state, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nPIRL Step-2 ViT pretraining complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
