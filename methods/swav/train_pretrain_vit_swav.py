"""Step-2 unified ViT-B/16 SwAV pretraining, in one process.

A faithful port of the capture's `train_step2_vit.py`: a ViT-B/16 (timm, from
scratch, dynamic image size) + projection head + prototypes is trained with the
multi-crop swapped-assignment SwAV objective. Because `models/vit_swav.py` mirrors
the native `ResNetSwAV` interface (list-of-crops `forward` returning
`(embeddings, scores)`, `normalize_prototypes`), this reuses the port's multi-crop
`train_epoch`, `swav_loss`, per-step `build_lr_schedule` and `get_swav_dataloader`
unchanged; the only differences are the ViT backbone and **AdamW** (vs the native
LARC-SGD), matching the unified Step-2 recipe. Checkpoints at each `save_at_epochs`
milestone (100/200/300) plus `checkpoint_latest.pth`. DDP/torchrun is dropped and
TensorBoard is left off (writer=None); the device is resolved, not sniffed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import get_swav_dataloader                              # noqa: E402
from models.vit_swav import build_vit_swav                        # noqa: E402
from train_pretrain_resnet import (build_lr_schedule, make_deterministic,  # noqa: E402
                                   resolve_device, set_lr, train_epoch)

# The ViT encoder is shaped by these; out_dim/hidden_mlp/nmb_prototypes shape
# only the projection head and prototypes (not the loaded encoder.*), so they
# default when absent -- the linear_eval config, which rebuilds only to read the
# backbone, omits them.
_ENCODER_INT = ("image_size", "patch_size", "embed_dim", "depth", "num_heads")
_FLOAT = ("mlp_ratio", "drop_rate", "attn_drop_rate")
_HEAD_INT = ("out_dim", "hidden_mlp", "nmb_prototypes")


def model_kwargs(m: dict) -> dict:
    """Build args for the ViT SwAV model, from a flat train/model dict."""
    out = {k: int(m[k]) for k in _ENCODER_INT}
    out.update({k: float(m[k]) for k in _FLOAT})
    out.update({k: int(m[k]) for k in _HEAD_INT if k in m})
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SwAV Step-2 ViT-B/16")
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
        cfg["data"]["train_path"] = str(Path(args.data_path) / "train")

    device = resolve_device(getattr(args, "device", "auto"))
    seed = int(cfg.get("seed", 42))
    make_deterministic(seed)

    save_dir = cfg["checkpoint"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    m = cfg["model"]
    d = cfg["data"]
    t = cfg["training"]

    model = build_vit_swav(**model_kwargs(m)).to(device)
    model.train()

    loader, _sampler = get_swav_dataloader(
        data_path=d["train_path"], size_crops=d["size_crops"],
        nmb_crops=d["nmb_crops"], min_scale_crops=d["min_scale_crops"],
        max_scale_crops=d["max_scale_crops"], batch_size=int(t["batch_size"]),
        num_workers=int(d["num_workers"]),
        color_jitter_strength=float(d["color_jitter_strength"]),
        distributed=False)

    base_lr = float(t["lr"])
    min_lr = float(t["min_lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=float(t["weight_decay"]),
                                  betas=(0.9, 0.999))
    total_epochs = int(t["epochs"])
    warmup = int(t["warmup_epochs"])
    save_at = {int(e) for e in t["save_at_epochs"]}
    lr_schedule = build_lr_schedule(
        base_lr=base_lr, final_lr=min_lr, start_warmup=0.0, epochs=total_epochs,
        steps_per_epoch=max(1, len(loader)), warmup_epochs=warmup)

    print("=" * 70)
    print("SwAV  pretrain: unified ViT-B/16 (Step 2 protocol, from scratch)")
    print(f"  device={device}  epochs={total_epochs}  "
          f"crops={d['nmb_crops']}@{d['size_crops']}  K={m['nmb_prototypes']}  "
          f"save_at={sorted(save_at)}")
    print("=" * 70)

    final_loss = None
    global_step = 0
    for epoch in range(total_epochs):
        set_lr(optimizer, lr_schedule[min(epoch * max(1, len(loader)),
                                          len(lr_schedule) - 1)])
        final_loss, global_step = train_epoch(
            model, loader, optimizer, epoch, cfg, None, False, device, 1,
            global_step, lr_schedule)
        print(f"  [{epoch}] swav_loss={final_loss}")

        state = {"epoch": epoch, "global_step": global_step,
                 "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(state, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nSwAV Step-2 ViT pretraining complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
