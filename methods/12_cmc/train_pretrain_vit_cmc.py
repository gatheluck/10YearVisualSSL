"""Step-2 unified ViT-B/16 CMC pretraining, in one process.

A faithful port of the capture's `train_step2_vit.py`: two ViT-B/16 branches
(timm, from scratch, `in_chans` 1 and 2) map the L and ab views to L2-normalised
embeddings; the cross-view NCE objective over two momentum memory banks pulls an
image's two views together and apart from negatives, reusing the port's
`CMCDataset` (RGB->Lab, index-carrying), `NCEAverage` and `NCECriterion`.
Optimiser AdamW with betas (0.9, 0.95); linear warmup then cosine decay to
`min_lr`. Checkpoints at each `save_at_epochs` milestone (100/200/300) plus
`checkpoint_latest.pth`; the two memory banks ride in the checkpoint under
`contrast_state_dict` and a resume restores them, as the native path does.
DDP/torchrun and TensorBoard are dropped; the device is resolved, not sniffed.
Matching the capture's ViT CMC loop, there is no AMP and no gradient clipping.
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

from data import CMCDataset                                       # noqa: E402
from nce import NCEAverage, NCECriterion                         # noqa: E402
from models.vit_cmc import build_vit_cmc                         # noqa: E402
from train_pretrain_cmc import make_deterministic, resolve_device  # noqa: E402

_INT = ("feat_dim", "hidden_dim", "image_size", "patch_size", "embed_dim",
        "depth", "num_heads")
_FLOAT = ("mlp_ratio", "drop_rate", "attn_drop_rate")


def model_kwargs(m: dict) -> dict:
    """Build args for the two-branch ViT CMC model, from a flat train/model
    dict. `img_size` maps to the builder's `image_size`."""
    out = {}
    for k in _INT:
        src = "img_size" if k == "image_size" else k
        out[k] = int(m[src])
    for k in _FLOAT:
        out[k] = float(m[k])
    return out


def lr_at(epoch: int, base_lr: float, min_lr: float, warmup: int,
          total: int) -> float:
    if warmup > 0 and epoch < warmup:
        return base_lr * float(epoch + 1) / float(warmup)
    span = max(1, total - warmup)
    progress = min(1.0, max(0.0, (epoch - warmup) / span))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CMC Step-2 ViT-B/16")
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
    n = cfg["nce"]
    d = cfg["data"]
    t = cfg["training"]

    model = build_vit_cmc(**model_kwargs(m)).to(device)
    model.train()

    dataset = CMCDataset(d["data_root"], mode="train",
                         image_size=int(d["img_size"]),
                         crop_low=float(d["crop_low"]))
    n_data = len(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(t["batch_size"]), shuffle=True,
        num_workers=int(t["num_workers"]), drop_last=True,
        generator=torch.Generator().manual_seed(seed))

    contrast = NCEAverage(
        feat_dim=int(m["feat_dim"]), n_data=n_data,
        K=int(n["num_negatives"]), T=float(n["temperature"]),
        momentum=float(n["nce_momentum"])).to(device)
    contrast.multinomial.to(device)
    criterion_l = NCECriterion(n_data).to(device)
    criterion_ab = NCECriterion(n_data).to(device)

    base_lr = float(t["lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=float(t["weight_decay"]),
                                  betas=(0.9, 0.95))
    total_epochs = int(t["epochs"])
    warmup = int(t["warmup_epochs"])
    min_lr = float(t["min_lr"])
    save_at = {int(e) for e in t["save_at_epochs"]}

    start_epoch = 0
    if getattr(args, "resume", None) and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_epoch = state["epoch"] + 1
        model.load_state_dict(state["model_state_dict"])
        contrast.load_state_dict(state["contrast_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        print(f"Resumed from epoch {state['epoch']}")

    print("=" * 70)
    print("Visual CMC  pretrain: unified two-branch ViT-B/16 (Step 2, scratch)")
    print(f"  device={device}  epochs={total_epochs}  images={n_data}  "
          f"feat_dim={m['feat_dim']}  K={n['num_negatives']}  "
          f"save_at={sorted(save_at)}")
    print("=" * 70)

    final_loss = None
    for epoch in range(start_epoch, total_epochs):
        lr = lr_at(epoch, base_lr, min_lr, warmup, total_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        running, count = 0.0, 0
        for imgs, _, idx in loader:
            imgs = imgs.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)
            feat_l, feat_ab = model(imgs)
            out_l, out_ab = contrast(feat_l, feat_ab, idx)
            loss = criterion_l(out_l) + criterion_ab(out_ab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
            count += imgs.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] lr={lr:.2e} nce_loss={final_loss}")

        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "contrast_state_dict": contrast.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(state, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nVisual CMC Step-2 ViT pretraining complete!")
    ran = total_epochs > start_epoch and final_loss is not None
    return {"epochs": total_epochs - start_epoch,
            "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
