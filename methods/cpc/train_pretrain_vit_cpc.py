"""CPC unified Step-2 pretraining: ViT-B/16 + column-GRU context, single process.

A port of the capture's `methods/11_cpc/train_step2_vit.py`. A timm ViT-B/16's
patch tokens become the CPC z-grid; a column-wise GRU gives the context; k linear
predictors score an InfoNCE loss. The capture ran it under DistributedDataParallel
with TensorBoard; this port owns a thin single-process fp32 loop, resolves the
device instead of assuming CUDA, and drops DDP / TensorBoard. The recipe -- AdamW
(the lr used directly; the capture baked lr = 1.5e-4 x 1024/256 = 6e-4 into the
config), a per-epoch linear-warmup->cosine schedule, weight decay 0.05 -- is kept
faithfully.

`encoder.pt` is the ViT trunk (`encoder.*`, so it loads into a plain
VisionTransformer); the column GRU and the InfoNCE predictors are training
machinery. Milestone `checkpoint_epoch_{N}.pth` is written at each
`training.save_at_epochs`.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_cpc_vit                                    # noqa: E402
from train_pretrain_cpc2018 import make_deterministic, resolve_device  # noqa: E402

MODEL_KEYS = ("z_dim", "c_dim", "pred_steps", "img_size", "patch_size",
              "embed_dim", "depth", "num_heads", "mlp_ratio", "drop_rate",
              "attn_drop_rate")
_FLOATS = ("mlp_ratio", "drop_rate", "attn_drop_rate")


def model_kwargs(train: dict) -> dict:
    """The build_cpc_vit kwargs, read from a flat train dict. load_encoder rebuilds
    with the same set, so it lives here once."""
    return {k: (float(train[k]) if k in _FLOATS else int(train[k]))
            for k in MODEL_KEYS}


def _train_loader(data_root: str, img_size: int, batch_size: int,
                  num_workers: int, seed: int):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(), norm,
    ])
    dataset = datasets.ImageFolder(str(Path(data_root) / "train"), transform=tf)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=True, generator=torch.Generator().manual_seed(seed))
    return dataset, loader


def warmup_cosine_lr(epoch: int, peak_lr: float, min_lr: float,
                     warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return peak_lr * (epoch + 1) / max(warmup_epochs, 1)
    p = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * p))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPC Step 2: ViT-B/16 + GRU context, single process")
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

    data_root = getattr(args, "data_path", None) or cfg["data"]["data_root"]
    device = resolve_device(getattr(args, "device", "auto"))
    seed = int(cfg.get("seed", 42))
    make_deterministic(seed)

    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    m = cfg["model"]
    t = cfg["training"]
    total_epochs = int(t["epochs"])
    peak_lr = float(t["lr"])
    min_lr = float(t["min_lr"])
    warmup_epochs = int(t["warmup_epochs"])
    temperature = float(cfg["cpc"]["temperature"])
    save_at = {int(n) for n in t.get("save_at_epochs", [])}

    model = build_cpc_vit(**model_kwargs(m)).to(device)
    model.train()

    dataset, loader = _train_loader(
        data_root, int(m["img_size"]), int(t["batch_size"]),
        int(t["num_workers"]), seed)

    optimizer = optim.AdamW(model.parameters(), lr=peak_lr,
                            weight_decay=float(t["weight_decay"]))

    print("=" * 72)
    print("CPC  Step 2: ViT-B/16 + column-GRU context  (InfoNCE)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"grid={model.grid_size}  z_dim={m['z_dim']}  peak_lr={peak_lr:.2e}  "
          f"save_at_epochs={sorted(save_at)}")
    print("=" * 72)

    final_loss = None
    for epoch in range(total_epochs):
        lr = warmup_cosine_lr(epoch, peak_lr, min_lr, warmup_epochs,
                              total_epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        running, count = 0.0, 0
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            z_grid, c_grid = model(imgs)
            loss = model.cpc_loss_fast(z_grid, c_grid, temperature=temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
            count += imgs.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] infonce_loss={final_loss}  lr={lr:.3g}")

        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": final_loss, "config": cfg}
        torch.save(ckpt, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(ckpt, os.path.join(save_dir,
                                          f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nCPC Step 2 training complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
