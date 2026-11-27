"""MAE step 1: masked-autoencoder pretraining (He et al., 2021; arXiv:2111.06377).

A self-contained re-implementation, ported from the lab's own MAE code. An image
is patchified, 75% of tokens are masked, the encoder sees the visible tokens, a
lightweight decoder reconstructs the masked pixels, and the loss is MSE on the
masked patches.

What the port owns, and what it drops. The lab wrapper trains under
`DistributedDataParallel` and logs to TensorBoard; neither is needed for a
single-process run, so the loop here is single-process and full-precision, the
device is **resolved** rather than assumed CUDA, and TensorBoard is dropped.
`encoder.pt` is the encoder side (patch embed, CLS token, encoder blocks and
norm); the decoder is reconstruction machinery and is excluded.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

import adapterlib

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_mae                         # noqa: E402

# The architecture arguments build_mae takes (everything but `arch`, which
# selects the variant). load_encoder rebuilds with the same set, so they live
# here once.
ARCH_KEYS = ("img_size", "patch_size", "enc_embed_dim", "enc_depth",
             "enc_num_heads", "dec_embed_dim", "dec_depth", "dec_num_heads",
             "mlp_ratio", "mask_ratio", "norm_pix_loss")


def model_kwargs(model: dict) -> dict:
    out = {}
    for k in ARCH_KEYS:
        v = model[k]
        if k in ("mlp_ratio", "mask_ratio"):
            out[k] = float(v)
        elif k == "norm_pix_loss":
            out[k] = bool(v)
        else:
            out[k] = int(v)
    return out


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


def _build_loader(data_root: str, img_size: int, batch_size: int,
                  num_workers: int, seed: int):
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    transform = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.2, 1.0),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dataset = ImageFolder(adapterlib.dataset_split_dir(data_root, "train"),
                          transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=True, generator=torch.Generator().manual_seed(seed))
    return dataset, loader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MAE step 1 (masked autoencoding)")
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--data_path", default=None,
                        help="Override the ImageFolder root of training images")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Added by the port; the lab wrapper assumed CUDA")
    return parser


def run(args, config: dict | None = None) -> dict:
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

    mk = model_kwargs(cfg["model"])
    model = build_mae(cfg["model"]["arch"], **mk).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        betas=(0.9, 0.95))

    dataset, loader = _build_loader(
        cfg["data"]["data_root"], mk["img_size"],
        int(cfg["training"]["batch_size"]),
        int(cfg["training"]["num_workers"]), seed)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_epoch = state["epoch"] + 1
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        print(f"Resumed from epoch {state['epoch']}")

    total_epochs = int(cfg["training"]["epochs"])
    print("=" * 70)
    print("MAE  Step 1: masked-autoencoder pretraining  (arXiv:2111.06377)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"arch={cfg['model']['arch']}  mask_ratio={mk['mask_ratio']}")
    print("=" * 70)

    final_loss = None
    for epoch in range(start_epoch, total_epochs):
        running, count = 0.0, 0
        for imgs, _labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            loss, _pred, _mask = model(imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
            count += imgs.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] mse_loss={final_loss}")
        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nMAE Step 1 training complete!")
    ran = total_epochs > start_epoch and final_loss is not None
    return {"epochs": total_epochs - start_epoch,
            "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
