"""BEiT step 1 (Bao et al., 2021; arXiv:2106.08254), masked image modeling.

A self-contained re-implementation, ported from the lab's own BEiT code. A dVAE
tokenizer turns each image into discrete visual tokens; a random block of patches
is replaced by a shared mask token in the ViT input; the ViT predicts the visual
tokens at the masked positions (cross-entropy over the dVAE vocabulary). AdamW
under a cosine LR schedule with warmup; gradient-norm clipping.

The tokenizer is the frozen DALL-E dVAE for a real run (a hash-pinned download,
imported lazily); the hermetic smoke uses a random tokenizer, so nothing is
downloaded. The lab wrapper trains under DistributedDataParallel and logs to
TensorBoard; none is needed for a single-process run, so the loop here is
single-process fp32, the device is resolved rather than assumed CUDA, and
TensorBoard / tqdm are dropped. `encoder.pt` is the BEiT backbone trunk; the shared
mask token and the MIM head are excluded.
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

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_beit, build_tokenizer      # noqa: E402
from data import get_beit_dataloader                # noqa: E402

MODEL_ARGS = ("img_size", "patch_size", "vocab_size", "embed_dim", "depth",
              "num_heads", "mlp_ratio", "drop_path_rate", "init_values")


def _model_kwargs(model_cfg: dict) -> dict:
    return {"img_size": int(model_cfg["img_size"]),
            "patch_size": int(model_cfg["patch_size"]),
            "vocab_size": int(model_cfg["vocab_size"]),
            "embed_dim": int(model_cfg["embed_dim"]),
            "depth": int(model_cfg["depth"]),
            "num_heads": int(model_cfg["num_heads"]),
            "mlp_ratio": float(model_cfg["mlp_ratio"]),
            "drop_path_rate": float(model_cfg["drop_path_rate"]),
            "init_values": float(model_cfg["init_values"])}


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


def cosine_lr_with_warmup(optimizer, step: int, total_steps: int, peak_lr: float,
                          warmup_steps: int) -> float:
    """Linear warmup from 0 to peak_lr, then cosine decay to 0."""
    if warmup_steps > 0 and step < warmup_steps:
        lr = peak_lr * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = 0.5 * peak_lr * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BEiT step 1 (MIM)")
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
    tk = cfg["tokenizer"]
    d = cfg["data"]
    mk = cfg["masking"]
    t = cfg["training"]

    model = build_beit(**_model_kwargs(m)).to(device)
    model.train()

    tokenizer = build_tokenizer(
        vocab_size=int(m["vocab_size"]), ckpt=str(tk.get("ckpt") or ""),
        stride=int(tk["token_size"]) // (int(m["img_size"]) // int(m["patch_size"])),
        device=device, input_is_mapped=bool(tk["input_is_mapped"]))

    loader, dataset = get_beit_dataloader(
        d["data_root"], batch_size=int(t["batch_size"]),
        num_workers=int(d["num_workers"]), img_size=int(m["img_size"]),
        patch_size=int(m["patch_size"]), token_size=int(tk["token_size"]),
        num_masking_patches=int(mk["num_masking_patches"]),
        min_masking_patches=int(mk["min_num_patches"]), seed=seed)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(t["lr"]),
        betas=(float(t["beta1"]), float(t["beta2"])), eps=float(t["eps"]),
        weight_decay=float(t["weight_decay"]))
    criterion = nn.CrossEntropyLoss()

    total_epochs = int(t["epochs"])
    steps_per_epoch = max(1, len(loader))
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = int(t["warmup_epochs"]) * steps_per_epoch
    peak_lr = float(t["lr"])
    clip_grad = float(t["clip_grad"])

    print("=" * 72)
    print("BEiT  Step 1: ViT + dVAE tokens + masked image modeling  (arXiv:2106.08254)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"embed_dim={m['embed_dim']}  vocab={m['vocab_size']}  "
          f"tokenizer={'dall-e' if tk.get('ckpt') else 'random (smoke)'}")
    print("=" * 72)

    global_step = 0
    final_loss = None
    for epoch in range(total_epochs):
        running, count = 0.0, 0
        for patch_imgs, token_imgs, masks, _labels in loader:
            lr = cosine_lr_with_warmup(optimizer, global_step, total_steps,
                                       peak_lr, warmup_steps)
            patch_imgs = patch_imgs.to(device, non_blocking=True)
            token_imgs = token_imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.no_grad():
                visual_tokens = tokenizer(token_imgs)
            labels = visual_tokens[masks]
            logits = model(patch_imgs, masks)
            loss = criterion(logits, labels)
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"BEiT loss became non-finite: {loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            running += loss.item() * patch_imgs.size(0)
            count += patch_imgs.size(0)
            global_step += 1
        final_loss = running / count if count else None
        print(f"  [{epoch}] beit_mim_loss={final_loss}  lr={lr:.6g}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": final_loss, "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nBEiT Step 1 training complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
