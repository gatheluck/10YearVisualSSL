"""Step-2 unified ViT-B/16 BEiT pretraining, in one process.

The capture's Step 2 is the *same* BEiT masked-image-modeling objective, ViT-B/16
architecture, blockwise masking and DALL-E dVAE tokenizer as Step 1 -- only the
schedule changes to the unified recipe (epochs 800->300, batch 2048->1024, lr
1.5e-3->6e-4; AdamW betas 0.9/0.999 and fixed weight decay 0.05 are retained,
as the capture's BEiT Step 2 does) and milestone checkpoints are written at
`save_at_epochs` (100/200/300). Selected by `recipe: unified` (absent = the native
paper recipe, byte-for-byte unchanged). Reuses `build_beit`, `build_tokenizer`,
`get_beit_dataloader` and the Step-1 helpers -- so it is the native loop plus
milestone saving; the ViT is the port's own (no timm), so no lock change.
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

from models import build_beit, build_tokenizer                     # noqa: E402
from data import get_beit_dataloader                              # noqa: E402
from train_pretrain_beit import (_model_kwargs, cosine_lr_with_warmup,  # noqa: E402
                                 make_deterministic, resolve_device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BEiT Step-2 unified ViT-B/16")
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
    save_at = {int(e) for e in t["save_at_epochs"]}

    print("=" * 72)
    print("BEiT  pretrain: unified ViT-B/16 (Step 2 protocol, from scratch)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"embed_dim={m['embed_dim']}  save_at={sorted(save_at)}  "
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
                raise FloatingPointError(
                    f"BEiT loss became non-finite: {loss.item()}")
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

        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(state, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nBEiT Step-2 ViT pretraining complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
