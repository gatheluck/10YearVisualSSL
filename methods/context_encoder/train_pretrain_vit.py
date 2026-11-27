"""Step-2 unified ViT-B/16 context-encoder pretraining, in one process.

A faithful port of the capture's ViT Step-2 path: the centre-hole inpainting task
on a ViT-B/16 encoder + transformer decoder, always adversarial. Each step the
encoder sees the image with the centred hole zeroed and the decoder predicts the
hole patches; a centre-hole discriminator scores real vs predicted hole pixels.
The generator loss is `reconstruction_weight * MSE(pred_hole, target_hole) +
adversarial_weight * BCE(D(pred_hole), 1)` with `reconstruction_weight =
1 - adversarial_weight` (the capture's 0.999 / 0.001). Both the generator and the
discriminator use AdamW (betas 0.9/0.999), linear warmup then cosine decay to
`min_lr`; mixed precision on CUDA; the generator's gradient is clipped. Reuses the
port's `create_dataloader`, `resolve_device` and `make_deterministic`.
Checkpoints at each `save_at_epochs` milestone (100/200/300). DDP/torchrun and
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

from datasets import create_dataloader                              # noqa: E402
from models.vit_context_encoder import (build_vit_context_encoder,  # noqa: E402
                                        Discriminator)
from train_pretrain import make_deterministic, resolve_device       # noqa: E402

_ENCODER_INT = ("image_size", "patch_size", "in_channels", "embed_dim", "depth",
                "num_heads", "decoder_dim", "decoder_depth", "decoder_heads",
                "hole_size")


def model_kwargs(m: dict) -> dict:
    """Args for the ViT context-encoder model, from a flat model dict."""
    out = {k: int(m[k]) for k in _ENCODER_INT}
    out["mlp_ratio"] = float(m["mlp_ratio"])
    return out


def lr_at(epoch: int, base_lr: float, min_lr: float, warmup: int,
          total: int) -> float:
    if warmup > 0 and epoch < warmup:
        return base_lr * float(epoch + 1) / float(warmup)
    span = max(1, total - warmup)
    progress = min(1.0, max(0.0, (epoch - warmup) / span))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Context Encoder Step-2 ViT-B/16")
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
        cfg["data"]["train_path"] = args.data_path

    device = resolve_device(getattr(args, "device", "auto"))
    seed = int(cfg.get("seed", 42))
    make_deterministic(seed)

    save_dir = cfg["checkpoint"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    m = cfg["model"]
    d = cfg["data"]
    t = cfg["training"]

    model = build_vit_context_encoder(**model_kwargs(m)).to(device)
    discriminator = Discriminator(channels=int(m["in_channels"]),
                                  img_size=int(m["hole_size"])).to(device)

    loader = create_dataloader(
        "inpainting", d["train_path"], split="train",
        batch_size=int(t["batch_size"]), num_workers=int(d["num_workers"]),
        img_size=int(m["image_size"]), mask_size=int(m["hole_size"]))

    base_lr = float(t["lr"])
    min_lr = float(t["min_lr"])
    warmup = int(t["warmup_epochs"])
    clip_grad = float(t["clip_grad"])
    wd = float(t["weight_decay"])
    adv_w = float(t["adversarial_weight"])
    recon_w = 1.0 - adv_w
    g_opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd,
                              betas=(0.9, 0.999))
    d_opt = torch.optim.AdamW(discriminator.parameters(), lr=base_lr,
                              weight_decay=wd, betas=(0.9, 0.999))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    total_epochs = int(t["epochs"])
    save_at = {int(e) for e in t["save_at_epochs"]}

    print("=" * 70)
    print("Context Encoder  pretrain: unified ViT-B/16 (Step 2, adversarial)")
    print(f"  device={device}  epochs={total_epochs}  hole={m['hole_size']}  "
          f"recon_w={recon_w:.3f} adv_w={adv_w:.3f}  save_at={sorted(save_at)}")
    print("=" * 70)

    recon_v = adv_v = total_v = None
    for epoch in range(total_epochs):
        lr = lr_at(epoch, base_lr, min_lr, warmup, total_epochs)
        for opt in (g_opt, d_opt):
            for group in opt.param_groups:
                group["lr"] = lr
        model.train()
        discriminator.train()
        r_sum = a_sum = tot_sum = 0.0
        count = 0
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            target_hole = model.extract_target_hole(images)

            # Discriminator: real hole vs detached predicted hole.
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred, _, _ = model(images)
                pred_hole_det = model.extract_predicted_hole(pred).detach()
            with torch.amp.autocast("cuda", enabled=use_amp):
                d_real = bce(discriminator(target_hole),
                             torch.ones(images.size(0), 1, device=device))
                d_fake = bce(discriminator(pred_hole_det),
                             torch.zeros(images.size(0), 1, device=device))
                d_loss = 0.5 * (d_real + d_fake)
            d_opt.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.step(d_opt)

            # Generator: reconstruction + adversarial.
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred, _, _ = model(images)
                pred_hole = model.extract_predicted_hole(pred)
                recon = mse(pred_hole, target_hole)
                adv = bce(discriminator(pred_hole),
                          torch.ones(images.size(0), 1, device=device))
                g_loss = recon_w * recon + adv_w * adv
            g_opt.zero_grad()
            scaler.scale(g_loss).backward()
            if clip_grad > 0:
                scaler.unscale_(g_opt)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(g_opt)
            scaler.update()

            n = images.size(0)
            r_sum += recon.item() * n
            a_sum += adv.item() * n
            tot_sum += g_loss.item() * n
            count += n
        recon_v = r_sum / count if count else None
        adv_v = a_sum / count if count else None
        total_v = tot_sum / count if count else None
        print(f"  [{epoch}] lr={lr:.2e} recon={recon_v} adv={adv_v} "
              f"total={total_v}")

        if (epoch + 1) in save_at:
            state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                     "discriminator_state_dict": discriminator.state_dict(),
                     "optimizer_state_dict": g_opt.state_dict(),
                     "loss": total_v, "config": cfg}
            torch.save(state, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nContext Encoder Step-2 ViT pretraining complete!")
    ran = total_epochs > 0 and total_v is not None
    return {"epochs": total_epochs,
            "final_loss": total_v if ran else None,
            "final_recon_loss": recon_v if ran else None,
            "final_adv_loss": adv_v if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
