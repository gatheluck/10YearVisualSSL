"""V-JEPA step 2 (Bardes et al., 2024; arXiv:2404.08471), ported as this port's
step 1 -- the from-scratch image adaptation of V-JEPA on ImageNet.

A context encoder and an EMA target encoder (the official facebookresearch/jepa
video ViT + predictor, run at num_frames=1 so the backbone is an image ViT) are
trained by latent prediction: the target encodes the full image, the context
encoder sees only the visible (encoder-mask) tokens, and a narrow predictor
predicts the target's representations at the masked (predictor-mask) positions,

    loss = mean_over_masks( mean|z_pred - layernorm(target)|^loss_exp / loss_exp )
           + reg_coeff * mean(relu(1 - std(z_pred)))

3D multi-block masks (2D at num_frames=1); AdamW under a warmup+cosine LR schedule;
the target encoder is an EMA of the context encoder. `encoder.pt` is the EMA target
encoder (the representation V-JEPA eval uses).

The lab wrapper imports the official repo and trains under DistributedDataParallel
+ TensorBoard; this single-process port imports the same model (init_video_model),
mask collator and apply_masks, drops DDP / TensorBoard / tqdm, resolves the device
rather than assuming CUDA, and owns a thin single-process loop. The `src`/`app`
imports are lazy (inside run/build) so the in-process test suite stays collision
-free with the other submodule ports that also expose a `src` package.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_JEPA_SUBMODULE = ROOT.parent.parent / "third_party" / "jepa"

from models import build_vjepa               # noqa: E402
from data import get_vjepa_dataloader        # noqa: E402

MODEL_ARGS = ("model_name", "crop_size", "patch_size", "num_frames",
              "tubelet_size", "pred_depth", "pred_embed_dim", "uniform_power",
              "use_mask_tokens", "zero_init_mask_tokens", "use_sdpa")


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


def cosine(start: float, end: float, step: int, total: int) -> float:
    if total <= 0:
        return end
    p = min(max(step / total, 0.0), 1.0)
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * p))


def lr_at(step, total_steps, warmup_steps, start_lr, peak_lr, final_lr):
    if warmup_steps > 0 and step < warmup_steps:
        return start_lr + (peak_lr - start_lr) * (step + 1) / warmup_steps
    return cosine(peak_lr, final_lr, step - warmup_steps,
                  max(1, total_steps - warmup_steps))


def _param_groups(*modules):
    decay, no_decay = [], []
    for mod in modules:
        for name, p in mod.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return decay, no_decay


@torch.no_grad()
def update_ema(target: nn.Module, source: nn.Module, momentum: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(momentum).add_(sp.detach().data, alpha=1.0 - momentum)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V-JEPA step 2 (image adaptation)")
    parser.add_argument("--config", default="configs/step1.yaml")
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

    from models.vjepa_model import _prepare_jepa_path
    _prepare_jepa_path()                        # src -> third_party/jepa
    from src.masks.utils import apply_masks    # noqa: E402  (upstream, lazy)

    m, d, t = cfg["model"], cfg["data"], cfg["training"]
    cfgs_mask = cfg["mask"]

    encoder, predictor = build_vjepa(m, num_mask_tokens=len(cfgs_mask),
                                     device=device)
    encoder.train()
    predictor.train()
    target_encoder = copy.deepcopy(encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad_(False)
    target_encoder.eval()

    loader, dataset = get_vjepa_dataloader(
        d["data_root"], batch_size=int(t["batch_size"]), cfgs_mask=cfgs_mask,
        crop_size=int(m["crop_size"]), num_frames=int(m["num_frames"]),
        patch_size=int(m["patch_size"]), tubelet_size=int(m["tubelet_size"]),
        use_color_jitter=bool(d["use_color_jitter"]),
        num_workers=int(d["num_workers"]), seed=seed)

    decay, no_decay = _param_groups(encoder, predictor)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(t["weight_decay"])},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=float(t["lr"]), betas=(float(t["beta1"]), float(t["beta2"])),
        eps=float(t["eps"]))

    loss_exp = float(cfg["loss"]["loss_exp"])
    reg_coeff = float(cfg["loss"]["reg_coeff"])
    clip_grad = float(t["clip_grad"])
    total_epochs = int(t["epochs"])
    ipe = max(1, len(loader))
    total_steps = total_epochs * ipe
    warmup_steps = int(t["warmup_epochs"]) * ipe
    ema_start, ema_final = float(t["ema_start"]), float(t["ema_final"])

    print("=" * 72)
    print("V-JEPA  Step 2 (image, num_frames=1): ViT + predictor + EMA target  (arXiv:2404.08471)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"model={m['model_name']}  masks={len(cfgs_mask)}  loss_exp={loss_exp}")
    print("=" * 72)

    global_step = 0
    final_loss = None
    for epoch in range(total_epochs):
        running, count = 0.0, 0
        for images, _labels, masks_enc, masks_pred in loader:
            lr = lr_at(global_step, total_steps, warmup_steps,
                       float(t["start_lr"]), float(t["lr"]), float(t["final_lr"]))
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            images = images.to(device, non_blocking=True)
            masks_enc = [mk.to(device, non_blocking=True) for mk in masks_enc]
            masks_pred = [mk.to(device, non_blocking=True) for mk in masks_pred]

            with torch.no_grad():
                h = target_encoder(images)
                h = F.layer_norm(h, (h.size(-1),))
                h_targets = apply_masks(h, masks_pred, concat=False)
            z_context = encoder(images, masks_enc)
            z_pred = predictor(z_context, h_targets, masks_enc, masks_pred)

            loss_jepa = images.new_zeros(())
            for zi, hi in zip(z_pred, h_targets):
                loss_jepa = loss_jepa + torch.mean(torch.abs(zi - hi) ** loss_exp) / loss_exp
            loss_jepa = loss_jepa / max(len(z_pred), 1)
            reg_loss = images.new_zeros(())
            if reg_coeff > 0:
                pstd = sum(torch.sqrt(zi.var(dim=1) + 1e-4) for zi in z_pred) / max(len(z_pred), 1)
                reg_loss = torch.mean(F.relu(1.0 - pstd))
            loss = loss_jepa + reg_coeff * reg_loss
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"V-JEPA loss became non-finite: {loss.item()}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()),
                    clip_grad)
            optimizer.step()
            momentum = cosine(ema_start, ema_final, global_step, total_steps)
            update_ema(target_encoder, encoder, momentum)

            running += loss.item() * images.size(0)
            count += images.size(0)
            global_step += 1
        final_loss = running / count if count else None
        print(f"  [{epoch}] vjepa_loss={final_loss}  jepa={loss_jepa.item():.4f}"
              f"  lr={lr:.3g}  ema={momentum:.4f}")
        torch.save({"epoch": epoch,
                    "target_encoder_state_dict": target_encoder.state_dict(),
                    "encoder_state_dict": encoder.state_dict(),
                    "loss": final_loss, "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nV-JEPA Step 2 training complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
