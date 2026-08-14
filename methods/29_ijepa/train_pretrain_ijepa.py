"""I-JEPA step 1 (Assran et al., 2023; arXiv:2301.08243), the ViT path.

A self-contained re-implementation, ported from the lab's own I-JEPA code. A
context encoder sees a large context block of patches; a narrow predictor predicts
the representations of several masked target blocks; the targets come from an EMA
**target encoder** (a momentum copy of the context encoder), and the loss is a
smooth-L1 in latent space. AdamW under cosine LR / weight-decay schedules with
warmup; the EMA momentum follows a cosine schedule to 1.0.

The lab wrapper trains under DistributedDataParallel with bf16 autocast and grad
accumulation and logs to TensorBoard; none is needed for a single-process run, so
the loop here is single-process fp32, the device is resolved rather than assumed
CUDA, and AMP / TensorBoard / tqdm / no_sync / accumulation are dropped.
`encoder.pt` is the target ViT encoder; the context encoder, predictor and mask
token are excluded.
"""

from __future__ import annotations

import argparse
import copy
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

from models import build_ijepa_encoder, build_ijepa_predictor   # noqa: E402
from masks import MultiBlockMaskCollator                        # noqa: E402
from data import get_imagenet_loader                            # noqa: E402
from utils import cosine_scheduler, ema_scheduler               # noqa: E402


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


@torch.no_grad()
def ema_update(target_model: nn.Module, source_model: nn.Module,
               momentum: float) -> None:
    """EMA update: target <- m*target + (1-m)*source."""
    for pt, ps in zip(target_model.parameters(), source_model.parameters()):
        pt.data.mul_(momentum).add_((1.0 - momentum) * ps.data)


def apply_masks(tokens: torch.Tensor, mask_ids: torch.Tensor) -> torch.Tensor:
    """Select patches at mask_ids: tokens [B,N,D], mask_ids [B,n] -> [B,n,D]."""
    idx = mask_ids.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    return torch.gather(tokens, dim=1, index=idx)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="I-JEPA step 1 (ViT)")
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--data_path", default=None,
                        help="Override the ImageNet root (must contain train/)")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Added by the port; the lab wrapper assumed CUDA")
    return parser


def _param_groups(encoder, predictor):
    decay, no_decay = [], []
    for model in (encoder, predictor):
        skip = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or name in skip:
                no_decay.append(p)
            else:
                decay.append(p)
    return [{"params": decay, "use_wd": True},
            {"params": no_decay, "weight_decay": 0.0, "use_wd": False}]


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
    p = cfg["predictor"]
    d = cfg["data"]
    mk = cfg["masking"]
    t = cfg["training"]
    em = cfg["ema"]
    name = str(m["name"])
    img_size = int(m["img_size"])
    patch_size = int(m["patch_size"])

    encoder = build_ijepa_encoder(name, img_size=img_size,
                                  patch_size=patch_size).to(device)
    target_encoder = copy.deepcopy(encoder).to(device)
    for param in target_encoder.parameters():
        param.requires_grad = False
    target_encoder.eval()
    predictor = build_ijepa_predictor(name, img_size=img_size,
                                      patch_size=patch_size,
                                      pred_dim=int(p["pred_dim"]),
                                      pred_depth=int(p["pred_depth"])).to(device)
    encoder.train()
    predictor.train()

    optimizer = torch.optim.AdamW(
        _param_groups(encoder, predictor), lr=float(t["lr"]),
        betas=(float(t["beta1"]), float(t["beta2"])),
        weight_decay=float(t["weight_decay"]))

    collator = MultiBlockMaskCollator(
        img_size=img_size, patch_size=patch_size,
        enc_mask_scale=tuple(float(s) for s in mk["enc_mask_scale"]),
        enc_mask_aspect=tuple(float(s) for s in mk["enc_mask_aspect"]),
        pred_mask_scale=tuple(float(s) for s in mk["pred_mask_scale"]),
        pred_mask_aspect=tuple(float(s) for s in mk["pred_mask_aspect"]),
        num_enc_masks=int(mk["num_enc_masks"]),
        num_pred_masks=int(mk["num_pred_masks"]),
        allow_overlap=bool(mk["allow_overlap"]), min_keep=int(mk["min_keep"]))

    loader, dataset = get_imagenet_loader(
        d["data_root"], split="train", augmentation=str(d["augmentation"]),
        img_size=img_size, batch_size=int(t["batch_size"]),
        num_workers=int(d["num_workers"]), collate_fn=collator,
        use_horizontal_flip=bool(d["use_horizontal_flip"]), seed=seed)

    total_epochs = int(t["epochs"])
    # Milestone checkpoints for the additive unified Step-2 recipe. The native
    # step-1 config never sets save_at_epochs, so this is empty and only
    # checkpoint_latest.pth is written -- the native behaviour is unchanged.
    save_at = {int(n) for n in t.get("save_at_epochs", [])}
    ipe = max(1, len(loader))
    ipe_scale = float(t["ipe_scale"])
    total_steps = max(1, int(total_epochs * ipe * ipe_scale))
    warmup_steps = int(t["warmup_epochs"]) * ipe
    clip_grad = float(t["clip_grad"])

    lr_schedule = cosine_scheduler(float(t["lr"]), float(t["final_lr"]),
                                   total_steps, warmup_steps,
                                   start_warmup_value=float(t["start_lr"]))
    wd_schedule = cosine_scheduler(float(t["weight_decay"]),
                                   float(t["final_wd"]), total_steps)
    ema_sched = ema_scheduler(float(em["start_ema"]), float(em["final_ema"]),
                              total_steps)

    print("=" * 72)
    print("I-JEPA  Step 1: ViT + EMA target + latent prediction  (arXiv:2301.08243)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"arch={name}  embed_dim={encoder.embed_dim}  "
          f"pred_masks={mk['num_pred_masks']}")
    print("=" * 72)

    def _sched(arr, step):
        return float(arr[min(step, len(arr) - 1)])

    global_step = 0
    final_loss = None
    for epoch in range(total_epochs):
        running, count = 0.0, 0
        for images, _labels, enc_masks, pred_masks in loader:
            lr = _sched(lr_schedule, global_step)
            wd = _sched(wd_schedule, global_step)
            mom = _sched(ema_sched, global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
                if pg.get("use_wd", True):
                    pg["weight_decay"] = wd

            images = images.to(device, non_blocking=True)
            enc_masks = enc_masks.to(device, non_blocking=True)
            pred_masks = [pm.to(device, non_blocking=True) for pm in pred_masks]

            with torch.no_grad():
                h_full = target_encoder(images)
                h_full = F.layer_norm(h_full, (h_full.shape[-1],))
                h_targets = [apply_masks(h_full, pm) for pm in pred_masks]

            z_ctx = encoder(images, enc_masks)
            total_loss = torch.tensor(0.0, device=device)
            for pm, ht in zip(pred_masks, h_targets):
                z_pred = predictor(z_ctx, enc_masks, pm)
                total_loss = total_loss + F.smooth_l1_loss(z_pred, ht,
                                                           reduction="mean")
            total_loss = total_loss / len(pred_masks)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()),
                    clip_grad)
            optimizer.step()
            ema_update(target_encoder, encoder, mom)

            running += total_loss.item() * images.size(0)
            count += images.size(0)
            global_step += 1
        final_loss = running / count if count else None
        print(f"  [{epoch}] ijepa_loss={final_loss}")
        model_state = {}
        for prefix, mod in (("target_encoder.", target_encoder),
                            ("encoder.", encoder), ("predictor.", predictor)):
            for k, v in mod.state_dict().items():
                model_state[prefix + k] = v
        ckpt = {"epoch": epoch, "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": final_loss, "config": cfg}
        torch.save(ckpt, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(ckpt, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nI-JEPA Step 1 training complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
