"""NEPA step 1 (Xu et al., 2025; arXiv:2512.16922), the ViT path.

A self-contained re-implementation, ported from the lab's own NEPA code. Patch
embeddings z = f(x) run through a causal autoregressive predictor h to give z_hat;
the loss is the negative cosine similarity between z_hat[:, :-1] and a
stop-gradient shifted target z[:, 1:]. An EMA copy of the model is kept for
evaluation. AdamW under a cosine LR schedule with warmup; the peak LR is the base
LR scaled by the global batch size.

The lab wrapper trains under DistributedDataParallel with bf16 autocast and logs
to TensorBoard; none is needed for a single-process run, so the loop here is
single-process fp32, the device is resolved rather than assumed CUDA, and AMP /
TensorBoard / tqdm are dropped. `encoder.pt` is the EMA model; the online model is
excluded.
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

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_nepa_model            # noqa: E402
from data import get_nepa_dataloader           # noqa: E402

MODEL_ARGS = ("embed_dim", "depth", "num_heads", "patch_size", "img_size",
              "mlp_ratio", "use_swiglu", "use_qk_norm", "use_rope",
              "use_cls_token", "patch_embed_norm", "layerscale_init",
              "layer_norm_eps", "rope_theta", "pos_embed_shift",
              "pos_embed_jitter", "pos_embed_rescale")


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


def _param_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for _name, p in model.named_parameters():
        if not p.requires_grad:      # excludes the frozen EMA model
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NEPA step 1 (ViT)")
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--data_path", default=None,
                        help="Override the ImageNet root (must contain train/)")
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
    d = cfg["data"]
    t = cfg["training"]
    model_kwargs = {k: m[k] for k in MODEL_ARGS}
    model_kwargs["ema_decay"] = float(cfg["ema"]["ema_decay"])

    model = build_nepa_model(**model_kwargs)
    model.setup_ema()
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        _param_groups(model, float(t["weight_decay"])), lr=float(t["base_lr"]),
        betas=(float(t["beta1"]), float(t["beta2"])))

    loader, dataset = get_nepa_dataloader(
        d["data_root"], augmentation=str(d["augmentation"]),
        batch_size=int(t["batch_size"]), num_workers=int(d["num_workers"]),
        img_size=int(m["img_size"]), seed=seed)

    total_epochs = int(t["epochs"])
    steps_per_epoch = max(1, len(loader))
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = int(t["warmup_epochs"]) * steps_per_epoch
    peak_lr = float(t["base_lr"]) * int(t["batch_size"]) / 256.0
    clip_grad = float(t["clip_grad"])

    print("=" * 72)
    print("NEPA  Step 1: ViT + causal AR predictor + EMA  (arXiv:2512.16922)")
    print(f"  device={device}  epochs={total_epochs}  images={len(dataset)}  "
          f"embed_dim={m['embed_dim']}  depth={m['depth']}  peak_lr={peak_lr:g}")
    print("=" * 72)

    global_step = 0
    final_loss = None
    for epoch in range(total_epochs):
        running, count = 0.0, 0
        for images, _labels in loader:
            lr = cosine_lr_with_warmup(optimizer, global_step, total_steps,
                                       peak_lr, warmup_steps)
            images = images.to(device, non_blocking=True)
            loss = model.nepa_loss(images)
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"NEPA loss became non-finite: {loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), clip_grad)
            optimizer.step()
            model.update_ema()
            running += loss.item() * images.size(0)
            count += images.size(0)
            global_step += 1
        final_loss = running / count if count else None
        print(f"  [{epoch}] nepa_loss={final_loss}  lr={lr:.6g}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": final_loss, "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nNEPA Step 1 training complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
