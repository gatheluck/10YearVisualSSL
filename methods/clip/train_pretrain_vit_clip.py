"""CLIP Step-2 label-text adaptation: ViT-B/16 from scratch, single process.

A port of the capture's `methods/38_clip/train_step2_vit.py`. The official CLIP
ViT-B/16 image tower and its 12-layer text tower are built from scratch (through
the pinned `openai/CLIP` constructor, `third_party/CLIP`, imported not copied) and
trained on ImageNet-1k: each labeled image is paired with a deterministic,
epoch-varying OpenAI ImageNet class-name prompt, and the symmetric image-text
contrastive loss is minimised.

**This is a supervised label-text adaptation, not unlabeled VSSL** -- every
checkpoint records `supervised_label_text_adaptation=true` /
`main_vssl_comparability=false`, exactly as the capture demands (see README and
provenance.json). It may be shown as a CLIP adaptation reference; it must not be
reported as a comparable self-supervised ImageNet result.

The capture ran it under DistributedDataParallel with an eight-GPU differentiable
all-gather, BF16 autocast and an atomic reservation-launcher resume; this port owns
a thin single-process fp32 loop, resolves the device instead of assuming CUDA, and
drops DDP / the all-gather / the launcher machinery. The recipe -- the per-step
warmup->cosine LR, AdamW, the deterministic per-sample prompt choice, gradient-norm
clipping and the logit-scale clamp -- is kept faithfully.

`encoder.pt` is the trained image tower (`visual.*`, extracted by the adapter);
the text tower and the logit scale are training machinery. Milestone
`checkpoint_epoch_{N}.pth` is written at each `training.save_at_epochs`.
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
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import (                                          # noqa: E402
    CLIP_MEAN,
    CLIP_STD,
    STEP2_PROTOCOL,
    build_clip,
    load_official_imagenet_metadata,
    tokenize_prompts,
)


def resolve_device(spec: str) -> "torch.device":
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device is 'cuda' but no CUDA device is visible. Ask for 'auto' "
                "to accept a CPU; getting a CPU silently would misreport what ran")
        return torch.device("cuda")
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


class IndexedImageFolder(datasets.ImageFolder):
    """ImageFolder that also returns the sample index (for the prompt choice)."""

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        return image, target, index


def build_transform(image_size: int):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size, scale=(0.08, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def lr_at_step(step: int, total_steps: int, warmup_steps: int,
               base_lr: float, min_lr: float) -> float:
    """Linear warmup to ``base_lr`` then cosine decay to ``min_lr`` (faithful)."""
    if step < warmup_steps:
        return base_lr * float(step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def choose_prompt_tokens(prompt_tokens: torch.Tensor, labels: torch.Tensor,
                         indices: torch.Tensor, epoch: int) -> torch.Tensor:
    """Pick one template per sample, deterministic in (sample index, epoch).

    Faithful to the capture: the same (index, epoch) always selects the same
    template, and a different epoch selects a different one."""
    template_count = prompt_tokens.shape[1]
    template_ids = (indices.to(torch.long) * 1103515245
                    + (epoch + 1) * 12345) % template_count
    return prompt_tokens[labels.to(torch.long), template_ids]


def clip_contrastive_loss(image_features: torch.Tensor,
                          text_features: torch.Tensor,
                          logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric image-text InfoNCE over the (single-process) batch."""
    logits = logit_scale * image_features @ text_features.t()
    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets)
                  + F.cross_entropy(logits.t(), targets))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLIP Step 2: ViT-B/16 label-text adaptation, single process")
    parser.add_argument("--config", default="configs/pretrain_vit.yaml")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser


def _prompt_table(cfg: dict, dataset) -> torch.Tensor:
    """The [C, T, context_length] prompt-token table.

    Real run: the official 1000 ImageNet class names and 80 templates (requires a
    1000-class dataset). Hermetic smoke: the dataset's own folder names with the
    config's templates, so a tiny 2-class run needs no 1000-class ImageNet."""
    prompts = cfg["prompts"]
    if prompts["use_official_imagenet"]:
        if len(dataset.classes) != 1000:
            raise RuntimeError(
                "prompts.use_official_imagenet is true but the dataset has "
                f"{len(dataset.classes)} classes, not the 1000 ImageNet needs")
        class_names, templates = load_official_imagenet_metadata()
    else:
        class_names = list(dataset.classes)
        templates = list(prompts["templates"])
    return tokenize_prompts(class_names, templates)


def run(args, config: "dict | None" = None) -> dict:
    if config is not None:
        cfg = config
    else:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    train_path = getattr(args, "data_path", None) or cfg["data"]["train_path"]
    device = resolve_device(getattr(args, "device", "auto"))
    seed = int(cfg.get("seed", 0))
    make_deterministic(seed)

    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    t = cfg["training"]
    total_epochs = int(t["epochs"])
    batch_size = int(t["batch_size"])
    save_at = {int(n) for n in t.get("save_at_epochs", [])}

    dataset = IndexedImageFolder(
        train_path, transform=build_transform(int(cfg["data"]["image_size"])))
    prompt_tokens = _prompt_table(cfg, dataset)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]), drop_last=True,
        generator=generator)

    model = build_clip(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(t["lr"]),
        betas=(float(t["beta1"]), float(t["beta2"])), eps=float(t["eps"]),
        weight_decay=float(t["weight_decay"]))

    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * int(t["warmup_epochs"])
    clip_grad = float(t["clip_grad_norm"])
    global_step = 0

    print("=" * 72)
    print("CLIP Step 2: official ViT-B/16 architecture, ImageNet label-text "
          "adaptation")
    print("WARNING: supervised_label_text_adaptation=true; not unlabeled VSSL")
    print(f"  device={device}  epochs={total_epochs}  batch={batch_size}  "
          f"lr={float(t['lr']):.2e}  vision_width={cfg['model']['vision_width']}  "
          f"save_at_epochs={sorted(save_at)}")
    print("=" * 72)

    final_loss = None
    for epoch in range(total_epochs):
        model.train()
        running, count = 0.0, 0
        for images, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            text = choose_prompt_tokens(prompt_tokens, labels, indices, epoch).to(
                device, non_blocking=True)
            lr = lr_at_step(global_step, total_steps, warmup_steps,
                            float(t["lr"]), float(t["min_lr"]))
            for group in optimizer.param_groups:
                group["lr"] = lr

            image_features = F.normalize(model.encode_image(images).float(), dim=-1)
            text_features = F.normalize(model.encode_text(text).float(), dim=-1)
            logit_scale = model.logit_scale.exp().float()
            loss = clip_contrastive_loss(image_features, text_features, logit_scale)
            if not math.isfinite(loss.item()):
                raise FloatingPointError(
                    f"CLIP loss became non-finite at epoch={epoch} step={global_step}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            with torch.no_grad():
                model.logit_scale.clamp_(0.0, math.log(100.0))

            running += loss.item() * images.shape[0]
            count += images.shape[0]
            global_step += 1
        final_loss = running / count if count else None
        print(f"  [{epoch}] clip_loss={final_loss}  lr={lr:.3g}  "
              f"scale={float(model.logit_scale.exp()):.3f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": final_loss,
            "config": cfg,
            "protocol": STEP2_PROTOCOL,
            "arch": "ViT-B/16",
            "feature_dim": int(cfg["model"]["embed_dim"]),
            "supervised_label_text_adaptation": True,
            "main_vssl_comparability": False,
        }
        torch.save(ckpt, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(ckpt, os.path.join(save_dir,
                                          f"checkpoint_epoch_{epoch + 1}.pth"))

    print("\nCLIP Step 2 training complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
