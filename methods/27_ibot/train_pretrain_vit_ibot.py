"""
iBOT Step 2 training: the unified ViT-B/16 recipe, single process.

This is the additive Step-2 path (`recipe: unified` in the config). It plugs the
**same** iBOT objective -- student/teacher ViT with a shared DINOHead, block-masked
patch MIM, EMA teacher -- into the unified ViT-Base/16 backbone, and follows the
capture's `train_step2_vit.py`. It is faithful to that recipe while dropping the
multi-GPU / RNG-checkpoint machinery a single-process port does not need, exactly
as `train_pretrain.py` did for step 1.

What differs from the native step-1 trainer, and why (all read from the capture's
step-2 config and trainer):

  - **the learning rate is used directly**, not rescaled by batch/256. The capture
    baked ``lr = 1.5e-4 * 1024/256 = 6e-4`` into the config, so the trainer must
    not scale it a second time.
  - **weight decay is fixed** at ``training.weight_decay`` (the capture's cosine
    ran 0.05 -> 0.05), rather than the native 0.04 -> 0.4 schedule.
  - **masking is set by ``ibot.mask_ratio_min``/``mask_ratio_max``**, the loader's
    step-2 path, rather than the native ``pred_ratio``/``pred_ratio_var``.
  - **grad_clip 0.3 and freeze_last_layer 3** are the official ViT-B stability
    settings (native step 1 used 3.0 and 1).
  - **milestone checkpoints**: ``checkpoint_epoch_{N}.pth`` is written at each
    ``training.save_at_epochs`` so the adapter can hand over a frozen encoder per
    milestone (the 100/200/300 sweep).

The pure helpers (device resolution, seeding, the cosine schedule, the LR setter,
gradient clipping, the meter) are imported from `train_pretrain.py` -- one
implementation per method, not a second copy.
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path

import torch
import torch.optim as optim

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import vit_base, DINOHead, iBOT, iBOTLoss
from data import get_ibot_dataloader
# One implementation per method: the schedule, the setters, the clip, the meter,
# the device resolution and the seeding are the native trainer's, imported here.
from train_pretrain import (
    AverageMeter, cosine_scheduler_value, set_lr, clip_gradients,
    resolve_device, make_deterministic,
)

# The unified recipe's backbone. `vit_small` is step 1; step 2 is ViT-Base/16.
_VIT_BUILDERS = {"vit_base": vit_base}

# iBOT beta is fixed at (0.9, 0.999); the capture's step 2 does not override it.
BETAS = (0.9, 0.999)


def train_one_epoch(model, criterion, loader, optimizer, epoch, cfg, device):
    """One epoch of the unified iBOT step-2 loop.

    The lr is the config value used directly (no batch/256 rescale); the weight
    decay is fixed at construction, so only the lr and the teacher momentum move
    per step.
    """
    model.train()
    criterion.train()

    total_epochs = cfg["training"]["epochs"]
    base_lr = cfg["training"]["lr"]
    min_lr = cfg["training"].get("min_lr", 1e-6)
    warmup_epochs = cfg["training"]["warmup_epochs"]
    freeze_last_layer = cfg["training"].get("freeze_last_layer", 3)
    grad_clip = cfg["training"]["grad_clip"]
    mom_base = cfg["ibot"]["teacher_momentum_start"]
    mom_final = cfg["ibot"]["teacher_momentum_end"]
    print_freq = cfg["training"]["print_freq"]
    n_global = cfg["data"]["n_global_crops"]

    n_steps = len(loader)
    total_steps = total_epochs * n_steps
    warmup_steps = warmup_epochs * n_steps

    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    patch_loss_meter = AverageMeter()
    t0 = time.time()

    for i, (crops, masks, _labels) in enumerate(loader):
        global_step = epoch * n_steps + i
        lr = cosine_scheduler_value(global_step, total_steps, base_lr,
                                    final=min_lr, warmup_steps=warmup_steps)
        teacher_mom = cosine_scheduler_value(global_step, total_steps, mom_base,
                                             final=mom_final)
        set_lr(optimizer, lr)

        crops = [c.to(device, non_blocking=True) for c in crops]
        masks = [m.to(device, non_blocking=True) for m in masks]

        student_cls, student_patch, teacher_cls, teacher_patch = model(
            crops, masks, n_global)

        loss, cls_loss, patch_loss = criterion(
            student_cls, teacher_cls, student_patch, teacher_patch, masks, epoch)

        if not math.isfinite(loss.item()):
            raise RuntimeError(
                f"non-finite iBOT loss at epoch={epoch} step={i}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_gradients([model.student, model.head], grad_clip)

        # freeze_last_layer: zero the head's final linear gradients for the first
        # N epochs, the DINO/iBOT anti-collapse guard.
        if epoch < freeze_last_layer:
            for name, param in model.head.named_parameters():
                if "last_layer" in name and param.grad is not None:
                    param.grad = None

        optimizer.step()
        model.update_teacher(teacher_mom)

        bs = crops[0].size(0)
        loss_meter.update(loss.item(), bs)
        cls_loss_meter.update(cls_loss.item(), bs)
        patch_loss_meter.update(patch_loss.item(), bs)

        if i % print_freq == 0:
            print(f"  [{epoch}][{i:5d}/{n_steps}]  loss={loss_meter.avg:.4f} "
                  f"(cls={cls_loss_meter.avg:.4f} patch={patch_loss_meter.avg:.4f})  "
                  f"lr={lr:.2e}  mom={teacher_mom:.5f}  t={time.time() - t0:.1f}s")

    return loss_meter.avg, cls_loss_meter.avg, patch_loss_meter.avg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="iBOT Step 2: unified ViT-Base/16, single process")
    parser.add_argument("--config", default="configs/pretrain_vit.yaml")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    return parser


def run(args, config: dict | None = None) -> dict:
    """Run the unified step-2 training and return the epoch loss + components."""
    if config is not None:
        cfg = config
    else:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    if getattr(args, "data_path", None):
        cfg["data"]["train_path"] = os.path.join(args.data_path, "train")

    device = resolve_device(getattr(args, "device", "auto"))
    make_deterministic(int(cfg.get("seed", 42)))

    save_dir = cfg["checkpoint"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    embed_dim = cfg["model"]["embed_dim"]     # 768 for ViT-B
    out_dim = cfg["ibot"]["out_dim"]
    arch = cfg["model"]["arch"]
    if arch not in _VIT_BUILDERS:
        raise ValueError(
            f"model.arch is {arch!r}; the unified step-2 recipe supports "
            f"{', '.join(sorted(_VIT_BUILDERS))} only")
    builder = _VIT_BUILDERS[arch]

    student_vit = builder(patch_size=cfg["model"]["patch_size"],
                          drop_path_rate=cfg["model"].get("drop_path_rate", 0.1),
                          use_mask_token=True)
    teacher_vit = builder(patch_size=cfg["model"]["patch_size"],
                          drop_path_rate=0.0, use_mask_token=False)

    head = DINOHead(
        in_dim=embed_dim, out_dim=out_dim,
        hidden_dim=cfg["ibot"]["head_hidden_dim"],
        bottleneck_dim=cfg["ibot"]["head_bottleneck_dim"],
        nlayers=cfg["ibot"]["head_nlayers"],
        norm_last_layer=cfg["ibot"].get("norm_last_layer", True),
    )

    model = iBOT(student_vit, teacher_vit, head).to(device)

    criterion = iBOTLoss(
        out_dim=out_dim, patch_out_dim=out_dim,
        student_temp=cfg["ibot"]["student_temp"],
        teacher_temp=cfg["ibot"]["teacher_temp"],
        teacher_patch_temp=cfg["ibot"].get("teacher_patch_temp",
                                           cfg["ibot"]["teacher_temp"]),
        teacher_temp_warmup=cfg["ibot"]["teacher_temp_warmup"],
        teacher_patch_temp_warmup=cfg["ibot"].get(
            "teacher_patch_temp_warmup", cfg["ibot"]["teacher_temp_warmup"]),
        teacher_temp_warmup_epochs=cfg["ibot"]["teacher_temp_warmup_epochs"],
        center_momentum=cfg["ibot"]["center_momentum"],
        center_momentum_patch=cfg["ibot"].get("center_momentum_patch",
                                              cfg["ibot"]["center_momentum"]),
        lambda_token=cfg["ibot"]["lambda_token"],
        n_global_crops=cfg["data"]["n_global_crops"],
        n_local_crops=cfg["data"]["n_local_crops"],
    ).to(device)

    # AdamW with no weight decay on 1-D / norm / token parameters. The decay is
    # fixed (set once here), not scheduled -- the unified recipe's wd is constant.
    base_wd = cfg["training"]["weight_decay"]
    params_with_wd, params_without_wd = [], []
    for name, p in (list(model.student.named_parameters())
                    + list(model.head.named_parameters())):
        if not p.requires_grad:
            continue
        if (p.ndim == 1 or "bias" in name or "norm" in name
                or "cls_token" in name or "pos_embed" in name):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    optimizer = optim.AdamW(
        [{"params": params_with_wd, "weight_decay": base_wd},
         {"params": params_without_wd, "weight_decay": 0.0}],
        lr=cfg["training"]["lr"],
        betas=BETAS,
    )

    train_loader, _ = get_ibot_dataloader(
        data_path=cfg["data"]["train_path"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        n_local_crops=cfg["data"]["n_local_crops"],
        global_size=cfg["data"]["global_size"],
        local_size=cfg["data"]["local_size"],
        patch_size=cfg["model"]["patch_size"],
        global_crops_scale=cfg["data"]["global_crops_scale"],
        local_crops_scale=cfg["data"]["local_crops_scale"],
        mask_ratio_min=cfg["ibot"]["mask_ratio_min"],
        mask_ratio_max=cfg["ibot"]["mask_ratio_max"],
        pred_shape=cfg["ibot"].get("pred_shape", "block"),
        step="step2",
        distributed=False,
    )

    total_epochs = cfg["training"]["epochs"]
    save_at = {int(e) for e in cfg["training"].get("save_at_epochs", [])}

    print("=" * 75)
    print("iBOT  Step 2: unified ViT-Base/16 on ImageNet-1k")
    print(f"  epochs={total_epochs}  batch={cfg['training']['batch_size']}  "
          f"lr={cfg['training']['lr']:.2e}  wd={base_wd}  "
          f"grad_clip={cfg['training']['grad_clip']}  "
          f"freeze_last_layer={cfg['training'].get('freeze_last_layer', 3)}")
    print(f"  mask_ratio=[{cfg['ibot']['mask_ratio_min']},"
          f"{cfg['ibot']['mask_ratio_max']}]  save_at_epochs={sorted(save_at)}")
    print("=" * 75)

    avg_loss = cls_loss = patch_loss = None
    for epoch in range(total_epochs):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        print(f"\n=== Epoch {epoch}/{total_epochs - 1} ===")
        avg_loss, cls_loss, patch_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, epoch, cfg, device)

        if (epoch + 1) in save_at:
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "criterion": criterion.state_dict(), "config": cfg},
                       ckpt_path)
            print(f"  [ckpt] Saved: {ckpt_path}")

    print("\niBOT Step 2 training complete!")
    return {
        "epochs": total_epochs,
        "final_loss": avg_loss,
        "final_cls_loss": cls_loss,
        "final_patch_loss": patch_loss,
    }


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
