"""DINOv3 step 2 (Simeoni et al., 2025; arXiv:2508.10104), the from-scratch
unified SSL comparison on ImageNet -- ported as this port's step 1.

A self-contained re-implementation, ported from the lab's own DINOv3 code. A
student ViT (with 4 register tokens and axial RoPE) and an EMA teacher see
multi-crop views (2 global + 8 local); the objective is DINOv3's core:

    loss = L_DINO + L_iBOT + koleo_weight * L_KoLeo

L_DINO is a cross-view distillation with Sinkhorn-Knopp (SwAV) centering of the
teacher assignments; L_iBOT is a masked-patch distillation (block masks on the
global crops); L_KoLeo spreads the batch's CLS features. The teacher is an EMA of
the student; `encoder.pt` is the teacher backbone (the DINO/iBOT heads are
training machinery and are excluded).

The default `core` profile preserves the original single-process schedules.
The explicit `step4_gram_components` profile wires the fixed 300-epoch clock,
frozen clean-crop Gram teacher and sparse refreshes; selectable heads implement
four Step-4 sharing designs. See docs/DINOV3_STEP4.md. Both profiles remain
single-process float32 components, not canonical distributed reproduction.

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

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import (DINOHead, IBOTHead, VisionTransformer,  # noqa: E402
                    vit_base_patch16)
from models.dino_head import (TokenTypeAffine, build_head_mlp,  # noqa: E402
                              build_prototypes, project_tokens)
from losses import DINOLoss, IBOTLoss, KoLeoLoss, GramLoss   # noqa: E402
import protocol as step_protocol                           # noqa: E402
from losses.ibot_loss import generate_block_mask            # noqa: E402
from data import MultiCropAugmentation, get_multicrop_dataloader  # noqa: E402

MODEL_ARGS = ("img_size", "patch_size", "embed_dim", "depth", "num_heads",
              "mlp_ratio", "n_register_tokens", "drop_path_rate", "use_rope",
              "rope_base")


def build_vit(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
              mlp_ratio=4.0, n_register_tokens=4, drop_path_rate=0.1,
              use_rope=True, rope_base=100.0) -> VisionTransformer:
    """The DINOv3 backbone from explicit dims (the config-driven path)."""
    return VisionTransformer(
        img_size=int(img_size), patch_size=int(patch_size),
        embed_dim=int(embed_dim), depth=int(depth), num_heads=int(num_heads),
        mlp_ratio=float(mlp_ratio), n_register_tokens=int(n_register_tokens),
        drop_path_rate=float(drop_path_rate), use_ibot_mask=True,
        use_rope=bool(use_rope), rope_base=float(rope_base))


def _vit_kwargs(m: dict) -> dict:
    return {k: m[k] for k in MODEL_ARGS}


class DINOv3Model(nn.Module):
    """Student/teacher container: a ViT backbone + DINO and iBOT projection heads."""

    def __init__(self, model_cfg: dict):
        super().__init__()
        self.head_layout = model_cfg.get('head_layout', 'separate')
        layouts = ('separate', 'shared', 'shared_prototypes', 'shared_mlp', 'token_affine')
        if self.head_layout not in layouts:
            raise ValueError(f'unknown head_layout: {self.head_layout}')
        dino = tuple(int(model_cfg[k]) for k in
                     ('dino_head_hidden_dim', 'dino_head_bottleneck_dim', 'dino_out_dim'))
        ibot = tuple(int(model_cfg[k]) for k in
                     ('ibot_head_hidden_dim', 'ibot_head_bottleneck_dim', 'ibot_out_dim'))
        incompatible = ((self.head_layout in ('shared', 'token_affine') and dino != ibot)
                        or (self.head_layout == 'shared_prototypes' and dino[1:] != ibot[1:])
                        or (self.head_layout == 'shared_mlp' and dino[:2] != ibot[:2]))
        if incompatible:
            raise ValueError(f'incompatible dimensions for {self.head_layout}')
        self.backbone = build_vit(**_vit_kwargs(model_cfg))
        embed_dim = self.backbone.embed_dim
        if self.head_layout == 'shared_prototypes':
            self.prototypes = build_prototypes(dino[1], dino[2])
            self.dino_mlp = build_head_mlp(embed_dim, *dino[:2])
            # The reference recursively reinitializes the supplied prototype
            # after each MLP. Preserve its seeded draws without registering aliases.
            DINOHead._init_weights(self.prototypes)
            self.ibot_mlp = build_head_mlp(embed_dim, *ibot[:2])
            DINOHead._init_weights(self.prototypes)
        elif self.head_layout == 'shared_mlp':
            self.shared_mlp = build_head_mlp(embed_dim, *dino[:2])
            self.dino_prototypes = build_prototypes(dino[1], dino[2])
            self.ibot_prototypes = build_prototypes(ibot[1], ibot[2])
        else:
            self.dino_head = DINOHead(embed_dim, dino[2], dino[0], dino[1])
            if self.head_layout == 'separate':
                self.ibot_head = IBOTHead(embed_dim, ibot[2], ibot[0], ibot[1])
            elif self.head_layout == 'token_affine':
                self.token_affine_cls = TokenTypeAffine(embed_dim)
                self.token_affine_patch = TokenTypeAffine(embed_dim)

    def project_dino(self, tokens):
        if self.head_layout == 'shared_prototypes':
            return project_tokens(self.dino_mlp, self.prototypes, tokens)
        if self.head_layout == 'shared_mlp':
            return project_tokens(self.shared_mlp, self.dino_prototypes, tokens)
        if self.head_layout == 'token_affine':
            tokens = self.token_affine_cls(tokens)
        return self.dino_head(tokens)

    def project_ibot(self, tokens):
        if self.head_layout == 'shared_prototypes':
            return project_tokens(self.ibot_mlp, self.prototypes, tokens)
        if self.head_layout == 'shared_mlp':
            return project_tokens(self.shared_mlp, self.ibot_prototypes, tokens)
        if self.head_layout == 'separate':
            return self.ibot_head(tokens)
        if self.head_layout == 'token_affine':
            tokens = self.token_affine_patch(tokens)
        return self.dino_head(tokens)

    def forward_backbone(self, crops, n_global, masks_global=None):
        B = crops[0].shape[0]
        cls_list, patch_list = [], []
        for i in range(n_global):
            mask_i = masks_global[i * B:(i + 1) * B] if masks_global is not None else None
            cls_i, pat_i = self.backbone(crops[i], mask=mask_i, is_global=True)
            cls_list.append(cls_i)
            patch_list.append(pat_i)
        for i in range(n_global, len(crops)):
            cls_i, _ = self.backbone(crops[i], mask=None, is_global=False)
            cls_list.append(cls_i)
        return torch.cat(cls_list, dim=0), torch.cat(patch_list, dim=0)

    def forward(self, crops, n_global=2, masks_global=None,
                ibot_selection_masks=None):
        # masks_global masks the student's *input* patches (mask token); the
        # teacher passes masks_global=None (sees full patches) but selects the
        # same positions via ibot_selection_masks -- so the teacher's unmasked
        # tokens are the targets for the student's masked predictions.
        cls_all, patches_global = self.forward_backbone(crops, n_global, masks_global)
        dino_logits = self.project_dino(cls_all)
        selection = (ibot_selection_masks if ibot_selection_masks is not None
                     else masks_global)
        selected = patches_global[selection] if selection is not None else patches_global
        ibot_logits = self.project_ibot(selected)
        return dino_logits, ibot_logits, cls_all, patches_global


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


def lr_at(step, total_steps, warmup_steps, peak_lr, min_lr):
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    return cosine(peak_lr, min_lr, step - warmup_steps,
                  max(1, total_steps - warmup_steps))


def teacher_temp_at(step, warmup_steps, t_start, t_end):
    if warmup_steps > 0 and step < warmup_steps:
        return t_start + (t_end - t_start) * (step + 1) / warmup_steps
    return t_end


@torch.no_grad()
def update_ema(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    signature = lambda module: [(name, tuple(p.shape)) for name, p in module.named_parameters()]
    if signature(teacher) != signature(student):
        raise ValueError('teacher/student parameter layouts differ')
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)


def core_objective(dino, ibot, koleo, loss_config, koleo_weight):
    weights = [float(loss_config.get(name, 1.)) for name in
               ('dino_loss_weight', 'ibot_loss_weight')]
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError('DINO/iBOT loss weights must be finite and nonnegative')
    return weights[0] * dino + weights[1] * ibot + koleo_weight * koleo


def validate_gram_schedule(cfg):
    """Prevent a fixed profile from silently ignoring contradictory settings."""
    expected = {
        'training': {
            'lr': step_protocol.BASE_LR, 'min_lr': step_protocol.MIN_LR,
            'warmup_epochs': step_protocol.WARMUP_EPOCHS,
            'teacher_momentum_start': step_protocol.CORE_TEACHER_MOMENTUM,
            'teacher_momentum_end': step_protocol.GRAM_TEACHER_MOMENTUM,
        },
        'loss': {
            'teacher_temp_start': step_protocol.TEACHER_TEMP_START,
            'teacher_temp_end': step_protocol.TEACHER_TEMP_END,
            'teacher_temp_warmup_epochs': step_protocol.TEACHER_TEMP_WARMUP_EPOCHS,
        },
    }
    for section, values in expected.items():
        for key, value in values.items():
            if cfg[section].get(key) != value:
                raise ValueError(f'step4_gram_components requires {section}.{key}={value}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DINOv3 step 2 (core objective)")
    parser.add_argument("--config", default="configs/pretrain.yaml")
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

    profile = cfg["training"].get("training_profile", "core")
    if profile not in {"core", "step4_gram_components"}:
        raise ValueError(f"unknown training_profile: {profile!r}")
    if getattr(args, "resume", None):
        raise ValueError("resume is not supported by this single-process trainer")
    use_gram = profile == "step4_gram_components"
    if use_gram:
        validate_gram_schedule(cfg)

    device = resolve_device(getattr(args, "device", "auto"))
    seed = int(cfg.get("seed", 42))
    make_deterministic(seed)

    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    m, d, t, ls = cfg["model"], cfg["data"], cfg["training"], cfg["loss"]

    student = DINOv3Model(m).to(device)
    student.train()
    teacher = copy.deepcopy(student)
    teacher.requires_grad_(False)
    teacher.eval()

    n_global = int(d["n_global_crops"])
    n_local = int(d["n_local_crops"])
    aug = MultiCropAugmentation(
        global_size=int(d["global_size"]), local_size=int(d["local_size"]),
        global_scale=tuple(d["global_scale"]), local_scale=tuple(d["local_scale"]),
        n_global=n_global, n_local=n_local, return_gram_teacher_crops=use_gram)
    loader = get_multicrop_dataloader(
        d["data_root"], aug, batch_size=int(t["batch_size"]),
        num_workers=int(d["num_workers"]), distributed=False, seed=seed)

    dino_loss = DINOLoss(n_crops_global=n_global, n_crops_local=n_local,
                         student_temp=float(ls["student_temp"]),
                         sk_n_iters=int(ls["sk_n_iters"])).to(device)
    ibot_loss = IBOTLoss(n_crops_global=n_global,
                         student_temp=float(ls["student_temp"]),
                         sk_n_iters=int(ls["sk_n_iters"])).to(device)
    koleo_loss = KoLeoLoss().to(device)
    gram_loss = GramLoss().to(device)
    gram_teacher = None

    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(t["lr"]),
                                  weight_decay=float(t["weight_decay"]))

    koleo_weight = float(t["koleo_loss_weight"])
    grad_clip = float(t["grad_clip"])
    patch_size = int(m["patch_size"])
    gsz = int(d["global_size"])
    n_ph = n_pw = gsz // patch_size
    total_epochs = int(t["epochs"])
    steps_per_epoch = max(1, len(loader))
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = int(t["warmup_epochs"]) * steps_per_epoch
    temp_warmup_steps = int(ls["teacher_temp_warmup_epochs"]) * steps_per_epoch

    print("=" * 72)
    print("DINOv3: ViT + DINO + iBOT + KoLeo  (arXiv:2508.10104)")
    print(f"  device={device}  epochs={total_epochs}  crops={n_global}g+{n_local}l"
          f"  embed_dim={m['embed_dim']}  profile={profile}"
          f"  head_layout={student.head_layout}  canonical_eligible=False")
    print("=" * 72)

    # Milestone checkpoints for the 100/200/300 frozen-backbone probe sweep. This
    # config is already the unified ViT-B/16 Step 2, so the milestones are part of
    # the recipe; an empty save_at_epochs writes only checkpoint_latest.pth.
    save_at = {int(n) for n in t.get("save_at_epochs", [])}

    global_step = 0
    final_loss = None
    for epoch in range(total_epochs):
        running, count = 0.0, 0
        for batch in loader:
            views, gram_views, _labels = batch if use_gram else (batch[0], None, batch[1])
            lr = lr_at(global_step, total_steps, warmup_steps,
                       float(t["lr"]), float(t["min_lr"]))
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            ttemp = teacher_temp_at(global_step, temp_warmup_steps,
                                    float(ls["teacher_temp_start"]),
                                    float(ls["teacher_temp_end"]))
            dino_loss.teacher_temp = ttemp
            ibot_loss.teacher_temp = ttemp
            momentum = cosine(float(t["teacher_momentum_start"]),
                              float(t["teacher_momentum_end"]),
                              global_step, total_steps)

            local_weight = 1.0
            if use_gram:
                schedule = step_protocol.optimizer_step_schedule(global_step, steps_per_epoch)
                for pg in optimizer.param_groups:
                    pg["lr"] = schedule.lr
                lr, momentum = schedule.lr, schedule.teacher_momentum
                dino_loss.teacher_temp = ibot_loss.teacher_temp = schedule.teacher_temp
                local_weight = schedule.dino_local_weight

            views = [v.to(device, non_blocking=True) for v in views]
            B = views[0].shape[0]
            masks_global = generate_block_mask(
                batch_size=B * n_global, n_patches_h=n_ph, n_patches_w=n_pw,
                mask_ratio_min=float(t["ibot_mask_ratio_min"]),
                mask_ratio_max=float(t["ibot_mask_ratio_max"]),
                mask_probability=float(t["ibot_mask_sample_probability"]),
                device=device)

            with torch.no_grad():
                t_dino, t_ibot, _, _ = teacher(
                    views[:n_global], n_global=n_global, masks_global=None,
                    ibot_selection_masks=masks_global)
            s_dino, s_ibot, s_cls, s_patch = student(
                views, n_global=n_global, masks_global=masks_global,
                ibot_selection_masks=masks_global)

            loss_dino = dino_loss(s_dino, t_dino, local_loss_weight=local_weight)
            loss_ibot = ibot_loss(s_ibot, t_ibot, masks_global)
            global_cls = s_cls[:B * n_global].reshape(n_global, B, -1)
            loss_koleo = sum(koleo_loss(c) for c in global_cls)
            loss = core_objective(loss_dino, loss_ibot, loss_koleo, ls, koleo_weight)
            if use_gram and epoch >= step_protocol.CORE_EPOCHS:
                if gram_teacher is None or gram_views is None:
                    raise RuntimeError("Gram stage requires a frozen teacher and clean crops")
                with torch.no_grad():
                    gram_targets = torch.cat([
                        gram_teacher(v.to(device), mask=None, is_global=True)[1]
                        for v in gram_views], dim=0)
                loss = loss + schedule.gram_weight * gram_loss(s_patch, gram_targets)
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"DINOv3 loss became non-finite: {loss.item()}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optimizer.step()
            update_ema(teacher, student, momentum=momentum)

            running += loss.item() * B
            count += B
            global_step += 1
        final_loss = running / count if count else None
        print(f"  [{epoch}] dinov3_loss={final_loss}  dino={loss_dino.item():.4f}"
              f"  ibot={loss_ibot.item():.4f}  koleo={loss_koleo.item():.4f}"
              f"  lr={lr:.3g}  m={momentum:.4f}")
        if use_gram and epoch + 1 == step_protocol.CORE_EPOCHS:
            gram_teacher = copy.deepcopy(teacher.backbone).requires_grad_(False).eval()
        if use_gram and epoch + 1 in step_protocol.GRAM_TEACHER_UPDATE_EPOCHS:
            if gram_teacher is None:
                raise RuntimeError("Gram teacher refresh requested before snapshot")
            gram_teacher.load_state_dict(teacher.backbone.state_dict())
        ckpt = {"epoch": epoch, "teacher_state_dict": teacher.state_dict(),
                "student_state_dict": student.state_dict(),
                "loss": final_loss, "config": cfg}
        if use_gram:
            ckpt.update(training_profile=profile, canonical_eligible=False,
                        optimizer_state_dict=optimizer.state_dict(),
                        optimizer_step=global_step, steps_per_epoch=steps_per_epoch)
            if gram_teacher is not None:
                ckpt["gram_teacher_state_dict"] = gram_teacher.state_dict()
        torch.save(ckpt, os.path.join(save_dir, "checkpoint_latest.pth"))
        if (epoch + 1) in save_at:
            torch.save(ckpt, os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print(f"\nDINOv3 training complete (profile={profile})!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
