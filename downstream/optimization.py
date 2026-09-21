"""Verified optimizer groups and reference-batch Basic5 schedule components."""
from __future__ import annotations

import os
import math

import torch

PROFILE = "basic5_frozen_v1"
FT_PROFILE = "basic5_finetune_v1"
REFERENCE_SCHEDULE = "basic5_reference_schedule_v1"
COCO_SCHEDULE = "coco_frozen_1x_v1"
DENSE_AP_SCHEDULE = "dense_ap_cosine_v1"
DENSE_AP_EPOCHS = {"ade20k_segmentation": 20, "nyuv2_depth": 30}

# Task-owned runners pass their TASK identity. These are task recipes, not
# model-specific tuning. Full finetuning needs independently verified groups.
_RECIPES = {
    "ade20k_segmentation": ("AdamW", .001, 8),
    "nyuv2_depth": ("AdamW", .001, 8),
    "coco_detection": ("SGD", .02, 16),
    "ssv2_video": ("SGD", .1, 256),
    "imagenet_classification": ("SGD", .1, 256),
}


def resolve_optimization(cfg: dict, task: str) -> dict | None:
    """Validate an opt-in selection and describe exactly what will execute.

The existing numeric lr belongs to the legacy optimizer. Requiring the explicit
sentinel avoids silently ignoring a caller's requested rate. This component
only supports one process and one full physical batch per optimizer update.
"""
    if "scheduler_profile" in cfg:
        if cfg["scheduler_profile"] == REFERENCE_SCHEDULE:
            if cfg.get("optimizer_profile") not in (PROFILE, FT_PROFILE):
                raise ValueError("reference schedule requires a Basic5 optimizer")
        elif cfg["scheduler_profile"] == DENSE_AP_SCHEDULE:
            if task not in DENSE_AP_EPOCHS or cfg.get("adaptation") != "attentive":
                raise ValueError("dense_ap_cosine_v1 requires ADE20K/NYUv2 attentive adaptation")
        elif cfg["scheduler_profile"] != COCO_SCHEDULE or task != "coco_detection":
            raise ValueError("scheduler_profile requires coco_frozen_1x_v1 on COCO")
        if cfg["scheduler_profile"] != REFERENCE_SCHEDULE and cfg.get("optimizer_profile") != PROFILE:
            raise ValueError("scheduler_profile requires basic5_frozen_v1 optimizer")
    if "optimizer_profile" not in cfg:
        return None
    if cfg["optimizer_profile"] not in (PROFILE, FT_PROFILE):
        raise ValueError(f"config.optimizer_profile: expected {PROFILE} or {FT_PROFILE}")
    if cfg.get("profile") != "capture_basic5_components":
        raise ValueError("optimizer_profile requires capture_basic5_components")
    adaptation = cfg.get("adaptation", "frozen")
    finetune = cfg["optimizer_profile"] == FT_PROFILE
    if finetune:
        from downstream.spatial_backbones import supports_finetune_groups
        if adaptation != "finetune" or not supports_finetune_groups(cfg.get("backbone", {}).get("kind")):
            raise ValueError("finetune optimizer requires a verified provider policy and finetune adaptation")
    elif adaptation not in ("frozen", "attentive"):
        raise ValueError("basic5_frozen_v1 supports frozen/attentive only; select basic5_finetune_v1 for FT")
    if cfg.get("task") != task or task not in _RECIPES:
        raise ValueError("optimizer_profile requires the runner's supported task identity")
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise ValueError("optimizer_profile requires WORLD_SIZE=1; distributed execution unsupported")
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError("optimizer_profile does not support a distributed process group")
    settings = cfg["detector" if task == "coco_detection" else "probe"]
    batch = settings["batch_size"]
    if type(batch) is not int or batch <= 0:
        raise ValueError("optimizer_profile requires a positive integer batch_size")
    for field, minimum in (("epochs", 1), ("max_steps_per_epoch", 0)):
        value = settings[field]
        if type(value) is not int or value < minimum:
            raise ValueError(f"optimizer_profile requires integer {field} >= {minimum}")
    if cfg.get("scheduler_profile") == COCO_SCHEDULE and settings["epochs"] > 12:
        raise ValueError("coco_frozen_1x_v1 requires epochs <= 12")
    if cfg.get("scheduler_profile") == DENSE_AP_SCHEDULE:
        if batch != 8 or settings["epochs"] != DENSE_AP_EPOCHS[task]:
            raise ValueError("dense_ap_cosine_v1 requires batch 8 and the task's full reference epochs")
    if settings["lr"] != "protocol":
        raise ValueError("optimizer_profile requires lr: protocol; numeric overrides unsupported")
    algorithm, base_lr, reference_batch = _RECIPES[task]
    if task in ("ssv2_video", "imagenet_classification") and adaptation == "attentive":
        algorithm, base_lr = "AdamW", .001
    weight_decay = .05 if adaptation == "attentive" and algorithm == "AdamW" else .0001
    if task == "imagenet_classification" and adaptation == "frozen":
        weight_decay = 0.
    if finetune:
        if task == "imagenet_classification":
            raise ValueError("ImageNet FT recipe conflicts remain unresolved")
        algorithm, base_lr, reference_batch, layer_decay = {
            "ade20k_segmentation": ("AdamW", .0001, 8, .8),
            "nyuv2_depth": ("AdamW", .0001, 8, .8),
            "coco_detection": ("SGD", .02, 16, 1.),
            "ssv2_video": ("AdamW", .0005, 256, .75),
        }[task]
        weight_decay = .0001 if algorithm == "SGD" else .05
    report = {
        "profile": cfg["optimizer_profile"], "optimizer": algorithm,
        "base_lr": base_lr, "reference_batch": reference_batch,
        "lr": base_lr * batch / reference_batch, "weight_decay": weight_decay,
        "effective_batch": batch, "world_size": 1, "accumulation_steps": 1,
        "drop_last": True, "schedule": "none",
    }
    report.update({"momentum": .9} if algorithm == "SGD" else {"betas": [.9, .999]})
    if finetune:
        report["layer_decay"] = layer_decay
    if cfg.get("scheduler_profile") == REFERENCE_SCHEDULE:
        epochs = {"ade20k_segmentation": 20, "nyuv2_depth": 30,
                  "coco_detection": 12, "ssv2_video": 50, "imagenet_classification": 100}[task]
        if batch != reference_batch or settings["epochs"] != epochs:
            raise ValueError("reference schedule requires the reference batch and full epoch horizon")
        report.update(scheduler_profile=REFERENCE_SCHEDULE, task=task,
                      adaptation=adaptation, epochs=epochs)
    return report


def build_optimizer(model, report: dict):
    """Optimize every trainable head/reader parameter exactly once."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("optimizer_profile found no trainable parameters")
    if report["profile"] == FT_PROFILE:
        params = _finetune_groups(model, report)
    kwargs = {"lr": report["lr"], "weight_decay": report["weight_decay"]}
    if report["optimizer"] == "SGD":
        return torch.optim.SGD(params, momentum=report["momentum"], **kwargs)
    return torch.optim.AdamW(params, betas=tuple(report["betas"]), **kwargs)


def _finetune_groups(model, report):
    providers = [m for m in model.modules() if callable(getattr(m, "finetune_group_policy", None))]
    if len(providers) != 1:
        raise ValueError("finetune optimizer requires exactly one parameter-policy provider")
    provider = providers[0]
    output_layer, policy = provider.finetune_group_policy()
    owned = {name: p for name, p in provider.named_parameters() if p.requires_grad}
    if set(policy) != set(owned):
        raise ValueError("finetune policy must cover every trainable encoder parameter")
    mapping = {id(p): policy[name] for name, p in owned.items()}
    groups, coverage = {}, {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        layer, no_decay = mapping.get(id(parameter),
            (output_layer, "bias" in name.lower() or "norm" in name.lower()))
        if type(layer) is not int or not 0 <= layer <= output_layer:
            raise ValueError("finetune policy layer outside encoder depth")
        scale = report["layer_decay"] ** (output_layer - layer)
        key = (layer, no_decay)
        group = groups.setdefault(key, dict(params=[], lr=report["lr"] * scale,
            weight_decay=0. if no_decay else report["weight_decay"], layer_id=layer,
            lr_scale=scale))
        group["params"].append(parameter)
        coverage[name] = dict(layer_id=layer, lr=group["lr"], weight_decay=group["weight_decay"])
    report["parameter_groups"] = coverage
    return list(groups.values())


def require_training_batches(loader, report: dict | None) -> None:
    """A dropped, empty training set is a failure, never a completed run."""
    if report is not None and len(loader) == 0:
        raise ValueError("optimizer_profile requires at least one full training batch")


def build_coco_scheduler(optimizer, steps_per_epoch: int):
    """Match captured COCO update indexing, including warmup precedence.

    The caller passes the full nonempty loader length, before a smoke step cap.
    LambdaLR initializes update zero; call step only after optimizer.step.
    Accumulation and resume are outside this component's supported interface.
    """
    def factor(update):
        if update < 500:
            return .001 + .999 * update / 500
        epoch = update / steps_per_epoch
        if epoch >= 11:
            return .01
        if epoch >= 8:
            return .1
        return 1.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def build_dense_ap_scheduler(optimizer, steps_per_epoch: int, epochs: int):
    """Reference-batch AP: one epoch warmup, then cosine to 1e-6.

    Runners validate the fixed batch and horizon before calling this builder.
    The full loader length, before a smoke cap, defines the schedule clock.
    """
    def factor(update):
        if update < steps_per_epoch:
            return .001 + .999 * update / steps_per_epoch
        progress = min(max((update - steps_per_epoch) / ((epochs - 1) * steps_per_epoch), 0.), 1.)
        return .001 + .999 * .5 * (1. + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def dense_ap_schedule_report(scheduler, steps_per_epoch, epochs, max_steps):
    """Describe realized updates without promoting component runs to records."""
    return {"profile": DENSE_AP_SCHEDULE, "steps_per_epoch": steps_per_epoch,
            "warmup_updates": steps_per_epoch, "warmup_start_lr": 1e-6,
            "minimum_lr": 1e-6, "reference_epochs": epochs,
            "updates_completed": scheduler.last_epoch,
            "next_update_lr": scheduler.get_last_lr()[0],
            "truncated": bool(max_steps and max_steps < steps_per_epoch)}


def build_task_scheduler(optimizer, report, steps_per_epoch):
    """Reference-batch schedules only: batch-scaled endpoints remain unresolved."""
    if report is None or report.get("scheduler_profile") != REFERENCE_SCHEDULE:
        return None
    if type(steps_per_epoch) is not int or steps_per_epoch < 1:
        raise ValueError("schedule requires positive steps_per_epoch")
    if report["task"] == "coco_detection":
        return build_coco_scheduler(optimizer, steps_per_epoch)
    classification = report["task"] in ("ssv2_video", "imagenet_classification")
    warmup_epochs = 5 if classification else 1
    if report["task"] == "imagenet_classification" and report["adaptation"] == "frozen":
        warmup_epochs = 0
    warmup = warmup_epochs * steps_per_epoch
    total = report["epochs"] * steps_per_epoch
    start = 1e-6 / report["base_lr"]
    floor = 0. if classification and report["adaptation"] == "frozen" else start
    def factor(update):
        if update < warmup:
            return start + (1. - start) * update / warmup
        progress = min(max((update - warmup) / (total - warmup), 0.), 1.)
        return floor + (1. - floor) * .5 * (1. + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def task_schedule_report(scheduler, report, steps_per_epoch, max_steps):
    return dict(profile=REFERENCE_SCHEDULE, reference_epochs=report["epochs"],
                steps_per_epoch=steps_per_epoch, updates_completed=scheduler.last_epoch,
                next_update_lr=scheduler.get_last_lr(),
                truncated=bool(max_steps and max_steps < steps_per_epoch))
