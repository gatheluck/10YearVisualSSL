"""Verified frozen/AP optimizer components, not complete Basic5 schedules."""
from __future__ import annotations

import os

import torch

PROFILE = "basic5_frozen_v1"

# Task-owned runners pass their TASK identity. These are task recipes, not
# model-specific tuning. Full finetuning needs independently verified groups.
_RECIPES = {
    "ade20k_segmentation": ("AdamW", .001, 8),
    "nyuv2_depth": ("AdamW", .001, 8),
    "coco_detection": ("SGD", .02, 16),
    "ssv2_video": ("SGD", .1, 256),
}


def resolve_optimization(cfg: dict, task: str) -> dict | None:
    """Validate an opt-in selection and describe exactly what will execute.

The existing numeric lr belongs to the legacy optimizer. Requiring the explicit
sentinel avoids silently ignoring a caller's requested rate. This component
only supports one process and one full physical batch per optimizer update.
"""
    if "optimizer_profile" not in cfg:
        return None
    if cfg["optimizer_profile"] != PROFILE:
        raise ValueError(f"config.optimizer_profile: expected {PROFILE}")
    if cfg.get("profile") != "capture_basic5_components":
        raise ValueError("optimizer_profile requires capture_basic5_components")
    adaptation = cfg.get("adaptation", "frozen")
    if adaptation not in ("frozen", "attentive"):
        raise ValueError("optimizer_profile supports frozen/attentive only; FT groups pending")
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
    if settings["lr"] != "protocol":
        raise ValueError("optimizer_profile requires lr: protocol; numeric overrides unsupported")
    algorithm, base_lr, reference_batch = _RECIPES[task]
    if task == "ssv2_video" and adaptation == "attentive":
        algorithm, base_lr = "AdamW", .001
    weight_decay = .05 if adaptation == "attentive" and algorithm == "AdamW" else .0001
    report = {
        "profile": PROFILE, "optimizer": algorithm,
        "base_lr": base_lr, "reference_batch": reference_batch,
        "lr": base_lr * batch / reference_batch, "weight_decay": weight_decay,
        "effective_batch": batch, "world_size": 1, "accumulation_steps": 1,
        "drop_last": True, "schedule": "none",
    }
    report.update({"momentum": .9} if algorithm == "SGD" else {"betas": [.9, .999]})
    return report


def build_optimizer(model, report: dict):
    """Optimize every trainable head/reader parameter exactly once."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("optimizer_profile found no trainable parameters")
    kwargs = {"lr": report["lr"], "weight_decay": report["weight_decay"]}
    if report["optimizer"] == "SGD":
        return torch.optim.SGD(params, momentum=report["momentum"], **kwargs)
    return torch.optim.AdamW(params, betas=tuple(report["betas"]), **kwargs)


def require_training_batches(loader, report: dict | None) -> None:
    """A dropped, empty training set is a failure, never a completed run."""
    if report is not None and len(loader) == 0:
        raise ValueError("optimizer_profile requires at least one full training batch")
