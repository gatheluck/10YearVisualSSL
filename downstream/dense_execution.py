"""Opt-in accumulated BasicFive dense-task training.

This preserves the inspected per-provider clipping and discarded-tail policy.
Portable continuation is separate; native checkpoints remain outside this
execution profile. Replicated DDP requires an inspected provider policy.
"""

import torch

from downstream import extended_execution as execution
from downstream.extended_distributed import launch, unwrap
from downstream.spatial_backbones import dense_execution_policy

HORIZONS = {"ade20k_segmentation": 20, "nyuv2_depth": 30, "coco_detection": 12}


def resolve(cfg):
    settings = cfg.get("execution")
    policy = dense_execution_policy(cfg.get("backbone", {}).get("kind"))
    if (
        not isinstance(settings, dict)
        or set(settings)
        != {"profile", "accumulation_steps", "precision", "tail_policy"}
        or settings["profile"] != "captured_dense_v1"
        or policy is None
        or cfg.get("task") not in HORIZONS
        or cfg.get("profile") != "capture_basic5_components"
        or cfg.get("optimizer_profile")
        not in ("basic5_frozen_v1", "basic5_finetune_v1")
    ):
        raise ValueError("explicit inspected dense execution and optimizer required")
    if (
        type(settings["accumulation_steps"]) is not int
        or settings["accumulation_steps"] < 1
        or settings["precision"] not in ("fp32", "bf16")
        or settings["tail_policy"] != policy["tail_policy"]
    ):
        raise ValueError("invalid dense accumulation or precision policy")
    if "resume" in cfg and (
        not isinstance(cfg["resume"], str) or not cfg["resume"].strip()
    ):
        raise ValueError("resume requires a checkpoint path")
    world = launch()[2]
    if world > 1 and not policy.get("distributed", False):
        raise ValueError(
            "provider dense distributed source behavior remains unverified"
        )
    if (
        world > 1
        and cfg.get("adaptation") == "finetune"
        and "distributed_finetune_tasks" in policy
        and cfg["task"] not in policy["distributed_finetune_tasks"]
    ):
        raise ValueError("provider dense distributed finetuning requires native FSDP")
    probe = cfg["detector" if cfg["task"] == "coco_detection" else "probe"]
    for key, minimum in (("batch_size", 1), ("epochs", 1), ("max_steps_per_epoch", 0)):
        if type(probe[key]) is not int or probe[key] < minimum:
            raise ValueError(f"dense execution requires integer {key} >= {minimum}")
    if (
        probe["max_steps_per_epoch"]
        or not 1 <= probe["epochs"] <= HORIZONS[cfg["task"]]
    ):
        raise ValueError(
            "dense execution requires complete epochs within the task horizon"
        )
    clipping = cfg.get("adaptation", "frozen") != "frozen" or (
        cfg["task"] == "coco_detection" and policy["clip_frozen_detection"]
    )
    plan = dict(
        settings,
        effective_batch=probe["batch_size"] * settings["accumulation_steps"] * world,
        world_size=world,
        clip_norm=1.0 if clipping else None,
        schedule_clock="optimizer_updates_with_microbatch_horizon",
    )
    if world > 1 and policy.get("gradient_sync") == "accumulated_mean":
        plan["gradient_sync"] = "accumulated_mean"
    return plan


def average_gradients(model):
    """Source-style sum/mean at update boundaries, with identical collectives.

    Missing gradients are zeros even when absent on every rank. This preserves
    the inspected optimizer behavior for unused trainable parameters.
    """
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
        parameter.grad.div_(torch.distributed.get_world_size())


def prepare(cfg, device, optimizer, scheduler, steps_per_epoch, scaled_lr):
    plan = resolve(cfg)
    execution.autocast_context(device, plan["precision"])
    if steps_per_epoch < plan["accumulation_steps"]:
        raise ValueError("epoch cannot produce an optimizer update")
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    runtime = execution.execution_report(
        plan,
        {"epochs": HORIZONS[cfg["task"]]},
        steps_per_epoch,
        scaled_lr,
    )
    return plan, scheduler, runtime


def train_epoch(model, loader, loss_for_batch, optimizer, scheduler, cfg, plan):
    def clip():
        if plan["clip_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], plan["clip_norm"]
            )

    backbone = getattr(unwrap(model), "backbone", None)
    if cfg["task"] == "coco_detection":
        backbone = getattr(backbone, "body", backbone)
    return execution.train_epoch(
        model,
        loader,
        loss_for_batch,
        optimizer,
        scheduler,
        accumulation_steps=plan["accumulation_steps"],
        tail_policy=plan["tail_policy"],
        adaptation=cfg.get("adaptation", "frozen"),
        trainable_backbone=cfg.get("adaptation") == "finetune",
        clip_gradients=clip,
        frozen_backbone=backbone,
        synchronize_gradients=(lambda: average_gradients(model))
        if plan.get("gradient_sync") == "accumulated_mean"
        else None,
    )


def batch_loss(model, batch, device, task, precision):
    """Autocast the forward; preserve the dense reductions in FP32."""
    if task == "coco_detection":
        images, targets = batch
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        with execution.autocast_context(device, precision):
            losses = model(images, targets)
        return sum(losses.values())
    images, *targets = [value.to(device) for value in batch]
    with execution.autocast_context(device, precision):
        prediction = model(images)
    if task == "ade20k_segmentation":
        return torch.nn.functional.cross_entropy(
            prediction.float(), targets[0], ignore_index=255
        )
    from downstream.nyuv2 import silog_loss

    return silog_loss(prediction.float(), targets[0].float(), targets[1])


autocast_context = execution.autocast_context
record_epoch = execution.record_epoch
