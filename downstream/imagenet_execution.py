"""Opt-in ImageNet accumulation, DDP and portable epoch continuation."""

import hashlib
import json
from pathlib import Path
from typing import cast

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from downstream import contract
from downstream import extended_distributed as distributed
from downstream import extended_execution as execution
from downstream.extended_resume import Continuation
from downstream.spatial_backbones import imagenet_execution_policy

PROFILE = "captured_imagenet_v1"


def resolve(cfg):
    settings = cfg.get("execution")
    keys = {"profile", "accumulation_steps", "precision", "tail_policy"}
    policy = imagenet_execution_policy(cfg.get("backbone", {}).get("kind"))
    if (
        not isinstance(settings, dict)
        or set(settings) != keys
        or settings["profile"] != PROFILE
        or cfg.get("task") != "imagenet_classification"
        or policy is None
    ):
        raise ValueError("explicit inspected ImageNet execution profile required")
    if (
        type(settings["accumulation_steps"]) is not int
        or settings["accumulation_steps"] < 1
        or settings["precision"] not in ("fp32", "bf16")
        or settings["tail_policy"] != policy["tail_policy"]
    ):
        raise ValueError("invalid ImageNet accumulation or precision policy")
    _rank, _local, world = distributed.launch()
    if world > 1 and not policy["distributed"]:
        raise ValueError("provider distributed source behavior remains unverified")
    if cfg["probe"]["max_steps_per_epoch"] or not 1 <= cfg["probe"]["epochs"] <= 100:
        raise ValueError(
            "execution requires complete epochs within the 100-epoch horizon"
        )
    if "resume" in cfg and (
        not isinstance(cfg["resume"], str) or not cfg["resume"].strip()
    ):
        raise ValueError("resume requires a checkpoint path")
    return dict(
        settings,
        world_size=world,
        effective_batch=cfg["probe"]["batch_size"]
        * world
        * settings["accumulation_steps"],
        rank_seed_stride=policy["rank_seed_stride"],
        schedule_clock="optimizer_updates_with_microbatch_horizon",
    )


class ImageNetContinuation(Continuation):
    format_name = "imagenet_epoch_v1"

    def model_state(self):
        if self.model.adaptation == "finetune":
            return self.model.state_dict()
        return super().model_state()


def _data(cfg, api):
    size = cfg["probe"]["image_size"]
    ft = api.imagenet_finetune.resolve(cfg) if cfg["adaptation"] == "finetune" else None
    preprocessing = api.resolve_preprocessing(cfg)
    train = api.ImageNetImages(
        cfg["data_root"], "train", size, finetune_recipe=ft, preprocessing=preprocessing
    )
    val = api.ImageNetImages(
        cfg["data_root"], "val", size, finetune_recipe=ft, preprocessing=preprocessing
    )
    if train.class_to_idx != val.class_to_idx or len(train.classes) > api.NUM_CLASSES:
        raise ValueError("ImageNet class mapping differs or exceeds 1000 classes")
    classes = train.classes
    selected = []
    identity = {"classes": classes}
    for split, data, limit in (
        ("train", train, cfg["probe"]["max_train_samples"]),
        ("val", val, cfg["probe"]["max_val_samples"]),
    ):
        rows = data.samples[:limit] if limit else data.samples
        digest = hashlib.sha256()
        for name, target in rows:
            path = Path(name)
            digest.update(
                json.dumps(
                    [str(path.relative_to(cfg["data_root"])), target],
                    separators=(",", ":"),
                ).encode()
            )
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        if not rows:
            raise ValueError("ImageNet split is empty")
        identity[split] = {"samples": len(rows), "sha256": digest.hexdigest()}
        selected.append(Subset(data, range(len(rows))) if limit else data)
    return *selected, identity, ft, preprocessing


def run(cfg, out, *, device_override=None):
    from downstream import imagenet as api

    with distributed.session(device_override or cfg["device"]) as context:
        plan = context.call(lambda: resolve(cfg))
        context.agree(cfg)
        out = Path(out)

        def prepare_output():
            if out.exists() and any(out.iterdir()):
                raise FileExistsError("ImageNet output directory must be empty")
            out.mkdir(parents=True, exist_ok=True)

        context.call(prepare_output, leader=True)
        device = context.device
        execution.autocast_context(device, plan["precision"])
        api.make_deterministic(cfg["seed"] + plan["rank_seed_stride"] * context.rank)
        train, val, membership, ft, preprocessing = context.call(
            lambda: _data(cfg, api)
        )
        context.agree(membership)
        settings = cfg["probe"]
        loader = context.call(lambda: context.loader(train, settings, cfg["seed"]))
        validation = DataLoader(
            val, batch_size=settings["batch_size"], num_workers=settings["num_workers"]
        )
        builder = (
            api.build_trainable_backbone
            if cfg["adaptation"] == "finetune"
            else api.build_frozen_backbone
        )
        raw = context.call(
            lambda: api.ImageClassifier(
                builder(cfg["backbone"], device),
                cfg["adaptation"],
                reader_profile=cfg.get("reader_profile"),
            ).to(device)
        )
        model = context.wrap(raw)
        report = cast(dict, api.resolve_optimization(cfg, api.TASK))
        opt = api.build_optimizer(raw, report)
        scheduler = api.build_task_scheduler(opt, report, len(loader))
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        runtime = execution.execution_report(
            plan, {"epochs": 100}, len(loader), report["lr"]
        )
        continuation = ImageNetContinuation(
            cfg, membership, raw, opt, scheduler, runtime, loader, context, out
        )

        def loss_for_batch(batch):
            images, labels = (value.to(device) for value in batch)
            if ft is not None:
                images, targets = api.imagenet_finetune.mix(
                    images, labels, api.NUM_CLASSES, ft
                )
            with execution.autocast_context(device, plan["precision"]):
                logits = model(images)
            if ft is not None:
                return -(targets * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()
            return F.cross_entropy(logits.float(), labels)

        for epoch in range(continuation.start_epoch, settings["epochs"]):
            if context.world > 1:
                loader.sampler.set_epoch(epoch)
            statistics = execution.train_epoch(
                model,
                loader,
                loss_for_batch,
                opt,
                scheduler,
                accumulation_steps=plan["accumulation_steps"],
                tail_policy=plan["tail_policy"],
                adaptation=cfg["adaptation"],
                trainable_backbone=cfg["adaptation"] == "finetune",
            )
            execution.record_epoch(runtime, statistics)
            continuation.save(epoch + 1)

        def finish():
            raw.eval()
            totals = [0.0, 0.0]
            count = 0
            with torch.no_grad():
                for images, labels in validation:
                    with execution.autocast_context(device, plan["precision"]):
                        logits = raw(images.to(device))
                    scores = api.accuracy(logits.float(), labels.to(device))
                    count += len(labels)
                    totals[:] = [
                        total + score * len(labels)
                        for total, score in zip(totals, scores)
                    ]
            result = {
                "top1": totals[0] / count,
                "top5": totals[1] / count,
                "images": count,
                "epochs": settings["epochs"],
            }
            if "scheduler_profile" in report:
                report["schedule"] = api.task_schedule_report(
                    scheduler, report, len(loader), 0
                )
            contract.write_metrics(out, result, api.METRIC_NAMES)
            if ft is not None:
                torch.save(
                    {
                        "model": raw.state_dict(),
                        "finetune_recipe": ft,
                        "epochs_completed": settings["epochs"],
                    },
                    out / "finetune_model.pt",
                )
            else:
                torch.save(continuation.model_state(), out / "probe.pt")
            result_report = {
                "task": api.TASK,
                "profile": cfg["profile"],
                "adaptation": cfg["adaptation"],
                "reader_profile": cfg.get("reader_profile"),
                "backbone": cfg["backbone"],
                "num_classes": api.NUM_CLASSES,
                "observed_classes": membership["classes"],
                "optimization": report,
                "final": result,
                "execution": runtime,
                "continuation": continuation.report(),
                "membership": membership,
                "canonical_eligible": False,
                "record_value": False,
            }
            if ft is not None:
                result_report["finetune_recipe"] = ft
            if preprocessing is not None:
                result_report["preprocessing"] = preprocessing
            (out / "results.json").write_text(
                json.dumps(result_report, indent=2, sort_keys=True) + "\n"
            )
            return result

        return context.call(finish, leader=True)
