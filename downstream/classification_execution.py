"""Shared accumulated image/video classification and epoch continuation."""

import json
from pathlib import Path
from typing import cast

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from downstream import contract
from downstream import extended_distributed as distributed
from downstream import extended_execution as execution


def run(cfg, out, adapter, *, device_override=None):
    api = adapter.api_module()
    finetune = cfg["adaptation"] == "finetune"

    with distributed.session(device_override or cfg["device"]) as context:
        plan = context.call(lambda: adapter.resolve(cfg))
        context.agree(cfg)
        out = Path(out)

        def prepare_output():
            if out.exists() and any(out.iterdir()):
                raise FileExistsError("classification output directory must be empty")
            out.mkdir(parents=True, exist_ok=True)

        context.call(prepare_output, leader=True)
        device = context.device
        execution.autocast_context(device, plan["precision"])
        api.make_deterministic(cfg["seed"] + plan["rank_seed_stride"] * context.rank)
        train, val, membership, ft, preprocessing = context.call(
            lambda: adapter.data(cfg, api)
        )
        context.agree(membership)
        settings = cfg["probe"]
        loader = context.call(lambda: context.loader(train, settings, cfg["seed"]))
        validation = DataLoader(
            val, batch_size=settings["batch_size"], num_workers=settings["num_workers"]
        )
        raw = context.call(lambda: adapter.model(cfg, device, api))
        model = context.wrap(raw)
        report = cast(dict, api.resolve_optimization(cfg, api.TASK))
        opt = api.build_optimizer(raw, report)
        scheduler = api.build_task_scheduler(opt, report, len(loader))
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        runtime = execution.execution_report(
            plan, {"epochs": adapter.HORIZON}, len(loader), report["lr"]
        )
        continuation = adapter.CONTINUATION(
            cfg, membership, raw, opt, scheduler, runtime, loader, context, out
        )

        def loss_for_batch(batch):
            images, labels = (value.to(device) for value in batch)
            if finetune:
                images, targets = adapter.mix(images, labels, api.NUM_CLASSES, ft)
            with execution.autocast_context(device, plan["precision"]):
                logits = model(images)
            if finetune:
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
                adapter.POPULATION_KEY: count,
                "epochs": settings["epochs"],
            }
            if "scheduler_profile" in report:
                report["schedule"] = api.task_schedule_report(
                    scheduler, report, len(loader), 0
                )
            contract.write_metrics(out, result, api.METRIC_NAMES)
            if finetune:
                torch.save(
                    {
                        "model": raw.state_dict(),
                        adapter.RECIPE_KEY: ft,
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
            result_report.update(adapter.describe(cfg, ft, preprocessing))
            (out / "results.json").write_text(
                json.dumps(result_report, indent=2, sort_keys=True) + "\n"
            )
            return result

        return context.call(finish, leader=True)
