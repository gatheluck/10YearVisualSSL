"""LP/AP execution for five explicit pose, localization and reasoning datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from downstream import contract
from downstream import extended_distributed as distributed
from downstream import extended_execution as execution
from downstream.extended_classification import PROTOCOLS, optimizer
from downstream.extended_classification import validate_config as validate_probe
from downstream.extended_resume import Continuation, _probe
from downstream.spatial_backbones import build_frozen_backbone
from downstream.structured_data import FAMILIES, collate, load_data
from downstream.structured_heads import Probe, camera_loss, heatmap_loss
from downstream.structured_metrics import (
    CameraScore,
    KeypointAP,
    decode_heatmaps,
    pckh_counts,
)

TASK = "extended_structured"
TRANSFORM = "captured_structured_v1"
RECIPE_IDS = {
    "pose": "pose_30",
    "localization": "localization_30",
    "reasoning": "visual_reasoning_50",
}


def recipe(dataset, adaptation):
    if dataset not in FAMILIES or adaptation not in ("frozen", "attentive"):
        raise ValueError("only verified structured LP/AP datasets are supported")
    ap = adaptation == "attentive"
    filename = "EXTEND_ATTENTIVE_v1.json" if ap else "EXTEND_LINEAR_v1.json"
    key = "recipe_overrides" if ap else "recipes"
    result = dict(
        json.loads((PROTOCOLS / filename).read_text())[key][
            RECIPE_IDS[FAMILIES[dataset]]
        ]["optimization"]
    )
    result["betas"] = [0.9, 0.999]
    result["warmup"] = {
        "1 epoch": "1 epoch from 1e-6",
        "5 epochs": "5 epochs from 1e-6",
    }[result["warmup"]]
    if result["schedule"] != "cosine":
        raise ValueError("unresolved structured schedule")
    result["schedule"] = "cosine to 0"
    return result


def validate_config(cfg):
    family = FAMILIES.get(cfg.get("dataset"))
    if family is None:
        raise ValueError("unknown structured dataset")
    plan = validate_probe(
        cfg,
        task=TASK,
        get_recipe=recipe,
        reader_attribute="EXTENDED_IMAGE_READER"
        if family == "reasoning"
        else "EXTENDED_DENSE_READER",
        max_epochs=50 if family == "reasoning" else 30,
    )
    if cfg["transform_profile"] != TRANSFORM:
        raise ValueError("explicit structured transform profile required")
    return plan


def evaluate(data, loader, forward):
    family = data.family
    seen = set()
    correct = count = 0
    predictions = {}
    camera = (
        CameraScore(data.dataset, data.scenes) if family == "localization" else None
    )
    for batch in loader:
        images, indices = batch[0], batch[-1]
        output = (
            forward(images, batch[1] if family == "reasoning" else None)
            .float()
            .detach()
            .cpu()
        )
        if len(output) != len(indices) or not torch.isfinite(output).all():
            raise ValueError("invalid evaluation prediction")
        for k, index in enumerate(indices.tolist()):
            if index in seen or not 0 <= index < len(data):
                raise ValueError("duplicate or out-of-range evaluation sample")
            seen.add(index)
            if family == "reasoning":
                row = data.items[index]
                if output.ndim != 2 or output.shape[1] < len(row["options"]):
                    raise ValueError("missing answer logits")
                correct += int(int(output[k].argmax()) == row["target"])
                count += 1
            elif camera is not None:
                row = data.items[index]
                if output.shape[1:] != (len(data.scenes), 12):
                    raise ValueError("invalid camera head output")
                value = output[k, data.scenes.index(row["scene"])].numpy()
                pose = np.eye(4)
                pose[:3, 3] = value[:3]
                pose[:3, :3] = value[3:].reshape(3, 3)
                camera.update(
                    row["scene"], pose, np.asarray(row["pose"], dtype=np.float32)
                )
            else:
                row, person = data.items[index]
                joints = 16 if data.dataset == "mpii_pose" else 14
                if output.shape[1:] != (joints, 56, 56):
                    raise ValueError("invalid pose heatmap output")
                xy, confidence = decode_heatmaps(output[k].numpy(), person["crop"])
                if data.dataset == "mpii_pose":
                    c, n = pckh_counts(
                        xy[None],
                        np.asarray(person["pos_gt_src"])[None],
                        np.asarray(person["jnt_missing"])[None],
                        np.asarray(person["headboxes_src"])[None],
                    )
                    correct += c
                    count += n
                else:
                    predictions.setdefault(row["id"], []).append(
                        {
                            "keypoints": np.column_stack([xy, confidence])
                            .reshape(-1)
                            .tolist(),
                            "score": float(confidence.mean()),
                        }
                    )
    if seen != set(range(len(data))):
        raise ValueError("incomplete evaluation population")
    if camera is not None:
        result = camera.result()
    elif data.dataset == "crowdpose":
        metric = KeypointAP()
        for row in data.rows:
            metric.add_image(
                row["id"],
                row["people"],
                predictions.get(row["id"], []),
                row["crowd_index"],
            )
        values = metric.result()
        result = {k: 100 * values[k] for k in ("keypoint_AP", "AP50", "AP75")}
        result["images"] = values["n"]
    else:
        if not count:
            raise ValueError("no evaluable targets")
        result = {
            "pckh_hrnet_gt_val" if family == "pose" else "accuracy": 100.0
            * correct
            / count,
            "targets": count,
        }
    result["evaluated_samples"] = len(seen)
    return result


def metric_names(dataset):
    if dataset == "mpii_pose":
        return {"pckh_hrnet_gt_val": "extended_mpii_pckh_hrnet_gt_val"}
    if dataset == "crowdpose":
        return {
            k: "extended_crowdpose_" + k.lower()
            for k in ("keypoint_AP", "AP50", "AP75")
        }
    if dataset == "three_d_srbench":
        return {"accuracy": "extended_3dsr_local_accuracy"}
    return {
        k: "extended_" + dataset + "_" + k
        for k in ("median_translation_m", "median_rotation_deg")
    }


def run(cfg, out):
    with distributed.session(cfg["device"]) as context:
        plan = context.call(lambda: validate_config(cfg))
        context.agree(cfg)
        context.seed(cfg["seed"])
        device = context.device
        execution.autocast_context(device, plan["precision"])
        train, val, membership = context.call(
            lambda: load_data(cfg["samples"], cfg["data_root"], cfg["dataset"])
        )
        context.agree(membership)
        settings = cfg["probe"]
        loader = context.call(
            lambda: context.loader(train, settings, cfg["seed"], collate_fn=collate)
        )
        validation = DataLoader(
            val,
            batch_size=settings["batch_size"],
            num_workers=settings["num_workers"],
            collate_fn=collate,
        )
        family = FAMILIES[cfg["dataset"]]
        raw = context.call(
            lambda: Probe(
                build_frozen_backbone(cfg["backbone"], device),
                family,
                membership["outputs"],
                cfg["reader_profile"],
            ).to(device)
        )
        model = context.wrap(raw)
        spec = recipe(cfg["dataset"], cfg["adaptation"])
        opt, scheduler = optimizer(
            raw, spec, batch_size=plan["effective_batch"], steps_per_epoch=len(loader)
        )
        runtime = execution.execution_report(
            plan, spec, len(loader), scheduler.base_lrs[0]
        )
        continuation = Continuation(
            cfg, membership, raw, opt, scheduler, runtime, loader, context, out
        )

        def forward(images, questions=None):
            with execution.autocast_context(device, plan["precision"]):
                return model(
                    images.to(device),
                    questions.to(device) if questions is not None else None,
                )

        def loss(batch):
            pred = forward(
                batch[0], batch[1] if family == "reasoning" else None
            ).float()
            target = batch[-1].to(device)
            if family == "reasoning":
                return F.cross_entropy(pred, target)
            return (
                heatmap_loss(pred, target)
                if family == "pose"
                else camera_loss(pred, target)
            )

        for epoch in range(continuation.start_epoch, settings["epochs"]):
            if context.world > 1:
                loader.sampler.set_epoch(epoch)
            stats = execution.train_epoch(
                model,
                loader,
                loss,
                opt,
                scheduler,
                accumulation_steps=plan["accumulation_steps"],
                tail_policy=plan["tail_policy"],
                adaptation=cfg["adaptation"],
            )
            execution.record_epoch(runtime, stats)
            continuation.save(epoch + 1)
        model = raw

        def publish():
            raw.eval()
            with torch.no_grad():
                result = evaluate(val, validation, forward)
            result["epochs"] = settings["epochs"]
            names = dict.fromkeys(result)
            names.update(metric_names(cfg["dataset"]))
            names["epochs"] = "epochs_completed"
            contract.write_metrics(
                out,
                {k: result[k] for k in (*metric_names(cfg["dataset"]), "epochs")},
                names,
            )
            torch.save(
                {k: v.cpu() for k, v in _probe(raw).items()}, Path(out) / "probe.pt"
            )
            report = {
                "task": TASK,
                "profile": cfg["profile"],
                "dataset": cfg["dataset"],
                "adaptation": cfg["adaptation"],
                "reader_profile": cfg["reader_profile"],
                "transform_profile": TRANSFORM,
                "backbone": cfg["backbone"],
                "membership": membership,
                "recipe": spec,
                "execution": runtime,
                "continuation": continuation.report(),
                "final": result,
                "canonical_eligible": False,
                "record_value": False,
                "limitations": [
                    "explicit sample membership is not release authenticity or historical score attribution",
                    "released-weight CUDA/BF16 and full-data result parity unverified",
                    "legacy native checkpoint import and whole-builder RNG parity unverified",
                ],
                "task_disclosure": {
                    "pose": "GT person crops; 56-grid heatmaps; MPII one-based HRNet sidecar; CrowdPose GT boxes, maxDets20, no NMS",
                    "localization": "absolute pose regression; not the companion retrieval/verification recipe; equal-scene medians",
                    "reasoning": "COCO-2100 local fitted holdout; full UTF-8 question/options; not CircularEval",
                }[family],
            }
            (Path(out) / "results.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n"
            )
            return result

        return context.call(publish, leader=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return distributed.cli(Path(args.config).read_bytes(), args.out, run, TASK)


if __name__ == "__main__":
    raise SystemExit(main())
