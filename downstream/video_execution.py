"""Inspected BasicFive video execution with portable epoch checkpoints."""

import hashlib
import json
import sys
from pathlib import Path

from torch.utils.data import Subset

from downstream import extended_distributed as distributed
from downstream import video_recipe
from downstream.imagenet_execution import ImageNetContinuation
from downstream.spatial_backbones import imagenet_execution_policy

HORIZON = 50
POPULATION_KEY = "videos"
RECIPE_KEY = "video_recipe"


class VideoContinuation(ImageNetContinuation):
    format_name = "ssv2_epoch_v1"


CONTINUATION = VideoContinuation


def resolve(cfg):
    video_recipe.resolve(cfg)
    settings = cfg.get(
        "execution",
        {
            "profile": "captured_video_v1",
            "accumulation_steps": 1,
            "precision": "fp32",
            "tail_policy": "discard",
        },
    )
    policy = imagenet_execution_policy(cfg["backbone"].get("kind"))
    if (
        not isinstance(settings, dict)
        or set(settings)
        != {"profile", "accumulation_steps", "precision", "tail_policy"}
        or settings["profile"] != "captured_video_v1"
        or policy is None
    ):
        raise ValueError("explicit inspected video execution required")
    if (
        type(settings["accumulation_steps"]) is not int
        or settings["accumulation_steps"] < 1
        or settings["precision"] not in ("fp32", "bf16")
        or settings["tail_policy"] != policy["tail_policy"]
    ):
        raise ValueError("invalid video accumulation or precision policy")
    _, _, world = distributed.launch()
    if world > 1 and not policy["distributed"]:
        raise ValueError("provider distributed source remains unverified")
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


def api_module():
    from downstream import ssv2

    return ssv2


def data(cfg, api):
    settings = cfg["probe"]
    recipe = video_recipe.resolve(cfg)
    datasets = [
        api.SSV2Clips(
            Path(cfg["data_root"]),
            split,
            settings["num_frames"],
            settings["image_size"],
            profile=cfg["profile"],
            recipe=recipe,
        )
        for split in ("train", "validation")
    ]
    labels = datasets[0].label_to_id
    if (
        labels != datasets[1].label_to_id
        or any(not 0 <= n < api.NUM_CLASSES for n in labels.values())
        or len(set(labels.values())) != len(labels)
    ):
        raise ValueError("video class mapping differs or exceeds 174 classes")
    identity = {"classes": labels}
    selected = []
    for split, dataset, limit in zip(
        ("train", "validation"),
        datasets,
        (settings["max_train_samples"], settings["max_val_samples"]),
    ):
        rows = dataset.samples[:limit] if limit else dataset.samples
        if not rows:
            raise ValueError("empty video population")
        digest = hashlib.sha256()
        for path, label in rows:
            content = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    content.update(chunk)
            digest.update(
                json.dumps(
                    [
                        str(path.relative_to(cfg["data_root"])),
                        label,
                        content.hexdigest(),
                    ],
                    separators=(",", ":"),
                ).encode()
            )
        identity[split] = {
            "samples": len(rows),
            "sha256": digest.hexdigest(),
            "annotation_sha256": dataset.entry_digest,
        }
        selected.append(Subset(dataset, range(len(rows))) if limit else dataset)
    return *selected, identity, recipe, None


def model(cfg, device, api):
    builder = (
        api.build_trainable_backbone
        if cfg["adaptation"] == "finetune"
        else api.build_frozen_backbone
    )
    return api.FrozenFrameAverageClassifier(
        builder(cfg["backbone"], device),
        adaptation=cfg["adaptation"],
        reader_profile=cfg.get("reader_profile"),
    ).to(device)


mix = video_recipe.mix


def describe(cfg, recipe, preprocessing):
    return {"video_recipe": recipe, "num_frames": cfg["probe"]["num_frames"]}


def run(cfg, out, *, device_override=None):
    from downstream.classification_execution import run as execute

    return execute(cfg, out, sys.modules[__name__], device_override=device_override)
