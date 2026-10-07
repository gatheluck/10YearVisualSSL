"""Opt-in ImageNet accumulation, DDP and portable epoch continuation."""

import hashlib
import json
from pathlib import Path

from torch.utils.data import Subset

from downstream import extended_distributed as distributed
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


HORIZON = 100
POPULATION_KEY = "images"
RECIPE_KEY = "finetune_recipe"
CONTINUATION = ImageNetContinuation
data = _data


def api_module():
    from downstream import imagenet

    return imagenet


def model(cfg, device, api):
    builder = (
        api.build_trainable_backbone
        if cfg["adaptation"] == "finetune"
        else api.build_frozen_backbone
    )
    return api.ImageClassifier(
        builder(cfg["backbone"], device),
        cfg["adaptation"],
        reader_profile=cfg.get("reader_profile"),
    ).to(device)


def mix(images, labels, classes, recipe):
    return api_module().imagenet_finetune.mix(images, labels, classes, recipe)


def describe(cfg, recipe, preprocessing):
    result = {"finetune_recipe": recipe} if recipe is not None else {}
    if preprocessing is not None:
        result["preprocessing"] = preprocessing
    return result


def run(cfg, out, *, device_override=None):
    import sys

    from downstream.classification_execution import run as execute

    return execute(cfg, out, sys.modules[__name__], device_override=device_override)
