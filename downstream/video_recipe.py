"""Inspected clip geometry and FT augmentation; historical attribution is separate."""

import itertools
import random
from typing import cast

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from downstream import imagenet_finetune
from downstream.spatial_backbones import video_recipe_policy

PROFILE = "captured_provider_video_v1"


def resolve(cfg):
    policy = video_recipe_policy(cfg["backbone"].get("kind"))
    if (
        cfg.get("video_recipe") != PROFILE
        or policy is None
        or cfg.get("profile") != "capture_basic5_components"
        or cfg.get("task") != "ssv2_video"
    ):
        raise ValueError("explicit inspected video recipe required")
    if cfg.get("adaptation") not in ("frozen", "attentive", "finetune") or cfg.get(
        "optimizer_profile"
    ) not in ("basic5_frozen_v1", "basic5_finetune_v1"):
        raise ValueError(
            "video recipe requires explicit adaptation and Basic5 optimization"
        )
    settings = cfg["probe"]
    for key, minimum in (
        ("num_frames", 1),
        ("image_size", 1),
        ("num_workers", 0),
        ("max_train_samples", 0),
        ("max_val_samples", 0),
    ):
        if type(settings[key]) is not int or settings[key] < minimum:
            raise ValueError("invalid video recipe input dimensions or population")
    if settings["max_steps_per_epoch"] or not 1 <= settings["epochs"] <= 50:
        raise ValueError(
            "video recipe requires complete epochs within the 50-epoch horizon"
        )
    if cfg["backbone"].get("arch") == "released" and (
        cfg["probe"]["image_size"] != 224 or cfg["probe"]["num_frames"] != 16
    ):
        raise ValueError("released video recipe requires 16 frames at 224 pixels")
    ft = cfg.get("adaptation") == "finetune"
    return dict(
        policy,
        profile=PROFILE,
        augmentation=policy["augmentation"] if ft else "plain",
        batch_mixing=policy["batch_mixing"] if ft else "none",
        horizontal_flip=False,
        label_smoothing=0.0,
        documented_label_smoothing=0.1,
        historical_run_verified=False,
        protocol_conflict=True,
    )


def sample_indices(count, segments, train, recipe):
    if any(type(x) is not int or x <= 0 for x in (count, segments)):
        raise ValueError("frame and segment counts must be positive integers")
    bounds = [round(i * count / segments) for i in range(segments + 1)]
    indices = []
    for start, stop in itertools.pairwise(bounds):
        end = min(max(stop, start + 1), count)
        start = min(start, end - 1)
        if not train:
            index = (start + end - 1) // 2
        elif recipe["temporal_rng"] == "python":
            index = random.randint(start, end - 1)
        else:
            index = int(torch.randint(start, end, (1,)))
        indices.append(index)
    return indices


def transform(frames, train, size, recipe):
    mode = recipe["augmentation"] if train else "plain"
    interpolation = (
        T.InterpolationMode.BICUBIC
        if recipe["interpolation"] == "bicubic"
        else T.InterpolationMode.BILINEAR
    )
    crop = (
        T.RandomResizedCrop.get_params(frames[0], [0.08, 1.0], [0.75, 4.0 / 3.0])
        if train
        else None
    )
    aug = T.RandAugment(num_ops=2, magnitude=9) if mode != "plain" else None
    raw_shared = mode in ("shared_random", "shared_half", "shared_unit")
    erase = (
        T.RandomErasing(p=0.25, value=0.5 if mode == "shared_half" else "random")
        if raw_shared
        else None
    )
    rae_erase = mode == "shared_zero" and torch.rand(1).item() < 0.25
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    state = torch.get_rng_state()
    images = []
    for image in frames:
        if crop is not None:
            top, left, height, width = crop
            image = TF.resized_crop(
                image, top, left, height, width, [size] * 2, interpolation
            )
        else:
            image = TF.center_crop(TF.resize(image, [256], interpolation), [size] * 2)
        if aug is not None:
            if mode == "shared_zero":
                random.setstate(python_state)
                np.random.set_state(numpy_state)
            torch.set_rng_state(state)
            image = aug(image)
        images.append(image)
    erase_state = torch.get_rng_state()
    tensors = []
    for image in images:
        tensor = TF.to_tensor(cast(Image.Image, image))
        if erase is not None:
            torch.set_rng_state(erase_state)
            tensor = erase(tensor)
        elif rae_erase:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(state)
            tensor = T.RandomErasing(p=1.0, value=0)(tensor)
        tensors.append(tensor)
    clip = torch.stack(tensors)
    if mode == "shared_unit":
        if float(clip.min()) < -1e-4:
            clip = clip * 0.5 + 0.5
        clip = clip.clamp(0.0, 1.0)
    clip = TF.normalize(clip, imagenet_finetune.MEAN, imagenet_finetune.STD)
    if mode == "independent_zero" and torch.rand(1).item() < 0.25:
        erase = T.RandomErasing(p=1.0)
        clip = torch.stack([erase(frame) for frame in clip])
    return clip


def mix(clips, labels, classes, recipe):
    if clips.ndim != 5:
        raise ValueError("video mixing requires B,T,C,H,W clips")
    if recipe["batch_mixing"] == "none":
        raise ValueError("video mixing requires FT")
    b, t, c, h, w = clips.shape
    mixed, targets = imagenet_finetune.mix(
        clips.reshape(b, t * c, h, w), labels, classes, recipe
    )
    return mixed.reshape(b, t, c, h, w), targets
