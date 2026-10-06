"""Named captured ImageNet FT components; protocol conflicts stay explicit."""

import math
import random

import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# Geometry, erase domain/value and RNG/mixing behavior are source-owned.
PROFILES = {
    "captured_bicubic_random_python_v1": ("bicubic", "raw", "random", "python"),
    "captured_bicubic_half_python_v1": ("bicubic", "raw", 0.5, "python"),
    "captured_bilinear_plain_torch_v1": ("bilinear", "none", None, "torch"),
    "captured_bilinear_raw_zero_torch_v1": ("bilinear", "raw", 0, "torch"),
    "captured_bilinear_normalized_zero_torch_v1": (
        "bilinear",
        "normalized",
        0,
        "torch",
    ),
    "captured_bicubic_unit_mixup_v1": ("bicubic", "raw", "random", "mixup"),
}


def resolve(cfg):
    from downstream.spatial_backbones import imagenet_finetune_recipe

    profile = cfg.get("finetune_recipe")
    expected = imagenet_finetune_recipe(cfg["backbone"].get("kind"))
    if (
        cfg.get("adaptation") != "finetune"
        or profile not in PROFILES
        or profile != expected
    ):
        raise ValueError("ImageNet FT requires its provider's explicit captured recipe")
    interpolation, domain, value, mixing = PROFILES[profile]
    return {
        "profile": profile,
        "interpolation": interpolation,
        "erase_domain": domain,
        "erase_value": value,
        "batch_mixing": mixing,
        "randaugment": domain != "none",
        "erasing_probability": 0.25 if domain != "none" else 0.0,
        "mixup_alpha": 0.8,
        "cutmix_alpha": 0.0 if mixing == "mixup" else 1.0,
        "label_smoothing": 0.0,
        "documented_label_smoothing": 0.1,
        "legacy_unit_repair": mixing == "mixup",
        "gradient_clip_norm": 1.0,
        "protocol_conflict": True,
        "historical_run_verified": False,
    }


def transform(image, train, image_size, recipe):
    interpolation = (
        T.InterpolationMode.BICUBIC
        if recipe["interpolation"] == "bicubic"
        else T.InterpolationMode.BILINEAR
    )
    if train:
        image = TF.resized_crop(
            image,
            *T.RandomResizedCrop.get_params(image, [0.08, 1.0], [0.75, 4.0 / 3.0]),
            [image_size] * 2,
            interpolation,
        )
        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
        if recipe["randaugment"]:
            image = T.RandAugment(num_ops=2, magnitude=9)(image)
    else:
        image = TF.center_crop(TF.resize(image, [256], interpolation), [image_size] * 2)
    tensor = T.ToTensor()(image)
    erase = T.RandomErasing(
        p=recipe["erasing_probability"], value=recipe["erase_value"] or 0
    )
    if train and recipe["erase_domain"] == "raw":
        tensor = erase(tensor)
    if recipe["legacy_unit_repair"]:
        # Preserve the inspected wrapper's whole-image heuristic, including its
        # response to negative random-erasing pixels. This is not a correction.
        if float(tensor.min()) < -1e-4:
            tensor = tensor * 0.5 + 0.5
        tensor = tensor.clamp(0.0, 1.0)
    tensor = TF.normalize(tensor, MEAN, STD)
    if train and recipe["erase_domain"] == "normalized":
        tensor = erase(tensor)
    return tensor


def mix(images, labels, num_classes, recipe):
    """Preserve the captured RNG order, clipped CutMix area and unsmoothed labels."""
    mode = recipe["batch_mixing"]
    if mode == "python":
        cutmix = random.random() >= 0.5
        alpha = 1.0 if cutmix else 0.8
        lam = float(np.random.beta(alpha, alpha))
        perm = torch.randperm(images.size(0), device=images.device)
    elif mode == "torch":
        torch.rand(1)  # Captured probability=1 application draw.
        cutmix = torch.rand(1).item() < 0.5
        perm = torch.randperm(images.size(0), device=images.device)
        alpha = 1.0 if cutmix else 0.8
        lam = float(torch.distributions.Beta(alpha, alpha).sample())
    else:
        cutmix = False
        lam = float(torch.distributions.Beta(0.8, 0.8).sample())
        perm = torch.randperm(images.size(0), device=images.device)
    if cutmix:
        h, w = images.shape[-2:]
        ratio = math.sqrt(1.0 - lam)
        cw, ch = int(w * ratio), int(h * ratio)
        if mode == "python":
            cx, cy = random.randint(0, w), random.randint(0, h)
        else:
            cx, cy = int(torch.randint(0, w, (1,))), int(torch.randint(0, h, (1,)))
        x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
        y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (h * w))
    else:
        images = images * lam + images[perm] * (1.0 - lam)
    targets = F.one_hot(labels, num_classes).float()
    return images, targets * lam + targets[perm] * (1.0 - lam)
