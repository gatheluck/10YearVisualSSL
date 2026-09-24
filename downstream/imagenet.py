"""Online ImageNet LP/AP components with explicitly verified provider readout.

FT model composition is available for parity tests, but the FT execution recipe
is refused while its augmentation interpretation remains unresolved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import RandomResizedCrop, InterpolationMode
from torchvision.transforms import functional as TF

from downstream import contract
from downstream.attention import QueryReader, task_spatial_features, clip_attentive_gradients, validate_adaptation
from downstream.spatial_backbones import build_frozen_backbone, supports_image_classification
from downstream.optimization import (resolve_optimization, build_optimizer,
    require_training_batches, build_task_scheduler, task_schedule_report, REFERENCE_SCHEDULE)
from downstream.ssv2 import accuracy, resolve_device, make_deterministic

TASK = "imagenet_classification"
NUM_CLASSES = 1000
METRIC_NAMES = {"top1": "imagenet_top1", "top5": "imagenet_top5", "images": None,
                "epochs": "epochs_completed"}


def validate_config(cfg):
    required = {"task", "seed", "device", "data_root", "profile", "adaptation",
                "optimizer_profile", "backbone", "probe"}
    if set(cfg) - (required | {"scheduler_profile"}) or required - set(cfg):
        raise ValueError("ImageNet config has missing or unknown fields")
    if cfg["profile"] != "capture_basic5_components" or cfg["adaptation"] not in ("frozen", "attentive"):
        raise ValueError("ImageNet supports frozen/attentive components; FT augmentation is unresolved")
    if not supports_image_classification(cfg["backbone"].get("kind")):
        raise ValueError("ImageNet requires an explicitly verified classification provider")
    validate_adaptation(cfg)
    if cfg["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    settings = cfg["probe"]
    keys = {"epochs", "batch_size", "lr", "num_workers", "image_size", "max_train_samples",
            "max_val_samples", "max_steps_per_epoch"}
    if set(settings) != keys:
        raise ValueError("ImageNet probe has missing or unknown fields")
    for key in keys - {"lr"}:
        if type(settings[key]) is not int or settings[key] < (1 if key in ("epochs", "batch_size", "image_size") else 0):
            raise ValueError(f"invalid integer probe setting: {key}")
    if "scheduler_profile" in cfg and cfg["scheduler_profile"] != REFERENCE_SCHEDULE:
        raise ValueError("ImageNet requires basic5_reference_schedule_v1")
    resolve_optimization(cfg, TASK)


class ImageNetImages(ImageFolder):
    """Captured bilinear geometry and normalization; no FT augmentation guess."""
    def __init__(self, root, split, image_size):
        if split not in ("train", "val"):
            raise ValueError("ImageNet split must be train or val")
        super().__init__(Path(root) / split)
        self.train = split == "train"
        self.image_size = image_size

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path)
        if self.train:
            image = TF.resized_crop(image, *RandomResizedCrop.get_params(image, (.08, 1.), (.75, 4./3.)),
                                    [self.image_size]*2, InterpolationMode.BILINEAR)
            if torch.rand(1).item() < .5:
                image = TF.hflip(image)
        else:
            image = TF.center_crop(TF.resize(image, 256, InterpolationMode.BILINEAR), [self.image_size]*2)
        return TF.normalize(TF.to_tensor(image), [.485, .456, .406], [.229, .224, .225]), label


class ImageClassifier(nn.Module):
    """Verified final patch mean/L2 or query reader, with zero initialized head."""
    def __init__(self, backbone, adaptation):
        super().__init__()
        if adaptation not in ("frozen", "attentive", "finetune"):
            raise ValueError("invalid classification adaptation")
        self.backbone, self.adaptation = backbone, adaptation
        self.reader = QueryReader(backbone.out_channels) if adaptation == "attentive" else None
        self.classifier = nn.Linear(512 if self.reader is not None else getattr(backbone, "global_channels", backbone.out_channels), NUM_CLASSES)
        std = getattr(backbone, "classifier_init_std", 0.)
        if std:
            nn.init.normal_(self.classifier.weight, std=std)
        else:
            nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, images):
        if callable(getattr(self.backbone, "classification_features", None)):
            from contextlib import nullcontext
            with nullcontext() if self.adaptation == "finetune" else torch.no_grad():
                features = self.backbone.classification_features(images, adaptation=self.adaptation)
            return self.classifier(features)
        tokens = task_spatial_features(self.backbone, images, adaptation=self.adaptation).float().flatten(2).transpose(1, 2)
        features = self.reader(tokens) if self.reader is not None else F.normalize(tokens.mean(1), dim=-1)
        return self.classifier(features)


def run(cfg, out, *, device_override=None):
    validate_config(cfg)
    device = resolve_device(device_override or cfg["device"])
    make_deterministic(cfg["seed"])
    settings, adaptation = cfg["probe"], cfg["adaptation"]
    report = resolve_optimization(cfg, TASK)
    train = ImageNetImages(cfg["data_root"], "train", settings["image_size"])
    val = ImageNetImages(cfg["data_root"], "val", settings["image_size"])
    if train.class_to_idx != val.class_to_idx:
        raise ValueError("ImageNet train/val class mapping differs")
    if len(train.classes) > NUM_CLASSES:
        raise ValueError("ImageNet has more than 1000 classes")
    classes = train.classes
    if settings["max_train_samples"]:
        train = Subset(train, range(min(len(train), settings["max_train_samples"])))
    if settings["max_val_samples"]:
        val = Subset(val, range(min(len(val), settings["max_val_samples"])))
    loader = DataLoader(train, batch_size=settings["batch_size"], shuffle=True,
                        drop_last=True, num_workers=settings["num_workers"],
                        generator=torch.Generator().manual_seed(cfg["seed"]))
    validation = DataLoader(val, batch_size=settings["batch_size"], num_workers=settings["num_workers"])
    require_training_batches(loader, report)
    model = ImageClassifier(build_frozen_backbone(cfg["backbone"], device), adaptation).to(device)
    optimizer = build_optimizer(model, report)
    scheduler = build_task_scheduler(optimizer, report, len(loader))
    cap = settings["max_steps_per_epoch"]
    for _ in range(settings["epochs"]):
        model.train()
        for step, (images, labels) in enumerate(loader):
            if cap and step >= cap:
                break
            loss = F.cross_entropy(model(images.to(device)), labels.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite ImageNet loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_attentive_gradients(model, adaptation)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
    model.eval()
    totals, count = [0., 0.], 0
    with torch.no_grad():
        for images, labels in validation:
            scores = accuracy(model(images.to(device)), labels.to(device))
            count += len(labels)
            totals = [total + score*len(labels) for total, score in zip(totals, scores)]
    if not count:
        raise ValueError("ImageNet validation is empty")
    if scheduler is not None:
        report["schedule"] = task_schedule_report(scheduler, report, len(loader), cap)
    raw = dict(top1=totals[0]/count, top5=totals[1]/count, images=count, epochs=settings["epochs"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    contract.write_metrics(out, raw, METRIC_NAMES)
    (out / "results.json").write_text(json.dumps(dict(task=TASK, profile=cfg["profile"],
        adaptation=adaptation, backbone=cfg["backbone"], num_classes=NUM_CLASSES,
        observed_classes=classes, optimization=report, final=raw,
        canonical_eligible=False, record_value=False), indent=2, sort_keys=True)+"\n")
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    args = parser.parse_args(argv)
    config_bytes = Path(args.config).read_bytes()
    cfg = json.loads(config_bytes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    now = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    started, error = now(), None
    try:
        run(cfg, out, device_override=args.device)
        status = "ok"
    except Exception:
        status, error = "failed", traceback.format_exc(limit=8).strip()
    contract.write_manifest(out, task=TASK, method_ref=str(cfg.get("backbone", {}).get("encoder") or "random-smoke"),
        status=status, config_sha256=contract.sha256_bytes(config_bytes), started_at=started,
        finished_at=now(), seed=cfg.get("seed", 0), backbone=cfg.get("backbone", {}), error=error)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
