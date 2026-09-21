"""COCO object detection on a frozen backbone (downstream task 2).

    python -m downstream.coco --config <resolved.json> --out <dir>

A port of the capture's `downstream/coco_frcnn.py`, wired to this repo's downstream
contract (`downstream.contract`). It freezes a backbone (a method's trained
`encoder.pt`, or a random tiny ViT for the hermetic smoke), attaches a torchvision
**Faster R-CNN** RPN/ROI head on its single spatial feature map, trains only the
head, and reports COCO **bbox mAP** (and mAP@50) via `pycocotools`.

Faithful to the capture: Faster R-CNN over a single feature map (`featmap_names
["0"]`, `size_divisible = patch_size`), 91 classes, SGD, `pycocotools` COCOeval,
and `max_*_samples` / `max_steps_per_epoch` subsetting that stamps the result
`record_value: false`. Changed for the port: the device is resolved (not assumed
CUDA), the run is seeded, the result is the contract's manifest + metrics, and the
anchor sizes are a config key so the hermetic smoke can use small anchors on a
tiny feature map while a real run keeps the paper's `(32,64,128,256,512)`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from downstream.optimization import resolve_optimization, build_optimizer, require_training_batches
from downstream.optimization import build_coco_scheduler
from downstream.attention import (SpatialAdapter, task_spatial_features,
                                  validate_adaptation, clip_attentive_gradients)
from downstream import contract                                    # noqa: E402
from downstream.spatial_backbones import build_frozen_backbone, build_trainable_backbone, KINDS  # noqa: E402

TASK = "coco_detection"
NUM_CLASSES = 91          # COCO category ids run 1..90; index 0 is background.

TOP_KEYS = frozenset({"task", "seed", "device", "data_root", "backbone", "detector"})
CAPTURE_PROFILE = "capture_basic5_components"
BACKBONE_REQUIRED = frozenset({"kind", "encoder", "arch", "img_size", "patch_size"})
BACKBONE_OPTIONAL = frozenset({"embed_dim", "depth", "num_heads"})
DETECTOR_KEYS = frozenset({"epochs", "batch_size", "lr", "num_workers", "min_size",
                           "max_size", "anchor_sizes", "max_train_samples",
                           "max_val_samples", "max_steps_per_epoch"})
DEVICES = ("auto", "cuda", "cpu")
METRIC_NAMES = {"bbox_mAP": "coco_map", "bbox_mAP_50": "coco_map_50",
                "bbox_mAP_75": "coco_map_75", "bbox_mAP_small": "coco_map_small",
                "bbox_mAP_medium": "coco_map_medium", "bbox_mAP_large": "coco_map_large",
                "detections": None, "epochs": "epochs_completed",
                "metrics_unavailable": "metrics_unavailable"}


class ConfigError(Exception):
    """A refusal, always naming what was refused."""


def _named(missing, unknown, where: str) -> None:
    if missing:
        raise ConfigError(f"{where}: missing {', '.join(sorted(missing))}")
    if unknown:
        raise ConfigError(f"{where}: unknown {', '.join(sorted(unknown))}")


def validate_config(cfg: dict) -> None:
    for key in ("output", "out", "result_dir"):
        if key in cfg:
            raise ConfigError(
                f"config: {key} is set; the output location is fixed at --out")
    _named(TOP_KEYS - set(cfg), set(cfg) - (TOP_KEYS | {"profile", "adaptation", "optimizer_profile", "scheduler_profile"}), "config")
    try:
        validate_adaptation(cfg)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if cfg.get("profile", "legacy") not in ("legacy", CAPTURE_PROFILE):
        raise ConfigError("config.profile: expected legacy or capture_basic5_components")
    if cfg["device"] not in DEVICES:
        raise ConfigError(f"config: device is {cfg['device']!r}; expected "
                          f"{', '.join(DEVICES)}")
    backbone = cfg["backbone"]
    if not isinstance(backbone, dict):
        raise ConfigError("config: backbone is not a mapping")
    _named(BACKBONE_REQUIRED - set(backbone),
           set(backbone) - (BACKBONE_REQUIRED | BACKBONE_OPTIONAL),
           "config.backbone")
    if backbone["kind"] not in KINDS:
        raise ConfigError(
            f"config.backbone: kind is {backbone['kind']!r}; ported kinds are "
            f"{', '.join(KINDS)}")
    detector = cfg["detector"]
    if not isinstance(detector, dict):
        raise ConfigError("config: detector is not a mapping")
    _named(DETECTOR_KEYS - set(detector), set(detector) - DETECTOR_KEYS,
           "config.detector")
    try:
        resolve_optimization(cfg, TASK)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def resolve_device(spec: str) -> "torch.device":
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device is 'cuda' but no CUDA device is visible. Ask for 'auto' "
                "to accept a CPU; getting a CPU silently would misreport what ran")
        return torch.device("cuda")
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unknown device {spec!r}; expected auto, cuda or cpu")


def make_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)


def collate(batch):
    return tuple(zip(*batch))


class CocoDetectionForFRCNN(CocoDetection):
    """COCO in the (image_tensor, target-dict) shape torchvision detection wants."""

    def __init__(self, *args, train: bool = False, profile: str = "legacy", **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = train and profile == CAPTURE_PROFILE

    def __getitem__(self, index: int):
        image, anns = super().__getitem__(index)
        image_id = self.ids[index]
        boxes, labels, area, iscrowd = [], [], [], []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            area.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))
        box_tensor = (torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
                      if boxes else torch.zeros((0, 4), dtype=torch.float32))
        if self.augment and torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            box_tensor[:, [0, 2]] = image.width - box_tensor[:, [2, 0]]
        target = {
            "boxes": box_tensor,
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.as_tensor(area, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        return TF.to_tensor(image), target


def _subset(dataset, maximum: int):
    if maximum and maximum < len(dataset):
        return Subset(dataset, list(range(maximum)))
    return dataset


class SimpleFeaturePyramid(nn.Module):
    """Four trainable projections of a verified stride-16 spatial map."""

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.lateral4 = nn.Conv2d(in_channels, out_channels, 1)
        self.lateral8 = nn.Conv2d(in_channels, out_channels, 1)
        self.lateral16 = nn.Conv2d(in_channels, out_channels, 1)
        self.lateral32 = nn.Conv2d(in_channels, out_channels, 1)
        self.out_channels = out_channels

    def forward(self, feat):
        p16 = self.lateral16(feat)
        p8 = self.lateral8(F.interpolate(feat, scale_factor=2.0, mode="bilinear", align_corners=False))
        p4 = self.lateral4(F.interpolate(feat, scale_factor=4.0, mode="bilinear", align_corners=False))
        p32 = self.lateral32(F.max_pool2d(feat, kernel_size=2, stride=2))
        return OrderedDict([("0", p4), ("1", p8), ("2", p16), ("3", p32)])


class FrozenPyramidBackbone(nn.Module):
    """Register the frozen encoder separately from the trainable detection head."""

    def __init__(self, body, *, adaptation: str = "frozen"):
        super().__init__()
        self.body = body
        self.adaptation = adaptation
        self.adapter = SpatialAdapter(body.out_channels) if adaptation == "attentive" else None
        self.fpn = SimpleFeaturePyramid(body.out_channels)
        self.out_channels = self.fpn.out_channels

    def forward(self, images):
        feat = task_spatial_features(self.body, images, self.adapter, adaptation=self.adaptation)
        expected = (images.shape[-2] // 16, images.shape[-1] // 16)
        if feat.ndim != 4 or tuple(feat.shape[-2:]) != expected:
            raise RuntimeError("pyramid body must produce a stride-16 spatial grid")
        return self.fpn(feat)


def build_frozen_detector(backbone_spec: dict, detector: dict,
                          device: "torch.device", *, profile: str = "legacy",
                          adaptation: str = "frozen") -> FasterRCNN:
    validate_adaptation({"profile": profile, "adaptation": adaptation})
    captured = profile == CAPTURE_PROFILE
    if captured:
        from downstream.spatial_backbones import supports_capture_pyramid
        if not supports_capture_pyramid(backbone_spec["kind"]):
            raise NotImplementedError("captured pyramid requires a verified stride-16 provider")
        if int(backbone_spec["patch_size"]) != 16:
            raise ValueError("captured pyramid requires stride 16")
        if len(detector["anchor_sizes"]) != 4 or any(int(s) <= 0 for s in detector["anchor_sizes"]):
            raise ValueError("captured pyramid requires four positive anchor sizes")
    builder = build_trainable_backbone if adaptation == "finetune" else build_frozen_backbone
    backbone = builder(backbone_spec, device)
    anchor_sizes = tuple(int(s) for s in detector["anchor_sizes"])
    if captured:
        # The existing timm provider expects ImageNet normalization. Keep it in
        # FasterRCNN's transform, once, rather than transplanting another model's
        # private normalization into this generic body.
        model = FasterRCNN(
            FrozenPyramidBackbone(backbone, adaptation=adaptation), num_classes=NUM_CLASSES,
            rpn_anchor_generator=AnchorGenerator(sizes=tuple((s,) for s in anchor_sizes),
                                                aspect_ratios=((0.5, 1.0, 2.0),) * 4),
            box_roi_pool=MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"],
                                            output_size=7, sampling_ratio=2),
            size_divisible=32,
            min_size=int(detector["min_size"]), max_size=int(detector["max_size"]))
        return model.to(device)
    anchor_generator = AnchorGenerator(sizes=(anchor_sizes,),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7,
                                    sampling_ratio=2)
    model = FasterRCNN(
        backbone, num_classes=NUM_CLASSES,
        rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,
        size_divisible=int(getattr(backbone, "patch_size", 16)),
        min_size=int(detector["min_size"]), max_size=int(detector["max_size"]))
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    return model.to(device)


def _train_one_epoch(model, loader, optimizer, device, max_steps,
                     *, adaptation="frozen", scheduler=None) -> float:
    model.train()
    total, steps = 0.0, 0
    for step, (images, targets) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses = model(images, targets)
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite COCO detection loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_attentive_gradients(model, adaptation)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += float(loss.detach().cpu())
        steps += 1
    return total / max(steps, 1)


@torch.no_grad()
def evaluate(model, loader, dataset, device, *, profile: str = "legacy") -> dict:
    from pycocotools.cocoeval import COCOeval
    model.eval()
    coco = dataset.dataset.coco if isinstance(dataset, Subset) else dataset.coco
    results, image_ids = [], []
    for images, targets in loader:
        outputs = model([img.to(device) for img in images])
        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())
            image_ids.append(image_id)
            for box, label, score in zip(output["boxes"].cpu(),
                                         output["labels"].cpu(),
                                         output["scores"].cpu()):
                x1, y1, x2, y2 = box.tolist()
                results.append({"image_id": image_id, "category_id": int(label),
                                "bbox": [x1, y1, max(0.0, x2 - x1),
                                         max(0.0, y2 - y1)],
                                "score": float(score)})
    names = ("bbox_mAP", "bbox_mAP_50")
    if profile == CAPTURE_PROFILE:
        names += ("bbox_mAP_75", "bbox_mAP_small", "bbox_mAP_medium", "bbox_mAP_large")
    if not results:
        return {**{name: 0.0 for name in names}, "detections": 0}
    evaluator = COCOeval(coco, coco.loadRes(results), "bbox")
    evaluator.params.imgIds = sorted(set(image_ids))
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {**{name: float(evaluator.stats[i]) for i, name in enumerate(names)},
            "detections": len(results)}


def run(cfg: dict, out: Path, device_override: str | None = None) -> dict:
    validate_config(cfg)
    device = resolve_device(device_override or cfg["device"])
    seed = int(cfg["seed"])
    make_deterministic(seed)
    detector = cfg["detector"]
    profile = cfg.get("profile", "legacy")
    adaptation = validate_adaptation(cfg)
    optimization = resolve_optimization(cfg, TASK)

    root = Path(cfg["data_root"])
    train_ds = _subset(CocoDetectionForFRCNN(
        str(root / "images/train2017"),
        str(root / "annotations/instances_train2017.json"), train=True, profile=profile),
        int(detector["max_train_samples"]))
    val_ds = _subset(CocoDetectionForFRCNN(
        str(root / "images/val2017"),
        str(root / "annotations/instances_val2017.json"), profile=profile),
        int(detector["max_val_samples"]))
    bs, nw = int(detector["batch_size"]), int(detector["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              drop_last=optimization is not None,
                              collate_fn=collate,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=nw,
                            collate_fn=collate)

    require_training_batches(train_loader, optimization)
    model = build_frozen_detector(cfg["backbone"], detector, device, profile=profile,
                                  adaptation=adaptation)
    encoder = model.backbone.body if profile == CAPTURE_PROFILE else model.backbone
    if adaptation != "finetune" and any(p.requires_grad for p in encoder.parameters()):
        raise RuntimeError("backbone is not frozen")
    trainable = [p for p in model.parameters() if p.requires_grad]
    if optimization is not None:
        optimizer = build_optimizer(model, optimization)
    else:
        optimizer = torch.optim.SGD(trainable, lr=float(detector["lr"]), momentum=0.9,
                                    weight_decay=0.0005)

    subset_mode = bool(detector["max_train_samples"] or detector["max_val_samples"]
                       or detector["max_steps_per_epoch"])
    epochs = int(detector["epochs"])
    max_steps = int(detector["max_steps_per_epoch"]) or None
    scheduler = (build_coco_scheduler(optimizer, len(train_loader))
                 if "scheduler_profile" in cfg else None)
    if scheduler is not None and epochs < 12:
        subset_mode = True
    print(f"COCO det  device={device}  backbone={cfg['backbone']['kind']}"
          f"({'trained' if cfg['backbone'].get('encoder') else 'random (smoke)'})"
          f"  epochs={epochs}")
    metrics = {"bbox_mAP": 0.0, "bbox_mAP_50": 0.0, "detections": 0}
    for epoch in range(epochs):
        loss = _train_one_epoch(model, train_loader, optimizer, device, max_steps,
                                adaptation=adaptation, scheduler=scheduler)
        metrics = evaluate(model, val_loader, val_ds, device, profile=profile)
        print(f"[{epoch + 1}/{epochs}] loss={loss:.4f} "
              f"mAP={metrics['bbox_mAP']:.4f} mAP50={metrics['bbox_mAP_50']:.4f}")

    if scheduler is not None:
        optimization["schedule"] = {
            "profile": cfg["scheduler_profile"], "warmup_updates": 500,
            "warmup_start_factor": .001, "milestone_epochs": [8, 11],
            "decay_factor": .1, "reference_epochs": 12,
            "steps_per_epoch": len(train_loader),
            "updates_completed": scheduler.last_epoch,
            "next_update_lr": optimizer.param_groups[0]["lr"],
            "truncated": epochs < 12 or bool(max_steps and max_steps < len(train_loader)),
        }
    raw = {"bbox_mAP": float(metrics["bbox_mAP"]),
           "bbox_mAP_50": float(metrics["bbox_mAP_50"]),
           "detections": int(metrics["detections"]), "epochs": epochs}
    if profile == CAPTURE_PROFILE:
        for key in ("bbox_mAP_75", "bbox_mAP_small", "bbox_mAP_medium", "bbox_mAP_large"):
            raw[key] = float(metrics.get(key, 0.0))
    contract.write_metrics(out, raw, METRIC_NAMES)
    (Path(out) / "results.json").write_text(
        json.dumps({"task": TASK, "backbone": cfg["backbone"],
                    "num_classes": NUM_CLASSES, "epochs": epochs, "final": raw,
                    "profile": profile, "adaptation": adaptation,
                    "canonical_eligible": False,
                    "metric_units": "ratio; -1 means undefined",
                    "record_value": not subset_mode and profile == "legacy",
                    **({"optimization": optimization} if optimization is not None else {}),
                    "subset_or_smoke": subset_mode},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return raw


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None, choices=[None, *DEVICES])
    return parser


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config_bytes = Path(args.config).read_bytes()
    cfg = json.loads(config_bytes)
    method_ref = str(cfg.get("backbone", {}).get("encoder") or "random-smoke")
    started = _now()
    error = None
    try:
        run(cfg, out, device_override=args.device)
        status = "ok"
    except ConfigError as exc:
        print(f"  *** {exc}", file=sys.stderr)
        return 2
    except Exception:
        import traceback
        error = traceback.format_exc(limit=8).strip()
        status = "failed"
    contract.write_manifest(
        out, task=TASK, method_ref=method_ref, status=status,
        config_sha256=contract.sha256_bytes(config_bytes),
        started_at=started, finished_at=_now(), seed=int(cfg.get("seed", 0)),
        backbone=cfg.get("backbone", {}), error=error)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
