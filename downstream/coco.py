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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from downstream import contract                                    # noqa: E402
from downstream.spatial_backbones import build_frozen_backbone, KINDS  # noqa: E402

TASK = "coco_detection"
NUM_CLASSES = 91          # COCO category ids run 1..90; index 0 is background.

TOP_KEYS = frozenset({"task", "seed", "device", "data_root", "backbone", "detector"})
BACKBONE_REQUIRED = frozenset({"kind", "encoder", "arch", "img_size", "patch_size"})
BACKBONE_OPTIONAL = frozenset({"embed_dim", "depth", "num_heads"})
DETECTOR_KEYS = frozenset({"epochs", "batch_size", "lr", "num_workers", "min_size",
                           "max_size", "anchor_sizes", "max_train_samples",
                           "max_val_samples", "max_steps_per_epoch"})
DEVICES = ("auto", "cuda", "cpu")
METRIC_NAMES = {"bbox_mAP": "coco_map", "bbox_mAP_50": "coco_map_50",
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
    _named(TOP_KEYS - set(cfg), set(cfg) - TOP_KEYS, "config")
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


def build_frozen_detector(backbone_spec: dict, detector: dict,
                          device: "torch.device") -> FasterRCNN:
    backbone = build_frozen_backbone(backbone_spec, device)
    anchor_sizes = tuple(int(s) for s in detector["anchor_sizes"])
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


def _train_one_epoch(model, loader, optimizer, device, max_steps) -> float:
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
        optimizer.step()
        total += float(loss.detach().cpu())
        steps += 1
    return total / max(steps, 1)


@torch.no_grad()
def evaluate(model, loader, dataset, device) -> dict:
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
    if not results:
        return {"bbox_mAP": 0.0, "bbox_mAP_50": 0.0, "detections": 0}
    evaluator = COCOeval(coco, coco.loadRes(results), "bbox")
    evaluator.params.imgIds = sorted(set(image_ids))
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {"bbox_mAP": float(evaluator.stats[0]),
            "bbox_mAP_50": float(evaluator.stats[1]), "detections": len(results)}


def run(cfg: dict, out: Path, device_override: str | None = None) -> dict:
    validate_config(cfg)
    device = resolve_device(device_override or cfg["device"])
    seed = int(cfg["seed"])
    make_deterministic(seed)
    detector = cfg["detector"]

    root = Path(cfg["data_root"])
    train_ds = _subset(CocoDetectionForFRCNN(
        str(root / "images/train2017"),
        str(root / "annotations/instances_train2017.json")),
        int(detector["max_train_samples"]))
    val_ds = _subset(CocoDetectionForFRCNN(
        str(root / "images/val2017"),
        str(root / "annotations/instances_val2017.json")),
        int(detector["max_val_samples"]))
    bs, nw = int(detector["batch_size"]), int(detector["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              collate_fn=collate,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=nw,
                            collate_fn=collate)

    model = build_frozen_detector(cfg["backbone"], detector, device)
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError("backbone is not frozen")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=float(detector["lr"]), momentum=0.9,
                                weight_decay=0.0005)

    subset_mode = bool(detector["max_train_samples"] or detector["max_val_samples"]
                       or detector["max_steps_per_epoch"])
    epochs = int(detector["epochs"])
    max_steps = int(detector["max_steps_per_epoch"]) or None
    print(f"COCO det  device={device}  backbone={cfg['backbone']['kind']}"
          f"({'trained' if cfg['backbone'].get('encoder') else 'random (smoke)'})"
          f"  epochs={epochs}")
    metrics = {"bbox_mAP": 0.0, "bbox_mAP_50": 0.0, "detections": 0}
    for epoch in range(epochs):
        loss = _train_one_epoch(model, train_loader, optimizer, device, max_steps)
        metrics = evaluate(model, val_loader, val_ds, device)
        print(f"[{epoch + 1}/{epochs}] loss={loss:.4f} "
              f"mAP={metrics['bbox_mAP']:.4f} mAP50={metrics['bbox_mAP_50']:.4f}")

    raw = {"bbox_mAP": float(metrics["bbox_mAP"]),
           "bbox_mAP_50": float(metrics["bbox_mAP_50"]),
           "detections": int(metrics["detections"]), "epochs": epochs}
    contract.write_metrics(out, raw, METRIC_NAMES)
    (Path(out) / "results.json").write_text(
        json.dumps({"task": TASK, "backbone": cfg["backbone"],
                    "num_classes": NUM_CLASSES, "epochs": epochs, "final": raw,
                    "record_value": not subset_mode,
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
