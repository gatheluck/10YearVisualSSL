"""ADE20K semantic segmentation on a frozen backbone (downstream pilot).

    python -m downstream.ade20k --config <resolved.json> --out <dir>

A port of the capture's `downstream/ade20k_segmentation.py`, wired to this repo's
downstream contract (`downstream.contract`) instead of the capture's cluster-side
results-registry. It freezes a backbone (a method's trained `encoder.pt`, or a
random tiny ViT for the hermetic smoke), attaches a single 1x1-conv readout head,
trains only the head, and reports **mIoU** and pixel accuracy.

Faithful to the capture: 150 classes, ignore index 255 (mask label 0 is
"unlabelled" -> ignore, other labels shift down by one), a confusion-matrix mIoU,
and `max_*_samples` / `max_steps_per_epoch` subsetting that stamps the result
`record_value: false` so a smoke number can never pass as a real one. Changed for
the port: the device is resolved (not assumed CUDA), the run is seeded, and the
result is the contract's manifest + metrics rather than a registry JSON.
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from downstream import contract                                    # noqa: E402
from downstream.spatial_backbones import build_frozen_backbone, KINDS  # noqa: E402

TASK = "ade20k_segmentation"
NUM_CLASSES = 150
IGNORE_INDEX = 255
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

TOP_KEYS = frozenset({"task", "seed", "device", "data_root", "backbone", "probe"})
BACKBONE_REQUIRED = frozenset({"kind", "encoder", "arch", "img_size", "patch_size"})
BACKBONE_OPTIONAL = frozenset({"embed_dim", "depth", "num_heads"})
PROBE_KEYS = frozenset({"epochs", "batch_size", "lr", "num_workers", "image_size",
                        "max_train_samples", "max_val_samples",
                        "max_steps_per_epoch"})
DEVICES = ("auto", "cuda", "cpu")
METRIC_NAMES = {"miou": "ade20k_miou", "pacc": "ade20k_pixel_accuracy",
                "epochs": "epochs_completed", "metrics_unavailable": "metrics_unavailable"}


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
    probe = cfg["probe"]
    if not isinstance(probe, dict):
        raise ConfigError("config: probe is not a mapping")
    _named(PROBE_KEYS - set(probe), set(probe) - PROBE_KEYS, "config.probe")


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


class ADE20kSegmentation(Dataset):
    def __init__(self, root: Path, split: str, image_size: int):
        if split not in {"training", "validation"}:
            raise ValueError(f"unknown ADE20k split: {split}")
        self.image_size = image_size
        image_dir = Path(root) / "images" / split
        mask_dir = Path(root) / "annotations" / split
        self.images = sorted(image_dir.glob("*.jpg"))
        if not self.images:
            raise FileNotFoundError(f"no ADE20k images under {image_dir}")
        self.masks = [mask_dir / f"{img.stem}.png" for img in self.images]
        missing = [str(m) for m in self.masks if not m.exists()]
        if missing:
            raise FileNotFoundError(f"missing ADE20k masks, first={missing[0]}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index])
        image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
        mask = TF.resize(mask, [self.image_size, self.image_size],
                         interpolation=TF.InterpolationMode.NEAREST)
        image_tensor = (TF.to_tensor(image) - IMAGENET_MEAN) / IMAGENET_STD
        target = torch.from_numpy(np.array(mask, dtype=np.int64))
        target[target == 0] = IGNORE_INDEX
        valid = target != IGNORE_INDEX
        target[valid] -= 1
        return image_tensor, target


def _subset(dataset: Dataset, maximum: int):
    if maximum and maximum < len(dataset):
        return Subset(dataset, list(range(maximum)))
    return dataset


class FrozenSegModel(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Conv2d(backbone.out_channels, num_classes, kernel_size=1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.backbone.forward_features(x)
        logits = self.head(feat)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear",
                             align_corners=False)


def _train_one_epoch(model, loader, optimizer, device, max_steps) -> float:
    model.train()
    total, steps = 0.0, 0
    for step, (images, targets) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite ADE20k train loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
        steps += 1
    return total / max(steps, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    total = correct = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        pred = model(images).argmax(dim=1)
        valid = targets != IGNORE_INDEX
        if not valid.any():
            continue
        p = pred[valid].view(-1).cpu()
        t = targets[valid].view(-1).cpu()
        total += int(t.numel())
        correct += int((p == t).sum())
        conf += torch.bincount(t * NUM_CLASSES + p,
                               minlength=NUM_CLASSES * NUM_CLASSES).view(
            NUM_CLASSES, NUM_CLASSES)
    inter = conf.diag().float()
    union = conf.sum(1).float() + conf.sum(0).float() - inter
    present = union > 0
    miou = float((inter[present] / union[present].clamp_min(1)).mean()) \
        if present.any() else 0.0
    return {"miou": miou * 100.0, "pacc": (correct / max(total, 1)) * 100.0}


def run(cfg: dict, out: Path, device_override: str | None = None) -> dict:
    validate_config(cfg)
    device = resolve_device(device_override or cfg["device"])
    seed = int(cfg["seed"])
    make_deterministic(seed)
    probe = cfg["probe"]
    image_size = int(probe["image_size"])

    root = Path(cfg["data_root"])
    train_ds = _subset(ADE20kSegmentation(root, "training", image_size),
                       int(probe["max_train_samples"]))
    val_ds = _subset(ADE20kSegmentation(root, "validation", image_size),
                     int(probe["max_val_samples"]))
    bs, nw = int(probe["batch_size"]), int(probe["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              drop_last=False,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)

    backbone = build_frozen_backbone(cfg["backbone"], device)
    model = FrozenSegModel(backbone).to(device)
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError("backbone is not frozen")
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=float(probe["lr"]),
                                  weight_decay=0.01)

    subset_mode = bool(probe["max_train_samples"] or probe["max_val_samples"]
                       or probe["max_steps_per_epoch"])
    epochs = int(probe["epochs"])
    max_steps = int(probe["max_steps_per_epoch"]) or None
    print(f"ADE20k seg  device={device}  backbone={cfg['backbone']['kind']}"
          f"({'trained' if cfg['backbone'].get('encoder') else 'random (smoke)'})"
          f"  epochs={epochs}  out_channels={backbone.out_channels}")
    metrics = {"miou": 0.0, "pacc": 0.0}
    for epoch in range(epochs):
        loss = _train_one_epoch(model, train_loader, optimizer, device, max_steps)
        metrics = evaluate(model, val_loader, device)
        print(f"[{epoch + 1}/{epochs}] loss={loss:.4f} "
              f"mIoU={metrics['miou']:.3f} pACC={metrics['pacc']:.3f}")

    raw = {"miou": float(metrics["miou"]), "pacc": float(metrics["pacc"]),
           "epochs": epochs}
    contract.write_metrics(out, raw, METRIC_NAMES)
    (Path(out) / "results.json").write_text(
        json.dumps({"task": TASK, "backbone": cfg["backbone"],
                    "num_classes": NUM_CLASSES, "ignore_index": IGNORE_INDEX,
                    "epochs": epochs, "final": raw,
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
        # A refused config is misuse, not a run result: no manifest, exit 2.
        return 2
    except Exception:                       # a run failure is a result
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
