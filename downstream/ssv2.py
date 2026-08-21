"""Something-Something-v2 video classification on a frozen backbone (task 4).

    python -m downstream.ssv2 --config <resolved.json> --out <dir>

A port of the capture's `downstream/ssv2_linear.py`, wired to this repo's downstream
contract (`downstream.contract`). It freezes a backbone (a method's trained
`encoder.pt`, or a random tiny ViT for the hermetic smoke), decodes a few frames
per video with **PyAV**, averages the per-frame frozen features, and fits a plain
linear classifier over the 174 SSv2 classes, reporting **top-1 / top-5**.

Faithful to the capture: uniform `num_frames` sampling, `labels/labels.json`
(bracket-stripped template -> id) + `labels/{split}.json` entries, `videos/<id>.webm`
(also the nested `20bn-something-something-v2/` layout), frame-average pooled
features + a plain linear head, top-1/top-5, and `max_*_samples` /
`max_steps_per_epoch` subsetting that stamps `record_value: false`. Changed for the
port: the device is resolved (not assumed CUDA), the run is seeded, the result is
the contract's manifest + metrics, and the capture's on-disk **feature-cache
sharding** (a cluster throughput optimisation) is dropped for a thin single-process
loop -- features are recomputed each epoch from the frozen backbone.
"""

from __future__ import annotations

import argparse
import hashlib
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

TASK = "ssv2_video"
NUM_CLASSES = 174
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

TOP_KEYS = frozenset({"task", "seed", "device", "data_root", "backbone", "probe"})
BACKBONE_REQUIRED = frozenset({"kind", "encoder", "arch", "img_size", "patch_size"})
BACKBONE_OPTIONAL = frozenset({"embed_dim", "depth", "num_heads"})
PROBE_KEYS = frozenset({"epochs", "batch_size", "lr", "num_workers", "image_size",
                        "num_frames", "max_train_samples", "max_val_samples",
                        "max_steps_per_epoch"})
DEVICES = ("auto", "cuda", "cpu")
METRIC_NAMES = {"top1": "ssv2_top1", "top5": "ssv2_top5", "videos": None,
                "epochs": "epochs_completed",
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


def normalize_template(template: str) -> str:
    return template.replace("[", "").replace("]", "")


def decode_video_frames(path: Path, num_frames: int) -> "list[Image.Image]":
    import av
    frames: "list[Image.Image]" = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_image().convert("RGB"))
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    if len(frames) > num_frames:
        positions = torch.linspace(0, len(frames) - 1,
                                   steps=num_frames).round().long().tolist()
        frames = [frames[i] for i in positions]
    if len(frames) < num_frames:
        frames = frames + [frames[-1]] * (num_frames - len(frames))
    return frames[:num_frames]


class SSV2Clips(Dataset):
    def __init__(self, root: Path, split: str, num_frames: int, image_size: int):
        if split not in {"train", "validation"}:
            raise ValueError(f"unknown SSV2 split: {split}")
        self.video_dir = Path(root) / "videos"
        self.num_frames = num_frames
        self.image_size = image_size
        labels = json.loads((Path(root) / "labels" / "labels.json").read_text())
        self.label_to_id = {name: int(idx) for name, idx in labels.items()}
        entries = json.loads(
            (Path(root) / "labels" / f"{split}.json").read_text())
        self.entry_digest = hashlib.sha256(
            json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest()
        self.samples: "list[tuple[Path, int]]" = []
        missing: "list[Path]" = []
        for entry in entries:
            label_name = normalize_template(entry["template"])
            if label_name not in self.label_to_id:
                raise KeyError(f"SSV2 label not found: {label_name}")
            video = self.video_dir / f"{entry['id']}.webm"
            if not video.exists():
                nested = (self.video_dir / "20bn-something-something-v2"
                          / f"{entry['id']}.webm")
                video = nested if nested.exists() else video
            (self.samples if video.exists() else missing).append(
                (video, self.label_to_id[label_name]) if video.exists() else video)
        if missing:
            raise FileNotFoundError(
                f"missing {len(missing)} SSV2 videos for split={split}; "
                f"first={missing[0]}")
        if not self.samples:
            raise FileNotFoundError(f"no SSV2 videos under {self.video_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        frames = []
        for image in decode_video_frames(path, self.num_frames):
            image = TF.resize(image, [self.image_size, self.image_size],
                              antialias=True)
            frames.append((TF.to_tensor(image) - IMAGENET_MEAN) / IMAGENET_STD)
        return torch.stack(frames, dim=0), torch.tensor(label, dtype=torch.long)


def _subset(dataset: Dataset, maximum: int):
    if maximum and maximum < len(dataset):
        return Subset(dataset, list(range(maximum)))
    return dataset


class FrozenFrameAverageClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.out_channels, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = clips.shape
        flat = clips.view(batch * frames, channels, height, width)
        with torch.no_grad():
            feat_map = self.backbone.forward_features(flat)
            feat = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        feat = feat.view(batch, frames, -1).mean(dim=1)
        return self.classifier(feat)


def accuracy(output: torch.Tensor, target: torch.Tensor,
             topk=(1, 5)) -> "list[float]":
    maxk = min(max(topk), output.size(1))
    _, pred = output.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    out = []
    for k in topk:
        kk = min(k, maxk)
        out.append(float(correct[:kk].reshape(-1).float().sum().mul_(
            100.0 / target.size(0)).cpu()))
    return out


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    top1 = top5 = 0.0
    total = 0
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        logits = model(clips)
        b = labels.size(0)
        a1, a5 = accuracy(logits, labels)
        top1 += a1 * b
        top5 += a5 * b
        total += b
    return {"top1": top1 / max(total, 1), "top5": top5 / max(total, 1),
            "videos": total}


def run(cfg: dict, out: Path, device_override: str | None = None) -> dict:
    validate_config(cfg)
    device = resolve_device(device_override or cfg["device"])
    seed = int(cfg["seed"])
    make_deterministic(seed)
    probe = cfg["probe"]
    root = Path(cfg["data_root"])
    num_frames, image_size = int(probe["num_frames"]), int(probe["image_size"])

    train_ds = _subset(SSV2Clips(root, "train", num_frames, image_size),
                       int(probe["max_train_samples"]))
    val_ds = _subset(SSV2Clips(root, "validation", num_frames, image_size),
                     int(probe["max_val_samples"]))
    bs, nw = int(probe["batch_size"]), int(probe["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)

    backbone = build_frozen_backbone(cfg["backbone"], device)
    model = FrozenFrameAverageClassifier(backbone).to(device)
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError("backbone is not frozen")
    optimizer = torch.optim.AdamW(model.classifier.parameters(),
                                  lr=float(probe["lr"]), weight_decay=0.01)

    subset_mode = bool(probe["max_train_samples"] or probe["max_val_samples"]
                       or probe["max_steps_per_epoch"])
    epochs = int(probe["epochs"])
    max_steps = int(probe["max_steps_per_epoch"]) or None
    print(f"SSv2 video  device={device}  backbone={cfg['backbone']['kind']}"
          f"({'trained' if cfg['backbone'].get('encoder') else 'random (smoke)'})"
          f"  epochs={epochs}  frames={num_frames}")
    metrics = {"top1": 0.0, "top5": 0.0, "videos": 0}
    for epoch in range(epochs):
        model.train()
        for step, (clips, labels) in enumerate(train_loader):
            if max_steps and step >= max_steps:
                break
            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite SSv2 loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, val_loader, device)
        print(f"[{epoch + 1}/{epochs}] top1={metrics['top1']:.3f} "
              f"top5={metrics['top5']:.3f}")

    raw = {"top1": float(metrics["top1"]), "top5": float(metrics["top5"]),
           "videos": int(metrics["videos"]), "epochs": epochs}
    contract.write_metrics(out, raw, METRIC_NAMES)
    (Path(out) / "results.json").write_text(
        json.dumps({"task": TASK, "backbone": cfg["backbone"],
                    "num_classes": NUM_CLASSES, "num_frames": num_frames,
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
