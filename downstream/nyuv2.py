"""NYUv2 depth estimation on a frozen backbone (downstream task 3).

    python -m downstream.nyuv2 --config <resolved.json> --out <dir>

A port of the capture's `downstream/nyuv2_depth.py`, wired to this repo's downstream
contract (`downstream.contract`). It freezes a backbone (a method's trained
`encoder.pt`, or a random tiny ViT for the hermetic smoke), attaches a DPT-style
progressive-upsampling depth head, trains only the head with a masked L1 loss, and
reports **RMSE** and **AbsRel** over the valid depth pixels.

Faithful to the capture: the labelled `nyu_depth_v2_labeled.mat` read via `h5py`,
the DPT head (1x1 project -> GroupNorm/GELU -> 4 residual refine blocks with
progressive x2 upsampling -> softplus), masked L1, RMSE/AbsRel over finite
positive depths, and `max_*_samples` / `max_steps_per_epoch` subsetting that stamps
`record_value: false`. Changed for the port: the device is resolved (not assumed
CUDA), the run is seeded, the result is the contract's manifest + metrics, and the
**train/val split adapts to the file size** -- the real labelled file (1449 images)
keeps the capture's first-795-train / remaining-654-val split, while a tiny
hermetic `.mat` is split in half, so a CPU smoke needs no 1449-image dataset.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from downstream import contract                                    # noqa: E402
from downstream.spatial_backbones import build_frozen_backbone, KINDS  # noqa: E402

TASK = "nyuv2_depth"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
# The capture's split of the labelled file (1449 images).
FULL_SIZE = 1449
TRAIN_SIZE = 795

TOP_KEYS = frozenset({"task", "seed", "device", "data_root", "backbone", "probe"})
BACKBONE_REQUIRED = frozenset({"kind", "encoder", "arch", "img_size", "patch_size"})
BACKBONE_OPTIONAL = frozenset({"embed_dim", "depth", "num_heads"})
PROBE_KEYS = frozenset({"epochs", "batch_size", "lr", "num_workers", "image_size",
                        "head_hidden_dim", "max_train_samples", "max_val_samples",
                        "max_steps_per_epoch"})
DEVICES = ("auto", "cuda", "cpu")
METRIC_NAMES = {"rmse": "nyuv2_rmse", "abs_rel": "nyuv2_absrel",
                "valid_pixels": None, "epochs": "epochs_completed",
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


def split_indices(n: int) -> "tuple[list[int], list[int]]":
    """Train/val indices into the labelled file.

    The real labelled file (1449 images) keeps the capture's first-795-train /
    remaining-654-val split; a smaller (hermetic) file is split in half."""
    if n >= FULL_SIZE:
        return list(range(TRAIN_SIZE)), list(range(TRAIN_SIZE, FULL_SIZE))
    half = max(1, n // 2)
    return list(range(half)), list(range(half, n))


class NYUv2Depth(Dataset):
    def __init__(self, mat_path: Path, indices: "list[int]", image_size: int):
        self.mat_path = str(mat_path)
        self.indices = indices
        self.image_size = image_size
        self._file = None

    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self.mat_path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        source_idx = self.indices[idx]
        image = torch.from_numpy(
            self.file["images"][source_idx].astype("float32") / 255.0)
        depth = torch.from_numpy(self.file["depths"][source_idx].astype("float32"))
        image = image.transpose(1, 2)
        depth = depth.t().unsqueeze(0)
        image = F.interpolate(image.unsqueeze(0),
                              size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False).squeeze(0)
        depth = F.interpolate(depth.unsqueeze(0),
                              size=(self.image_size, self.image_size),
                              mode="nearest").squeeze(0)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        valid = torch.isfinite(depth) & (depth > 0)
        return image, depth, valid.float()


class DPTRefineBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels), nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.block(x))


class DPTDepthHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 256):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim), nn.GELU())
        self.refine = nn.ModuleList(DPTRefineBlock(hidden_dim) for _ in range(4))
        self.out = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(), nn.Conv2d(hidden_dim // 2, 1, kernel_size=1))

    def forward(self, feat: torch.Tensor,
                output_size: "tuple[int, int]") -> torch.Tensor:
        x = self.project(feat)
        for block in self.refine:
            x = block(x)
            if x.shape[-2] < output_size[0] or x.shape[-1] < output_size[1]:
                x = F.interpolate(
                    x, size=(min(output_size[0], x.shape[-2] * 2),
                             min(output_size[1], x.shape[-1] * 2)),
                    mode="bilinear", align_corners=False)
        if x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear",
                              align_corners=False)
        return F.softplus(self.out(x))


class FrozenDepthModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int = 256):
        super().__init__()
        self.backbone = backbone
        self.head = DPTDepthHead(backbone.out_channels, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.backbone.forward_features(x)
        return self.head(feat, x.shape[-2:])


def masked_l1(pred, target, valid):
    mask = valid > 0
    if not mask.any():
        return pred.sum() * 0.0
    return (pred[mask] - target[mask]).abs().mean()


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    sq_sum = abs_rel_sum = 0.0
    count = 0
    for image, depth, valid in loader:
        image, depth = image.to(device), depth.to(device)
        pred = model(image)
        mask = (valid.to(device) > 0) & torch.isfinite(depth) & (depth > 0)
        if not mask.any():
            continue
        diff = pred[mask] - depth[mask]
        sq_sum += float((diff ** 2).sum().cpu())
        abs_rel_sum += float((diff.abs() / depth[mask].clamp_min(1e-6)).sum().cpu())
        count += int(mask.sum().cpu())
    return {"rmse": (sq_sum / max(count, 1)) ** 0.5,
            "abs_rel": abs_rel_sum / max(count, 1), "valid_pixels": count}


def _mat_length(mat_path: Path) -> int:
    with h5py.File(str(mat_path), "r") as handle:
        return int(handle["images"].shape[0])


def run(cfg: dict, out: Path, device_override: str | None = None) -> dict:
    validate_config(cfg)
    device = resolve_device(device_override or cfg["device"])
    seed = int(cfg["seed"])
    make_deterministic(seed)
    probe = cfg["probe"]
    image_size = int(probe["image_size"])

    mat_path = Path(cfg["data_root"]) / "labeled/nyu_depth_v2_labeled.mat"
    if not mat_path.is_file():
        raise FileNotFoundError(f"NYUv2 labelled file not found: {mat_path}")
    train_idx, val_idx = split_indices(_mat_length(mat_path))
    if int(probe["max_train_samples"]):
        train_idx = train_idx[:int(probe["max_train_samples"])]
    if int(probe["max_val_samples"]):
        val_idx = val_idx[:int(probe["max_val_samples"])]

    bs, nw = int(probe["batch_size"]), int(probe["num_workers"])
    train_ds = NYUv2Depth(mat_path, train_idx, image_size)
    val_ds = NYUv2Depth(mat_path, val_idx, image_size)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)

    backbone = build_frozen_backbone(cfg["backbone"], device)
    model = FrozenDepthModel(backbone, hidden_dim=int(probe["head_hidden_dim"])).to(device)
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError("backbone is not frozen")
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=float(probe["lr"]),
                                  weight_decay=0.01)

    subset_mode = bool(probe["max_train_samples"] or probe["max_val_samples"]
                       or probe["max_steps_per_epoch"])
    epochs = int(probe["epochs"])
    max_steps = int(probe["max_steps_per_epoch"]) or None
    print(f"NYUv2 depth  device={device}  backbone={cfg['backbone']['kind']}"
          f"({'trained' if cfg['backbone'].get('encoder') else 'random (smoke)'})"
          f"  epochs={epochs}  train={len(train_ds)} val={len(val_ds)}")
    metrics = {"rmse": 0.0, "abs_rel": 0.0, "valid_pixels": 0}
    for epoch in range(epochs):
        model.train()
        for step, (image, depth, valid) in enumerate(train_loader):
            if max_steps and step >= max_steps:
                break
            image, depth, valid = (image.to(device), depth.to(device),
                                   valid.to(device))
            pred = model(image)
            loss = masked_l1(pred, depth, valid)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite NYUv2 depth loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, val_loader, device)
        print(f"[{epoch + 1}/{epochs}] rmse={metrics['rmse']:.4f} "
              f"abs_rel={metrics['abs_rel']:.4f}")

    raw = {"rmse": float(metrics["rmse"]), "abs_rel": float(metrics["abs_rel"]),
           "valid_pixels": int(metrics["valid_pixels"]), "epochs": epochs}
    contract.write_metrics(out, raw, METRIC_NAMES)
    (Path(out) / "results.json").write_text(
        json.dumps({"task": TASK, "backbone": cfg["backbone"],
                    "split": "labelled: first 795 train / remaining val (full); "
                             "half/half for a smaller hermetic file",
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
