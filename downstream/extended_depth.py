"""Extended NYUv2 components; distinct from the BasicFive depth recipe."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from downstream import contract
from downstream import extended_distributed as distributed
from downstream import extended_execution as execution
from downstream.attention import SpatialAdapter
from downstream.captured_readers import (
    CROSS_SELF,
    SINGLE_BLOCK,
    SingleBlockSpatialAdapter,
)
from downstream.extended_classification import PROTOCOLS, optimizer
from downstream.extended_classification import validate_config as validate_probe
from downstream.extended_resume import Continuation, _probe
from downstream.spatial_backbones import build_frozen_backbone

TASK = "extended_depth"
TRANSFORM = "captured_nyu_fixed224_v1"
METRICS = {
    "rmse": "extended_depth_rmse",
    "absrel": "extended_depth_absrel",
    "delta1": "extended_depth_delta1",
    "pixels": None,
    "images": None,
    "epochs": "epochs_completed",
}


def recipe(dataset, adaptation):
    if dataset != "nyuv2" or adaptation not in ("frozen", "attentive"):
        raise ValueError("only verified Extended NYUv2 LP/AP recipes are supported")
    attentive = adaptation == "attentive"
    path = PROTOCOLS / (
        "EXTEND_ATTENTIVE_v1.json" if attentive else "EXTEND_LINEAR_v1.json"
    )
    key = "recipe_overrides" if attentive else "recipes"
    result = dict(json.loads(path.read_text())[key]["depth_30"]["optimization"])
    result["betas"] = [0.9, 0.999]
    return result


class DepthProbe(nn.Module):
    def __init__(self, backbone, reader_profile):
        super().__init__()
        if reader_profile not in (None, SINGLE_BLOCK, CROSS_SELF):
            raise ValueError("unknown depth reader profile")
        self.backbone = backbone.requires_grad_(False).eval()
        self.head = nn.Conv2d(backbone.out_channels, 1, 1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.get_parameter("bias"))
        self.adapter = (
            (
                SingleBlockSpatialAdapter
                if reader_profile == SINGLE_BLOCK
                else SpatialAdapter
            )(backbone.out_channels)
            if reader_profile is not None
            else None
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images, out_hw):
        with torch.no_grad():
            normalized = TF.normalize(
                images, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
            features = self.backbone.forward_features(normalized).float()
        if self.adapter is not None:
            features = self.adapter(features)
        positive = F.softplus(self.head(features)) + 0.001
        return F.interpolate(
            positive, size=out_hw, mode="bilinear", align_corners=False
        ).squeeze(1)


def _mask(pred, target, valid):
    if (
        pred.ndim != 3
        or pred.shape != target.shape
        or pred.shape != valid.shape
        or valid.dtype != torch.bool
        or not pred.numel()
        or not torch.isfinite(pred).all()
        or not torch.isfinite(target[valid]).all()
        or not (target[valid] > 0).all()
    ):
        raise ValueError("invalid depth tensors or valid targets")
    return valid


def depth_loss(pred, target, valid):
    mask = _mask(pred, target, valid)
    if not mask.any():
        raise ValueError("training batch has no valid depth pixels")
    # Extended uses full log variance, unlike the BasicFive half-mean penalty.
    delta = torch.log(pred.float().clamp(min=1e-4)) - torch.log(
        target.float().clamp(min=1e-4)
    )
    delta = delta[mask]
    return delta.square().mean() - delta.mean().square()


class DepthScore:
    def __init__(self):
        self.squared = self.relative = self.correct = 0.0
        self.pixels = self.images = 0

    def update(self, pred, target, valid):
        mask = _mask(pred, target, valid)
        self.images += len(pred)
        if not mask.any():
            return
        p, t = pred[mask].float().clamp(1e-3, 80), target[mask].float().clamp(1e-3, 80)
        self.squared += (p - t).square().sum().item()
        self.relative += ((p - t).abs() / t).sum().item()
        self.correct += (torch.maximum(p / t, t / p) < 1.25).sum().item()
        self.pixels += int(mask.sum())

    def result(self):
        if not self.pixels:
            raise ValueError("evaluation has no valid depth pixels")
        return {
            "rmse": (self.squared / self.pixels) ** 0.5,
            "absrel": self.relative / self.pixels,
            "delta1": 100 * self.correct / self.pixels,
            "pixels": self.pixels,
            "images": self.images,
        }


class NYUPairs(Dataset):
    def __init__(self, path, indices, channels_first, *, train):
        self.path, self.indices, self.channels_first = (
            Path(path),
            indices,
            channels_first,
        )
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        j = self.indices[index]
        # Each read owns its handle, including worker/spawn execution.
        with h5py.File(self.path, "r") as f:
            if self.channels_first:
                rgb, depth = (
                    np.array(f["images"][:, :, :, j]).transpose(1, 2, 0),
                    np.array(f["depths"][:, :, j]),
                )
            else:
                rgb, depth = (
                    np.array(f["images"][j]).transpose(1, 2, 0),
                    np.array(f["depths"][j]),
                )
        if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
            raise ValueError("invalid RGB values")
        if rgb.max() <= 1:
            rgb = rgb * 255
        image = Image.fromarray(rgb.astype("uint8")).resize(
            (224, 224), Image.Resampling.BICUBIC
        )
        target = Image.fromarray(depth.astype(np.float32)).resize(
            (224, 224), Image.Resampling.BILINEAR
        )
        if self.train and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        target = torch.from_numpy(np.array(target, dtype=np.float32))
        valid = (target > 0.1) & (target < 10) & torch.isfinite(target)
        return TF.to_tensor(image), target, valid


def load_data(path, splits):
    path, splits = Path(path), Path(splits)
    raw = splits.read_bytes()
    data = loadmat(splits)
    with h5py.File(path, "r") as f:
        im, depth = f["images"].shape, f["depths"].shape
    if len(im) != 4:
        raise ValueError("invalid NYUv2 image geometry")
    channels_first = im[0] == 3
    count = im[3] if channels_first else im[0]
    expected = im[1:] if channels_first else (im[0], im[2], im[3])
    if (
        not count
        or min(im) < 1
        or depth != expected
        or (not channels_first and im[1] != 3)
    ):
        raise ValueError("NYUv2 depth/image geometry disagreement")
    indices = []
    for key in ("trainNdxs", "testNdxs"):
        values = np.asarray(data.get(key, [])).reshape(-1)
        if (
            values.dtype.kind not in "iu"
            or not len(values)
            or not ((values >= 1) & (values <= count)).all()
        ):
            raise ValueError("invalid one-based NYUv2 split indices")
        indices.append((values.astype(np.int64) - 1).tolist())
    joined = indices[0] + indices[1]
    if len(joined) != count or len(set(joined)) != count:
        raise ValueError("duplicate, overlapping or incomplete NYUv2 splits")
    membership = {
        "split_sha256": contract.sha256_bytes(raw),
        "data_sha256": contract.sha256_file(path),
        "train_indices": indices[0],
        "validation_indices": indices[1],
        "train_count": len(indices[0]),
        "validation_count": len(indices[1]),
        "official_membership_verified": False,
    }
    return (
        NYUPairs(path, indices[0], channels_first, train=True),
        NYUPairs(path, indices[1], channels_first, train=False),
        membership,
    )


def validate_config(cfg):
    plan = validate_probe(
        cfg,
        task=TASK,
        get_recipe=recipe,
        reader_attribute="EXTENDED_DENSE_READER",
        max_epochs=30,
    )
    if cfg["transform_profile"] != TRANSFORM:
        raise ValueError("explicit captured NYUv2 transform required")
    return plan


def run(cfg, out):
    with distributed.session(cfg["device"]) as context:
        plan = context.call(lambda: validate_config(cfg))
        context.agree(cfg)
        context.seed(cfg["seed"])
        device = context.device
        execution.autocast_context(device, plan["precision"])
        train, validation, membership = context.call(
            lambda: load_data(cfg["data_root"], cfg["samples"])
        )
        context.agree(membership)
        settings = cfg["probe"]
        loader = context.call(lambda: context.loader(train, settings, cfg["seed"]))
        val_loader = DataLoader(
            validation,
            batch_size=settings["batch_size"],
            num_workers=settings["num_workers"],
        )
        raw = context.call(
            lambda: DepthProbe(
                build_frozen_backbone(cfg["backbone"], device), cfg["reader_profile"]
            ).to(device)
        )
        model = context.wrap(raw)
        spec = recipe(cfg["dataset"], cfg["adaptation"])
        opt, scheduler = optimizer(
            raw, spec, batch_size=plan["effective_batch"], steps_per_epoch=len(loader)
        )
        runtime = execution.execution_report(
            plan, spec, len(loader), scheduler.base_lrs[0]
        )
        continuation = Continuation(
            cfg, membership, raw, opt, scheduler, runtime, loader, context, out
        )

        def forward(images, hw):
            with execution.autocast_context(device, plan["precision"]):
                return model(images.to(device), hw)

        def loss_for_batch(batch):
            images, target, valid = batch
            return depth_loss(
                forward(images, target.shape[-2:]).float(),
                target.to(device),
                valid.to(device),
            )

        for epoch in range(continuation.start_epoch, settings["epochs"]):
            if context.world > 1:
                loader.sampler.set_epoch(epoch)
            stats = execution.train_epoch(
                model,
                loader,
                loss_for_batch,
                opt,
                scheduler,
                accumulation_steps=plan["accumulation_steps"],
                tail_policy=plan["tail_policy"],
                adaptation=cfg["adaptation"],
            )
            execution.record_epoch(runtime, stats)
            continuation.save(epoch + 1)
        model = raw

        def evaluate_and_save():
            model.eval()
            score = DepthScore()
            with torch.no_grad():
                for images, target, valid in val_loader:
                    score.update(
                        forward(images, target.shape[-2:]).float().cpu(), target, valid
                    )
            result = score.result()
            result["epochs"] = settings["epochs"]
            contract.write_metrics(out, result, METRICS)
            torch.save(
                {k: v.cpu() for k, v in _probe(raw).items()}, Path(out) / "probe.pt"
            )
            report = {
                "task": TASK,
                "profile": cfg["profile"],
                "dataset": cfg["dataset"],
                "adaptation": cfg["adaptation"],
                "reader_profile": cfg["reader_profile"],
                "transform_profile": TRANSFORM,
                "backbone": cfg["backbone"],
                "membership": membership,
                "recipe": spec,
                "execution": runtime,
                "continuation": continuation.report(),
                "evaluation_grid": [224, 224],
                "depth_range": [0.1, 10.0],
                "metric_alignment": "none",
                "final": result,
                "canonical_eligible": False,
                "record_value": False,
                "limitations": [
                    "official split authenticity and historical run correspondence unverified",
                    "released-weight CUDA/BF16 and full-score parity unverified",
                    "legacy native checkpoint import not supported",
                ],
            }
            (Path(out) / "results.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n"
            )
            return result

        return context.call(evaluate_and_save, leader=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return distributed.cli(Path(args.config).read_bytes(), args.out, run, TASK)


if __name__ == "__main__":
    raise SystemExit(main())
