"""Explicit captured flow components, not a certification of paper protocols."""

from __future__ import annotations

import argparse
import io
import itertools
import json
import re
import struct
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
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

TASK = "extended_flow"
TRANSFORM = "captured_flow_input_pixels_v2"
DATASETS = ("mpi_sintel", "middlebury_flow", "spring")


def _native(flow, valid=None):
    flow = np.asarray(flow)
    if (
        flow.ndim != 3
        or flow.shape[-1] != 2
        or min(flow.shape[:2]) < 1
        or flow.dtype.kind != "f"
    ):
        raise ValueError("flow requires floating H,W,2 values")
    mask = np.isfinite(flow).all(-1)
    if valid is not None:
        valid = np.asarray(valid)
        if valid.shape != flow.shape[:2] or not np.isin(valid, [0, 1]).all():
            raise ValueError("flow validity requires a binary H,W mask")
        mask &= valid.astype(bool)
    if not mask.any():
        raise ValueError("no valid native flow pixels")
    return flow, mask


def decode_flo(raw):
    if len(raw) < 12:
        raise ValueError("truncated flow header")
    magic, w, h = struct.unpack("<fii", raw[:12])
    if magic != 202021.25 or min(w, h) < 1 or len(raw) != 12 + 8 * w * h:
        raise ValueError("invalid flow magic, dimensions or payload")
    flow = np.frombuffer(raw, dtype="<f4", offset=12).reshape(h, w, 2).copy()
    return _native(flow, (np.abs(flow) < 1e9).all(-1))


def decode_flo5(raw):
    with h5py.File(io.BytesIO(raw), "r") as f:
        if "flow" not in f or set(f) - {"flow", "valid"}:
            raise ValueError("unknown flo5 schema or validity semantics")
        return _native(
            np.array(f["flow"]), np.array(f["valid"]) if "valid" in f else None
        )


def resize_flow(flow, valid, size=(224, 224), image_hw=None):
    flow, valid = _native(flow, valid)
    ih, iw = image_hw if image_hw is not None else valid.shape
    oh, ow = size
    if min(ih, iw, oh, ow) < 1:
        raise ValueError("invalid native image or output dimensions")
    values = torch.from_numpy(
        np.where(valid[..., None], flow, 0).astype(np.float32)
    ).permute(2, 0, 1)
    mask = torch.from_numpy(valid.copy()).float()[None, None]
    target = F.interpolate(
        values[None], size=size, mode="bilinear", align_corners=False
    )[0]
    support = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)[0, 0]
    nearest = F.interpolate(mask, size=size, mode="nearest-exact")[0, 0] > 0.5
    keep = nearest & (support >= 1.0 - 1e-6)
    target[0] *= ow / iw
    target[1] *= oh / ih
    target[:, ~keep] = 0
    if not torch.isfinite(target).all() or not keep.any():
        raise ValueError("no finite valid flow after resizing")
    return target, keep


def _batch(pred, target, valid):
    if (
        target.ndim != 4
        or target.shape[1] != 2
        or min(target.shape) < 1
        or valid.shape != target.shape[:1] + target.shape[2:]
        or valid.dtype != torch.bool
        or not valid.flatten(1).any(1).all()
        or pred.shape != target.shape
        or not torch.isfinite(pred).all()
        or not torch.isfinite(target.permute(0, 2, 3, 1)[valid]).all()
    ):
        raise ValueError("invalid flow prediction, target or per-sample mask")


def flow_loss(pred, target, valid):
    _batch(pred, target, valid)
    return (pred.float() - target.float()).abs().mean(1)[valid].mean()


class FlowScore:
    def __init__(self):
        self.groups = {}
        self.native_sum = 0.0
        self.images = 0
        self.nonzero = False

    def update(self, pred, target, valid, metadata):
        _batch(pred, target, valid)
        tags, sizes = metadata["rendering"], metadata["native_hw"]
        if (
            len(tags) != len(pred)
            or sizes.shape != (len(pred), 2)
            or not torch.isfinite(sizes).all()
            or not (sizes > 0).all()
        ):
            raise ValueError("complete positive native image geometry required")
        if any(tag not in ("", "clean", "final") for tag in tags):
            raise ValueError("unknown flow rendering")
        self.nonzero |= bool(torch.count_nonzero(target.permute(0, 2, 3, 1)[valid]))
        delta = pred.float() - target.float()
        for i, tag in enumerate(tags):
            errors = torch.linalg.vector_norm(delta[i], dim=0)[valid[i]]
            native = delta[i].clone()
            native[0] *= sizes[i, 1].item() / target.shape[-1]
            native[1] *= sizes[i, 0].item() / target.shape[-2]
            self.native_sum += (
                torch.linalg.vector_norm(native, dim=0)[valid[i]].double().sum().item()
            )
            for group in ("all", tag) if tag else ("all",):
                acc = self.groups.setdefault(group, [0.0, 0, 0])
                acc[0] += errors.double().sum().item()
                acc[1] += int((errors > 1.0).sum())
                acc[2] += errors.numel()
        self.images += len(pred)

    def result(self):
        if not self.nonzero or not self.groups:
            raise ValueError("empty or all-zero flow evaluation population")
        result = {"images": self.images, "pixels": self.groups["all"][2]}
        for group, (total, outliers, count) in self.groups.items():
            suffix = "" if group == "all" else "_" + group
            result["epe" + suffix] = total / count
            result["outlier_gt1px_at224_pct" + suffix] = 100.0 * outliers / count
            result["pixels" + suffix] = count
        result["epe_native_pixels_on_224_grid"] = self.native_sum / result["pixels"]
        return result


class FlowHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels, 128, 1)
        self.flow = nn.Conv2d(128, 2, 1)

    def forward(self, a, b):
        return self.flow(self.proj(b) - self.proj(a))


class FlowProbe(nn.Module):
    def __init__(self, backbone, reader_profile):
        super().__init__()
        if reader_profile not in (None, SINGLE_BLOCK, CROSS_SELF):
            raise ValueError("unknown flow reader profile")
        self.backbone = backbone.requires_grad_(False).eval()
        # Unlike the depth constructor, the original flow adapter precedes its head.
        self.adapter = (
            (
                SingleBlockSpatialAdapter
                if reader_profile == SINGLE_BLOCK
                else SpatialAdapter
            )(backbone.out_channels)
            if reader_profile
            else None
        )
        self.head = FlowHead(backbone.out_channels)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, a, b, out_hw):
        def features(x):
            with torch.no_grad():
                f = self.backbone.forward_features(
                    TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ).float()
            return self.adapter(f) if self.adapter is not None else f

        return F.interpolate(
            self.head(features(a), features(b)),
            size=out_hw,
            mode="bilinear",
            align_corners=False,
        )


def _path(root, value):
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or ".." in Path(value).parts
    ):
        raise ValueError("flow paths must be relative under data_root")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("missing or escaping flow input")
    return path


def _read(root, reference):
    if isinstance(reference, str):
        return _path(root, reference).read_bytes()
    if not isinstance(reference, dict) or set(reference) != {"archive", "member"}:
        raise ValueError("input requires a file or archive/member reference")
    member = reference["member"]
    if (
        not isinstance(member, str)
        or not member
        or Path(member).is_absolute()
        or ".." in Path(member).parts
    ):
        raise ValueError("invalid archive member")
    with zipfile.ZipFile(_path(root, reference["archive"])) as z:
        if z.namelist().count(member) != 1:
            raise ValueError("missing or duplicate archive member")
        return z.read(member)


class FlowPairs(Dataset):
    def __init__(self, root, rows, dataset, *, train):
        self.root, self.rows, self.dataset, self.train = root, rows, dataset, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        raw = _read(self.root, row["flow"])
        native, valid = (decode_flo5 if self.dataset == "spring" else decode_flo)(raw)
        images, sizes = [], []
        for key in ("a", "b"):
            with Image.open(io.BytesIO(_read(self.root, row[key]))) as im:
                sizes.append((im.height, im.width))
                images.append(
                    TF.to_tensor(
                        im.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
                    )
                )
        if sizes[0] != sizes[1]:
            raise ValueError("paired RGB geometry differs")
        allowed = {sizes[0]}
        if self.dataset == "spring":
            allowed.add(tuple(2 * x for x in sizes[0]))
        if native.shape[:2] not in allowed:
            raise ValueError("native flow grid and RGB geometry differ")
        target, mask = resize_flow(native, valid, image_hw=sizes[0])
        sample = (*images, target, mask)
        return (
            sample
            if self.train
            else sample
            + ({"rendering": row["rendering"], "native_hw": torch.tensor(sizes[0])},)
        )


def load_data(root, manifest, dataset):
    if dataset not in DATASETS:
        raise ValueError("flow dataset identity is unresolved")
    root = Path(root).resolve()
    raw = Path(manifest).read_bytes()
    data = json.loads(raw)
    if (
        set(data)
        != {"schema_version", "dataset", "split_evidence", "train", "validation"}
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or data["dataset"] != dataset
        or not isinstance(data["split_evidence"], str)
        or not data["split_evidence"].strip()
    ):
        raise ValueError("explicit flow membership and evidence required")
    split_scenes, split_targets, split_images, fingerprints = [], [], [], {}
    for split in ("train", "validation"):
        rows = data[split]
        if not isinstance(rows, list) or not rows:
            raise ValueError("nonempty train and validation flow splits required")
        scenes, targets, pairs, images = set(), set(), set(), set()
        for row in rows:
            if (
                not isinstance(row, dict)
                or set(row) != {"a", "b", "flow", "scene", "rendering"}
                or not isinstance(row["scene"], str)
                or not row["scene"].strip()
                or row["rendering"] not in ("clean", "final", "")
                or (dataset == "mpi_sintel") != bool(row["rendering"])
            ):
                raise ValueError("invalid flow row or rendering identity")
            ids = []
            for key in ("a", "b", "flow"):
                ref = row[key]
                # Validate and hash each source file once, including whole archives.
                _read(root, ref)
                path = _path(root, ref if isinstance(ref, str) else ref["archive"])
                identity = str(path) + (
                    "::" + ref["member"] if isinstance(ref, dict) else ""
                )
                ids.append(identity)
                if str(path) not in fingerprints:
                    fingerprints[str(path)] = contract.sha256_file(path)
            pair = tuple(ids)
            if pair in pairs or ids[0] == ids[1]:
                raise ValueError("duplicate flow pair or identical source frames")
            pairs.add(pair)
            targets.add(ids[2])
            scenes.add(row["scene"])
            images.update(ids[:2])
        split_scenes.append(scenes)
        split_targets.append(targets)
        split_images.append(images)
    if (
        split_scenes[0] & split_scenes[1]
        or split_targets[0] & split_targets[1]
        or split_images[0] & split_images[1]
    ):
        raise ValueError("flow scenes, targets or RGB images overlap across splits")
    train = FlowPairs(root, data["train"], dataset, train=True)
    val = FlowPairs(root, data["validation"], dataset, train=False)
    for ds in (train, val):
        if not any(bool(torch.count_nonzero(t[:, m])) for _, _, t, m, *_ in ds):
            raise ValueError("all-zero flow supervision split")
    return (
        train,
        val,
        {
            "manifest_sha256": contract.sha256_bytes(raw),
            "source_sha256": {
                str(Path(k).relative_to(root)): v
                for k, v in sorted(fingerprints.items())
            },
            "train_count": len(train),
            "validation_count": len(val),
            "split_evidence": data["split_evidence"],
            "official_membership_verified": False,
        },
    )


def build_manifest(root, dataset):
    """Convert captured local held-out partitions; never claim an official split."""
    root = Path(root).resolve()
    if dataset not in ("mpi_sintel", "middlebury_flow"):
        raise ValueError("native membership builder not verified for this dataset")
    if dataset == "mpi_sintel":
        bases = [
            p
            for p in (root, root / "training", root / "raw" / "training")
            if (p / "flow").is_dir()
        ]
        if len(bases) != 1:
            raise ValueError("expected exactly one Sintel flow root")
        base = bases[0]
        scenes = sorted(p.name for p in (base / "flow").iterdir() if p.is_dir())
        items = []
        for tag in ("clean", "final"):
            if not (base / tag).is_dir() or {
                p.name for p in (base / tag).iterdir() if p.is_dir()
            } != set(scenes):
                raise ValueError("Sintel flow/clean/final scene sets differ")
        for scene in scenes:
            frames = sorted((base / "clean" / scene).glob("frame_*.png"))
            final = sorted((base / "final" / scene).glob("frame_*.png"))
            if len(frames) < 2 or [p.name for p in frames] != [p.name for p in final]:
                raise ValueError("incomplete Sintel renderings")
            for a, b in itertools.pairwise(frames):
                if (
                    not re.fullmatch(r"frame_\d+", a.stem)
                    or not re.fullmatch(r"frame_\d+", b.stem)
                    or int(b.stem.split("_")[-1]) != int(a.stem.split("_")[-1]) + 1
                ):
                    raise ValueError("nonconsecutive Sintel frames")
                target = base / "flow" / scene / (a.stem + ".flo")
                if not target.is_file():
                    raise ValueError("missing Sintel target in selected population")
                for tag in ("clean", "final"):
                    items.append(
                        {
                            "a": str((base / tag / scene / a.name).relative_to(root)),
                            "b": str((base / tag / scene / b.name).relative_to(root)),
                            "flow": str(target.relative_to(root)),
                            "scene": scene,
                            "rendering": tag,
                        }
                    )
    else:
        targets_zip = "archives/other-gt-flow.zip"
        images_zip = "archives/other-color-allframes.zip"

        def index(archive, filename):
            with zipfile.ZipFile(_path(root, archive)) as z:
                names = z.namelist()
            if len(names) != len(set(names)):
                raise ValueError("duplicate archive members")
            found = {}
            for name in names:
                p = Path(name)
                if p.name == filename:
                    if p.parent.name in found:
                        raise ValueError("ambiguous Middlebury scene")
                    found[p.parent.name] = {"archive": archive, "member": name}
            return found

        targets = index(targets_zip, "flow10.flo")
        a, b = index(images_zip, "frame10.png"), index(images_zip, "frame11.png")
        if not targets or not set(targets) <= set(a) or not set(targets) <= set(b):
            raise ValueError("Middlebury target lacks frame10/frame11")
        scenes = sorted(targets)
        items = [
            {"a": a[s], "b": b[s], "flow": targets[s], "scene": s, "rendering": ""}
            for s in scenes
        ]
    if len(scenes) < 2:
        raise ValueError("at least two scenes required for held-out partition")
    chosen = set(scenes[: max(1, int(0.8 * len(scenes)))])
    return {
        "schema_version": 1,
        "dataset": dataset,
        "split_evidence": "captured local sorted scene 80/20 partition; not an official benchmark split",
        "train": [r for r in items if r["scene"] in chosen],
        "validation": [r for r in items if r["scene"] not in chosen],
    }


def recipe(dataset, adaptation):
    if dataset not in DATASETS or adaptation not in ("frozen", "attentive"):
        raise ValueError("only explicit captured flow LP/AP components supported")
    ap = adaptation == "attentive"
    path = PROTOCOLS / ("EXTEND_ATTENTIVE_v1.json" if ap else "EXTEND_LINEAR_v1.json")
    result = dict(
        json.loads(path.read_text())["recipe_overrides" if ap else "recipes"][
            "flow_30"
        ]["optimization"]
    )
    # Resolve the source trainer's actual interpretation, not a new schedule.
    if result["warmup"] != "1 epoch" or result["schedule"] != "cosine":
        raise ValueError("flow schedule source changed; reconcile before execution")
    result.update(
        warmup="1 epoch from 1e-6", schedule="cosine to 0", betas=[0.9, 0.999]
    )
    return result


def validate_config(cfg):
    plan = validate_probe(
        cfg,
        task=TASK,
        get_recipe=recipe,
        reader_attribute="EXTENDED_DENSE_READER",
        max_epochs=30,
    )
    if cfg["transform_profile"] != TRANSFORM:
        raise ValueError("explicit captured input-pixel flow profile required")
    return plan


def run(cfg, out):
    with distributed.session(cfg["device"]) as context:
        plan = context.call(lambda: validate_config(cfg))
        context.agree(cfg)
        context.seed(cfg["seed"])
        device = context.device
        execution.autocast_context(device, plan["precision"])
        train, val, membership = context.call(
            lambda: load_data(cfg["data_root"], cfg["samples"], cfg["dataset"])
        )
        context.agree(membership)
        settings = cfg["probe"]
        loader = context.call(lambda: context.loader(train, settings, cfg["seed"]))
        validation = DataLoader(
            val, batch_size=settings["batch_size"], num_workers=settings["num_workers"]
        )
        raw = context.call(
            lambda: FlowProbe(
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

        def forward(a, b, hw):
            with execution.autocast_context(device, plan["precision"]):
                return model(a.to(device), b.to(device), hw)

        def loss_for_batch(batch):
            a, b, target, valid = batch
            return flow_loss(
                forward(a, b, target.shape[-2:]).float(),
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
            score = FlowScore()
            with torch.no_grad():
                for a, b, target, valid, meta in validation:
                    score.update(
                        forward(a, b, target.shape[-2:]).float().cpu(),
                        target,
                        valid,
                        meta,
                    )
            result = score.result()
            result["epochs"] = settings["epochs"]
            vocabulary = {
                key: (
                    "extended_flow_" + key
                    if key.startswith(("epe", "outlier_"))
                    else "epochs_completed"
                    if key == "epochs"
                    else None
                )
                for key in result
            }
            contract.write_metrics(out, result, vocabulary)
            torch.save(
                {k: v.cpu() for k, v in _probe(raw).items()}, Path(out) / "probe.pt"
            )
            report = {
                "task": TASK,
                "profile": cfg["profile"],
                "dataset": cfg["dataset"],
                "adaptation": cfg["adaptation"],
                "reader_profile": cfg["reader_profile"],
                "backbone": cfg["backbone"],
                "transform_profile": TRANSFORM,
                "membership": membership,
                "recipe": spec,
                "execution": runtime,
                "continuation": continuation.report(),
                "final": result,
                "flow_units": "224x224 input-image pixels",
                "evaluation_grid": [224, 224],
                "record_value": False,
                "canonical_eligible": False,
                "source_protocol_conflicts": [
                    "source feature difference and L1 versus companion correlation and robust endpoint loss",
                    "source local held-out partitions and 224-pixel metrics require final-table run reconciliation",
                ],
                "limitations": [
                    "released-weight CUDA/BF16/full-data score parity unverified",
                    "native-pixel EPE is evaluated on the 224 grid, not native-grid benchmark evaluation",
                    "legacy checkpoint import and FT not supported",
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
