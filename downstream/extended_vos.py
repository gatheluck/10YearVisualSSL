"""Captured first-mask video segmentation components and execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from downstream import contract
from downstream import extended_distributed as distributed
from downstream import extended_execution as execution
from downstream.extended_classification import PROTOCOLS, optimizer
from downstream.extended_classification import validate_config as validate_probe
from downstream.extended_resume import Continuation, _probe
from downstream.paired_spatial import FrozenSpatialPair
from downstream.spatial_backbones import build_frozen_backbone
from downstream.vos_data import Frames, load_data


class MaskHead(nn.Module):
    """Shared projection, first-mask pooled template, channelwise correlation,
    and a single affine 1x1 mask prediction. BCE uses a fixed zero background
    logit, so independent object logits can compete with background at inference.
    """

    def __init__(self, in_dim, width=256):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, width, 1)
        self.mask = nn.Conv2d(width, 1, 1)

    def forward(self, template, search, first_mask):
        if first_mask.ndim != 4 or first_mask.shape[:2] != (len(template), 1):
            raise ValueError("DAVIS: first_mask must be B x 1 x H x W")
        if (
            not torch.isfinite(first_mask).all()
            or ((first_mask < 0) | (first_mask > 1)).any()
        ):
            raise ValueError("DAVIS: invalid first-mask occupancy")
        t = F.normalize(self.proj(template), dim=1)
        s = F.normalize(self.proj(search), dim=1)
        weights = F.interpolate(first_mask.float(), size=t.shape[-2:], mode="area")
        mass = weights.sum((-2, -1), keepdim=True)
        if (mass <= 0).any():
            raise ValueError("DAVIS: empty supervised template object")
        descriptor = (t * weights).sum((-2, -1), keepdim=True) / mass
        correlation = s * descriptor.to(s.dtype)
        logits = self.mask(correlation)
        logits = F.interpolate(
            logits.float(),
            size=first_mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return (logits, correlation.sum(1))


def mask_loss(logits, target):
    """Pixelwise foreground/background cross-entropy (binary logistic BCE).

    Standard supervised VOS mask classification loss, equivalent to two-class
    softmax CE with background logit zero. Void label 255 contributes no gradient.
    Equal weight per valid pixel; no fitted threshold, Dice or pseudo-mask loss.
    """
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("DAVIS: expected matching B x 1 x H x W mask logits/targets")
    if (
        not torch.isfinite(logits).all()
        or not ((target == 0) | (target == 1) | (target == 255)).all()
    ):
        raise ValueError("DAVIS: nonfinite logits or invalid mask target")
    valid = target != 255
    if not valid.any():
        raise ValueError("DAVIS: mask batch has no nonvoid supervision")
    return F.binary_cross_entropy_with_logits(
        logits.float()[valid], target.float()[valid]
    )


def boolean(x, name="flags"):
    x = np.asarray(x)
    if not np.isin(x, [0, 1]).all():
        raise ValueError(f"{name}: expected boolean or binary flags")
    return x.astype(bool)


def boundary(mask):
    seg = np.asarray(mask, bool)
    e, s, se = (np.zeros_like(seg), np.zeros_like(seg), np.zeros_like(seg))
    e[:, :-1], s[:-1], se[:-1, :-1] = (seg[:, 1:], seg[1:], seg[1:, 1:])
    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = False
    return b


def dilate_disk(mask, radius):
    h, w = mask.shape
    out = np.zeros_like(mask)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            y0, y1 = (max(0, dy), min(h, h + dy))
            x0, x1 = (max(0, dx), min(w, w + dx))
            if y1 > y0 and x1 > x0:
                out[y0:y1, x0:x1] |= mask[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    return out


def frame_score(gt, pred, void=None):
    gt, pred = (boolean(gt, "GT mask"), boolean(pred, "predicted mask"))
    if gt.ndim != 2 or gt.shape != pred.shape or (not gt.size):
        raise ValueError("DAVIS: invalid masks")
    void = np.zeros_like(gt) if void is None else boolean(void, "void mask")
    if void.shape != gt.shape:
        raise ValueError("DAVIS: invalid void mask")
    g, p = (gt & ~void, pred & ~void)
    union = (g | p).sum()
    j = (g & p).sum() / union if union else 1.0
    gb, pb = (boundary(g), boundary(p))
    radius = int(np.ceil(0.008 * np.linalg.norm(gt.shape)))
    ng, npred = (gb.sum(), pb.sum())
    if npred == 0 and ng > 0:
        precision, recall = (1.0, 0.0)
    elif npred > 0 and ng == 0:
        precision, recall = (0.0, 1.0)
    elif npred == 0 and ng == 0:
        precision = recall = 1.0
    else:
        precision = (pb & dilate_disk(gb, radius)).sum() / npred
        recall = (gb & dilate_disk(pb, radius)).sum() / ng
    f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (float(j), float(f))


def _mask(value):
    a = np.asarray(value)
    if (
        a.ndim != 2
        or not a.size
        or a.dtype.kind not in "iu"
        or a.min() < 0
        or a.max() > 255
    ):
        raise ValueError("integer native-resolution object mask required")
    return a


def _objects(first):
    objects = [int(i) for i in np.unique(_mask(first)) if i not in (0, 255)]
    if not objects:
        raise ValueError("first frame requires supervised objects")
    return objects


def image_tensor(rgb):
    image = (
        Image.fromarray(np.asarray(rgb))
        .convert("RGB")
        .resize((224, 224), Image.Resampling.BILINEAR)
    )
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255


def mask_tensor(mask, occupancy=False):
    value = torch.as_tensor(np.array(mask, copy=True), dtype=torch.float32)[None, None]
    return F.interpolate(
        value, size=(224, 224), mode="area" if occupancy else "nearest"
    )[0]


def mask_stream(frames, first, forward, device="cpu"):
    objects = _objects(first)
    frames = iter(frames)
    template = image_tensor(next(frames)).unsqueeze(0).to(device)
    masks = [mask_tensor(first == obj, True).unsqueeze(0).to(device) for obj in objects]
    yield np.where(first == 255, 0, first).astype(np.uint8)
    for frame in frames:
        search = image_tensor(frame).unsqueeze(0).to(device)
        shape = np.asarray(frame).shape[:2]
        best, labels = np.zeros(shape, np.float32), np.zeros(shape, np.uint8)
        for obj, mask in zip(objects, masks):
            output = forward(template, search, mask)
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise ValueError("expected mask logits and correlation")
            logits = output[0]
            if (
                not torch.is_tensor(logits)
                or tuple(logits.shape) != (1, 1, 224, 224)
                or not torch.isfinite(logits).all()
            ):
                raise ValueError("finite 224-grid object mask logits required")
            value = (
                F.interpolate(
                    logits.float(), size=shape, mode="bilinear", align_corners=False
                )[0, 0]
                .detach()
                .cpu()
                .numpy()
            )
            replace = value > best
            labels[replace], best[replace] = obj, value[replace]
        yield labels


class VideoScore:
    """Captured object-macro J/F, scoring neither endpoint; void is background."""

    def __init__(self):
        self.scores, self.seen, self.frames = [], set(), 0

    def add(self, name, truth, predictions):
        if name in self.seen or len(truth) < 3:
            raise ValueError("duplicate sequence or fewer than three annotated frames")
        first = _mask(truth[0])
        objects = _objects(first)
        sums = np.zeros((len(objects), 2))
        stream = iter(predictions)
        sentinel = object()
        for t, target in enumerate(truth):
            target = _mask(target)
            pred = next(stream, sentinel)
            if pred is sentinel:
                raise ValueError("prediction stream ended early")
            pred = _mask(pred)
            if (
                target.shape != first.shape
                or not np.isin(target, [0, 255] + objects).all()
            ):
                raise ValueError("target changes shape or introduces a new object")
            if pred.shape != first.shape or not np.isin(pred, [0] + objects).all():
                raise ValueError("prediction shape or object IDs disagree")
            if 0 < t < len(truth) - 1:
                for i, obj in enumerate(objects):
                    sums[i] += frame_score(target == obj, pred == obj)
        if next(stream, sentinel) is not sentinel:
            raise ValueError("prediction stream contains extra frames")
        self.scores.extend(sums / (len(truth) - 2))
        self.frames += len(truth) - 2
        self.seen.add(name)

    def result(self):
        if not self.scores:
            raise ValueError("empty video evaluation population")
        j, f = np.mean(self.scores, axis=0)
        return {
            "J": float(100 * j),
            "F": float(100 * f),
            "J_and_F": float(50 * (j + f)),
            "objects": len(self.scores),
            "sequences": len(self.seen),
            "scored_frames": self.frames,
        }


TASK = "extended_vos"
TRANSFORM = "captured_first_mask_v1"


class Probe(FrozenSpatialPair):
    def __init__(self, backbone, reader_profile):
        super().__init__(backbone, reader_profile)
        self.head = MaskHead(backbone.out_channels)

    def forward(self, template, search, first_mask):
        return self.head(self.features(template), self.features(search), first_mask)


def recipe(dataset, adaptation):
    if dataset != "davis2017" or adaptation not in ("frozen", "attentive"):
        raise ValueError("only explicit DAVIS LP/AP is supported")
    ap = adaptation == "attentive"
    filename = "EXTEND_ATTENTIVE_v1.json" if ap else "EXTEND_LINEAR_v1.json"
    result = dict(
        json.loads((PROTOCOLS / filename).read_text())[
            "recipe_overrides" if ap else "recipes"
        ]["tracking_30"]["optimization"]
    )
    if result["warmup"] != "1 epoch" or result["schedule"] != "cosine":
        raise ValueError("unresolved captured schedule")
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
        raise ValueError("explicit first-mask input profile required")
    return plan


@torch.no_grad()
def evaluate(data, indices, forward, device):
    score = VideoScore()
    seen = set()
    for index in indices:
        if type(index) is not int or not 0 <= index < len(data):
            raise ValueError("invalid or repeated sequence index")
        seen.add(index)
        row = data.rows[index]
        frames, truth = (
            Frames(data.root, row["frames"]),
            Frames(data.root, row["masks"], True),
        )
        prediction = mask_stream(frames, truth[0], forward, device)
        score.add(row["name"], truth, prediction)
    if seen != set(range(len(data))):
        raise ValueError("incomplete sequence population")
    return score.result()


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
        raw = context.call(
            lambda: Probe(
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

        def forward(a, b, mask):
            with execution.autocast_context(device, plan["precision"]):
                return model(a.to(device), b.to(device), mask.to(device))

        def loss_for_batch(batch):
            a, b, mask, target = batch
            return mask_loss(forward(a, b, mask)[0], target.to(device))

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
            result = evaluate(val, range(len(val)), forward, device)
            result["epochs"] = settings["epochs"]
            contract.write_metrics(
                out,
                result,
                {
                    "J": "extended_davis_j_mean",
                    "F": "extended_davis_f_mean",
                    "J_and_F": "extended_davis_jf_mean",
                    "epochs": "epochs_completed",
                    "objects": None,
                    "sequences": None,
                    "scored_frames": None,
                },
            )
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
                "canonical_eligible": False,
                "record_value": False,
                "evaluation": "native resolution; fixed first-mask templates; object macro; exclude first/last frames",
                "limitations": [
                    "captured evaluation treats void as background; benchmark void-ignore parity unverified",
                    "released weights, GPU/BF16/NCCL and full-data score reproduction unverified",
                    "historical run-to-table attribution and legacy checkpoint import unverified",
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
