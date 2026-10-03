"""Extended detection components; explicit geometry, original-annotation metrics."""

from __future__ import annotations

import copy
import json
import random
from datetime import UTC
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from downstream.attention import SpatialAdapter
from downstream.captured_readers import (
    CROSS_SELF,
    SINGLE_BLOCK,
    SingleBlockSpatialAdapter,
)

BENCHMARKS = {
    "coco2017": (False, 100, 100),
    "livecell": (True, 3000, 2000),
    "lvis_v1": (True, 300, 300),
}
GEOMETRIES = {"captured_224_256": (224, 256), "captured_800_1333": (800, 1333)}


def _image_path(root, record):
    name = record.get("file_name")
    if not name:
        parts = Path(urlparse(record.get("coco_url", "")).path).parts
        if len(parts) < 2 or parts[-2] not in ("train2017", "val2017"):
            raise ValueError("image needs a local file_name or COCO split URL")
        name = "/".join(parts[-2:])
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("image path must remain under data root")
    path = root / relative
    if any(
        (root / Path(*relative.parts[:i])).is_symlink()
        for i in range(1, len(relative.parts) + 1)
    ):
        raise ValueError("image symlinks are not accepted")
    if not path.is_file():
        raise ValueError("annotation image is missing")
    return path


class ExtendedDetectionData(torch.utils.data.Dataset):
    """Captured 512-side/64-instance training inputs; never truncate scoring GT."""

    def __init__(self, root, annotation, *, train, mask):
        from pycocotools.coco import COCO

        self.root = Path(root).resolve()
        self.annotation = Path(annotation)
        self.train, self.mask = train, mask
        source = json.loads(self.annotation.read_text())
        for key in ("images", "annotations", "categories"):
            values = source[key]
            ids = [x["id"] for x in values]
            if any(type(x) is not int or x < 1 for x in ids) or len(set(ids)) != len(
                ids
            ):
                raise ValueError(f"{key}: invalid or duplicate IDs")
        if not source["images"] or not source["categories"]:
            raise ValueError("empty image population or category ontology")
        images = {x["id"]: x for x in source["images"]}
        categories = {x["id"] for x in source["categories"]}
        for ann in source["annotations"]:
            if ann["image_id"] not in images or ann["category_id"] not in categories:
                raise ValueError("annotation refers to unknown image/category")
            if len(ann["bbox"]) != 4 or not np.isfinite(ann["bbox"]).all():
                raise ValueError("invalid annotation box")
        self.paths = {i: _image_path(self.root, row) for i, row in images.items()}
        self.coco = COCO()
        self.coco.dataset = source
        self.coco.createIndex()
        self.ids = sorted(images)
        self.category_to_label = {c: i + 1 for i, c in enumerate(sorted(categories))}
        self.num_classes = len(categories) + 1

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        iid = self.ids[index]
        record = self.coco.imgs[iid]
        with Image.open(self.paths[iid]) as raw:
            image = raw.convert("RGB")
        width, height = image.size
        if (record["width"], record["height"]) != (width, height):
            raise ValueError("image and annotation geometry disagree")
        anns = [
            a
            for a in self.coco.imgToAnns.get(iid, [])
            if not a.get("iscrowd", 0) and a["bbox"][2] > 1 and a["bbox"][3] > 1
        ]
        if len(anns) > 64:
            if self.train:
                random.shuffle(anns)
            anns = sorted(
                anns,
                key=lambda a: a.get("area", a["bbox"][2] * a["bbox"][3]),
                reverse=True,
            )[:64]
        boxes = [[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in anns)]
        masks = [self.coco.annToMask(a) for a in anns] if self.mask else []
        if any(m.shape != (height, width) for m in masks):
            raise ValueError("annotation mask geometry disagrees")
        if max(width, height) > 512:
            scale = 512 / max(width, height)
            width, height = max(1, round(width * scale)), max(1, round(height * scale))
            image = image.resize((width, height), Image.Resampling.BICUBIC)
            boxes = [[v * scale for v in box] for box in boxes]
            masks = [
                np.asarray(
                    Image.fromarray(m).resize((width, height), Image.Resampling.NEAREST)
                )
                for m in masks
            ]
        if self.train and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            boxes = [[width - x2, y1, width - x1, y2] for x1, y1, x2, y2 in boxes]
            masks = [np.ascontiguousarray(m[:, ::-1]) for m in masks]
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(
                [self.category_to_label[a["category_id"]] for a in anns],
                dtype=torch.int64,
            ),
            "image_id": torch.tensor([iid]),
            "area": torch.tensor(
                [a.get("area", a["bbox"][2] * a["bbox"][3]) for a in anns],
                dtype=torch.float32,
            ),
            "iscrowd": torch.zeros(len(anns), dtype=torch.int64),
        }
        if self.mask:
            target["masks"] = (
                torch.tensor(np.stack(masks), dtype=torch.uint8)
                if masks
                else torch.zeros((0, height, width), dtype=torch.uint8)
            )
        return TF.to_tensor(image), target


class ExtendedPyramid(nn.Module):
    def __init__(self, body, reader_profile):
        super().__init__()
        from downstream.native_detection import TransposedFeaturePyramid

        self.body = body.requires_grad_(False).eval()
        self.adapter = (
            None
            if reader_profile is None
            else (
                SingleBlockSpatialAdapter
                if reader_profile == SINGLE_BLOCK
                else SpatialAdapter
            )(body.out_channels)
        )
        self.fpn = TransposedFeaturePyramid(body.out_channels)
        self.out_channels = 256

    def train(self, mode=True):
        super().train(mode)
        self.body.eval()
        return self

    def forward(self, images):
        mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        with torch.no_grad():
            features = self.body.forward_features((images - mean) / std).float()
        if features.ndim != 4 or min(features.shape[-2:]) < 2:
            raise ValueError("detector needs a spatial grid at least 2 by 2")
        if self.adapter is not None:
            features = self.adapter(features)
        return self.fpn(features)


def build_extended_detector(
    body, num_classes, *, dataset, reader_profile, geometry_profile
):
    from torchvision.models.detection import FasterRCNN, MaskRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign

    if dataset not in BENCHMARKS or geometry_profile not in GEOMETRIES:
        raise ValueError("unverified detection benchmark or geometry profile")
    if reader_profile not in (None, SINGLE_BLOCK, CROSS_SELF):
        raise ValueError("unverified detection reader profile")
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("num_classes includes one background class")
    mask, cap, _ = BENCHMARKS[dataset]
    low, high = GEOMETRIES[geometry_profile]
    model = (MaskRCNN if mask else FasterRCNN)(
        ExtendedPyramid(body, reader_profile),
        num_classes=num_classes,
        rpn_anchor_generator=AnchorGenerator(
            ((32,), (64,), (128,), (256,)), ((0.5, 1.0, 2.0),) * 4
        ),
        box_roi_pool=MultiScaleRoIAlign(["0", "1", "2", "3"], 7, 2),
        min_size=low,
        max_size=high,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        box_detections_per_img=cap,
    )
    if dataset == "livecell":
        from torchvision.models.detection.rpn import RegionProposalNetwork

        assert isinstance(model.rpn, RegionProposalNetwork)
        model.rpn._pre_nms_top_n["testing"] = cap
        model.rpn._post_nms_top_n["testing"] = cap
    return model


def _predictions(data, outputs, dataset):
    """Map labels and input geometry back to official annotations, once per image."""
    from pycocotools import mask as mask_utils

    inverse = {v: k for k, v in data.category_to_label.items()}
    seen, rows = set(), []
    with_masks, _, cap = BENCHMARKS[dataset]
    for iid, input_hw, prediction in outputs:
        if iid not in data.ids or iid in seen:
            raise ValueError("unknown or repeated evaluation image")
        seen.add(iid)
        ih, iw = input_hw
        if ih <= 0 or iw <= 0:
            raise ValueError("invalid input image geometry")
        record = data.coco.imgs[iid]
        h, w = record["height"], record["width"]
        boxes = prediction["boxes"].detach().cpu().float()
        labels = prediction["labels"].detach().cpu()
        scores = prediction["scores"].detach().cpu().float()
        n = len(boxes)
        if (
            boxes.shape != (n, 4)
            or labels.shape != (n,)
            or scores.shape != (n,)
            or labels.dtype != torch.int64
            or not torch.isfinite(boxes).all()
            or not torch.isfinite(scores).all()
            or not ((scores >= 0) & (scores <= 1)).all()
            or not all(int(x) in inverse for x in labels)
            or not (boxes[:, 2:] > boxes[:, :2]).all()
        ):
            raise ValueError("invalid detector output")
        boxes *= boxes.new_tensor([w / iw, h / ih, w / iw, h / ih])
        # LVIS limits each image globally; COCO limits each category inside COCOeval.
        order = torch.arange(n)
        if dataset == "livecell":
            # Official useCats=0 pools categories before stable score sorting.
            order = order[torch.argsort(labels, stable=True)]
        order = order[torch.argsort(scores[order], descending=True, stable=True)]
        if dataset != "coco2017":
            order = order[:cap]
        masks = prediction.get("masks")
        if with_masks and (
            masks is None
            or masks.shape != (n, 1, ih, iw)
            or not torch.isfinite(masks).all()
            or not ((masks >= 0) & (masks <= 1)).all()
        ):
            raise ValueError("instance evaluation needs finite masks in input geometry")
        for start in range(0, len(order), 32):
            indices = order[start : start + 32]
            encoded = []
            if with_masks:
                resized = (
                    F.interpolate(
                        masks[indices.to(masks.device)].float(),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )[:, 0]
                    > 0.5
                )
                encoded = [
                    mask_utils.encode(
                        np.asfortranarray(m.cpu().numpy().astype(np.uint8))
                    )
                    for m in resized
                ]
            for offset, idx in enumerate(indices.tolist()):
                x1, y1, x2, y2 = boxes[idx].tolist()
                row = {
                    "image_id": iid,
                    "category_id": inverse[int(labels[idx])],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(scores[idx]),
                }
                if with_masks:
                    row["segmentation"] = encoded[offset]
                rows.append(row)
    if seen != set(data.ids):
        raise ValueError("evaluation must cover every annotation image exactly once")
    return rows


def evaluate_extended_detection(data, outputs, *, dataset):
    """Use benchmark evaluators, not a category-presence classification proxy."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if dataset not in BENCHMARKS or data.train:
        raise ValueError("unverified benchmark or training-mode evaluation input")
    masks, _, cap = BENCHMARKS[dataset]
    if masks and not data.mask:
        raise ValueError("instance segmentation requires mask annotations")
    if dataset == "lvis_v1" and not all(
        "neg_category_ids" in x and "not_exhaustive_category_ids" in x
        for x in data.coco.imgs.values()
    ):
        raise ValueError("LVIS needs federated negative and not-exhaustive categories")
    rows = _predictions(data, outputs, dataset)
    result = {
        "images": len(data.ids),
        "detections": len(rows),
        "evaluation_max_detections": cap,
        "use_categories": dataset != "livecell",
        "record_value": False,
        "canonical_eligible": False,
    }
    for kind in ("bbox", "segm") if masks else ("bbox",):
        records = copy.deepcopy(rows)
        if kind == "segm":
            for row in records:
                row.pop("bbox", None)
        if dataset == "lvis_v1":
            from lvis import LVIS, LVISEval, LVISResults

            gt = LVIS(str(data.annotation))
            if not records:
                # LVISResults indexes the first record; construct its empty equivalent.
                dt = object.__new__(LVISResults)
                dt.dataset = copy.deepcopy(gt.dataset)
                dt.logger = gt.logger
                dt.dataset["annotations"] = []
                dt._create_index()
            else:
                dt = LVISResults(gt, records, max_dets=cap)
            evaluator = LVISEval(gt, dt, iou_type=kind)
            # lvis 0.5.3 uses removed np.float in accumulate. Bind a private globals
            # mapping for this method; never mutate NumPy or the installed module.
            import types

            class LegacyNumpy:
                def __getattr__(self, name):
                    return float if name == "float" else getattr(np, name)

            original = type(evaluator).accumulate
            namespace = dict(original.__globals__, np=LegacyNumpy())
            compatible = types.FunctionType(
                original.__code__,
                namespace,
                original.__name__,
                original.__defaults__,
                original.__closure__,
            )

            class CompatibleLVIS(LVISEval):
                def accumulate(self, _compatible=compatible):
                    return _compatible(self)

            evaluator = CompatibleLVIS(gt, dt, iou_type=kind)
            evaluator.run()
            values = evaluator.get_results()
            numbers = [100 * float(values[k]) for k in ("AP", "AP50", "AP75")]
        else:
            if records:
                dt = data.coco.loadRes(records)
            else:
                dt = COCO()
                dt.dataset = {
                    "images": copy.deepcopy(data.coco.dataset["images"]),
                    "categories": copy.deepcopy(data.coco.dataset["categories"]),
                    "annotations": [],
                }
                dt.createIndex()
            evaluator = COCOeval(data.coco, dt, kind)
            evaluator.params.imgIds = data.ids
            evaluator.params.useCats = int(dataset != "livecell")
            evaluator.params.maxDets = (
                [100, 500, cap] if dataset == "livecell" else [1, 10, cap]
            )
            evaluator.evaluate()
            evaluator.accumulate()
            precision = np.asarray(evaluator.eval["precision"])[:, :, :, 0, -1]

            def average(values):
                valid = values[values >= 0]
                if not valid.size:
                    raise ValueError("evaluation has no scored ground truth")
                return float(valid.mean() * 100)

            numbers = [average(precision), average(precision[0]), average(precision[5])]
        if any(not np.isfinite(x) or x < 0 for x in numbers):
            raise ValueError("evaluation has no valid AP")
        prefix = "mask" if kind == "segm" else "bbox"
        result.update(
            {
                prefix + "_ap": numbers[0],
                prefix + "_ap50": numbers[1],
                prefix + "_ap75": numbers[2],
            }
        )
    return result


# Training reuses the component evaluator without asserting historical score identity.
from downstream import extended_distributed as distributed
from downstream import extended_execution as execution
from downstream.extended_classification import PROTOCOLS, vision_no_decay
from downstream.extended_resume import Continuation, _probe
from downstream.spatial_backbones import (
    _load_provider,
    build_frozen_backbone,
    discover_providers,
)

TRAIN_TASK = "extended_detection_training"


def training_recipe(dataset, adaptation):
    if dataset not in BENCHMARKS or adaptation not in ("frozen", "attentive"):
        raise ValueError("only verified detection LP/AP recipes are supported")
    catalog = json.loads((PROTOCOLS / "EXTEND_LINEAR_v1.json").read_text())
    key = catalog["datasets"][dataset]["recipe"]
    if adaptation == "attentive":
        catalog = json.loads((PROTOCOLS / "EXTEND_ATTENTIVE_v1.json").read_text())
        return dict(catalog["recipe_overrides"][key]["optimization"])
    return dict(catalog["recipes"][key]["optimization"])


def training_optimizer(model, spec, *, effective_batch, updates_per_epoch):
    if any(type(x) is not int or x < 1 for x in (effective_batch, updates_per_epoch)):
        raise ValueError("positive effective batch and updates per epoch required")
    if (
        spec["optimizer"] != "SGD"
        or spec["epochs"] != 12
        or spec["schedule"] != "step x0.1 at epochs 8 and 11"
        or spec["warmup"] not in ("500 iterations", "500 iterations, factor 0.001")
    ):
        raise ValueError("unverified detection optimization recipe")
    lr = spec["base_lr"] * effective_batch / spec["lr_reference_effective_batch"]
    groups = []
    for no_decay in (False, True):
        params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and vision_no_decay(n) == no_decay
        ]
        if params:
            groups.append(
                {
                    "params": params,
                    "weight_decay": 0.0 if no_decay else spec["weight_decay"],
                }
            )
    opt = torch.optim.SGD(groups, lr=lr, momentum=spec["momentum"])

    def factor(step):
        if step < 500:
            return 0.001 + (1.0 - 0.001) * (step / 500)
        epoch = step // updates_per_epoch
        return 0.01 if epoch >= 11 else 0.1 if epoch >= 8 else 1.0

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, factor)


def detection_collate(rows):
    images, targets = zip(*rows)
    return list(images), list(targets)


def training_batch(batch, device):
    images, targets = batch
    images = [im.to(device) for im in images]
    clean = []
    for im, source in zip(images, targets):
        target = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in source.items()
        }
        boxes = target["boxes"]
        h, w = im.shape[-2:]
        keep = (
            (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
            & (boxes[:, 2] > 0)
            & (boxes[:, 3] > 0)
            & (boxes[:, 0] < w)
            & (boxes[:, 1] < h)
        )
        for key in ("boxes", "labels", "masks"):
            if key in target:
                target[key] = target[key][keep][:64]
        clean.append(target)
    return images, clean


def load_training_data(cfg):
    mask = BENCHMARKS[cfg["dataset"]][0]
    train = ExtendedDetectionData(
        cfg["data_root"], cfg["train_annotations"], train=True, mask=mask
    )
    validation = ExtendedDetectionData(
        cfg["data_root"], cfg["validation_annotations"], train=False, mask=mask
    )
    if cfg["dataset"] == "lvis_v1" and not all(
        "neg_category_ids" in row and "not_exhaustive_category_ids" in row
        for data in (train, validation)
        for row in data.coco.imgs.values()
    ):
        raise ValueError("LVIS requires federated metadata before training")
    ontology = lambda data: {k: v["name"] for k, v in data.coco.cats.items()}
    spec = json.loads((PROTOCOLS / "EXTEND_LINEAR_v1.json").read_text())["datasets"][
        cfg["dataset"]
    ]
    if (
        ontology(train) != ontology(validation)
        or train.num_classes != spec["head_outputs"] + 1
    ):
        raise ValueError(
            "annotation ontology disagrees with dataset recipe or evaluation split"
        )
    if set(train.ids) & set(validation.ids) or set(train.paths.values()) & set(
        validation.paths.values()
    ):
        raise ValueError("training and validation membership overlap")
    from downstream import contract

    membership = {
        "train_sha256": contract.sha256_bytes(train.annotation.read_bytes()),
        "validation_sha256": contract.sha256_bytes(validation.annotation.read_bytes()),
        "categories": ontology(train),
        "train_count": len(train),
        "validation_count": len(validation),
        "official_membership_verified": False,
    }
    return train, validation, membership


class DetectionProbe(nn.Module):
    """Expose only the frozen encoder as backbone; retain the trainable FPN."""

    BACKBONE_STATE_PREFIX = "detector.backbone.body."

    def __init__(self, body, classes, cfg):
        super().__init__()
        self.detector = build_extended_detector(
            body,
            classes,
            dataset=cfg["dataset"],
            reader_profile=cfg["reader_profile"],
            geometry_profile=cfg["geometry_profile"],
        )

    @property
    def backbone(self):
        return self.detector.backbone.body

    def forward(self, images, targets=None):
        return self.detector(images, targets)


def validate_training_config(cfg):
    required = {
        "task",
        "profile",
        "dataset",
        "seed",
        "device",
        "data_root",
        "train_annotations",
        "validation_annotations",
        "adaptation",
        "reader_profile",
        "geometry_profile",
        "backbone",
        "probe",
    }
    if (
        set(cfg) - {"execution", "resume"} != required
        or cfg["task"] != TRAIN_TASK
        or cfg["profile"] != "capture_extended_components"
        or type(cfg["seed"]) is not int
        or cfg["seed"] != 0
    ):
        raise ValueError("explicit Extended detection training config required")
    spec = training_recipe(cfg["dataset"], cfg["adaptation"])
    if cfg["geometry_profile"] not in GEOMETRIES:
        raise ValueError("explicit verified detector geometry required")
    probe = cfg["probe"]
    if (
        not isinstance(probe, dict)
        or set(probe) != {"epochs", "batch_size", "num_workers"}
        or any(
            type(v) is not int or v < (0 if k == "num_workers" else 1)
            for k, v in probe.items()
        )
        or probe["epochs"] > spec["epochs"]
    ):
        raise ValueError("invalid physical probe settings")
    path = discover_providers().get(cfg["backbone"].get("kind"))
    provider = _load_provider(path) if path else None
    reader = getattr(provider, "EXTENDED_DENSE_READER", None)
    if reader not in (SINGLE_BLOCK, CROSS_SELF) or cfg["reader_profile"] != (
        reader if cfg["adaptation"] == "attentive" else None
    ):
        raise ValueError("provider and detector reader disagree")
    plan = execution.resolve_execution(cfg, provider)
    if plan["precision"] != "fp32":
        raise ValueError("captured detector training requires FP32; no BF16 fallback")
    plan["schedule_clock"] = "optimizer_updates_warmup_then_epoch_milestones"
    return plan


def train_run(cfg, out):
    from torch.utils.data import DataLoader

    from downstream import contract

    with distributed.session(cfg["device"]) as context:
        plan = context.call(lambda: validate_training_config(cfg))
        context.agree(cfg)
        context.seed(cfg["seed"])
        train, validation, membership = context.call(lambda: load_training_data(cfg))
        context.agree(membership)
        settings = cfg["probe"]
        loader = context.call(
            lambda: context.loader(
                train, settings, cfg["seed"], collate_fn=detection_collate
            )
        )
        val_loader = DataLoader(
            validation,
            batch_size=settings["batch_size"],
            num_workers=settings["num_workers"],
            collate_fn=detection_collate,
        )
        raw = context.call(
            lambda: DetectionProbe(
                build_frozen_backbone(cfg["backbone"], context.device),
                train.num_classes,
                cfg,
            ).to(context.device)
        )
        model = context.wrap(raw)
        updates = len(loader) // plan["accumulation_steps"]
        if (
            plan["tail_policy"] == "flush_scaled"
            and len(loader) % plan["accumulation_steps"]
        ):
            updates += 1
        spec = training_recipe(cfg["dataset"], cfg["adaptation"])
        opt, schedule = training_optimizer(
            raw,
            spec,
            effective_batch=plan["effective_batch"],
            updates_per_epoch=updates,
        )
        runtime = execution.execution_report(
            plan, spec, len(loader), schedule.base_lrs[0]
        )
        continuation = Continuation(
            cfg, membership, raw, opt, schedule, runtime, loader, context, out
        )

        def loss_for_batch(batch):
            images, targets = training_batch(batch, context.device)
            return sum(model(images, targets).values())

        for epoch in range(continuation.start_epoch, settings["epochs"]):
            if context.world > 1:
                loader.sampler.set_epoch(epoch)
            stats = execution.train_epoch(
                model,
                loader,
                loss_for_batch,
                opt,
                schedule,
                accumulation_steps=plan["accumulation_steps"],
                tail_policy=plan["tail_policy"],
                adaptation=cfg["adaptation"],
            )
            execution.record_epoch(runtime, stats)
            continuation.save(epoch + 1)

        def evaluate_and_save():
            raw.eval()

            def predictions():
                with torch.no_grad():
                    for images, targets in val_loader:
                        output = raw([im.to(context.device) for im in images])
                        for im, target, prediction in zip(images, targets, output):
                            yield (
                                int(target["image_id"].item()),
                                tuple(im.shape[-2:]),
                                prediction,
                            )

            metrics = evaluate_extended_detection(
                validation, predictions(), dataset=cfg["dataset"]
            )
            metrics["epochs"] = settings["epochs"]
            names = {
                k: f"extended_{cfg['dataset']}_{k}"
                if k.startswith(("bbox_", "mask_"))
                else "epochs_completed"
                if k == "epochs"
                else None
                for k in metrics
            }
            contract.write_metrics(
                out, {k: v for k, v in metrics.items() if type(v) is not bool}, names
            )
            torch.save(
                {k: v.cpu() for k, v in _probe(raw).items()}, Path(out) / "probe.pt"
            )
            report = {
                "task": TRAIN_TASK,
                "dataset": cfg["dataset"],
                "adaptation": cfg["adaptation"],
                "geometry_profile": cfg["geometry_profile"],
                "reader_profile": cfg["reader_profile"],
                "backbone": cfg["backbone"],
                "membership": membership,
                "recipe": spec,
                "execution": runtime,
                "continuation": continuation.report(),
                "final": metrics,
                "canonical_eligible": False,
                "record_value": False,
                "limitations": [
                    "source/catalog geometry unresolved",
                    "FP32 detector execution; released-weight CUDA parity unverified",
                    "official split identity and historical score correspondence unverified",
                ],
            }
            (Path(out) / "results.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n"
            )
            return metrics

        return context.call(evaluate_and_save, leader=True)


def main(argv=None):
    """Train explicit LP/AP components or score a prediction archive."""
    import argparse
    from datetime import datetime

    from downstream import contract

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    config_bytes = Path(args.config).read_bytes()
    cfg = json.loads(config_bytes)
    if cfg.get("task") == TRAIN_TASK:
        return distributed.cli(config_bytes, args.out, train_run, TRAIN_TASK)
    out = Path(args.out)
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return 1
    started = datetime.now(UTC).isoformat()
    status, error = "ok", None
    try:
        if (
            set(cfg)
            != {"task", "dataset", "data_root", "annotation", "predictions", "profile"}
            or cfg["task"] != "extended_detection_evaluation"
            or cfg["profile"] != "official_annotations_v1"
            or cfg["dataset"] not in BENCHMARKS
        ):
            raise ValueError("explicit supported evaluation config required")
        data = ExtendedDetectionData(
            cfg["data_root"],
            cfg["annotation"],
            train=False,
            mask=BENCHMARKS[cfg["dataset"]][0],
        )
        outputs = torch.load(cfg["predictions"], map_location="cpu", weights_only=True)
        metrics = evaluate_extended_detection(data, outputs, dataset=cfg["dataset"])
        names = {
            key: f"extended_{cfg['dataset']}_{key}"
            if key.startswith(("bbox_", "mask_"))
            else None
            for key in metrics
        }
        contract.write_metrics(
            out, {k: v for k, v in metrics.items() if type(v) is not bool}, names
        )
        report = {
            "dataset": cfg["dataset"],
            "metrics": metrics,
            "annotation_sha256": contract.sha256_bytes(
                Path(cfg["annotation"]).read_bytes()
            ),
            "predictions_sha256": contract.sha256_bytes(
                Path(cfg["predictions"]).read_bytes()
            ),
            "record_value": False,
            "canonical_eligible": False,
            "limitations": [
                "official split identity and prediction provenance unverified",
                "training recipe and released-weight reproduction not certified",
            ],
        }
        (out / "results.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    except Exception as exc:  # noqa: BLE001 -- CLI must record a failed manifest for every evaluator error.
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    contract.write_manifest(
        out,
        task="extended_detection_evaluation",
        method_ref="external_predictions",
        status=status,
        config_sha256=contract.sha256_bytes(config_bytes),
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(),
        seed=0,
        backbone={},
        error=error,
    )
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
