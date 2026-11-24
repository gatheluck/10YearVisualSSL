"""Native SpaceNet and SUN RGB-D sample readers and saved-prediction evaluation."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import zipfile

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF

from downstream.extended_segmentation import SegmentationScore

TASK = "figure5_native_semantic_evaluation"


def _mapping(items):
    result = dict(items)
    if len(result) != len(items):
        raise ValueError("duplicate identifier in JSON")
    return result


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _asset(root, name):
    path = Path(name)
    resolved = (root / path).resolve()
    if (
        path.is_absolute()
        or ".." in path.parts
        or not resolved.is_relative_to(root.resolve())
        or not resolved.is_file()
    ):
        raise ValueError("asset must be an existing relative file inside its root")
    return resolved


def _population(counts, expected, population):
    if population not in ("fixture", "full") or min(counts) < 1:
        raise ValueError("nonempty fixture or full population required")
    if population == "full" and counts != expected:
        raise ValueError("incomplete official population")


class SpaceNetSamples:
    def __init__(self, cfg, train):
        from pycocotools.coco import COCO

        self.root = Path(cfg["data_root"]) / "RGB-PanSharpen_jpg"
        self.train = train
        splits, hashes = [], {}
        for tag in ("train", "val"):
            raw = (
                self.root.parent / "annotations" / f"instance_{tag}.json"
            ).read_bytes()
            data = json.loads(raw)
            if {c["name"] for c in data["categories"]} != {"building"}:
                raise ValueError("expected building ontology")
            names = [image["file_name"] for image in data["images"]]
            if len(set(names)) != len(names) or len(
                {i["id"] for i in data["images"]}
            ) != len(names):
                raise ValueError("duplicate COCO image")
            splits.append(set(names))
            hashes[tag] = _digest(raw)
            if (tag == "train") == train:
                self.coco = COCO()
                self.coco.dataset = data
                self.coco.createIndex()
        if splits[0] & splits[1]:
            raise ValueError("overlapping SpaceNet splits")
        _population(tuple(map(len, splits)), (3080, 771), cfg["population"])
        self.hashes = hashes
        self.ids = tuple(str(i) for i in sorted(self.coco.imgs))

    def __getitem__(self, index):
        identifier = int(self.ids[index])
        info = self.coco.imgs[identifier]
        with Image.open(_asset(self.root, info["file_name"])) as raw:
            image = raw.convert("RGB")
        width, height = image.size
        if (info["width"], info["height"]) != image.size:
            raise ValueError("SpaceNet image dimensions disagree with annotation")
        anns = [
            a
            for a in self.coco.imgToAnns[identifier]
            if not a.get("iscrowd", 0)
            and a["category_id"] in self.coco.cats
            and min(a["bbox"][2:]) > 1
        ]
        if len(anns) > 64:
            if self.train:
                random.shuffle(anns)
            anns = sorted(
                anns,
                key=lambda a: a.get("area", a["bbox"][2] * a["bbox"][3]),
                reverse=True,
            )[:64]
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in anns:
            mask |= self.coco.annToMask(ann)
        mask = Image.fromarray(mask)
        if max(image.size) > 512:
            scale = 512 / max(image.size)
            size = tuple(max(1, round(v * scale)) for v in image.size)
            image = image.resize(size, Image.Resampling.BICUBIC)
            mask = mask.resize(size, Image.Resampling.NEAREST)
        if self.train and random.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        x = TF.resize(TF.to_tensor(image), [224, 224], antialias=True)
        y = TF.resize(
            torch.from_numpy(np.array(mask))[None].float(),
            [224, 224],
            interpolation=TF.InterpolationMode.NEAREST,
        )[0].long()
        return x, y


def _sun_id(value):
    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    if "SUNRGBD" not in parts or ".." in parts:
        raise ValueError("SUNRGBD metadata requires an unambiguous sample path")
    return "/".join(parts[parts.index("SUNRGBD") :])


class SunSamples:
    def __init__(self, cfg, train, stack):
        import h5py
        from scipy.io import loadmat

        self.train = train
        self.archive = stack.enter_context(zipfile.ZipFile(cfg["archive"]))
        with zipfile.ZipFile(cfg["toolbox"]) as toolbox:
            split = toolbox.read("SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat")
            metadata = toolbox.read("SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat")
        self.hashes = {"split": _digest(split), "metadata": _digest(metadata)}
        splits = loadmat(io.BytesIO(split), simplify_cells=True)
        rows = [
            tuple(_sun_id(x) for x in np.atleast_1d(splits[k]))
            for k in ("alltrain", "alltest")
        ]
        _population(tuple(map(len, rows)), (5285, 5050), cfg["population"])
        self.ids = rows[0 if train else 1]
        records = np.atleast_1d(
            loadmat(io.BytesIO(metadata), simplify_cells=True)["SUNRGBDMeta"]
        )
        self.by_id = {}
        for n, record in enumerate(records):
            rgb, depth = (_sun_id(record[k]) for k in ("rgbpath", "depthpath"))
            key, sep, _ = rgb.rpartition("/image/")
            if (
                not sep
                or "/depth/" not in depth
                or key != depth.rsplit("/depth/", 1)[0]
                or key in self.by_id
            ):
                raise ValueError("SUN RGB/depth scene identity mismatch or duplicate")
            self.by_id[key] = (rgb, depth, n)
        if len(set(rows[0] + rows[1])) != sum(map(len, rows)) or set(
            rows[0] + rows[1]
        ) != set(self.by_id):
            raise ValueError("split must partition all official metadata scenes")
        mat = Path(cfg["semantic_mat"])
        with mat.open("rb") as handle:
            self.hashes["semantic_mat"] = hashlib.file_digest(
                handle, "sha256"
            ).hexdigest()
        self.hdf = h5py.is_hdf5(mat)
        if self.hdf:
            self.labels = stack.enter_context(h5py.File(mat, "r"))
            keys = [
                k for k in ("seglabel", "SUNRGBD2Dseg/seglabel") if k in self.labels
            ]
            if len(keys) != 1:
                raise ValueError("expected one 37-class seglabel cell array")
            self.cells = self.labels[keys[0]]
            if (
                self.cells.ndim != 2
                or 1 not in self.cells.shape
                or h5py.check_dtype(ref=self.cells.dtype) is not h5py.Reference
            ):
                raise ValueError("invalid MATLAB semantic reference vector")
        else:
            self.cells = np.asarray(loadmat(mat)["seglabel"], dtype=object).reshape(-1)
        if self.cells.size != len(records):
            raise ValueError("semantic cells and metadata must have the same length")

    def __getitem__(self, index):
        rgb, depth, n = self.by_id[self.ids[index]]
        with Image.open(io.BytesIO(self.archive.read(rgb))) as image:
            color = image.convert("RGB")
        with Image.open(io.BytesIO(self.archive.read(depth))) as raw:
            packed = np.array(raw)
        if self.hdf:
            ref = self.cells[n, 0] if self.cells.shape[1] == 1 else self.cells[0, n]
            labels = np.asarray(self.labels[ref]).T
        else:
            labels = self.cells[n]
        if (
            labels.ndim != 2
            or labels.shape != packed.shape
            or color.size != labels.shape[::-1]
            or packed.dtype.kind not in "ui"
            or packed.min() < 0
            or packed.max() > 65535
            or labels.dtype.kind not in "uif"
            or not np.isfinite(labels).all()
            or labels.min() < 0
            or labels.max() > 37
            or not np.equal(labels, labels.astype(np.int64)).all()
        ):
            raise ValueError(
                "invalid registered SUN RGB, packed depth and 37-class target"
            )
        packed = packed.astype(np.uint16)
        meters = ((packed >> 3) | (packed << 13)).astype(np.float32) / 1000
        x = TF.to_tensor(color.resize((224, 224), Image.Resampling.BICUBIC))
        meters = torch.from_numpy(
            np.array(
                Image.fromarray(meters).resize((224, 224), Image.Resampling.NEAREST)
            )
        )
        x = torch.cat((x, (meters / 10)[None], (meters > 0).float()[None]))
        y = np.where(labels == 0, 255, labels.astype(np.int64) - 1).astype(np.uint8)
        if self.train:
            y = np.array(
                Image.fromarray(y).resize((224, 224), Image.Resampling.NEAREST)
            )
        return x, torch.from_numpy(y).long()


@contextmanager
def native_samples(cfg, *, train):
    """Yield the official split's ordered IDs and aligned (input, mask) pairs."""
    with ExitStack() as stack:
        if cfg["dataset"] == "spacenet2_vegas":
            yield SpaceNetSamples(cfg, train)
        elif cfg["dataset"] == "sun_rgbd":
            yield SunSamples(cfg, train, stack)
        else:
            raise ValueError("unknown Figure 5 semantic setting")


def evaluate(cfg, out):
    from downstream.extended_distributed import launch

    keys = {"task", "dataset", "device", "population", "predictions"}
    keys.update(
        {"archive", "toolbox", "semantic_mat"}
        if cfg["dataset"] == "sun_rgbd"
        else {"data_root"}
    )
    if set(cfg) != keys or cfg["device"] != "cpu" or launch()[2] != 1:
        raise ValueError(
            "saved-map evaluation requires one CPU process and explicit native paths"
        )
    directory = Path(cfg["predictions"])
    raw = (directory / "index.json").read_bytes()
    files = json.loads(raw, object_pairs_hook=_mapping)
    with native_samples(cfg, train=False) as samples:
        if not isinstance(files, dict) or set(files) != set(samples.ids):
            raise ValueError("prediction IDs must cover exactly the validation split")
        score = SegmentationScore(37 if cfg["dataset"] == "sun_rgbd" else 2)
        seen = set()
        fingerprints = hashlib.sha256()
        for i, key in enumerate(samples.ids):
            path = _asset(directory, files[key])
            if path in seen:
                raise ValueError("each sample requires a separate predicted map")
            seen.add(path)
            with path.open("rb") as handle:
                fingerprints.update(hashlib.file_digest(handle, "sha256").digest())
                handle.seek(0)
                predicted = np.load(handle, allow_pickle=False)
            _, target = samples[i]
            if predicted.dtype.kind not in "iu" or predicted.shape != tuple(
                target.shape
            ):
                raise ValueError(
                    "expected integer class map on the source evaluation grid"
                )
            score.update(
                torch.from_numpy(predicted.astype(np.int64))[None], target[None]
            )
        report = dict(
            dataset=cfg["dataset"],
            population=cfg["population"],
            metrics=score.result(),
            source_hashes=samples.hashes,
            prediction_index_sha256=_digest(raw),
            ordered_prediction_digest=fingerprints.hexdigest(),
            paper_score_verified=False,
        )
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
