"""Native DAVIS sequence lists and explicit, content-bound portable membership."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from downstream.structured_data import asset_bytes


def read_image(root, asset, mask=False):
    with Image.open(io.BytesIO(asset_bytes(root, asset))) as im:
        return np.array(im if mask else im.convert("RGB"), copy=True)


class Frames:
    def __init__(self, root, assets, mask=False):
        self.root, self.assets, self.mask = root, assets, mask

    def __len__(self):
        return len(self.assets)

    def __getitem__(self, index):
        return read_image(self.root, self.assets[index], self.mask)


class Samples(Dataset):
    def __init__(self, root, rows, train):
        self.root, self.rows, self.train = root, rows, train
        self.pairs = []
        if train:
            from downstream.extended_vos import _objects

            for i, row in enumerate(rows):
                first = read_image(root, row["masks"][0], True)
                self.pairs.extend(
                    (i, obj, t)
                    for obj in _objects(first)
                    for t in range(len(row["frames"]))
                )

    def __len__(self):
        return len(self.pairs) if self.train else len(self.rows)

    def __getitem__(self, index):
        if not self.train:
            return index
        from downstream.extended_vos import image_tensor, mask_tensor

        seq, obj, t = self.pairs[index]
        row = self.rows[seq]
        first = read_image(self.root, row["masks"][0], True)
        target = read_image(self.root, row["masks"][t], True)
        binary = np.where(target == 255, 255, target == obj).astype(np.float32)
        return (
            image_tensor(read_image(self.root, row["frames"][0])),
            image_tensor(read_image(self.root, row["frames"][t])),
            mask_tensor(first == obj, True),
            mask_tensor(binary),
        )


def load_data(root, path, dataset):
    from downstream.extended_vos import _mask, _objects

    root = Path(root)
    raw = Path(path).read_bytes()
    obj = json.loads(raw)
    if (
        dataset != "davis2017"
        or not isinstance(obj, dict)
        or set(obj)
        != {"schema_version", "dataset", "split_evidence", "train", "validation"}
        or type(obj["schema_version"]) is not int
        or obj["schema_version"] != 1
        or obj["dataset"] != dataset
        or not isinstance(obj["split_evidence"], str)
        or not obj["split_evidence"].strip()
    ):
        raise ValueError("explicit DAVIS schema and split evidence required")
    names, images, identities = set(), {}, []
    for split in ("train", "validation"):
        rows = obj[split]
        if not isinstance(rows, list) or not rows:
            raise ValueError("nonempty independent sequence lists required")
        for row in rows:
            if (
                not isinstance(row, dict)
                or set(row) != {"name", "frames", "masks"}
                or not isinstance(row["name"], str)
                or not row["name"].strip()
                or row["name"] in names
                or not isinstance(row["frames"], list)
                or not isinstance(row["masks"], list)
                or len(row["frames"]) < 3
                or len(row["frames"]) != len(row["masks"])
            ):
                raise ValueError("unique aligned annotated sequence required")
            names.add(row["name"])
            first = _mask(read_image(root, row["masks"][0], True))
            objects = _objects(first)
            for kind in ("frames", "masks"):
                seen = set()
                for asset in row[kind]:
                    key = json.dumps(asset, sort_keys=True)
                    if key in seen:
                        raise ValueError("repeated frame or mask asset")
                    seen.add(key)
                    data = asset_bytes(root, asset)
                    digest = hashlib.sha256(data).hexdigest()
                    identities.append([split, row["name"], kind, key, digest])
                    image = read_image(root, asset, kind == "masks")
                    if image.shape[:2] != first.shape:
                        raise ValueError("frame or mask changes native dimensions")
                    if kind == "masks":
                        if not np.isin(_mask(image), [0, 255] + objects).all():
                            raise ValueError("object introduced after initial mask")
                    else:
                        if digest in images and images[digest] != split:
                            raise ValueError("cross-split image-byte overlap")
                        images[digest] = split
    membership = {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "assets_sha256": hashlib.sha256(json.dumps(identities).encode()).hexdigest(),
        "train_sequences": len(obj["train"]),
        "validation_sequences": len(obj["validation"]),
        "split_evidence": obj["split_evidence"],
        "official_membership_authenticated": False,
    }
    return (
        Samples(root, obj["train"], True),
        Samples(root, obj["validation"], False),
        membership,
    )


def build_membership(root):
    root = Path(root)
    resolution = (
        "480p"
        if all((root / p / "480p").is_dir() for p in ("JPEGImages", "Annotations"))
        else "Full-Resolution"
    )
    result = {
        "schema_version": 1,
        "dataset": "davis2017",
        "split_evidence": "supplied DAVIS 2017 train/val lists; release authenticity not verified",
    }
    names = set()
    for split, source in [("train", "train"), ("validation", "val")]:
        path = root / "ImageSets/2017" / f"{source}.txt"
        listed = (
            asset_bytes(root, {"path": path.relative_to(root).as_posix()})
            .decode()
            .split()
        )
        rows = []
        if not listed:
            raise ValueError("empty native sequence list")
        for name in listed:
            if (
                Path(name).name != name
                or name in {".", ".."}
                or "\\" in name
                or name in names
            ):
                raise ValueError("invalid or repeated native sequence name")
            names.add(name)
            frames = sorted((root / "JPEGImages" / resolution / name).glob("*.jpg"))
            if len(frames) < 3:
                raise ValueError("at least three frames per sequence required")
            masks = [
                root / "Annotations" / resolution / name / (p.stem + ".png")
                for p in frames
            ]
            if not all(p.is_file() for p in masks):
                raise ValueError("missing aligned mask")
            if set(masks) != set(
                (root / "Annotations" / resolution / name).glob("*.png")
            ):
                raise ValueError("unaligned extra mask")
            rows.append(
                {
                    "name": name,
                    "frames": [
                        {"path": p.relative_to(root).as_posix()} for p in frames
                    ],
                    "masks": [{"path": p.relative_to(root).as_posix()} for p in masks],
                }
            )
        result[split] = rows
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = build_membership(args.data_root)
    with Path(args.out).open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
