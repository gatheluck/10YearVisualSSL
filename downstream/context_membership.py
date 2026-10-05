"""Stage captured PC59 targets without changing native inputs."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from downstream import semantic_staging
from downstream.structured_data import asset_bytes

CLASSES = (
    "aeroplane",
    "bag",
    "bed",
    "bedclothes",
    "bench",
    "bicycle",
    "bird",
    "boat",
    "book",
    "bottle",
    "building",
    "bus",
    "cabinet",
    "car",
    "cat",
    "ceiling",
    "chair",
    "cloth",
    "computer",
    "cow",
    "cup",
    "curtain",
    "dog",
    "door",
    "fence",
    "floor",
    "flower",
    "food",
    "grass",
    "ground",
    "horse",
    "keyboard",
    "light",
    "motorbike",
    "mountain",
    "mouse",
    "person",
    "plate",
    "platform",
    "pottedplant",
    "road",
    "rock",
    "sheep",
    "shelves",
    "sidewalk",
    "sign",
    "sky",
    "snow",
    "sofa",
    "table",
    "track",
    "train",
    "tree",
    "truck",
    "tvmonitor",
    "wall",
    "water",
    "window",
    "wood",
)

NATIVE_IDS = (
    2,
    9,
    18,
    19,
    22,
    23,
    25,
    31,
    33,
    34,
    44,
    45,
    46,
    59,
    65,
    68,
    72,
    80,
    85,
    98,
    104,
    105,
    113,
    115,
    144,
    158,
    159,
    162,
    187,
    189,
    207,
    220,
    232,
    258,
    259,
    260,
    284,
    295,
    296,
    308,
    324,
    326,
    347,
    349,
    354,
    355,
    360,
    366,
    368,
    397,
    415,
    416,
    420,
    424,
    427,
    440,
    445,
    454,
    458,
)


def ontology(raw):
    rows = {}
    names = set()
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        fields = line.split(":", 1)
        if len(fields) != 2:
            raise ValueError("native ID:name ontology required")
        ident, name = int(fields[0].strip()), fields[1].strip()
        if not 0 <= ident <= 459 or not name or ident in rows or name in names:
            raise ValueError("invalid or duplicate ontology entry")
        rows[ident] = name
        names.add(name)
    if rows.get(0) != "background" or any(
        rows.get(i) != n for i, n in zip(NATIVE_IDS, CLASSES)
    ):
        raise ValueError("native PC59 IDs and semantic names disagree")
    return {i: NATIVE_IDS.index(i) if i in NATIVE_IDS else 255 for i in rows}


def map_target(raw, mapping):
    data = loadmat(io.BytesIO(raw))
    if "LabelMap" not in data:
        raise ValueError("MAT LabelMap is required")
    target = np.asarray(data["LabelMap"])
    if target.ndim != 2 or not target.size or target.dtype.kind not in "iu":
        raise ValueError("nonempty integer native ID mask required")
    seen = set(map(int, np.unique(target)))
    if not seen <= set(mapping):
        raise ValueError("unrecognized native ontology ID")
    out = np.full(target.shape, 255, np.uint8)
    for value in seen:
        out[target == value] = mapping[value]
    return out


def convert(voc_root, context_root, out, *, fixture_counts=None):
    voc_root, context_root, out = map(Path, (voc_root, context_root, out))
    if out.exists():
        raise FileExistsError("staging output must be a new directory")
    counts = (4998, 5105) if fixture_counts is None else fixture_counts
    if len(counts) != 2 or any(type(v) is not int or v < 1 for v in counts):
        raise ValueError("positive split counts required")
    sources, plan, ids, images = {}, [], set(), {}

    def read(kind, name):
        root = voc_root if kind == "voc" else context_root
        raw = asset_bytes(root, {"path": name})
        key = kind + "/" + name
        digest = hashlib.sha256(raw).hexdigest()
        if key in sources and sources[key] != digest:
            raise ValueError("native input changed during conversion")
        sources[key] = digest
        return raw

    mapping = ontology(read("context", "labels.txt"))
    data = semantic_staging.manifest(
        CLASSES,
        "VOC2010 ImageSets/Main train/val; native PC59 ontology verified; release authenticity not verified",
    )
    for split, native, expected in zip(
        ("train", "validation"), ("train", "val"), counts
    ):
        rows = read("voc", f"ImageSets/Main/{native}.txt").decode().splitlines()
        listed = [line.strip() for line in rows if line.strip()]
        if len(listed) != expected:
            raise ValueError("native split population differs")
        for ident in listed:
            if (
                len(ident.split()) != 1
                or "/" in ident
                or "\\" in ident
                or ident in {".", ".."}
                or ident in ids
            ):
                raise ValueError("invalid, repeated or overlapping native sample ID")
            ids.add(ident)
            image = f"JPEGImages/{ident}.jpg"
            target = ident + ".mat"
            rgb = read("voc", image)
            digest = hashlib.sha256(rgb).hexdigest()
            if digest in images and images[digest] != split:
                raise ValueError("cross-split image-byte overlap")
            images[digest] = split
            mask = map_target(read("context", target), mapping)
            with Image.open(io.BytesIO(rgb)) as im:
                if mask.shape != (im.height, im.width):
                    raise ValueError("image and native mask geometry differ")
            plan.append((split, ident, image, target))
            data[split].append(
                {
                    "image": f"images/{split}/{ident}.jpg",
                    "mask": f"masks/{split}/{ident}.png",
                }
            )

    # Validate all native rows before creating output. A later I/O failure leaves
    # no success manifest; the caller must select a new output for a retry.
    def assets():
        for split, ident, image, target in plan:
            yield f"images/{split}/{ident}.jpg", read("voc", image)
            mask = map_target(read("context", target), mapping)
            buffer = io.BytesIO()
            Image.fromarray(mask).save(buffer, format="PNG")
            yield f"masks/{split}/{ident}.png", buffer.getvalue()

    semantic_staging.write(out, data, sources, assets())
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voc-root", required=True)
    parser.add_argument("--context-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    convert(args.voc_root, args.context_root, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
