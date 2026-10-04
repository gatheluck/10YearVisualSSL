"""Full v1.2 numeric-label joins from the two captured Mapillary archives."""

from __future__ import annotations

import argparse
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from downstream import semantic_staging


def convert(images, labels, out, *, fixture_counts=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError("staging output must be a new directory")
    counts = (18000, 2000) if fixture_counts is None else fixture_counts
    if len(counts) != 2 or any(type(v) is not int or v < 1 for v in counts):
        raise ValueError("positive split counts required")
    sources, plan, seen = {}, [], {}
    with zipfile.ZipFile(images) as im, zipfile.ZipFile(labels) as lab:
        for archive in (im, lab):
            names = archive.namelist()
            if len(set(names)) != len(names):
                raise ValueError("duplicate archive member")

        def index(archive, prefix, suffix):
            found = {}
            for name in archive.namelist():
                if not name.startswith(prefix) or not name.endswith(suffix):
                    continue
                key = name[len(prefix) : -len(suffix)]
                if (
                    not key
                    or "/" in key
                    or "\\" in key
                    or key in {".", ".."}
                    or key in found
                ):
                    raise ValueError("invalid or duplicate native image ID")
                found[key] = name
            return found

        def read(archive, key, tag):
            raw = archive.read(key)
            digest = hashlib.sha256(raw).hexdigest()
            name = tag + "/" + key
            if name in sources and sources[name] != digest:
                raise ValueError("archive input changed")
            sources[name] = digest
            return raw

        data = semantic_staging.manifest(
            [str(i) for i in range(66)],
            "full training/validation archive joins; v1.2 numeric labels; release authenticity not verified",
        )
        for native, split, count in zip(
            ("training", "validation"), ("train", "validation"), counts
        ):
            a = index(im, native + "/images/", ".jpg")
            b = index(lab, native + "/v1.2/labels/", ".png")
            if set(a) != set(b) or len(a) != count:
                raise ValueError("native split pair population differs")
            for ident in sorted(a):
                image = read(im, a[ident], "images")
                target = read(lab, b[ident], "labels")
                digest = hashlib.sha256(image).hexdigest()
                if digest in seen and seen[digest] != split:
                    raise ValueError("cross-split image-byte overlap")
                seen[digest] = split
                with (
                    Image.open(io.BytesIO(target)) as mask,
                    Image.open(io.BytesIO(image)) as rgb,
                ):
                    array = np.asarray(mask)
                    if (
                        array.ndim != 2
                        or array.dtype.kind not in "iu"
                        or not np.isin(array, list(range(66)) + [255]).all()
                        or rgb.size != mask.size
                    ):
                        raise ValueError(
                            "native numeric label or image geometry invalid"
                        )
                row = {
                    "image": f"images/{split}/{ident}.jpg",
                    "mask": f"masks/{split}/{ident}.png",
                }
                data[split].append(row)
                plan.append((row, a[ident], b[ident]))

        def assets():
            for row, image, target in plan:
                yield row["image"], read(im, image, "images")
                yield row["mask"], read(lab, target, "labels")

        semantic_staging.write(out, data, sources, assets())
        return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    convert(args.images, args.labels, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
