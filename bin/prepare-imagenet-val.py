#!/usr/bin/env python3
"""Turn HuggingFace `ILSVRC/imagenet-1k` val parquet shards into an ImageFolder.

The feature dump (`bin/extract-features.py`) reads ImageNet val as an
`ImageFolder`: `<DATA_ROOT>/val/<wnid>/*.JPEG`, one directory per class named by
its WordNet id. The canonical download ships val as parquet shards whose rows
carry the original JPEG bytes and an integer label `0..999`. This tool writes the
one out as the other, and refuses to call a partial or mislabelled result done.

    python3 bin/prepare-imagenet-val.py \
        --parquet-dir <dir with val-*.parquet> \
        --classes-py  <dir>/classes.py \
        --out         /data/visual_ssl/datasets/imagenet \
        --split val

Design (see tests/test_prepare_imagenet_val.py and DESIGN 2.4):

* The label -> wnid map is read from the dataset's own `classes.py`, in label
  order. That file is parsed, never executed, and only a run of wnid-shaped keys
  is accepted -- a wrong parse fails loudly rather than producing a plausible
  wrong answer.
* Nothing is dropped in silence: a shard missing a column, a row with no image
  bytes, or a final layout that is not exactly `expect_classes` classes of
  `expect_per_class` images each, is a hard error.
* The image bytes are written through unchanged -- the file on disk is the
  ImageNet original, not a re-encode.

pyarrow is imported lazily, so the pure logic (parsing, mapping, verification)
imports and runs in the base environment; only the parquet reader needs the
`.venvs/_dataprep` stack.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path

WNID_RE = re.compile(r"^n\d{8}$")

# Full ImageNet-1k val: 1000 classes, 50 images each.
DEFAULT_EXPECT_CLASSES = 1000
DEFAULT_EXPECT_PER_CLASS = 50


class PrepareError(Exception):
    """A fault that must stop the run -- never a silent skip."""


class SchemaError(PrepareError):
    """A parquet shard does not carry the columns/values we require."""


class LayoutError(PrepareError):
    """The written layout is not the exact, complete val set."""


# -- the label -> wnid map, from the dataset's own classes.py ----------------

def _keys_in_source_order(node):
    """The string keys of a dict literal or an OrderedDict/dict([...]) call,
    in source order, or None if `node` is not such a mapping."""
    if isinstance(node, ast.Dict):
        keys = []
        for k in node.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.append(k.value)
            else:
                return None
        return keys
    if isinstance(node, ast.Call):
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name not in ("OrderedDict", "dict"):
            return None
        if len(node.args) != 1:
            return None
        arg = node.args[0]
        if not isinstance(arg, (ast.List, ast.Tuple)):
            return None
        keys = []
        for elt in arg.elts:
            if (isinstance(elt, ast.Tuple) and elt.elts
                    and isinstance(elt.elts[0], ast.Constant)
                    and isinstance(elt.elts[0].value, str)):
                keys.append(elt.elts[0].value)
            else:
                return None
        return keys
    return None


def parse_wnid_order(classes_py_text):
    """The wnids from a `classes.py`, in label order.

    Parses (does not execute) the file, takes the first module-level assignment
    whose value is a mapping of wnid-shaped string keys, and returns those keys
    in source order. Raises PrepareError if no such mapping is found or a key is
    not wnid-shaped -- a wrong or empty parse is an error, not a wrong answer.
    """
    try:
        tree = ast.parse(classes_py_text)
    except SyntaxError as exc:
        raise PrepareError(f"classes.py does not parse: {exc}") from exc

    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign):
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value = node.value
        if value is None:
            continue
        keys = _keys_in_source_order(value)
        if not keys:
            continue
        if all(WNID_RE.match(k) for k in keys):
            return keys

    raise PrepareError(
        "no wnid mapping found in classes.py: expected a module-level "
        "assignment of an ordered mapping whose keys are wnids (n########)")


def label_to_wnid(wnids, label):
    """The wnid for an integer label, or PrepareError if out of range."""
    if not isinstance(label, int) or label < 0 or label >= len(wnids):
        raise PrepareError(
            f"label {label!r} is outside 0..{len(wnids) - 1} "
            f"({len(wnids)} wnids known)")
    return wnids[label]


# -- output naming -----------------------------------------------------------

def output_filename(row_path, wnid, seq):
    """The filename to write a row under.

    Keep the row's own basename when it is a JPEG (so the file matches the
    ImageNet original); otherwise synthesise a stable, unique JPEG name.
    """
    if row_path:
        base = os.path.basename(str(row_path))
        if base and base.lower().endswith((".jpeg", ".jpg")):
            return base
    return f"{wnid}_{seq:08d}.JPEG"


# -- the parquet reader (pyarrow) --------------------------------------------

def iter_rows(parquet_path):
    """Yield (label:int, image_bytes:bytes, row_path:str|None) for a shard.

    Raises SchemaError -- never skips -- on a missing column, an unexpected
    image column type, or a row carrying no image bytes.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    names = table.column_names
    for required in ("image", "label"):
        if required not in names:
            raise SchemaError(
                f"{parquet_path}: missing required column {required!r} "
                f"(has {names})")

    images = table.column("image").to_pylist()
    labels = table.column("label").to_pylist()
    for idx, (im, lab) in enumerate(zip(images, labels)):
        if isinstance(im, dict):
            data = im.get("bytes")
            path = im.get("path")
        elif isinstance(im, (bytes, bytearray)):
            data = im
            path = None
        else:
            raise SchemaError(
                f"{parquet_path} row {idx}: image column is neither a "
                f"struct nor binary ({type(im).__name__})")
        if data is None:
            raise SchemaError(
                f"{parquet_path} row {idx}: no image bytes")
        if not isinstance(lab, int):
            raise SchemaError(
                f"{parquet_path} row {idx}: label is not an integer "
                f"({type(lab).__name__})")
        yield lab, bytes(data), path


# -- the write, and the completeness check -----------------------------------

def prepare(parquet_paths, wnids, out_root, split):
    """Write every row of every shard under `<out_root>/<split>/<wnid>/`.

    Returns the per-wnid image count. Raises on any faulty row (via iter_rows)
    or out-of-range label; the caller verifies completeness with verify_layout.
    """
    split_root = Path(out_root) / split
    counts = {}
    seq = 0
    for shard in parquet_paths:
        for label, data, row_path in iter_rows(shard):
            wnid = label_to_wnid(wnids, label)
            dest_dir = split_root / wnid
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / output_filename(row_path, wnid, seq)).write_bytes(data)
            counts[wnid] = counts.get(wnid, 0) + 1
            seq += 1
    return counts


def verify_layout(counts, *, expect_classes, expect_per_class):
    """Raise LayoutError unless `counts` is exactly the complete val set."""
    problems = []
    if len(counts) != expect_classes:
        problems.append(
            f"got {len(counts)} classes, expected {expect_classes}")
    wrong = {w: c for w, c in counts.items() if c != expect_per_class}
    if wrong:
        sample = ", ".join(f"{w}={c}" for w, c in sorted(wrong.items())[:5])
        problems.append(
            f"{len(wrong)} class(es) do not have exactly {expect_per_class} "
            f"images (e.g. {sample})")
    total = sum(counts.values())
    if total != expect_classes * expect_per_class:
        problems.append(
            f"got {total} images, expected "
            f"{expect_classes * expect_per_class}")
    if problems:
        raise LayoutError("; ".join(problems))


def _sample_decode_check(split_root, n):
    """Decode n files with Pillow as a sanity check; raise on any failure.

    Deterministic (walks in sorted order, strides evenly) so a rerun checks the
    same files and there is no unseeded randomness.
    """
    if n <= 0:
        return 0
    from PIL import Image

    files = sorted(Path(split_root).rglob("*.JPEG"))
    if not files:
        return 0
    stride = max(1, len(files) // n)
    checked = 0
    for f in files[::stride][:n]:
        with Image.open(f) as im:
            im.verify()
        checked += 1
    return checked


def _find_shards(parquet_dir, split):
    shards = sorted(Path(parquet_dir).glob(f"{split}-*.parquet"))
    if not shards:
        raise PrepareError(
            f"no {split}-*.parquet shards found under {parquet_dir}")
    return shards


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parquet-dir", required=True,
                    help="directory holding the <split>-*.parquet shards")
    ap.add_argument("--classes-py", required=True,
                    help="path to the dataset's classes.py (label -> wnid)")
    ap.add_argument("--out", required=True,
                    help="dataset root; files go under <out>/<split>/<wnid>/")
    ap.add_argument("--split", default="val")
    ap.add_argument("--expect-classes", type=int,
                    default=DEFAULT_EXPECT_CLASSES)
    ap.add_argument("--expect-per-class", type=int,
                    default=DEFAULT_EXPECT_PER_CLASS)
    ap.add_argument("--decode-check", type=int, default=0,
                    help="decode this many written files with Pillow as a "
                         "sanity check (0 = skip)")
    args = ap.parse_args(argv)

    wnids = parse_wnid_order(Path(args.classes_py).read_text(encoding="utf-8"))
    print(f"classes.py: {len(wnids)} wnids in label order "
          f"({wnids[0]} .. {wnids[-1]})")

    shards = _find_shards(args.parquet_dir, args.split)
    print(f"{len(shards)} {args.split} shard(s) under {args.parquet_dir}")

    counts = prepare(shards, wnids, args.out, args.split)
    print(f"wrote {sum(counts.values())} images across {len(counts)} classes "
          f"to {Path(args.out) / args.split}")

    verify_layout(counts, expect_classes=args.expect_classes,
                  expect_per_class=args.expect_per_class)
    print(f"layout verified: {args.expect_classes} classes x "
          f"{args.expect_per_class} images")

    if args.decode_check:
        checked = _sample_decode_check(Path(args.out) / args.split,
                                       args.decode_check)
        print(f"decode check: {checked} file(s) opened cleanly")

    print("OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PrepareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
