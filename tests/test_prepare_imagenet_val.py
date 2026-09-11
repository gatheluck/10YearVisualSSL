#!/usr/bin/env python3
"""Specification for bin/prepare-imagenet-val.py.

The feature dump (`bin/extract-features.py`) reads ImageNet val as an
`ImageFolder`: a per-class subdirectory under `<DATA_ROOT>/val/`, one directory
per class, its name the WordNet id (`n01440764`, ...). The canonical source we
pull from -- HuggingFace `ILSVRC/imagenet-1k` -- ships val instead as a handful
of parquet shards whose rows carry the original JPEG bytes and an integer label
`0..999`. This tool turns the one into the other.

Three properties matter, and each is pinned here:

1. **The label -> wnid map is authoritative, not invented.** It is read from the
   dataset's own `classes.py` (an ordered wnid -> description mapping in label
   order), so we never hand-transcribe 1000 ids. The parser reads the ordering
   out of that file's source (it does not execute it), and refuses anything that
   is not a run of wnid-shaped keys -- a wrong or empty parse is a red error, not
   a plausible-looking wrong answer.

2. **Nothing is dropped in silence** (DESIGN 2.4). A row whose schema does not
   match, a shard that cannot be read, a class that ends up with the wrong number
   of images -- each is a loud failure. "50,000 images across 1000 classes, 50
   each" is verified mechanically; a partial result never passes as complete.

3. **The image bytes are copied, not re-encoded.** The parquet stores the
   original JPEG; we write those bytes through unchanged, so the file on disk is
   byte-for-byte the ImageNet original.

The pure logic -- parsing the wnid order, mapping a label, and verifying the
final layout -- is standard-library only and runs in the base environment. The
parquet reader needs pyarrow, so the reader-backed tests are gated on pyarrow and
skip (announced, never silent) where it is absent; they run under
`.venvs/_dataprep`.

This is a data tool, not a method: it names no method.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
TOOL = BIN / "prepare-imagenet-val.py"
_MOD_NAME = "prepare_imagenet_val_tool"


def tool():
    """Load bin/prepare-imagenet-val.py by path (its name is not importable).

    A missing tool is a red test, never a silent skip.
    """
    if _MOD_NAME in sys.modules:
        return sys.modules[_MOD_NAME]
    spec = importlib.util.spec_from_file_location(_MOD_NAME, TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MOD_NAME] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[_MOD_NAME]
        raise
    return mod


def _have_pyarrow():
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
        return True
    except ImportError:
        return False


# A tiny classes.py in each of the two shapes the upstream file has used.
_CLASSES_DICT = (
    "IMAGENET2012_CLASSES = {\n"
    '    "n01440764": "tench, Tinca tinca",\n'
    '    "n01443537": "goldfish, Carassius auratus",\n'
    '    "n01484850": "great white shark",\n'
    "}\n")

_CLASSES_ORDEREDDICT = (
    "from collections import OrderedDict\n"
    "IMAGENET2012_CLASSES = OrderedDict(\n"
    "    [\n"
    '        ("n01440764", "tench, Tinca tinca"),\n'
    '        ("n01443537", "goldfish, Carassius auratus"),\n'
    '        ("n01484850", "great white shark"),\n'
    "    ]\n"
    ")\n")


# -- the wnid order parser (pure) --------------------------------------------

class TestParseWnidOrder(unittest.TestCase):
    def test_positive_dict_literal_is_parsed_in_order(self):
        t = tool()
        self.assertEqual(
            t.parse_wnid_order(_CLASSES_DICT),
            ["n01440764", "n01443537", "n01484850"])

    def test_positive_ordereddict_of_tuples_is_parsed_in_order(self):
        t = tool()
        self.assertEqual(
            t.parse_wnid_order(_CLASSES_ORDEREDDICT),
            ["n01440764", "n01443537", "n01484850"])

    def test_negative_no_mapping_is_an_error(self):
        t = tool()
        with self.assertRaises(t.PrepareError):
            t.parse_wnid_order("x = 1\n")

    def test_negative_non_wnid_keys_are_an_error(self):
        # A different assignment picked up by mistake must not pass as the map.
        t = tool()
        with self.assertRaises(t.PrepareError):
            t.parse_wnid_order('LABELS = {"tench": 0, "goldfish": 1}\n')

    def test_negative_a_source_that_does_not_parse_is_an_error(self):
        t = tool()
        with self.assertRaises(t.PrepareError):
            t.parse_wnid_order("def (:\n")


# -- label -> wnid (pure) ----------------------------------------------------

class TestLabelToWnid(unittest.TestCase):
    WNIDS = ["n01440764", "n01443537", "n01484850"]

    def test_positive_maps_by_index(self):
        t = tool()
        self.assertEqual(t.label_to_wnid(self.WNIDS, 0), "n01440764")
        self.assertEqual(t.label_to_wnid(self.WNIDS, 2), "n01484850")

    def test_negative_out_of_range_is_an_error(self):
        t = tool()
        with self.assertRaises(t.PrepareError):
            t.label_to_wnid(self.WNIDS, 3)
        with self.assertRaises(t.PrepareError):
            t.label_to_wnid(self.WNIDS, -1)


# -- layout verification (pure) ----------------------------------------------

class TestVerifyLayout(unittest.TestCase):
    def test_positive_exact_counts_pass(self):
        t = tool()
        counts = {f"n{i:08d}": 2 for i in range(3)}
        t.verify_layout(counts, expect_classes=3, expect_per_class=2)  # no raise

    def test_negative_wrong_class_count_fails(self):
        t = tool()
        counts = {f"n{i:08d}": 2 for i in range(2)}
        with self.assertRaises(t.LayoutError):
            t.verify_layout(counts, expect_classes=3, expect_per_class=2)

    def test_negative_a_short_class_fails(self):
        t = tool()
        counts = {"n00000000": 2, "n00000001": 1, "n00000002": 2}
        with self.assertRaises(t.LayoutError):
            t.verify_layout(counts, expect_classes=3, expect_per_class=2)

    def test_negative_an_overfull_class_fails(self):
        t = tool()
        counts = {"n00000000": 2, "n00000001": 3, "n00000002": 2}
        with self.assertRaises(t.LayoutError):
            t.verify_layout(counts, expect_classes=3, expect_per_class=2)


# -- output filename (pure) --------------------------------------------------

class TestOutputFilename(unittest.TestCase):
    def test_it_keeps_a_jpeg_basename_from_the_row(self):
        t = tool()
        self.assertEqual(
            t.output_filename("ILSVRC2012_val_00000001.JPEG", "n01440764", 0),
            "ILSVRC2012_val_00000001.JPEG")

    def test_it_strips_any_directory_from_the_row_path(self):
        t = tool()
        self.assertEqual(
            t.output_filename("val/n01440764/x.JPEG", "n01440764", 0),
            "x.JPEG")

    def test_it_synthesises_a_jpeg_name_when_the_row_has_none(self):
        t = tool()
        name = t.output_filename(None, "n01440764", 7)
        self.assertTrue(name.endswith(".JPEG"))
        self.assertIn("n01440764", name)


# -- the parquet reader and the end-to-end prepare (pyarrow-gated) -----------

@unittest.skipUnless(_have_pyarrow(),
                     "pyarrow is not installed here; run under .venvs/_dataprep")
class TestPrepareEndToEnd(unittest.TestCase):
    WNIDS = ["n01440764", "n01443537", "n01484850"]

    def _write_shard(self, path, rows):
        """rows: list of (label:int, img_bytes:bytes, path:str|None)."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
        images = pa.array(
            [{"bytes": b, "path": p} for (_l, b, p) in rows], type=image_type)
        labels = pa.array([l for (l, _b, _p) in rows], type=pa.int64())
        pq.write_table(pa.table({"image": images, "label": labels}), path)

    def test_rows_write_original_bytes_into_wnid_dirs(self):
        t = tool()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shard = tmp / "val-00000-of-00001.parquet"
            self._write_shard(shard, [
                (0, b"BYTES-A", "ILSVRC2012_val_00000001.JPEG"),
                (2, b"BYTES-B", "ILSVRC2012_val_00000002.JPEG"),
                (0, b"BYTES-C", "ILSVRC2012_val_00000003.JPEG"),
            ])
            out = tmp / "imagenet"
            counts = t.prepare([shard], self.WNIDS, out, "val")
            self.assertEqual(counts, {"n01440764": 2, "n01484850": 1})
            a = out / "val" / "n01440764" / "ILSVRC2012_val_00000001.JPEG"
            self.assertEqual(a.read_bytes(), b"BYTES-A")
            b = out / "val" / "n01484850" / "ILSVRC2012_val_00000002.JPEG"
            self.assertEqual(b.read_bytes(), b"BYTES-B")

    def test_a_plain_binary_image_column_is_read(self):
        # Some exports store the image column as raw binary rather than a struct.
        import pyarrow as pa
        import pyarrow.parquet as pq
        t = tool()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shard = tmp / "val-00000-of-00001.parquet"
            tbl = pa.table({
                "image": pa.array([b"RAW-A", b"RAW-B"], type=pa.binary()),
                "label": pa.array([0, 1], type=pa.int64())})
            pq.write_table(tbl, shard)
            got = list(t.iter_rows(shard))
            self.assertEqual([(l, b) for (l, b, _n) in got],
                             [(0, b"RAW-A"), (1, b"RAW-B")])

    def test_a_shard_missing_a_required_column_is_a_loud_error(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        t = tool()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shard = tmp / "val-00000-of-00001.parquet"
            # No `label` column -> the reader must refuse, never skip.
            pq.write_table(
                pa.table({"image": pa.array([b"X"], type=pa.binary())}), shard)
            with self.assertRaises(t.SchemaError):
                list(t.iter_rows(shard))

    def test_a_row_with_no_image_bytes_is_a_loud_error(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        t = tool()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shard = tmp / "val-00000-of-00001.parquet"
            image_type = pa.struct([("bytes", pa.binary()),
                                    ("path", pa.string())])
            tbl = pa.table({
                "image": pa.array([{"bytes": None, "path": "x.JPEG"}],
                                  type=image_type),
                "label": pa.array([0], type=pa.int64())})
            pq.write_table(tbl, shard)
            with self.assertRaises(t.SchemaError):
                list(t.iter_rows(shard))


if __name__ == "__main__":
    unittest.main()
