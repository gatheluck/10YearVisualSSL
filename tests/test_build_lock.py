#!/usr/bin/env python3
"""Specification for bin/build-lock.py.

The tool renders a resolved dependency set into a CPU lock in this repository's
format -- `# <wheel-filename>` comments for the three target platforms and the
hashes for exactly those wheels. `uv` does not emit that shape, so the tool
exists to; and because a lock with a missing platform installs on some machines
and fails on others, the tool must **refuse** rather than emit a lock with a
hole.

These tests exercise the parsing, selection and formatting -- the parts that
decide correctness -- with fixture data, so they need no network. The network
fetches are the thin edges the pure functions are kept separate from.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "bin" / "build-lock.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("build_lock", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_lock"] = mod
    spec.loader.exec_module(mod)
    return mod


bl = load_tool()


def wheel(name: str, sha: str = "abc") -> dict:
    return {"filename": name, "sha256": sha}


class TestParseResolved(unittest.TestCase):
    def test_names_and_versions_are_read_hashes_ignored(self):
        text = ("# a header\n"
                "torch==2.13.0+cpu \\\n"
                "    --hash=sha256:deadbeef \\\n"
                "    --hash=sha256:cafef00d\n"
                "numpy==2.5.1\n")
        self.assertEqual(bl.parse_resolved(text),
                         [("torch", "2.13.0+cpu"), ("numpy", "2.5.1")])

    def test_constraint_and_option_lines_are_skipped(self):
        text = "-c /tmp/pins.txt\n--index-url https://x\nscipy==1.18.0\n"
        self.assertEqual(bl.parse_resolved(text), [("scipy", "1.18.0")])

    def test_markers_are_dropped_from_the_version(self):
        self.assertEqual(bl.parse_resolved("nvidia-cublas==13.1.1.3 ; x\n"),
                         [("nvidia-cublas", "13.1.1.3")])


class TestCp312Compatible(unittest.TestCase):
    def test_cp312_and_abi3_and_pure_python_are_accepted(self):
        self.assertTrue(bl.cp312_compatible("p-1-cp312-cp312-linux_x86_64.whl"))
        self.assertTrue(bl.cp312_compatible("p-1-cp37-abi3-linux_x86_64.whl"))
        self.assertTrue(bl.cp312_compatible("p-1-py3-none-any.whl"))

    def test_other_interpreters_and_sdists_are_rejected(self):
        self.assertFalse(bl.cp312_compatible("p-1-cp310-cp310-linux_x86_64.whl"))
        self.assertFalse(bl.cp312_compatible("p-1-pp39-pypy_x86_64.whl"))
        self.assertFalse(bl.cp312_compatible("p-1.tar.gz"))


class TestSelectWheels(unittest.TestCase):
    def test_one_wheel_per_target_is_chosen(self):
        files = [
            wheel("p-1-cp312-cp312-manylinux_2_28_x86_64.whl", "x"),
            wheel("p-1-cp312-cp312-manylinux_2_28_aarch64.whl", "a"),
            wheel("p-1-cp312-cp312-macosx_14_0_arm64.whl", "m"),
            wheel("p-1-cp312-cp312-win_amd64.whl", "w"),        # not a target
        ]
        chosen, missing = bl.select_wheels(files)
        self.assertEqual(missing, [])
        self.assertEqual([w["sha256"] for w in chosen], ["x", "a", "m"])

    def test_a_pure_python_wheel_serves_every_platform_alone(self):
        chosen, missing = bl.select_wheels([wheel("p-1-py3-none-any.whl", "u")])
        self.assertEqual(missing, [])
        self.assertEqual([w["sha256"] for w in chosen], ["u"])

    def test_the_highest_macos_version_is_chosen_deterministically(self):
        files = [
            wheel("p-1-cp312-cp312-manylinux_2_28_x86_64.whl", "x"),
            wheel("p-1-cp312-cp312-manylinux_2_28_aarch64.whl", "a"),
            wheel("p-1-cp312-cp312-macosx_11_0_arm64.whl", "m11"),
            wheel("p-1-cp312-cp312-macosx_14_0_arm64.whl", "m14"),
        ]
        chosen, _ = bl.select_wheels(files)
        self.assertIn("m14", [w["sha256"] for w in chosen])
        self.assertNotIn("m11", [w["sha256"] for w in chosen])

    def test_musllinux_does_not_stand_in_for_manylinux(self):
        """A musllinux wheel also contains 'x86_64'; it must not be accepted as
        the linux x86_64 wheel, or the lock would claim a coverage it lacks."""
        files = [wheel("p-1-cp312-cp312-musllinux_1_2_x86_64.whl", "musl")]
        chosen, missing = bl.select_wheels(files)
        self.assertIn("linux x86_64", missing)
        self.assertEqual(chosen, [])

    def test_a_missing_platform_is_reported_not_hidden(self):
        """The negative control: only x86_64 present, the other two reported."""
        files = [wheel("p-1-cp312-cp312-manylinux_2_28_x86_64.whl", "x")]
        chosen, missing = bl.select_wheels(files)
        self.assertEqual([w["sha256"] for w in chosen], ["x"])
        self.assertEqual(set(missing), {"linux aarch64", "macOS arm64"})


class TestFormatEntry(unittest.TestCase):
    def test_the_local_build_tag_is_stripped_from_the_pin(self):
        entry = bl.format_entry("torch", "2.13.0+cpu",
                                [wheel("torch-2.13.0+cpu-...x86_64.whl", "h")])
        self.assertIn("torch==2.13.0 \\", entry)
        self.assertNotIn("2.13.0+cpu \\", entry)
        # the +cpu WHEEL is still what is pinned
        self.assertIn("# torch-2.13.0+cpu-...x86_64.whl", entry)

    def test_hash_count_equals_wheel_count(self):
        wheels = [wheel("p-1-cp312-cp312-manylinux_2_28_x86_64.whl", "x"),
                  wheel("p-1-cp312-cp312-manylinux_2_28_aarch64.whl", "a"),
                  wheel("p-1-cp312-cp312-macosx_14_0_arm64.whl", "m")]
        entry = bl.format_entry("p", "1", wheels)
        self.assertEqual(entry.count("# "), 3)
        self.assertEqual(entry.count("--hash=sha256:"), 3)

    def test_only_the_last_hash_line_has_no_trailing_backslash(self):
        wheels = [wheel("p-1-cp312-cp312-manylinux_2_28_x86_64.whl", "x"),
                  wheel("p-1-cp312-cp312-macosx_14_0_arm64.whl", "m")]
        lines = bl.format_entry("p", "1", wheels).splitlines()
        hash_lines = [ln for ln in lines if "--hash" in ln]
        self.assertTrue(hash_lines[0].endswith("\\"))
        self.assertFalse(hash_lines[-1].endswith("\\"))


if __name__ == "__main__":
    unittest.main()
