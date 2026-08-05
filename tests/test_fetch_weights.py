#!/usr/bin/env python3
"""Specification for bin/fetch-weights.py: a pinned, hash-checked download.

External pretrained weights -- the VAR VQVAE tokeniser, and the frozen-backbone
foundation models to come -- are neither shipped nor fetched in CI; a real or
GPU run obtains them here. The recorded sha256 is the contract: a download that
does not match is a failure, not a silent substitution, and it must not leave a
file behind that later reads as verified.

No network is used here. The artifact is served over a `file://` URL, so the
positive and negative controls both run offline in the standard suite.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "bin" / "fetch-weights.py"


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="fw-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.blob = self.tmp / "weights.bin"
        self.blob.write_bytes(b"pretend-weights\n")
        self.sha = hashlib.sha256(self.blob.read_bytes()).hexdigest()

    def provenance(self, sha: str) -> Path:
        p = self.tmp / "provenance.json"
        p.write_text(json.dumps({"tokenizer_artifact": {
            "url": self.blob.as_uri(),
            "filename": "weights.bin",
            "sha256": sha}}), encoding="utf-8")
        return p

    def run_tool(self, prov: Path, out: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(TOOL), "--provenance", str(prov),
             "--out", str(out)],
            capture_output=True, text=True)


class TestFetchWeights(Base):
    def test_a_matching_hash_downloads_and_verifies(self):
        out = self.tmp / "dl"
        r = self.run_tool(self.provenance(self.sha), out)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        got = out / "weights.bin"
        self.assertTrue(got.is_file(), "the verified file was not written")
        self.assertEqual(
            hashlib.sha256(got.read_bytes()).hexdigest(), self.sha)

    def test_a_wrong_hash_is_refused(self):
        out = self.tmp / "dl"
        r = self.run_tool(self.provenance("0" * 64), out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("sha256", (r.stdout + r.stderr).lower())

    def test_a_rejected_download_leaves_no_file_claiming_success(self):
        # The no-silent-failure property: a hash mismatch must not leave a file
        # that a later step would read as the verified artifact.
        out = self.tmp / "dl"
        self.run_tool(self.provenance("0" * 64), out)
        self.assertFalse((out / "weights.bin").exists(),
                         "a rejected download left its file behind")

    def test_a_missing_artifact_section_is_refused(self):
        p = self.tmp / "provenance.json"
        p.write_text(json.dumps({"upstream": {"repo": "x"}}), encoding="utf-8")
        r = self.run_tool(p, self.tmp / "dl")
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
