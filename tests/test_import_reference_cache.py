"""Pinned cache delivery preserves sample identity and refuses invalid inputs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

TOOL = Path(__file__).resolve().parents[1] / "bin/import-reference-cache.py"


@unittest.skipIf(np is None, "numpy required")
class TestCacheImport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.assertTrue(TOOL.is_file(), "cache import is not implemented")
        spec = importlib.util.spec_from_file_location("cache_import", TOOL)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.features = np.array([[3, 4], [5, 12], [8, 15], [7, 24]], dtype=np.float16)
        self.labels = np.array([0, 0, 1, 1], dtype=np.int64)
        self.profile = dict(id="example", count=4, feat_dim=2,
                            expected_labels=self.artifact("expected", self.labels), shards=[])
        for rank in range(2):
            ix = np.arange(rank, 4, 2)
            self.profile["shards"].append(dict(
                features=self.artifact(f"f{rank}", self.features[ix]),
                labels=self.artifact(f"y{rank}", self.labels[ix]),
                indices=self.artifact(f"i{rank}", ix)))

    def artifact(self, name, value):
        path = self.root / (name + ".npy")
        np.save(path, value)
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def test_restores_order_normalizes_and_keeps_sources_unchanged(self):
        before = {p: p.read_bytes() for p in self.root.glob("*.npy")}
        out = self.root / "out"
        self.module.import_cache(self.profile, out)
        x = np.load(out / "features.npy")
        self.assertEqual(x.dtype, np.float32)
        np.testing.assert_allclose(x, self.features.astype(np.float32) / np.array([5,13,17,25])[:,None], atol=1e-7)
        np.testing.assert_array_equal(np.load(out / "labels.npy"), self.labels)
        self.assertEqual(json.loads((out / "result.json").read_text())["status"], "ok")
        self.assertEqual(json.loads((out / "meta.json").read_text())["source_profile"], self.profile)
        for path, data in before.items():
            self.assertEqual(path.read_bytes(), data)

    def test_refuses_overwriting_an_existing_delivery(self):
        out = self.root / "out"
        out.mkdir()
        (out / "keep").write_text("existing")
        with self.assertRaises(FileExistsError):
            self.module.import_cache(self.profile, out)
        self.assertEqual((out / "keep").read_text(), "existing")

    def test_source_hash_change_is_rejected_before_publication(self):
        Path(self.profile["shards"][0]["features"]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.module.import_cache(self.profile, self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_invalid_order_labels_values_and_width_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "width"):
            self.module.import_cache({**self.profile, "feat_dim": 3}, self.root / "wrong-width")
        self.assertFalse((self.root / "wrong-width").exists())
        cases = [("indices", [0,0]), ("indices", [0.,2.]),
                 ("labels", [1,1]), ("labels", [0.,1.]),
                 ("features", [[0.,0.],[1.,2.]]),
                 ("features", [[float("nan"),1.],[1.,2.]]),
                 ("features", [[1.],[2.]])]
        for number, (key, value) in enumerate(cases):
            with self.subTest(key=key, value=value):
                profile = copy.deepcopy(self.profile)
                profile["shards"][0][key] = self.artifact("invalid", np.asarray(value))
                out = self.root / f"bad{number}"
                with self.assertRaises(ValueError):
                    self.module.import_cache(profile, out)
                self.assertFalse(out.exists())

    def test_partial_and_empty_shards_are_rejected(self):
        for shards in ([], self.profile["shards"][:1]):
            profile = {**self.profile, "shards": shards}
            with self.assertRaises(ValueError):
                self.module.import_cache(profile, self.root / "out")

    def test_cli_delivers_from_a_private_profile(self):
        import subprocess
        import sys
        profile = self.root / "profile.json"
        profile.write_text(json.dumps(self.profile))
        result = subprocess.run([sys.executable, str(TOOL), "--profile", str(profile),
                                 "--out", str(self.root / "cli")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        np.testing.assert_array_equal(np.load(self.root / "cli/labels.npy"), self.labels)
