#!/usr/bin/env python3
"""Specification for bin/extract-features.py.

One command must walk every method that can produce features, extract one
feature vector per image over a dataset split, and save the result under a
per-method directory -- so a paper figure can be regenerated from a single
run. The number of methods is large and grows, so the driver **discovers**
the methods that provide an extractor rather than carrying a hand-kept list
(the list is the exact thing this repository keeps drifting on).

This file is shared machinery -- it is about the driver, not any one method --
so it names no method: the synthetic directories below use invented names, and
the one real-tree check discovers rather than names. A given method's provider
is proven end to end in that method's own test file, where naming it is
allowed (`tests/test_method_14_simclrv1.py`).

The pure control logic -- discovery, run planning, the manifest, and the
exit-status decision -- is tested here with the standard library only, using
synthetic method directories and a fake provider. The numpy-backed feature
save needs numpy and is covered by a numpy-gated test that skips (announced,
never silent) where the stack is absent.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
TOOL = BIN / "extract-features.py"
_MOD_NAME = "extract_features_tool"


def tool():
    """Load bin/extract-features.py by path (its name is not importable).

    Fails loudly if the tool is absent -- a missing driver is a red test,
    never a silent skip.
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


def _method(root: Path, name: str, *, provider: bool = True,
            encoder: bool = False) -> None:
    """A synthetic method directory, optionally carrying a provider and/or a
    checkpoint. The provider body is never executed by the pure-logic tests --
    only its presence, matched by exact filename, is read."""
    d = root / name
    (d / "adapter").mkdir(parents=True, exist_ok=True)
    if provider:
        (d / "feature_provider.py").write_text(
            "def extract_val_features(**kw):\n"
            "    raise AssertionError('not called in pure-logic tests')\n",
            encoding="utf-8")
    if encoder:
        (d / "encoder.pt").write_bytes(b"not a real checkpoint")


class TestProviderDiscovery(unittest.TestCase):
    """A method is included iff it ships feature_provider.py -- matched as a
    whole filename, never as a substring."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="featdisc-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.methods = self.tmp / "methods"
        self.methods.mkdir()

    def test_discovers_only_methods_with_a_provider(self):
        _method(self.methods, "aa_with_provider", provider=True)
        _method(self.methods, "zz_without_provider", provider=False)
        found = tool().discover_providers(self.methods)
        self.assertIn("aa_with_provider", found)
        self.assertNotIn("zz_without_provider", found,
                         "a method with no provider was picked up")

    def test_a_decoy_filename_does_not_count(self):
        """The negative control a substring match would fail: a file whose
        name merely contains the provider filename is not a provider."""
        d = self.methods / "cc_decoy"
        d.mkdir()
        (d / "test_feature_provider.py").write_text("# not the provider\n",
                                                    encoding="utf-8")
        (d / "feature_provider.py.bak").write_text("# not it either\n",
                                                   encoding="utf-8")
        found = tool().discover_providers(self.methods)
        self.assertNotIn("cc_decoy", found,
                         "a decoy filename was mistaken for a provider")

    def test_discovery_is_not_vacuous_on_the_real_tree(self):
        """Against the real methods/ at least one provider must be found, or
        every run below would pass by discovering nothing."""
        found = tool().discover_providers(ROOT / "methods")
        self.assertTrue(found, "no feature providers found under methods/; "
                        "the driver would silently do nothing")


class TestRunPlanning(unittest.TestCase):
    """Every discovered method appears in the plan: one with a checkpoint is
    ready, one without is skipped with a stated reason. Nothing is dropped."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="featplan-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.methods = self.tmp / "methods"
        self.methods.mkdir()
        _method(self.methods, "has_ckpt", provider=True, encoder=True)
        _method(self.methods, "no_ckpt", provider=True, encoder=False)

    def _plan(self):
        t = tool()
        encoders = {"has_ckpt": str(self.methods / "has_ckpt" / "encoder.pt")}
        return {r["method"]: r for r in t.plan(self.methods, encoders)}

    def test_a_method_with_an_encoder_is_ready(self):
        rec = self._plan()["has_ckpt"]
        self.assertEqual(rec["status"], "ready")
        self.assertTrue(rec["encoder"], "a ready method has no encoder path")

    def test_a_method_without_an_encoder_is_skipped_with_a_reason(self):
        rec = self._plan()["no_ckpt"]
        self.assertEqual(rec["status"], "skipped")
        self.assertTrue(rec.get("reason"),
                        "a skip must state why (DESIGN 2.4: never silent)")

    def test_no_discovered_method_is_dropped_from_the_plan(self):
        plan = self._plan()
        self.assertEqual(set(plan), {"has_ckpt", "no_ckpt"},
                         "a discovered method vanished from the plan")


class TestManifestAndExitStatus(unittest.TestCase):
    """The manifest records every method's outcome, and the exit status is
    nonzero unless every discovered method produced features."""

    def test_manifest_lists_every_record_and_the_split(self):
        t = tool()
        records = [{"method": "a", "status": "ok"},
                   {"method": "b", "status": "skipped", "reason": "no encoder"}]
        man = t.build_manifest(records, data_root="/data/imagenet", split="val")
        self.assertEqual(man["data_root"], "/data/imagenet")
        self.assertEqual(man["split"], "val")
        self.assertEqual({r["method"] for r in man["records"]}, {"a", "b"})

    def test_exit_is_zero_only_when_all_ok(self):
        t = tool()
        ok = t.build_manifest([{"method": "a", "status": "ok"}],
                              data_root="/d", split="val")
        self.assertEqual(t.exit_status(ok), 0)

    def test_a_skip_makes_the_run_fail_by_default(self):
        """The paper run asks for every method; a silently missing one must
        not read as success."""
        t = tool()
        man = t.build_manifest(
            [{"method": "a", "status": "ok"},
             {"method": "b", "status": "skipped", "reason": "no encoder"}],
            data_root="/d", split="val")
        self.assertNotEqual(t.exit_status(man), 0)

    def test_allow_missing_forgives_a_skip_but_never_an_error(self):
        """--allow-missing downgrades a missing checkpoint to a warning, but an
        error (a provider that ran and failed) must still fail the run."""
        t = tool()
        skipped = t.build_manifest(
            [{"method": "a", "status": "ok"},
             {"method": "b", "status": "skipped", "reason": "no encoder"}],
            data_root="/d", split="val")
        self.assertEqual(t.exit_status(skipped, allow_missing=True), 0)
        errored = t.build_manifest(
            [{"method": "a", "status": "error", "reason": "boom"}],
            data_root="/d", split="val")
        self.assertNotEqual(t.exit_status(errored, allow_missing=True), 0)

    def test_an_error_makes_the_run_fail(self):
        t = tool()
        man = t.build_manifest(
            [{"method": "a", "status": "error", "reason": "boom"}],
            data_root="/d", split="val")
        self.assertNotEqual(t.exit_status(man), 0)


class TestFeatureSaveLayout(unittest.TestCase):
    """The on-disk layout a downstream visualisation reads. numpy-backed, so
    gated on numpy rather than run in the base environment."""

    def setUp(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy is not installed in this environment "
                          "(announced, not silent)")
        self.tmp = Path(tempfile.mkdtemp(prefix="featsave-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_it_writes_features_labels_and_meta(self):
        import numpy as np
        t = tool()
        feats = np.arange(12, dtype="float32").reshape(4, 3)
        labels = np.array([0, 1, 0, 1])
        out = self.tmp / "some_method"
        t.save_features(out, feats, labels, {"method": "some_method",
                                             "feat_dim": 3, "count": 4})
        got_f = np.load(out / "features.npy")
        got_l = np.load(out / "labels.npy")
        meta = json.loads((out / "meta.json").read_text())
        self.assertEqual(got_f.shape, (4, 3))
        self.assertEqual(got_l.tolist(), [0, 1, 0, 1])
        self.assertEqual(meta["feat_dim"], 3)
        self.assertEqual(meta["count"], 4)


class TestInterpreterSelection(unittest.TestCase):
    """Each method runs in its own interpreter: its own venv if it has one,
    else the current one. Discovered from the venvs tree, never listed."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="featpy-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_it_prefers_the_methods_own_venv(self):
        venvs = self.tmp / ".venvs"
        py = venvs / "some_method" / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("#!/bin/sh\n", encoding="utf-8")
        got = tool().python_for("some_method", venvs)
        self.assertEqual(Path(got), py)

    def test_it_falls_back_to_the_current_interpreter(self):
        venvs = self.tmp / ".venvs"        # empty: no per-method venv
        venvs.mkdir()
        got = tool().python_for("some_method", venvs)
        self.assertEqual(got, sys.executable)


class TestWorkerCommand(unittest.TestCase):
    """The argv the driver hands the isolated worker carries every input the
    provider needs, and marks the process as a worker."""

    def test_it_carries_the_worker_flag_and_all_inputs(self):
        cmd = tool().worker_command(
            "/usr/bin/python", provider="/m/feature_provider.py",
            encoder="/e/encoder.pt", data_root="/data", split="val",
            out_method_dir="/out/some_method", device="cpu",
            batch_size=64, num_workers=4)
        self.assertEqual(cmd[0], "/usr/bin/python")
        self.assertIn("--worker", cmd)
        for token in ("/m/feature_provider.py", "/e/encoder.pt", "/data",
                      "val", "/out/some_method", "cpu", "64", "4"):
            self.assertIn(token, cmd, f"{token!r} missing from worker command")


class TestWorkerResultMapping(unittest.TestCase):
    """A worker reports back through its exit code and a result.json. A clean
    exit with an ok result is ok; a crash, or a missing result, is an error
    that carries the worker's stderr -- never a silent success."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="featres-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.rec = {"method": "some_method", "status": "ready",
                    "provider": "/m/feature_provider.py", "encoder": "/e.pt"}

    def test_a_clean_exit_with_an_ok_result_is_ok(self):
        res = self.tmp / "result.json"
        res.write_text(json.dumps({"status": "ok", "feat_dim": 3, "count": 2}),
                       encoding="utf-8")
        got = tool()._record_from_worker(self.rec, 0, res, "")
        self.assertEqual(got["status"], "ok")
        self.assertEqual(got["feat_dim"], 3)
        self.assertEqual(got["count"], 2)

    def test_a_nonzero_exit_is_an_error_carrying_stderr(self):
        got = tool()._record_from_worker(
            self.rec, 1, self.tmp / "missing.json", "Traceback: boom")
        self.assertEqual(got["status"], "error")
        self.assertIn("boom", got["reason"])

    def test_a_result_reporting_failure_is_an_error(self):
        res = self.tmp / "result.json"
        res.write_text(json.dumps({"status": "error", "reason": "no encoder"}),
                       encoding="utf-8")
        got = tool()._record_from_worker(self.rec, 1, res, "")
        self.assertEqual(got["status"], "error")
        self.assertIn("no encoder", got["reason"])


def _synthetic_provider(body: str) -> str:
    """A standalone feature_provider.py, importable in a fresh interpreter."""
    return body


class TestIsolatedRunIntegration(unittest.TestCase):
    """The driver runs each method in a real subprocess and collects the
    result. Proven with a synthetic provider (no torch, no real method named):
    it returns tiny numpy features and records its own pid, so the test can
    assert the provider truly ran in another process."""

    def setUp(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy absent (announced, not silent)")
        self.tmp = Path(tempfile.mkdtemp(prefix="featiso-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.methods = self.tmp / "methods"
        self.methods.mkdir()

    def _method_with(self, name: str, provider_body: str) -> str:
        d = self.methods / name
        (d / "adapter").mkdir(parents=True)
        (d / "feature_provider.py").write_text(provider_body, encoding="utf-8")
        enc = d / "encoder.pt"
        enc.write_bytes(b"dummy encoder")
        return str(enc)

    def test_a_provider_runs_in_a_separate_process_and_is_saved(self):
        import numpy as np
        body = (
            "import os\n"
            "import numpy as np\n"
            "def extract_val_features(*, encoder_path, data_root, split,\n"
            "                         device, batch_size, num_workers):\n"
            "    feats = np.arange(6, dtype='float32').reshape(2, 3)\n"
            "    labels = np.array([0, 1])\n"
            "    meta = {'representation': 'raw', 'feat_dim': 3, 'count': 2,\n"
            "            'provider_pid': os.getpid()}\n"
            "    return feats, labels, meta\n")
        enc = self._method_with("good_method", body)
        out = self.tmp / "features"
        manifest = tool().run(
            self.methods, data_root=str(self.tmp / "data"), split="val",
            out=out, encoders={"good_method": enc}, encoders_root=None,
            device="cpu", batch_size=2, num_workers=0, venvs_root=None)

        rec = {r["method"]: r for r in manifest["records"]}["good_method"]
        self.assertEqual(rec["status"], "ok", rec.get("reason", ""))
        method_out = out / "good_method"
        feats = np.load(method_out / "features.npy")
        self.assertEqual(feats.shape, (2, 3))
        meta = json.loads((method_out / "meta.json").read_text())
        self.assertEqual(meta["encoder_sha256"],
                         tool().sha256_of(Path(enc)))
        self.assertNotEqual(meta["provider_pid"], os.getpid(),
                            "the provider ran in this process, not isolated")

    def test_a_crashing_provider_is_recorded_as_error_not_silently_dropped(self):
        body = (
            "def extract_val_features(**kw):\n"
            "    raise RuntimeError('boom in worker')\n")
        enc = self._method_with("bad_method", body)
        out = self.tmp / "features"
        manifest = tool().run(
            self.methods, data_root=str(self.tmp / "data"), split="val",
            out=out, encoders={"bad_method": enc}, encoders_root=None,
            device="cpu", batch_size=2, num_workers=0, venvs_root=None)
        rec = {r["method"]: r for r in manifest["records"]}["bad_method"]
        self.assertEqual(rec["status"], "error")
        self.assertTrue(rec.get("reason"), "an error must carry a reason")


if __name__ == "__main__":
    unittest.main()
