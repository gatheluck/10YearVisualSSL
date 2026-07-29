#!/usr/bin/env python3
"""The whole chain, run for real: resolve-config -> adapter -> contract-test.

Each piece has its own tests. **This file exists because pieces that each pass
their own tests can still fail to meet.** The chain is what a port actually
has to survive, so it is exercised end to end, through subprocesses, judged by
exit status.

`methods/_reference` is the adapter under test here. It trains nothing. Its
whole purpose is to be a known-good implementation of the contract, so that
when a real method fails `contract-test` the chain itself is not in question.
It also gives every later adapter something to copy that is known to pass.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
REFERENCE = ROOT / "methods" / "_reference"


class Chain(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="e2e-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.cfg_dir = self.tmp / "configs"
        self.cfg_dir.mkdir()

    # -- the three steps ---------------------------------------------------

    def resolve(self, authoring: dict, **sets) -> Path:
        src = self.cfg_dir / "run.json"
        src.write_text(json.dumps(authoring), encoding="utf-8")
        out = self.tmp / "resolved.json"
        args = [sys.executable, str(BIN / "resolve-config.py"),
                "--config", str(src), "--out", str(out)]
        for k, v in sets.items():
            args += ["--set", f"{k}={v}"]
        r = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return out

    def adapt(self, config: Path, out: Path, env_extra: dict | None = None):
        env = {**os.environ, "PYTHONPATH": str(ROOT), **(env_extra or {})}
        return subprocess.run(
            [sys.executable, "-m", "adapter",
             "--config", str(config), "--out", str(out)],
            cwd=REFERENCE, env=env, capture_output=True, text=True)

    def verify(self, out: Path, config: Path, exit_status: int):
        return subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"),
             "--out", str(out), "--config", str(config),
             "--exit-status", str(exit_status)],
            capture_output=True, text=True)

    def full(self, authoring: dict, name: str = "run", **sets):
        cfg = self.resolve(authoring, **sets)
        out = self.tmp / name
        a = self.adapt(cfg, out)
        v = self.verify(out, cfg, a.returncode)
        return cfg, out, a, v


class TestTheChainHolds(Chain):
    def test_a_conforming_run_passes_the_contract(self):
        _, _, a, v = self.full({"seed": 0, "metrics": {"top1": 42.5}})
        self.assertEqual(a.returncode, 0, a.stderr)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    def test_the_manifest_records_the_config_that_was_handed_over(self):
        """The link the whole chain hangs from."""
        cfg, out, _, _ = self.full({"seed": 0, "metrics": {"top1": 1.0}})
        man = json.loads((out / "run_manifest.json").read_text())
        want = subprocess.run(
            [sys.executable, str(BIN / "resolve-config.py"),
             "--config", str(self.cfg_dir / "run.json"), "--print-hash"],
            capture_output=True, text=True).stdout.strip()
        self.assertEqual(man["config_sha256"], want)

    def test_includes_and_substitutions_travel_the_whole_way(self):
        (self.cfg_dir / "base.json").write_text(
            json.dumps({"seed": 3, "metrics": {"top1": 9.0}}), encoding="utf-8")
        _, out, a, v = self.full(
            {"include": ["base.json"], "data_root": "${DATA}"},
            DATA="/mnt/d")
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)
        self.assertEqual(json.loads((out / "run_manifest.json").read_text())
                         ["seed"], 3)

    def test_the_metrics_reach_the_output(self):
        _, out, _, _ = self.full({"seed": 0, "metrics": {"top1": 42.5}})
        self.assertEqual(
            json.loads((out / "metrics.json").read_text())["metrics"],
            {"top1": 42.5})

    def test_world_size_from_the_launcher_is_recorded(self):
        cfg = self.resolve({"seed": 0, "metrics": {"top1": 1.0}})
        out = self.tmp / "w"
        self.adapt(cfg, out, {"WORLD_SIZE": "8"})
        self.assertEqual(json.loads((out / "run_manifest.json").read_text())
                         ["world_size"], 8)


class TestReproducibility(Chain):
    """The property the repository exists for, measured rather than asserted."""

    def test_the_same_config_produces_the_same_artifacts(self):
        cfg = self.resolve({"seed": 0, "metrics": {"top1": 42.5}})
        digests = []
        for name in ("a", "b"):
            out = self.tmp / name
            self.adapt(cfg, out)
            man = json.loads((out / "run_manifest.json").read_text())
            digests.append({x["path"]: x["sha256"] for x in man["artifacts"]})
        self.assertEqual(digests[0], digests[1])
        self.assertIn("encoder.pt", digests[0])

    def test_a_different_config_produces_different_artifacts(self):
        """Artifacts that never change would pass the test above."""
        digests = []
        for i, name in enumerate(("a", "b")):
            cfg = self.resolve({"seed": i, "metrics": {"top1": 1.0}})
            out = self.tmp / name
            self.adapt(cfg, out)
            man = json.loads((out / "run_manifest.json").read_text())
            digests.append({x["path"]: x["sha256"] for x in man["artifacts"]})
        self.assertNotEqual(digests[0]["encoder.pt"], digests[1]["encoder.pt"])


class TestTheChainCatchesTrouble(Chain):
    """A chain that only ever passes proves nothing."""

    def test_a_failing_run_is_reported_as_a_failure_by_both(self):
        _, out, a, v = self.full({"seed": 0, "fail": "deliberate"})
        self.assertNotEqual(a.returncode, 0)
        self.assertEqual(json.loads((out / "run_manifest.json").read_text())
                         ["status"], "failed")
        self.assertNotEqual(v.returncode, 0)

    def test_a_correctly_reported_failure_is_not_a_contract_violation(self):
        """Failing is allowed. Failing dishonestly is not."""
        _, _, a, v = self.full({"seed": 0, "fail": "deliberate"})
        self.assertIn("correctly", v.stdout)

    def test_tampering_with_an_artifact_afterwards_is_caught(self):
        cfg, out, a, _ = self.full({"seed": 0, "metrics": {"top1": 1.0}})
        (out / "encoder.pt").write_bytes(b"swapped")
        v = self.verify(out, cfg, a.returncode)
        self.assertNotEqual(v.returncode, 0)
        self.assertIn("artifact-sha256", v.stdout)

    def test_a_stray_file_appearing_afterwards_is_caught(self):
        cfg, out, a, _ = self.full({"seed": 0, "metrics": {"top1": 1.0}})
        (out / "stray.log").write_text("who wrote me")
        v = self.verify(out, cfg, a.returncode)
        self.assertNotEqual(v.returncode, 0)
        self.assertIn("unlisted-file", v.stdout)

    def test_editing_the_config_after_the_run_is_caught(self):
        cfg, out, a, _ = self.full({"seed": 0, "metrics": {"top1": 1.0}})
        cfg.write_text('{"seed":999}\n', encoding="utf-8")
        v = self.verify(out, cfg, a.returncode)
        self.assertNotEqual(v.returncode, 0)
        self.assertIn("config-mismatch", v.stdout)

    def test_a_config_with_no_seed_never_reaches_a_run(self):
        cfg = self.resolve({"metrics": {"top1": 1.0}})
        out = self.tmp / "noseed"
        a = self.adapt(cfg, out)
        self.assertNotEqual(a.returncode, 0)
        self.assertIn("seed", a.stderr + a.stdout)
        self.assertFalse((out / "run_manifest.json").exists())


class TestTheAdapterNeedsNothingInstalled(Chain):
    def test_it_runs_without_site_packages(self):
        """`-S` removes site-packages, so nothing installed can be reached.

        That is the property that matters: the adapter has to work inside a
        method environment pinned to its own dependencies, where none of ours
        are present.

        Not `-I`: isolated mode also discards PYTHONPATH, so the run failed
        with "No module named adapter" and would have proved nothing about
        installed packages.
        """
        cfg = self.resolve({"seed": 0, "metrics": {"top1": 1.0}})
        out = self.tmp / "iso"
        r = subprocess.run(
            [sys.executable, "-S", "-m", "adapter",
             "--config", str(cfg), "--out", str(out)],
            cwd=REFERENCE, capture_output=True, text=True,
            env={"PYTHONPATH": f"{ROOT}{os.pathsep}{REFERENCE}",
                 "PATH": os.environ.get("PATH", "")})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.verify(out, cfg, r.returncode).returncode, 0)

    def test_site_packages_really_was_absent(self):
        """Otherwise the test above would pass even if -S did nothing."""
        r = subprocess.run(
            [sys.executable, "-S", "-c",
             "import sys; print(any('site-packages' in p for p in sys.path))"],
            capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
