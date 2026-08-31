#!/usr/bin/env python3
"""LeJEPA wired into the ARSSL harness (docs/STEP3_PORTING_PLAN.md, item A3:lejepa).

Phase A3 wires the lineage backbones already present as Step-1&2 ports into the
A1 ARSSL harness (`downstream/arssl.py`) so their Step-3 numbers reproduce here.
MAE and I-JEPA came first; each ships a self-contained, hand-written encoder that
is *not* timm-loadable, so each declares its own `downstream_backbone.py` provider
(a new spatial-backbone `KIND`) that the shared layer discovers.

LeJEPA is different, and measurement -- not the file name -- says so: it trains a
**standard timm ViT-B/16** backbone, and its `encoder.pt` is the bare backbone
(the `backbone.` prefix stripped at save time), so it loads straight into the
shared `vit` spatial-backbone kind, whose default arch is that very
`vit_base_patch16_224`. A dedicated provider would therefore be an empty
duplicate of the `vit` kind -- which this repo forbids ("never implement the same
rule twice"; a guard with no killed mutant is not a guard). So LeJEPA's wiring is
a **config, not code**: a method whose trained backbone is a shared kind declares
its frozen-backbone spec in its own `configs/downstream_arssl.json`, and the
ARSSL driver (JSON-native) runs that spec through the dense-task probes.

**The configs are discovered, not listed.** This test globs
`methods/*/configs/downstream_arssl.json` and validates every one it finds, so it
names no method's directory (`tests/test_no_hard_coded_methods.py`) -- exactly as
the MAE/I-JEPA provider tests reach their providers through discovery. The plan
records the artifact path (a prose-exempt `.md`); the machinery here never does.

Each discovered config must:

* declare a shared spatial-backbone `kind` (currently the timm `vit`) with a real,
  buildable arch -- config-based wiring is only for a backbone the shared layer
  already builds; a hand-written encoder ships a provider instead;
* accept a bare-key `encoder.pt` through that kind and reproduce it exactly (this
  is LeJEPA's saved format), while an alien checkpoint is refused, not
  half-loaded;
* drive the real ARSSL harness end to end over a dense task (ADE20K) to one `ok`.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Discovered by structure, so this test names no method (CLAUDE.md; the
# anti-pattern `tests/test_no_hard_coded_methods.py` guards).
CONFIG_GLOB = "methods/*/configs/downstream_arssl.json"


def discover_downstream_configs() -> "list[Path]":
    return sorted(ROOT.glob(CONFIG_GLOB))


try:
    import torch                                              # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# Imported unguarded when torch is present, so the module-under-test being absent
# is a real error (RED), never silently a skip.
if HAVE_TORCH:
    from downstream import spatial_backbones

try:
    import timm                                               # noqa: F401
    from tests.test_downstream_ade20k import tiny_ade, SMOKE_PROBE
    HAVE_TIMM = HAVE_TORCH and True
except ImportError:
    HAVE_TIMM = False

needs_timm = unittest.skipUnless(
    HAVE_TIMM, "the end-to-end ARSSL wiring drives the ADE20K runner (timm)")

# Overrides that turn a config's real ViT-B/16 into a hermetic tiny model: a
# 32x32 image at 16px patches is a 2x2 grid, 64-d features. The config's kind,
# arch and patch_size pass through unchanged and so stay under test.
TINY = {"img_size": 32, "embed_dim": 64, "depth": 2, "num_heads": 2}


def _backbone(cfg_path: Path) -> dict:
    doc = json.loads(cfg_path.read_text(encoding="utf-8"))
    return doc["backbone"]


def _tiny_spec(backbone: dict, **over) -> dict:
    # Drop documentation keys (a leading underscore) and the run-time ${ENCODER}
    # placeholder; shrink to the hermetic tiny model; apply the caller's overrides.
    spec = {k: v for k, v in backbone.items() if not k.startswith("_")}
    spec.update(TINY)
    spec["encoder"] = over.pop("encoder", "")
    spec.update(over)
    return spec


class TestTheWiringConfigsAreDiscovered(unittest.TestCase):
    """Structure-only checks: no torch needed, so the base suite runs them and a
    missing/miswired config is a real RED, never a skip."""

    def test_at_least_one_downstream_config_is_discovered(self):
        # The RED anchor: with no wiring config on disk this fails outright, so
        # the loop tests below can never pass vacuously over an empty set.
        self.assertGreaterEqual(len(discover_downstream_configs()), 1,
                                "no methods/*/configs/downstream_arssl.json found")

    def test_each_config_declares_a_shared_vit_backbone(self):
        cfgs = discover_downstream_configs()
        self.assertTrue(cfgs, "no downstream_arssl configs discovered")
        for cfg in cfgs:
            with self.subTest(cfg=str(cfg.relative_to(ROOT))):
                b = _backbone(cfg)
                # Config-based wiring is only for a shared kind the layer builds.
                self.assertEqual(b.get("kind"), "vit")
                self.assertIsInstance(b.get("arch"), str)
                self.assertTrue(b.get("arch"))
                self.assertEqual(int(b.get("patch_size")), 16)
                self.assertIn("img_size", b)


# Building the backbone goes through the shared `vit` kind, whose `_build_vit`
# imports timm -- so these need timm, not merely torch. A lock that has torch but
# not timm (most methods) must skip here, never error.
@needs_timm
class TestABareKeyEncoderLoadsThroughTheSharedKind(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lejepa-bb-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _cfgs(self):
        cfgs = discover_downstream_configs()
        self.assertTrue(cfgs, "no downstream_arssl configs discovered")
        return cfgs

    def test_each_config_builds_a_frozen_spatial_map(self):
        # Proves the declared arch is a real, buildable timm model: a bogus arch
        # would raise here.
        for cfg in self._cfgs():
            with self.subTest(cfg=str(cfg.relative_to(ROOT))):
                bb = spatial_backbones.build_frozen_backbone(
                    _tiny_spec(_backbone(cfg), encoder=""), torch.device("cpu"))
                self.assertEqual(bb.out_channels, 64)
                feat = bb.forward_features(torch.randn(2, 3, 32, 32))
                self.assertEqual(tuple(feat.shape), (2, 64, 2, 2))

    def test_a_bare_key_encoder_pt_loads_and_reproduces(self):
        # LeJEPA saves the bare backbone (backbone. prefix stripped) -> bare timm
        # ViT keys. Loading them must reconstruct the identical backbone.
        for cfg in self._cfgs():
            with self.subTest(cfg=str(cfg.relative_to(ROOT))):
                src = spatial_backbones.build_frozen_backbone(
                    _tiny_spec(_backbone(cfg), encoder=""), torch.device("cpu"))
                enc = self.tmp / "encoder.pt"
                torch.save(src.vit.state_dict(), enc)
                got = spatial_backbones.build_frozen_backbone(
                    _tiny_spec(_backbone(cfg), encoder=str(enc)),
                    torch.device("cpu"))
                torch.manual_seed(0)
                x = torch.randn(2, 3, 32, 32)
                self.assertTrue(torch.allclose(
                    src.forward_features(x), got.forward_features(x), atol=1e-5))

    def test_an_alien_checkpoint_is_refused(self):
        for cfg in self._cfgs():
            with self.subTest(cfg=str(cfg.relative_to(ROOT))):
                src = spatial_backbones.build_frozen_backbone(
                    _tiny_spec(_backbone(cfg), encoder=""), torch.device("cpu"))
                state = dict(src.vit.state_dict())
                state["not_a_vit_key"] = torch.zeros(1)
                enc = self.tmp / "alien.pt"
                torch.save(state, enc)
                with self.assertRaises(RuntimeError) as e:
                    spatial_backbones.build_frozen_backbone(
                        _tiny_spec(_backbone(cfg), encoder=str(enc)),
                        torch.device("cpu"))
                self.assertIn("not_a_vit_key", str(e.exception))


@needs_timm
class TestTheConfigDrivesTheARSSLHarness(unittest.TestCase):
    """Drive the real ARSSL harness over one frozen backbone built from a
    discovered wiring config: the ADE20K dense task runs as a subprocess, is
    checked by the downstream contract, and the battery aggregates to one ok."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lejepa-arssl-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def _ade_key(self) -> str:
        from downstream import arssl
        return next(k for k, m in arssl.discover_tasks().items() if m == "ade20k")

    def test_a_discovered_config_drives_the_ade20k_task(self):
        cfgs = discover_downstream_configs()
        self.assertTrue(cfgs, "no downstream_arssl configs discovered")
        cfg = cfgs[0]
        src = spatial_backbones.build_frozen_backbone(
            _tiny_spec(_backbone(cfg), encoder=""), torch.device("cpu"))
        enc = self.tmp / "encoder.pt"
        torch.save(src.vit.state_dict(), enc)
        data = tiny_ade(self.tmp / "data")
        run_cfg = {"task": "arssl", "seed": 0, "device": "cpu",
                   "backbone": _tiny_spec(_backbone(cfg), encoder=str(enc)),
                   "tasks": {self._ade_key(): {"data_root": str(data),
                                               "probe": dict(SMOKE_PROBE)}}}
        cfg_path = self.tmp / "arssl.json"
        cfg_path.write_text(json.dumps(run_cfg), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run(
            [sys.executable, "-m", "downstream.arssl", "--config", str(cfg_path),
             "--out", str(self.out)],
            cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        results = json.loads((self.out / "arssl_results.json").read_text())
        self.assertEqual(results["status"], "ok")
        self.assertIn("ade20k_miou", results["metrics"])


if __name__ == "__main__":
    unittest.main()
