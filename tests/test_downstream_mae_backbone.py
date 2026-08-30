#!/usr/bin/env python3
"""MAE wired into the ARSSL harness (docs/STEP3_PORTING_PLAN.md, item A3:mae).

Phase A3 wires the lineage backbones already present as Step-1&2 ports into the
A1 ARSSL harness (`downstream/arssl.py`) so their Step-3 numbers reproduce here.
MAE is the first: its frozen encoder must drive the downstream dense-task battery
(`downstream/{ade20k,nyuv2,ssv2}.py`) through one frozen spatial backbone.

The wiring is a new spatial-backbone *kind*, ``mae_vit``, provided by the method
in its **own** directory (a `downstream_backbone.py` declaring a module-level
`KIND` and a `build`) and **discovered** by the shared layer
(`downstream/spatial_backbones.py`) -- so the shared machinery names no method
(`tests/test_no_hard_coded_methods.py`). The provider **reuses** MAE's own model
instead of re-implementing it: MAE's ``encoder.pt`` is not timm-compatible -- its
keys are ``enc_blocks.*``/``enc_norm.*`` and its 2-D sincos position embedding is
a buffer regenerated at build time and *not stored* in the checkpoint -- and MAE's
``forward_encoder`` shuffles patches even at ``mask_ratio=0``. Only MAE's own
non-shuffling ``MAEEncoder`` produces a faithful, correctly ordered spatial map.

Two mechanisms keep the reuse honest and safe:

* the MAE model module is loaded **by file path under a unique module name**
  (never via ``sys.path``), so it cannot collide with another method's
  ``models`` package -- the cross-method import bug this repo has hit before;
* the patch-token grid is read with a **forward hook on the encoder's final
  norm**, so ``MAEEncoder.forward`` is reused verbatim (one implementation,
  invoked -- CLAUDE.md), not copied.

``mae_vit`` is a drop-in for the shared downstream backbone schema
(``kind, encoder, arch, img_size, patch_size`` + optional
``embed_dim, depth, num_heads``): the timm-named optional keys map onto MAE's
``enc_embed_dim/enc_depth/enc_num_heads``, the decoder is unused for features, and
``mlp_ratio`` is the standard 4.0. So the four task runners accept it with no
change beyond the shared ``KINDS`` tuple.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch                                            # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# Imported unguarded when torch is present, so the module-under-test being absent
# is a real error (RED), never silently a skip. The MAE provider is reached the
# way production reaches it -- through discovery -- so this test never names the
# method's directory either.
if HAVE_TORCH:
    from downstream import spatial_backbones
    MAE = spatial_backbones._load_provider(
        spatial_backbones.discover_providers()["mae_vit"])

try:
    import timm                                             # noqa: F401
    from tests.test_downstream_ade20k import tiny_ade, SMOKE_PROBE
    HAVE_TIMM = HAVE_TORCH and True
except ImportError:
    HAVE_TIMM = False

needs_torch = unittest.skipUnless(HAVE_TORCH, "mae_vit backbone needs torch")
needs_timm = unittest.skipUnless(
    HAVE_TIMM, "the end-to-end ARSSL wiring drives the ADE20K runner (timm)")

# A tiny MAE ViT: 32x32 image, 16px patches -> a 2x2 patch grid, 64-d features.
TINY_SPEC = {"kind": "mae_vit", "encoder": "", "arch": "vit_base_patch16",
             "img_size": 32, "patch_size": 16,
             "embed_dim": 64, "depth": 2, "num_heads": 2}


def _spec(**over) -> dict:
    s = dict(TINY_SPEC)
    s.update(over)
    return s


def _write_encoder_pt(tmp: Path, extra=None, drop=None) -> "tuple":
    """Build a tiny MAE, save its encoder-side weights as encoder.pt, and return
    (the built MAE, the path). ``extra`` injects an alien key; ``drop`` removes an
    encoder key -- both to exercise the strict-load refusals."""
    mae = MAE._build_mae(TINY_SPEC)
    state = {k: v for k, v in mae.state_dict().items()
             if k.startswith(MAE.ENC_PREFIXES)}
    if drop:
        state = {k: v for k, v in state.items() if k != drop}
    if extra:
        state[extra] = torch.zeros(1)
    p = tmp / "encoder.pt"
    torch.save(state, p)
    return mae, p


@needs_torch
class TestTheKindIsDiscoveredNotNamed(unittest.TestCase):
    """The shared layer discovers the provider by structure; it names no method."""

    def test_mae_vit_is_a_discovered_ported_backbone_kind(self):
        self.assertEqual(MAE.KIND, "mae_vit")
        self.assertIn(MAE.KIND, spatial_backbones.KINDS)
        providers = spatial_backbones.discover_providers()
        self.assertIn("mae_vit", providers)
        # The provider lives in a method's own directory, as its own file.
        self.assertEqual(providers["mae_vit"].name,
                         spatial_backbones.PROVIDER_FILE)
        self.assertEqual(providers["mae_vit"].parent.parent.name, "methods")

    def test_an_unknown_kind_is_still_refused(self):
        # Negative control: discovery must not turn the dispatch into a silent
        # accept-anything.
        with self.assertRaises(ValueError):
            spatial_backbones.build_frozen_backbone(
                {"kind": "not_a_backbone"}, torch.device("cpu"))


@needs_torch
class TestTheProviderDetectorHasControls(unittest.TestCase):
    """`_provider_kind` decides what gets registered, so it carries a positive and
    negative controls -- a file that IS a provider, and files that only look like
    one (the detector's own tests, kept off the real methods tree)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mae-prov-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write(self, body: str) -> Path:
        p = self.tmp / spatial_backbones.PROVIDER_FILE
        p.write_text(textwrap.dedent(body), encoding="utf-8")
        return p

    def test_a_module_with_kind_and_build_is_a_provider(self):    # positive
        p = self._write('KIND = "made_up"\ndef build(spec):\n    return 1\n')
        self.assertEqual(spatial_backbones._provider_kind(p), "made_up")

    def test_a_kind_without_build_is_not_a_provider(self):        # negative
        p = self._write('KIND = "made_up"\ndef other(spec):\n    return 1\n')
        self.assertIsNone(spatial_backbones._provider_kind(p))

    def test_a_build_without_kind_is_not_a_provider(self):        # negative
        p = self._write('def build(spec):\n    return 1\n')
        self.assertIsNone(spatial_backbones._provider_kind(p))

    def test_a_non_string_kind_is_not_a_provider(self):           # negative
        p = self._write('KIND = 7\ndef build(spec):\n    return 1\n')
        self.assertIsNone(spatial_backbones._provider_kind(p))

    def test_a_nested_build_does_not_count(self):                 # negative
        p = self._write('KIND = "made_up"\n'
                        'def wrap():\n    def build(spec):\n        return 1\n')
        self.assertIsNone(spatial_backbones._provider_kind(p))


@needs_torch
class TestBuildingFromAnEncoder(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mae-bb-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_it_builds_a_frozen_spatial_map_of_the_expected_shape(self):
        _mae, enc = _write_encoder_pt(self.tmp)
        bb = spatial_backbones.build_frozen_backbone(
            _spec(encoder=str(enc)), torch.device("cpu"))
        self.assertEqual(bb.out_channels, 64)
        x = torch.randn(2, 3, 32, 32)
        feat = bb.forward_features(x)
        self.assertEqual(tuple(feat.shape), (2, 64, 2, 2))

    def test_the_backbone_is_frozen_and_stays_in_eval(self):
        _mae, enc = _write_encoder_pt(self.tmp)
        bb = spatial_backbones.build_frozen_backbone(
            _spec(encoder=str(enc)), torch.device("cpu"))
        self.assertFalse(any(p.requires_grad for p in bb.parameters()))
        bb.train()                       # a frozen backbone never leaves eval
        self.assertFalse(bb.training)

    def test_the_features_reproduce_maes_own_encoder(self):
        # The whole point of reuse: the spatial map's tokens ARE MAE's encoder
        # tokens. Global-average-pooling the map must equal MAE's own avg-pooled
        # encoder output (which pools the same patch tokens). A re-implementation
        # that drifted (wrong pos-embed, shuffled patches) would fail here.
        mae, enc = _write_encoder_pt(self.tmp)
        bb = spatial_backbones.build_frozen_backbone(
            _spec(encoder=str(enc)), torch.device("cpu"))
        torch.manual_seed(0)
        x = torch.randn(2, 3, 32, 32)
        pooled_map = bb.forward_features(x).mean(dim=(2, 3))
        reference = mae.get_encoder(pool="avg").eval()
        with torch.no_grad():
            expected = reference(x)
        self.assertTrue(torch.allclose(pooled_map, expected, atol=1e-5),
                        (pooled_map - expected).abs().max().item())

    def test_an_empty_encoder_builds_a_random_tiny_backbone(self):
        # The hermetic smoke: no encoder path, no file, no download -- a tiny
        # random MAE is built so CI can drive the harness offline.
        bb = spatial_backbones.build_frozen_backbone(
            _spec(encoder=""), torch.device("cpu"))
        feat = bb.forward_features(torch.randn(1, 3, 32, 32))
        self.assertEqual(tuple(feat.shape), (1, 64, 2, 2))


@needs_torch
class TestTheStrictLoadRefuses(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mae-bb-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_an_encoder_carrying_alien_keys_is_refused(self):
        _mae, enc = _write_encoder_pt(self.tmp, extra="not_an_mae_key")
        with self.assertRaises(RuntimeError) as e:
            spatial_backbones.build_frozen_backbone(
                _spec(encoder=str(enc)), torch.device("cpu"))
        self.assertIn("not_an_mae_key", str(e.exception))

    def test_an_encoder_missing_the_backbone_weights_is_refused(self):
        _mae, enc = _write_encoder_pt(self.tmp, drop="patch_embed.proj.weight")
        with self.assertRaises(RuntimeError) as e:
            spatial_backbones.build_frozen_backbone(
                _spec(encoder=str(enc)), torch.device("cpu"))
        self.assertIn("patch_embed.proj.weight", str(e.exception))


@needs_timm
class TestTheEndToEndARSSLWiring(unittest.TestCase):
    """Drive the real ARSSL harness over one frozen MAE backbone: the ADE20K
    dense task runs as a subprocess, is checked by the downstream contract, and
    the battery aggregates to one ok result. This is MAE reproducing in A1."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mae-arssl-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def _ade_key(self) -> str:
        from downstream import arssl
        return next(k for k, m in arssl.discover_tasks().items() if m == "ade20k")

    def test_a_mae_backbone_drives_the_ade20k_task_through_the_battery(self):
        _mae, enc = _write_encoder_pt(self.tmp)
        data = tiny_ade(self.tmp / "data")
        cfg = {"task": "arssl", "seed": 0, "device": "cpu",
               "backbone": _spec(encoder=str(enc)),
               "tasks": {self._ade_key(): {"data_root": str(data),
                                           "probe": dict(SMOKE_PROBE)}}}
        cfg_path = self.tmp / "arssl.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
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
