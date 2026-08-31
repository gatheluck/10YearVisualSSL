#!/usr/bin/env python3
"""I-JEPA wired into the ARSSL harness (docs/STEP3_PORTING_PLAN.md, item A3:ijepa).

Phase A3 wires the lineage backbones already present as Step-1&2 ports into the
A1 ARSSL harness (`downstream/arssl.py`) so their Step-3 numbers reproduce here.
I-JEPA is the second (MAE was first): its frozen target encoder must drive the
downstream dense-task battery through one frozen spatial backbone.

The wiring is a new spatial-backbone *kind*, ``ijepa_vit``, provided by the method
in its **own** directory (a `downstream_backbone.py` declaring a module-level
`KIND` and a `build`) and **discovered** by the shared layer
(`downstream/spatial_backbones.py`) -- so the shared machinery names no method
(`tests/test_no_hard_coded_methods.py`) and needs no edit to admit a new lineage
backbone. This is the pattern MAE established as the first lineage provider;
I-JEPA reuses it.

I-JEPA is markedly simpler to reuse than MAE. Its ``VisionTransformer.forward(x)``
already returns every patch token in raster order with the final norm applied and
**no masking / no shuffling** (the mask branch is skipped when ``mask_ids=None``),
its position embedding is a stored parameter that ships in ``encoder.pt``, and it
has **no CLS token** (features are the mean of the patch tokens). So the provider
reuses that forward verbatim -- no forward hook, no non-shuffling clone, no prefix
to drop -- and ``encoder.pt`` carries bare ``VisionTransformer`` keys (the
``target_encoder.`` prefix is stripped at save time), which load with an exact
match. The model module is still loaded **by file path under a unique module
name** (never via ``sys.path``), so it cannot collide with another method's
``models`` package -- the cross-method import bug this repo has hit before.

``ijepa_vit`` is a drop-in for the shared downstream backbone schema
(``kind, encoder, arch, img_size, patch_size`` + optional
``embed_dim, depth, num_heads``): the timm-named optional keys map onto I-JEPA's
``embed_dim/depth/num_heads`` directly, and ``mlp_ratio`` is the standard 4.0.
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

try:
    import torch                                            # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# Imported unguarded when torch is present, so the module-under-test being absent
# is a real error (RED), never silently a skip. The provider is reached the way
# production reaches it -- through discovery -- so this test never names the
# method's directory either.
if HAVE_TORCH:
    from downstream import spatial_backbones
    IJEPA = spatial_backbones._load_provider(
        spatial_backbones.discover_providers()["ijepa_vit"])

try:
    import timm                                             # noqa: F401
    from tests.test_downstream_ade20k import tiny_ade, SMOKE_PROBE
    HAVE_TIMM = HAVE_TORCH and True
except ImportError:
    HAVE_TIMM = False

needs_torch = unittest.skipUnless(HAVE_TORCH, "ijepa_vit backbone needs torch")
needs_timm = unittest.skipUnless(
    HAVE_TIMM, "the end-to-end ARSSL wiring drives the ADE20K runner (timm)")

# A tiny I-JEPA ViT: 32x32 image, 16px patches -> a 2x2 patch grid, 64-d features.
TINY_SPEC = {"kind": "ijepa_vit", "encoder": "", "arch": "vit_base",
             "img_size": 32, "patch_size": 16,
             "embed_dim": 64, "depth": 2, "num_heads": 2}


def _spec(**over) -> dict:
    s = dict(TINY_SPEC)
    s.update(over)
    return s


def _write_encoder_pt(tmp: Path, extra=None, drop=None) -> "tuple":
    """Build a tiny I-JEPA encoder, save its (bare-key) weights as encoder.pt, and
    return (the built encoder, the path). ``extra`` injects an alien key; ``drop``
    removes an encoder key -- both to exercise the strict-load refusals."""
    model = IJEPA._build_ijepa(TINY_SPEC)
    state = dict(model.state_dict())
    if drop:
        state = {k: v for k, v in state.items() if k != drop}
    if extra:
        state[extra] = torch.zeros(1)
    p = tmp / "encoder.pt"
    torch.save(state, p)
    return model, p


@needs_torch
class TestTheKindIsDiscoveredNotNamed(unittest.TestCase):
    """The shared layer discovers the provider by structure; it names no method."""

    def test_ijepa_vit_is_a_discovered_ported_backbone_kind(self):
        self.assertEqual(IJEPA.KIND, "ijepa_vit")
        self.assertIn(IJEPA.KIND, spatial_backbones.KINDS)
        providers = spatial_backbones.discover_providers()
        self.assertIn("ijepa_vit", providers)
        # The provider lives in a method's own directory, as its own file.
        self.assertEqual(providers["ijepa_vit"].name,
                         spatial_backbones.PROVIDER_FILE)
        self.assertEqual(providers["ijepa_vit"].parent.parent.name, "methods")

    def test_an_unknown_kind_is_still_refused(self):
        # Negative control: adding ijepa_vit must not turn the dispatch into a
        # silent accept-anything.
        with self.assertRaises(ValueError):
            spatial_backbones.build_frozen_backbone(
                {"kind": "not_a_backbone"}, torch.device("cpu"))


@needs_torch
class TestBuildingFromAnEncoder(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ijepa-bb-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_it_builds_a_frozen_spatial_map_of_the_expected_shape(self):
        _model, enc = _write_encoder_pt(self.tmp)
        bb = spatial_backbones.build_frozen_backbone(
            _spec(encoder=str(enc)), torch.device("cpu"))
        self.assertEqual(bb.out_channels, 64)
        x = torch.randn(2, 3, 32, 32)
        feat = bb.forward_features(x)
        self.assertEqual(tuple(feat.shape), (2, 64, 2, 2))

    def test_the_backbone_is_frozen_and_stays_in_eval(self):
        _model, enc = _write_encoder_pt(self.tmp)
        bb = spatial_backbones.build_frozen_backbone(
            _spec(encoder=str(enc)), torch.device("cpu"))
        self.assertFalse(any(p.requires_grad for p in bb.parameters()))
        bb.train()                       # a frozen backbone never leaves eval
        self.assertFalse(bb.training)

    def test_the_features_reproduce_ijepas_own_encoder(self):
        # The whole point of reuse: the spatial map's tokens ARE I-JEPA's encoder
        # tokens. Global-average-pooling the map must equal I-JEPA's own
        # forward_features (the mean over the same patch tokens). A re-implementation
        # that drifted (wrong pos-embed, dropped norm) would fail here.
        model, enc = _write_encoder_pt(self.tmp)
        bb = spatial_backbones.build_frozen_backbone(
            _spec(encoder=str(enc)), torch.device("cpu"))
        torch.manual_seed(0)
        x = torch.randn(2, 3, 32, 32)
        pooled_map = bb.forward_features(x).mean(dim=(2, 3))
        model.eval()
        with torch.no_grad():
            expected = model.forward_features(x)
        self.assertTrue(torch.allclose(pooled_map, expected, atol=1e-5),
                        (pooled_map - expected).abs().max().item())

    def test_an_empty_encoder_builds_a_random_tiny_backbone(self):
        # The hermetic smoke: no encoder path, no file, no download -- a tiny
        # random I-JEPA is built so CI can drive the harness offline.
        bb = spatial_backbones.build_frozen_backbone(
            _spec(encoder=""), torch.device("cpu"))
        feat = bb.forward_features(torch.randn(1, 3, 32, 32))
        self.assertEqual(tuple(feat.shape), (1, 64, 2, 2))


@needs_torch
class TestTheStrictLoadRefuses(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ijepa-bb-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_an_encoder_carrying_alien_keys_is_refused(self):
        _model, enc = _write_encoder_pt(self.tmp, extra="not_an_ijepa_key")
        with self.assertRaises(RuntimeError) as e:
            spatial_backbones.build_frozen_backbone(
                _spec(encoder=str(enc)), torch.device("cpu"))
        self.assertIn("not_an_ijepa_key", str(e.exception))

    def test_an_encoder_missing_the_backbone_weights_is_refused(self):
        _model, enc = _write_encoder_pt(self.tmp, drop="pos_embed")
        with self.assertRaises(RuntimeError) as e:
            spatial_backbones.build_frozen_backbone(
                _spec(encoder=str(enc)), torch.device("cpu"))
        self.assertIn("pos_embed", str(e.exception))


@needs_timm
class TestTheEndToEndARSSLWiring(unittest.TestCase):
    """Drive the real ARSSL harness over one frozen I-JEPA backbone: the ADE20K
    dense task runs as a subprocess, is checked by the downstream contract, and
    the battery aggregates to one ok result. This is I-JEPA reproducing in A1."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ijepa-arssl-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def _ade_key(self) -> str:
        from downstream import arssl
        return next(k for k, m in arssl.discover_tasks().items() if m == "ade20k")

    def test_an_ijepa_backbone_drives_the_ade20k_task_through_the_battery(self):
        _model, enc = _write_encoder_pt(self.tmp)
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
