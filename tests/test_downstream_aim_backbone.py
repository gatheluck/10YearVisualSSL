#!/usr/bin/env python3
"""AIM wired into the ARSSL harness (docs/STEP3_PORTING_PLAN.md, item A3:aim).

Phase A3 wires the lineage backbones already present as Step-1&2 ports into the
A1 ARSSL harness (`downstream/arssl.py`) so their Step-3 numbers reproduce here.
AIM follows MAE, I-JEPA, LeJEPA and iGPT. Like MAE and I-JEPA it is hand-written
and non-timm, so it ships its own provider (a `downstream_backbone.py` declaring a
module-level `KIND` and a `build`) which the shared layer
(`downstream/spatial_backbones.py`) **discovers** by structure -- the shared
machinery names no method (`tests/test_no_hard_coded_methods.py`).

AIM is not a drop-in like I-JEPA. Its ``forward(x, prefix_len)`` returns a training
tuple ``(loss, pred, target)`` under a prefix-LM causal mask, not tokens; clean
tokens come only from ``AIMViT.forward_features(x, layer_ids)``, which runs the
trunk bidirectionally and **averages the last few layers** (AIM's generative-model
protocol). The method's own linear probe reads exactly that -- the last
``num_feature_layers`` blocks averaged, then patch-mean-pooled (the method's own
``evaluate_linear_aim`` probe). So the provider reproduces that read:
it calls ``forward_features`` with the same last-N layer ids and reshapes the
per-position tokens to a ``[B, C, h, w]`` map. Global-average-pooling that map
therefore equals AIM's own probe feature (one representation, two readers).

**The provider absorbs what the shared ViT schema has no slot for** (the pattern
iGPT established): ``num_feature_layers`` is not a backbone-schema key -- the four
task runners reject unknown keys -- so it is fixed here to AIM's protocol value 6
(the method's own ``linear_eval_vit`` probe config), and the four runners stay
unchanged. AIM has **no CLS token** (no prefix to drop), and its ``encoder.pt``
excludes the prediction head (``predictor.*``), so a strict load tolerates those
missing head keys but refuses any alien key or any missing *trunk* weight.
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
    AIM = spatial_backbones._load_provider(
        spatial_backbones.discover_providers()["aim_vit"])

try:
    import timm                                             # noqa: F401
    from tests.test_downstream_ade20k import tiny_ade, SMOKE_PROBE
    HAVE_TIMM = HAVE_TORCH and True
except ImportError:
    HAVE_TIMM = False

needs_torch = unittest.skipUnless(HAVE_TORCH, "aim_vit backbone needs torch")
needs_timm = unittest.skipUnless(
    HAVE_TIMM, "the end-to-end ARSSL wiring drives the ADE20K runner (timm)")

# A tiny AIM ViT: 32x32 image, 16px patches -> a 2x2 patch grid, 64-d features.
# depth 2 is shallower than AIM's 6-layer feature window, so the provider must
# clamp the layer ids (as the probe does) rather than index off the end.
TINY_SPEC = {"kind": "aim_vit", "encoder": "", "arch": "aim_base",
             "img_size": 32, "patch_size": 16,
             "embed_dim": 64, "depth": 2, "num_heads": 2}

# A deeper AIM: 8 layers, so "the last 6" differs from "the first 6" -- this is
# what makes the layer-selection reproduce a falsifiable claim (a provider that
# read the wrong layers, or all of them, would not match the probe here).
DEEP_SPEC = dict(TINY_SPEC, depth=8)


def _spec(base=TINY_SPEC, **over) -> dict:
    s = dict(base)
    s.update(over)
    return s


def _write_encoder_pt(tmp: Path, base=TINY_SPEC, extra=None, drop=None) -> "tuple":
    """Build a tiny AIM, save its trunk weights as encoder.pt (the prediction head
    ``predictor.*`` excluded, exactly as the method's adapter does), and return
    (the built model, the path). ``extra`` injects an alien key; ``drop`` removes a
    trunk key -- both to exercise the strict-load refusals."""
    model = AIM._build_aim(base)
    state = {k: v for k, v in model.state_dict().items()
             if not k.startswith("predictor.")}
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

    def test_aim_vit_is_a_discovered_ported_backbone_kind(self):
        self.assertEqual(AIM.KIND, "aim_vit")
        self.assertIn(AIM.KIND, spatial_backbones.KINDS)
        providers = spatial_backbones.discover_providers()
        self.assertIn("aim_vit", providers)
        # The provider lives in a method's own directory, as its own file.
        self.assertEqual(providers["aim_vit"].name,
                         spatial_backbones.PROVIDER_FILE)
        self.assertEqual(providers["aim_vit"].parent.parent.name, "methods")

    def test_an_unknown_kind_is_still_refused(self):
        # Negative control: adding aim_vit must not turn the dispatch into a
        # silent accept-anything.
        with self.assertRaises(ValueError):
            spatial_backbones.build_frozen_backbone(
                {"kind": "not_a_backbone"}, torch.device("cpu"))


@needs_torch
class TestBuildingFromAnEncoder(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="aim-bb-"))
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

    def test_the_pooled_map_reproduces_aims_own_probe_feature(self):
        # The whole point of reuse: global-average-pooling the spatial map must
        # equal AIM's own probe feature -- the last `num_feature_layers` blocks
        # averaged (forward_features) then patch-mean-pooled. A deeper model makes
        # this bite: reading the wrong layers, or dropping the transpose before the
        # grid reshape, would fail the comparison.
        model, enc = _write_encoder_pt(self.tmp, base=DEEP_SPEC)
        bb = spatial_backbones.build_frozen_backbone(
            _spec(base=DEEP_SPEC, encoder=str(enc)), torch.device("cpu"))
        torch.manual_seed(0)
        x = torch.randn(2, 3, 32, 32)
        pooled_map = bb.forward_features(x).mean(dim=(2, 3))
        n_blocks = len(model.blocks)
        k = min(AIM.DEFAULT_NUM_FEATURE_LAYERS, n_blocks)
        layer_ids = list(range(n_blocks - k, n_blocks))
        model.eval()
        with torch.no_grad():
            expected = model.forward_features(x, layer_ids=layer_ids).mean(dim=1)
        self.assertTrue(torch.allclose(pooled_map, expected, atol=1e-5),
                        (pooled_map - expected).abs().max().item())

    def test_an_empty_encoder_builds_a_random_tiny_backbone(self):
        # The hermetic smoke: no encoder path, no file, no download -- a tiny
        # random AIM is built so CI can drive the harness offline.
        bb = spatial_backbones.build_frozen_backbone(
            _spec(encoder=""), torch.device("cpu"))
        feat = bb.forward_features(torch.randn(1, 3, 32, 32))
        self.assertEqual(tuple(feat.shape), (1, 64, 2, 2))


@needs_torch
class TestTheStrictLoadRefuses(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="aim-bb-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_an_encoder_carrying_alien_keys_is_refused(self):
        _model, enc = _write_encoder_pt(self.tmp, extra="not_an_aim_key")
        with self.assertRaises(RuntimeError) as e:
            spatial_backbones.build_frozen_backbone(
                _spec(encoder=str(enc)), torch.device("cpu"))
        self.assertIn("not_an_aim_key", str(e.exception))

    def test_an_encoder_missing_a_trunk_weight_is_refused(self):
        # The head (predictor.*) is expected to be missing from encoder.pt; a
        # missing *trunk* weight means the checkpoint is not this encoder.
        _model, enc = _write_encoder_pt(self.tmp, drop="norm.weight")
        with self.assertRaises(RuntimeError) as e:
            spatial_backbones.build_frozen_backbone(
                _spec(encoder=str(enc)), torch.device("cpu"))
        self.assertIn("norm.weight", str(e.exception))


@needs_timm
class TestTheEndToEndARSSLWiring(unittest.TestCase):
    """Drive the real ARSSL harness over one frozen AIM backbone: the ADE20K dense
    task runs as a subprocess, is checked by the downstream contract, and the
    battery aggregates to one ok result. This is AIM reproducing in A1."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="aim-arssl-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def _ade_key(self) -> str:
        from downstream import arssl
        return next(k for k, m in arssl.discover_tasks().items() if m == "ade20k")

    def test_an_aim_backbone_drives_the_ade20k_task_through_the_battery(self):
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
