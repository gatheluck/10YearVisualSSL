#!/usr/bin/env python3
"""iGPT wired into the ARSSL harness (docs/STEP3_PORTING_PLAN.md, item A3:igpt).

Phase A3 wires the lineage backbones already present as Step-1&2 ports into the
A1 ARSSL harness (`downstream/arssl.py`) so their Step-3 numbers reproduce here.
iGPT follows MAE and I-JEPA: a hand-written, non-timm encoder, so it ships its
own spatial-backbone provider (a `downstream_backbone.py` declaring a module-level
`KIND` and a `build`) discovered by the shared layer -- the shared machinery names
no method (`tests/test_no_hard_coded_methods.py`).

iGPT is the first provider whose input is **not an image tensor**: it is a causal
transformer over discrete colour-cluster tokens. So its provider does what the
method's own linear probe does before reading features -- resize to the model's
token grid, quantise pixels to colour tokens with the clusters the model was
trained on, then read a **middle** transformer layer. The linear probe mean-pools
that layer to one vector; a dense task instead keeps the per-position features and
reshapes them to a `[B, C, h, w]` grid. Pooling the grid back therefore has to
equal the probe's own vector -- the reproduction test pins exactly that.

Two things the shared ViT backbone schema has no slot for, which the provider
absorbs so the four task runners stay unchanged (the chosen A3:igpt shape):

* **vocab_size** -- the colour vocabulary. For a real encoder it is *inferred*
  from `encoder.pt` (the token-embedding row count), so it can never disagree
  with the trained model; the hermetic smoke (no encoder) uses a default.
* **clusters** -- the colour centres pixels are quantised to. For a real encoder
  they are read from `clusters.npy` **beside** `encoder.pt` (where the adapter
  writes them); the smoke generates a deterministic set.

The provider is reached the way production reaches it -- through discovery -- so
this test never names the method's directory; the backbone `kind` string "igpt"
is not the method dir name.
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
# method's directory either. iGPT's provider is torch + numpy only (it does not
# touch timm), so building it is gated on torch, like MAE/I-JEPA.
if HAVE_TORCH:
    import numpy as np
    from downstream import spatial_backbones
    IGPT = spatial_backbones._load_provider(
        spatial_backbones.discover_providers()["igpt"])
    IGPT_MOD = IGPT.load_igpt_module()
    QUANT_MOD = IGPT.load_quantize_module()

try:
    import timm                                             # noqa: F401
    from tests.test_downstream_ade20k import tiny_ade, SMOKE_PROBE
    HAVE_TIMM = HAVE_TORCH and True
except ImportError:
    HAVE_TIMM = False

needs_torch = unittest.skipUnless(HAVE_TORCH, "the igpt backbone needs torch")
needs_timm = unittest.skipUnless(
    HAVE_TIMM, "the end-to-end ARSSL wiring drives the ADE20K runner (timm)")

# A tiny iGPT: an 8x8 token grid, 8-colour vocabulary, 32-d features -> a frozen
# spatial map [B, 32, 8, 8]. patch_size/arch are schema-required but informational
# for iGPT (one token per grid cell), so they carry placeholders.
TINY_VOCAB = 8
TINY_SPEC = {"kind": "igpt", "encoder": "", "arch": "igpt",
             "img_size": 8, "patch_size": 1,
             "embed_dim": 32, "depth": 2, "num_heads": 2}


def _spec(**over) -> dict:
    s = dict(TINY_SPEC)
    s.update(over)
    return s


def _tiny_model():
    """Build the tiny iGPT the encoder.pt is saved from, at the spec's dims and
    the test's vocabulary."""
    return IGPT_MOD.IGPT(vocab_size=TINY_VOCAB, img_size=TINY_SPEC["img_size"],
                         n_layer=TINY_SPEC["depth"], n_head=TINY_SPEC["num_heads"],
                         n_embd=TINY_SPEC["embed_dim"])


def _clusters(n=TINY_VOCAB):
    """A deterministic set of `n` colour centres in the [0, 1] range ToTensor
    produces -- what the probe would quantise with."""
    return np.random.RandomState(0).uniform(0.0, 1.0, size=(n, 3)).astype("float32")


def _write_encoder_pt(tmp: Path, extra=None, drop=None, with_clusters=True):
    """Build a tiny iGPT, save its representation (bare keys, no generative head)
    as encoder.pt, and -- unless told not to -- clusters.npy beside it. Returns
    (model, encoder_path). ``extra`` injects an alien key; ``drop`` removes an
    encoder weight -- both to exercise the strict-load refusals."""
    model = _tiny_model()
    # encoder.pt is the representation side only: the generative head is excluded,
    # exactly as the adapter writes it.
    state = {k: v for k, v in model.state_dict().items()
             if not k.startswith("head.")}
    if drop:
        state = {k: v for k, v in state.items() if k != drop}
    if extra:
        state[extra] = torch.zeros(1)
    enc = tmp / "encoder.pt"
    torch.save(state, enc)
    if with_clusters:
        np.save(tmp / "clusters.npy", _clusters())
    return model, enc


@needs_torch
class TestTheKindIsDiscoveredNotNamed(unittest.TestCase):
    """The shared layer discovers the provider by structure; it names no method."""

    def test_igpt_is_a_discovered_ported_backbone_kind(self):
        self.assertEqual(IGPT.KIND, "igpt")
        self.assertIn(IGPT.KIND, spatial_backbones.KINDS)
        providers = spatial_backbones.discover_providers()
        self.assertIn("igpt", providers)
        # The provider lives in a method's own directory, as its own file.
        self.assertEqual(providers["igpt"].name, spatial_backbones.PROVIDER_FILE)
        self.assertEqual(providers["igpt"].parent.parent.name, "methods")

    def test_an_unknown_kind_is_still_refused(self):
        # Negative control: adding igpt must not turn the dispatch into a silent
        # accept-anything.
        with self.assertRaises(ValueError):
            spatial_backbones.build_frozen_backbone(
                {"kind": "not_a_backbone"}, torch.device("cpu"))


@needs_torch
class TestBuildingFromAnEncoder(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="igpt-bb-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_it_builds_a_frozen_spatial_map_of_the_expected_shape(self):
        _model, enc = _write_encoder_pt(self.tmp)
        bb = spatial_backbones.build_frozen_backbone(
            _spec(encoder=str(enc)), torch.device("cpu"))
        self.assertEqual(bb.out_channels, 32)
        feat = bb.forward_features(torch.rand(2, 3, 8, 8))
        self.assertEqual(tuple(feat.shape), (2, 32, 8, 8))

    def test_the_vocabulary_is_inferred_from_the_encoder(self):
        # A real encoder settles the colour vocabulary: the provider reads it from
        # the token-embedding rows, so it can never disagree with the trained
        # model. Build at a non-default vocabulary and prove it was picked up.
        _model, enc = _write_encoder_pt(self.tmp)
        bb = spatial_backbones.build_frozen_backbone(
            _spec(encoder=str(enc)), torch.device("cpu"))
        self.assertEqual(bb.model.vocab_size, TINY_VOCAB)
        self.assertNotEqual(TINY_VOCAB, IGPT.DEFAULT_VOCAB_SIZE)

    def test_the_backbone_is_frozen_and_stays_in_eval(self):
        _model, enc = _write_encoder_pt(self.tmp)
        bb = spatial_backbones.build_frozen_backbone(
            _spec(encoder=str(enc)), torch.device("cpu"))
        self.assertFalse(any(p.requires_grad for p in bb.parameters()))
        bb.train()                       # a frozen backbone never leaves eval
        self.assertFalse(bb.training)

    def test_the_pooled_map_reproduces_igpts_own_probe_feature(self):
        # The whole point of reuse: global-average-pooling the dense map must equal
        # iGPT's own extract_features (the middle layer mean-pooled over the same
        # tokens). A drifted reshape (wrong order, transposed grid) fails here.
        model, enc = _write_encoder_pt(self.tmp)
        bb = spatial_backbones.build_frozen_backbone(
            _spec(encoder=str(enc)), torch.device("cpu"))
        torch.manual_seed(0)
        x = torch.rand(2, 3, 8, 8)                     # already the token grid
        pooled_map = bb.forward_features(x).mean(dim=(2, 3))
        clusters = _clusters()
        tokens = QUANT_MOD.quantize_images(x, clusters)
        model.eval()
        with torch.no_grad():
            expected = model.extract_features(tokens)
        self.assertTrue(torch.allclose(pooled_map, expected, atol=1e-5),
                        (pooled_map - expected).abs().max().item())

    def test_an_empty_encoder_builds_a_random_tiny_backbone(self):
        # The hermetic smoke: no encoder path, no file, no download -- a tiny
        # random iGPT is built and clusters are generated, so CI drives the
        # harness offline.
        bb = spatial_backbones.build_frozen_backbone(
            _spec(encoder=""), torch.device("cpu"))
        feat = bb.forward_features(torch.rand(1, 3, 8, 8))
        self.assertEqual(tuple(feat.shape), (1, 32, 8, 8))


@needs_torch
class TestTheStrictLoadRefuses(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="igpt-bb-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_an_encoder_carrying_alien_keys_is_refused(self):
        _model, enc = _write_encoder_pt(self.tmp, extra="not_an_igpt_key")
        with self.assertRaises(RuntimeError) as e:
            spatial_backbones.build_frozen_backbone(
                _spec(encoder=str(enc)), torch.device("cpu"))
        self.assertIn("not_an_igpt_key", str(e.exception))

    def test_an_encoder_missing_the_backbone_weights_is_refused(self):
        _model, enc = _write_encoder_pt(self.tmp, drop="ln_f.weight")
        with self.assertRaises(RuntimeError) as e:
            spatial_backbones.build_frozen_backbone(
                _spec(encoder=str(enc)), torch.device("cpu"))
        self.assertIn("ln_f.weight", str(e.exception))

    def test_a_missing_clusters_file_beside_the_encoder_is_refused(self):
        # A real run must quantise with the clusters the model was trained on;
        # their absence is refused, not silently replaced with random ones.
        _model, enc = _write_encoder_pt(self.tmp, with_clusters=False)
        with self.assertRaises(RuntimeError) as e:
            spatial_backbones.build_frozen_backbone(
                _spec(encoder=str(enc)), torch.device("cpu"))
        self.assertIn("clusters.npy", str(e.exception))

    def test_clusters_of_the_wrong_size_are_refused(self):
        # clusters.npy must match the encoder's colour vocabulary; a set sized for
        # a different vocabulary is from another run and is refused, not used (an
        # oversized token would index past the embedding).
        _model, enc = _write_encoder_pt(self.tmp, with_clusters=False)
        np.save(self.tmp / "clusters.npy", _clusters(TINY_VOCAB + 1))
        with self.assertRaises(RuntimeError) as e:
            spatial_backbones.build_frozen_backbone(
                _spec(encoder=str(enc)), torch.device("cpu"))
        self.assertIn("different runs", str(e.exception))


@needs_timm
class TestTheEndToEndARSSLWiring(unittest.TestCase):
    """Drive the real ARSSL harness over one frozen iGPT backbone: the ADE20K
    dense task runs as a subprocess, is checked by the downstream contract, and
    the battery aggregates to one ok result. This is iGPT reproducing in A1."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="igpt-arssl-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def _ade_key(self) -> str:
        from downstream import arssl
        return next(k for k, m in arssl.discover_tasks().items() if m == "ade20k")

    def test_an_igpt_backbone_drives_the_ade20k_task_through_the_battery(self):
        data = tiny_ade(self.tmp / "data")
        cfg = {"task": "arssl", "seed": 0, "device": "cpu",
               "backbone": _spec(encoder=""),          # hermetic random smoke
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
