#!/usr/bin/env python3
"""VAR wired into the ARSSL harness (docs/STEP3_PORTING_PLAN.md, item A3:var).

Phase A3 wires the lineage backbones already present as Step-1&2 ports into the
A1 ARSSL harness (`downstream/arssl.py`) so their Step-3 numbers reproduce here.
VAR follows MAE, I-JEPA, LeJEPA, iGPT and AIM. Like them it ships its own provider
(a `downstream_backbone.py` declaring a module-level `KIND` and a `build`) which
the shared layer (`downstream/spatial_backbones.py`) **discovers** by structure --
the shared machinery names no method (`tests/test_no_hard_coded_methods.py`).

**What VAR's representation is, measured -- not the name** (docs/EVAL_DOWNLOAD.md
section 2, and the method's own `evaluate_linear_var.py`): the probed feature is
the **VQVAE tokeniser's encoder** output, global-average-pooled -- *not* the VAR
transformer that step 1 trains, and *not* `encoder.pt`. `vae.encoder(x)` already
returns a `[B, Cvae, H/16, W/16]` spatial map (the VQGAN encoder is fully
convolutional, stride 16), so unlike the ViT providers there is no token grid to
reshape: the provider returns that map directly, and global-average-pooling it
reproduces VAR's own probe feature (one representation, two readers).

So for VAR the backbone spec's ``encoder`` is the **VQVAE tokeniser checkpoint**
(the pinned download `vae_ch160v4096z32.pth`), not a trained encoder. The VQVAE
architecture the shared ViT schema has no slot for is **inferred from the
checkpoint** (Cvae, the vocabulary V, and the base width ch -- the iGPT pattern),
so a config can never disagree with the trained tokeniser. The hermetic smoke
leaves ``encoder`` empty and builds a tiny random VQVAE, so CI downloads nothing.
The tokeniser is built and loaded through the method's own
`train_pretrain_var.build_vqvae` -- one place knows how the tokeniser is built --
whose strict load refuses a checkpoint that is not this VQVAE.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
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
    VAR = spatial_backbones._load_provider(
        spatial_backbones.discover_providers()["var_vqvae"])

try:
    import timm                                             # noqa: F401
    from tests.test_downstream_ade20k import tiny_ade, SMOKE_PROBE
    HAVE_TIMM = HAVE_TORCH and True
except ImportError:
    HAVE_TIMM = False

needs_torch = unittest.skipUnless(HAVE_TORCH, "var_vqvae backbone needs torch")
needs_timm = unittest.skipUnless(
    HAVE_TIMM, "the end-to-end ARSSL wiring drives the ADE20K runner (timm)")

# This module's tests draw random inputs (`torch.randn`), which advances the global
# torch RNG. The whole suite runs in one process in discovery order, and this file
# (``test_downstream_...``) sorts before every ``test_method_...``; several of those
# assert a freshly built model's loss/feature is finite *without seeding*, so they
# depend on the RNG state they inherit. Leaving this module's draws in the global
# RNG would shift that state and can tip such an assertion into a NaN it would not
# otherwise hit. So the module snapshots the RNG on entry and restores it on exit:
# it leaves the global RNG exactly as it found it, with no cross-module footprint.
# (The underlying fragility is those tests' own unseeded draws; this only keeps A3
# from being the perturbation that exposes it -- the provider itself is already
# RNG-hermetic, see TestTheProviderDoesNotLeakUpstreamImports.)
_RNG_AT_MODULE_ENTRY = None


def setUpModule():
    global _RNG_AT_MODULE_ENTRY
    if HAVE_TORCH:
        _RNG_AT_MODULE_ENTRY = torch.get_rng_state()


def tearDownModule():
    if HAVE_TORCH and _RNG_AT_MODULE_ENTRY is not None:
        torch.set_rng_state(_RNG_AT_MODULE_ENTRY)

# A tiny VQVAE: the VQGAN encoder downsamples by 16, so a 32x32 image yields a
# 2x2 feature map. ch/Cvae/V are tiny so the smoke builds and runs on a CPU with
# no download. arch/img_size/patch_size are schema-required but informational for
# VAR's fully-convolutional encoder (patch_size 16 matches the real stride).
TINY_SPEC = {"kind": "var_vqvae", "encoder": "", "arch": "var_d16",
             "img_size": 32, "patch_size": 16}


def _spec(**over) -> dict:
    s = dict(TINY_SPEC)
    s.update(over)
    return s


def _write_vqvae_ckpt(tmp: Path, extra=None, drop=None) -> "tuple":
    """Build a tiny random VQVAE through the provider (the smoke path), save the
    tokeniser's ``state_dict`` as a VQVAE checkpoint, and return (the built
    backbone, the path). ``extra`` injects an alien key and ``drop`` removes a
    real one -- both to exercise the strict load's refusals."""
    bb = spatial_backbones.build_frozen_backbone(_spec(), torch.device("cpu"))
    state = dict(bb.vae.state_dict())
    if drop:
        del state[drop]
    if extra:
        state[extra] = torch.zeros(1)
    p = tmp / "vqvae.pt"
    torch.save(state, p)
    return bb, p


@needs_torch
class TestTheKindIsDiscoveredNotNamed(unittest.TestCase):
    """The shared layer discovers the provider by structure; it names no method."""

    def test_var_vqvae_is_a_discovered_ported_backbone_kind(self):
        self.assertEqual(VAR.KIND, "var_vqvae")
        self.assertIn(VAR.KIND, spatial_backbones.KINDS)
        providers = spatial_backbones.discover_providers()
        self.assertIn("var_vqvae", providers)
        # The provider lives in a method's own directory, as its own file.
        self.assertEqual(providers["var_vqvae"].name,
                         spatial_backbones.PROVIDER_FILE)
        self.assertEqual(providers["var_vqvae"].parent.parent.name, "methods")

    def test_an_unknown_kind_is_still_refused(self):
        # Negative control: adding var_vqvae must not turn the dispatch into a
        # silent accept-anything.
        with self.assertRaises(ValueError):
            spatial_backbones.build_frozen_backbone(
                {"kind": "not_a_backbone"}, torch.device("cpu"))


@needs_torch
class TestBuildingTheBackbone(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="var-bb-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_an_empty_encoder_builds_a_random_tiny_backbone(self):
        # The hermetic smoke: no encoder path, no file, no download -- a tiny
        # random VQVAE is built so CI can drive the harness offline. The VQGAN
        # encoder downsamples 32x32 by 16 to a 2x2 map of Cvae channels.
        bb = spatial_backbones.build_frozen_backbone(
            _spec(), torch.device("cpu"))
        feat = bb.forward_features(torch.randn(1, 3, 32, 32))
        self.assertEqual(feat.ndim, 4)
        self.assertEqual(feat.shape[0], 1)
        self.assertEqual(feat.shape[1], bb.out_channels)
        self.assertEqual(tuple(feat.shape[-2:]), (2, 2))

    def test_it_loads_a_vqvae_checkpoint_and_matches_it(self):
        # A real run names the VQVAE tokeniser checkpoint as ``encoder``; the arch
        # is inferred from it, so the rebuilt tokeniser is bit-identical to the one
        # that was saved.
        smoke, ckpt = _write_vqvae_ckpt(self.tmp)
        bb = spatial_backbones.build_frozen_backbone(
            _spec(encoder=str(ckpt)), torch.device("cpu"))
        self.assertEqual(bb.out_channels, smoke.out_channels)
        x = torch.randn(2, 3, 32, 32)
        self.assertTrue(
            torch.allclose(bb.forward_features(x), smoke.forward_features(x),
                           atol=1e-6),
            "a checkpoint loaded backbone must reproduce the tokeniser it saved")

    def test_the_backbone_is_frozen_and_stays_in_eval(self):
        bb = spatial_backbones.build_frozen_backbone(
            _spec(), torch.device("cpu"))
        self.assertFalse(any(p.requires_grad for p in bb.parameters()))
        bb.train()                       # a frozen backbone never leaves eval
        self.assertFalse(bb.training)

    def test_the_pooled_map_reproduces_vars_own_probe_feature(self):
        # The whole point of reuse: global-average-pooling the spatial map must
        # equal VAR's own probe feature. That feature is the VQVAE *encoder*
        # output, pooled (what the method's own linear probe reads); it is
        # reconstructed here from the backbone's own public `vae.encoder` -- an
        # independent path from `forward_features` -- rather than by loading the
        # method's eval module, so this shared test names no method. A provider
        # that read the quantised (post quant_conv) map, or the decoder, would
        # diverge from `vae.encoder(x)` and fail here.
        _smoke, ckpt = _write_vqvae_ckpt(self.tmp)
        bb = spatial_backbones.build_frozen_backbone(
            _spec(encoder=str(ckpt)), torch.device("cpu"))
        torch.manual_seed(0)
        x = torch.randn(2, 3, 32, 32)
        pooled_map = bb.forward_features(x).mean(dim=(2, 3))
        with torch.no_grad():
            expected = bb.vae.encoder(x).mean(dim=(2, 3))   # VAR's own probe read
        self.assertTrue(torch.allclose(pooled_map, expected, atol=1e-6),
                        (pooled_map - expected).abs().max().item())


@needs_torch
class TestTheStrictLoadRefuses(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="var-bb-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_a_checkpoint_that_is_not_a_vqvae_tokeniser_is_refused(self):
        # The arch is inferred from the tokeniser's own weights; a checkpoint with
        # no encoder input convolution is not this tokeniser and cannot be built.
        _smoke, ckpt = _write_vqvae_ckpt(self.tmp, drop="encoder.conv_in.weight")
        with self.assertRaises(RuntimeError) as e:
            spatial_backbones.build_frozen_backbone(
                _spec(encoder=str(ckpt)), torch.device("cpu"))
        self.assertIn("encoder.conv_in.weight", str(e.exception))

    def test_a_checkpoint_carrying_alien_keys_is_refused(self):
        _smoke, ckpt = _write_vqvae_ckpt(self.tmp, extra="not_a_vqvae_key")
        with self.assertRaises(RuntimeError) as e:
            spatial_backbones.build_frozen_backbone(
                _spec(encoder=str(ckpt)), torch.device("cpu"))
        self.assertIn("not_a_vqvae_key", str(e.exception))

    def test_a_checkpoint_missing_a_tokeniser_weight_is_refused(self):
        # A weight not read for inference, dropped: the strict load in build_vqvae
        # refuses it rather than half-loading a different tokeniser.
        _smoke, ckpt = _write_vqvae_ckpt(self.tmp, drop="encoder.conv_out.weight")
        with self.assertRaises(RuntimeError) as e:
            spatial_backbones.build_frozen_backbone(
                _spec(encoder=str(ckpt)), torch.device("cpu"))
        self.assertIn("encoder.conv_out.weight", str(e.exception))


@needs_torch
class TestTheProviderDoesNotLeakUpstreamImports(unittest.TestCase):
    """VAR's tokeniser is built through its own ``build_vqvae``, whose collision-
    safe upstream loader puts ``third_party/var`` first on ``sys.path`` and binds
    the upstream's ``models`` package under the shared ``models`` key -- correct
    inside the adapter's own process, but the in-process test suite (and the
    downstream harness) holds every method at once. Each ViT method imports its
    *own* ``models`` package (``from models import ...``); if the build leaves
    ``third_party/var`` on the path or the upstream ``models`` cached, every later
    ViT method resolves the wrong package and its models come out NaN / not round-
    tripping. So the provider must build hermetically: whatever global import state
    the build touches, it restores. Discovered, not asserted for VAR alone -- this
    is the invariant the shared layer relies on for every provider."""

    # The submodule root -- named generically, so this test hard-codes no method
    # (tests/test_no_hard_coded_methods.py): any upstream a provider reaches lives
    # under here, and a leak is a new ``sys.modules`` entry whose file is under it.
    _THIRD_PARTY = str(ROOT / "third_party")

    def test_building_leaves_no_upstream_models_on_the_import_path(self):
        path_before = list(sys.path)
        mods_before = set(sys.modules)
        spatial_backbones.build_frozen_backbone(_spec(), torch.device("cpu"))
        # The submodule root must not be left on the path (a later
        # `from models import ...` would otherwise resolve to the upstream's).
        self.assertEqual(sys.path, path_before, "sys.path was not restored")
        # ...nor any upstream module left bound (it would shadow another method's
        # top-level package of the same name -- the cross-method import bug).
        leaked = sorted(
            m for m in set(sys.modules) - mods_before
            if str(getattr(sys.modules[m], "__file__", "") or "")
            .startswith(self._THIRD_PARTY))
        self.assertEqual(leaked, [],
                         f"the build leaked upstream modules: {leaked}")

    def test_building_restores_a_shared_module_it_overwrites(self):
        # A leak has two shapes: a *new* key (covered above) and an *overwritten*
        # one -- a method that imported its own top-level ``models`` package first
        # would find the upstream's bound in its place. Stand a sentinel in that
        # shared key and confirm the build leaves that exact object behind, not the
        # upstream module it binds there while running.
        key = "models"
        original = sys.modules.get(key)
        sentinel = types.ModuleType(key)
        sys.modules[key] = sentinel

        def _restore():
            if original is not None:
                sys.modules[key] = original
            else:
                sys.modules.pop(key, None)

        self.addCleanup(_restore)
        spatial_backbones.build_frozen_backbone(_spec(), torch.device("cpu"))
        self.assertIs(
            sys.modules.get(key), sentinel,
            "the build overwrote a shared module and did not restore it")

    def test_building_does_not_perturb_the_global_rng(self):
        # The smoke draws seeded-normal weights, which advances the global RNG; a
        # later test drawing an unseeded input would otherwise see a shifted draw.
        torch.manual_seed(0)
        before = torch.get_rng_state()
        spatial_backbones.build_frozen_backbone(_spec(), torch.device("cpu"))
        self.assertTrue(torch.equal(torch.get_rng_state(), before),
                        "the build left the global RNG advanced")

    def test_building_restores_torchs_default_parameter_init(self):
        # The upstream disables torch's built-in init for its own build speed --
        # setattr(nn.<cls>, 'reset_parameters', a no-op) over eight nn classes,
        # globally. Left leaked, every module a later method builds keeps its
        # uninitialised torch.empty memory (NaN / denormal) and comes out not
        # finite -- the real cause of the cross-method breakage. Confirm the
        # provider restores it exactly: a class whose init it replaced is put
        # back, one it *added* an override to is cleared, and a module built after
        # is actually initialised.
        import torch.nn as nn
        before = {c: c.__dict__.get("reset_parameters")
                  for c in vars(nn).values()
                  if isinstance(c, type) and issubclass(c, nn.Module)}
        spatial_backbones.build_frozen_backbone(_spec(), torch.device("cpu"))
        changed = sorted(c.__name__ for c in before
                         if c.__dict__.get("reset_parameters") is not before[c])
        self.assertEqual(changed, [],
                         f"the build left torch.nn init patched on: {changed}")
        weight = nn.Linear(32, 32).weight
        self.assertTrue(torch.isfinite(weight).all(),
                        "a module built after the provider is uninitialised")
        self.assertGreater(float(weight.std()), 0.0)


@needs_timm
class TestTheEndToEndARSSLWiring(unittest.TestCase):
    """Drive the real ARSSL harness over one frozen VAR (VQVAE) backbone: the
    ADE20K dense task runs as a subprocess, is checked by the downstream contract,
    and the battery aggregates to one ok result. This is VAR reproducing in A1."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="var-arssl-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def _ade_key(self) -> str:
        from downstream import arssl
        return next(k for k, m in arssl.discover_tasks().items() if m == "ade20k")

    def test_a_var_backbone_drives_the_ade20k_task_through_the_battery(self):
        data = tiny_ade(self.tmp / "data")
        cfg = {"task": "arssl", "seed": 0, "device": "cpu",
               "backbone": _spec(),                 # empty encoder: hermetic smoke
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
