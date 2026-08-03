#!/usr/bin/env python3
"""Specification for mar (Li et al., 2024; arXiv:2406.11838).

**The first port whose model is a pinned submodule, not code copied in.** The
upstream (`gatheluck/mar`, a fork of `LTH14/mar` carrying a two-line device
patch) lives under `third_party/mar` and is imported, so this file also pins the
things a submodule brings: that the working tree's submodule commit is the one
the adapter records, and that the adapter imports *that* upstream rather than
another method's `models` package.

Everything else follows the earlier ports: the config is translated and refused
by name, the device is resolved rather than sniffed, the output stays under
`--out`, and a hermetic smoke runs a real training step -- here on **fabricated
cached latents**, so no VAE and no download, which is what lets it run offline
and on a CPU.

`encoder.pt` is the MAE-encoder side of MAR; the round trip is proved here, in
the one place the method's dependencies are guaranteed present.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from _method_import import load_from        # noqa: E402
from _checkout import needs_checkout         # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
METHOD = ROOT / "methods" / "mar"
BIN = ROOT / "bin"
UPSTREAM = ROOT / "third_party" / "mar"

# The upstream model reads timm and scipy at import, and numpy throughout; a
# smoke without them would *fail* rather than skip, which reports a defect where
# the environment is merely incomplete. The condition names what is needed.
try:
    import torch                                       # noqa: F401
    import numpy                                       # noqa: F401
    import timm                                        # noqa: F401
    import scipy                                       # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

needs_deps = unittest.skipUnless(
    HAVE_DEPS, "mar needs torch, numpy, timm and scipy (the upstream's stack)")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("mar_adapter", METHOD / "adapter" / "__init__.py")

# A model small enough to train a step on a CPU in a moment. img_size 16 over a
# vae_stride of 16 is a single latent token; the encoder dimensions are fixed by
# mar_base and are not shrinkable, so this stays the real architecture.
TRAIN = {"img_size": 16, "vae_stride": 16, "patch_size": 1, "vae_embed_dim": 16,
         "class_num": 4, "buffer_size": 64, "diffloss_d": 1, "diffloss_w": 64,
         "mask_ratio_min": 0.7, "label_drop_prob": 0.1,
         "epochs": 1, "batch_size": 2, "num_workers": 0, "lr": 1.0e-4,
         "weight_decay": 0.02, "grad_clip": 3.0}


def tiny_cached_latents(root: Path, n: int = 4) -> Path:
    """A cached-latents directory in the upstream CachedFolder format.

    Class subdirectories of `.npz` files, each carrying `moments` and
    `moments_flip` of shape `[2*vae_embed_dim, h, w]`. Fabricated rather than
    produced by a VAE, so nothing is downloaded and the smoke runs offline.
    """
    import numpy as np
    cls = root / "class0"
    cls.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    embed2 = 2 * TRAIN["vae_embed_dim"]
    h = TRAIN["img_size"] // TRAIN["vae_stride"]
    for i in range(n):
        np.savez(cls / f"{i}.npz",
                 moments=rng.randn(embed2, h, h).astype(np.float32),
                 moments_flip=rng.randn(embed2, h, h).astype(np.float32))
    return root


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="mar-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def config(self, **over) -> dict:
        cfg = {"stage": "step1", "seed": 0,
               "data_root": str(self.tmp / "cached"),
               "device": "cpu", "train": dict(TRAIN)}
        for k, v in over.items():
            if k == "train" and v is not None:
                cfg["train"] = {**cfg["train"], **v}
            else:
                cfg[k] = v
        return cfg


class TestThePinnedUpstream(unittest.TestCase):
    """A submodule method is only reproducible if the commit it records is the
    commit that is checked out, and if it imports that upstream and no other."""

    @needs_checkout
    def test_the_adapter_records_the_checked_out_commit(self):
        """The gitlink in the working tree must be the commit the adapter names
        in every manifest. A drift here means a run recorded against code that
        is not what ran. Gated on a checkout: where there is no git (the
        container image), the pin is not readable and the test skips."""
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=UPSTREAM,
                           capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("the submodule is not checked out here")
        self.assertEqual(r.stdout.strip(), adapter.UPSTREAM["commit"],
                         "the adapter's upstream commit is not the one checked "
                         "out under third_party/mar")

    def test_provenance_agrees_with_the_adapter(self):
        prov = json.loads((METHOD / "provenance.json").read_text())
        self.assertEqual(prov["upstream"]["commit"], adapter.UPSTREAM["commit"])
        self.assertEqual(prov["upstream"]["repo"], adapter.UPSTREAM["repo"])

    def test_the_fork_is_pinned_not_the_bare_upstream(self):
        """DESIGN section 2.8: the patch lives in our fork, pinned. The URL must
        be the fork, and it must differ from what it forked."""
        prov = json.loads((METHOD / "provenance.json").read_text())
        self.assertNotEqual(prov["upstream"]["repo"], prov["upstream"]["fork_of"])
        self.assertNotEqual(prov["upstream"]["commit"],
                            prov["upstream"]["base_commit"])

    @needs_deps
    def test_the_pinned_upstream_is_imported_not_another_methods_models(self):
        """`models` is a package name three methods define; the upstream defines
        it too. Caching another method's `models` first must not decide which
        `mar_base` the adapter gets."""
        two = ROOT / "methods" / "02_vae"
        load_from(two, "models", two / "models" / "__init__.py")  # poison cache
        trainer = load("mar_trainer", METHOD / "train_step1_mar.py")
        trainer._load_upstream()
        self.assertTrue(
            sys.modules["models.mar"].__file__.startswith(str(UPSTREAM)),
            "the upstream models.mar was not imported from third_party/mar")


class TestConfigTranslation(Base):
    def test_every_setting_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["training"]["epochs"], 1)
        self.assertEqual(built["model"]["vae_embed_dim"], 16)
        self.assertEqual(built["model"]["class_num"], 4)
        self.assertEqual(built["data"]["batch_size"], 2)
        self.assertEqual(built["seed"], 0)

    def test_a_missing_setting_is_refused_by_name(self):
        for key in TRAIN:
            with self.subTest(key=key):
                cfg = self.config()
                cfg["train"] = {k: v for k, v in TRAIN.items() if k != key}
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(cfg, out=self.out)
                self.assertIn(key, str(e.exception))

    def test_an_unknown_setting_is_refused_by_name(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(train={"momentum": 0.9}),
                                  out=self.out)
        self.assertIn("momentum", str(e.exception))

    def test_an_unknown_stage_is_refused(self):
        with self.assertRaises(adapter.ConfigError):
            adapter.to_run_config(self.config(stage="linear_eval"),
                                  out=self.out)

    def test_the_checkpoint_dir_is_under_out(self):
        built = adapter.to_run_config(self.config(), out=Path("/tmp/x/out"))
        self.assertTrue(Path(built["output"]["checkpoint_dir"])
                        .is_relative_to(Path("/tmp/x/out")))

    def test_a_config_that_sets_output_is_refused(self):
        cfg = self.config()
        cfg["output"] = {"checkpoint_dir": "/anywhere"}
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("--out", str(e.exception))


class TestExtractingTheEncoder(unittest.TestCase):
    """Pure, so the guards can be reached without building the model."""

    def test_only_the_encoder_side_comes_out(self):
        got = adapter.extract_encoder({
            "z_proj.weight": 1, "encoder_blocks.0.norm1.weight": 2,
            "class_emb.weight": 3, "fake_latent": 4, "encoder_norm.weight": 5,
            "decoder_blocks.0.norm1.weight": 6, "diffloss.net.0.weight": 7,
            "mask_token": 8})
        self.assertEqual(
            set(got), {"z_proj.weight", "encoder_blocks.0.norm1.weight",
                       "class_emb.weight", "fake_latent", "encoder_norm.weight"})

    def test_the_decoder_and_diffusion_head_are_left_out(self):
        got = adapter.extract_encoder({"z_proj.weight": 1,
                                       "decoder_blocks.0.w": 2,
                                       "diffloss.net.0.w": 3})
        self.assertNotIn("decoder_blocks.0.w", got)
        self.assertNotIn("diffloss.net.0.w", got)

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"decoder_blocks.0.w": 1})
        self.assertIn("encoder", str(e.exception).lower())

    def test_an_empty_checkpoint_is_refused(self):
        with self.assertRaises(RuntimeError):
            adapter.extract_encoder({})


class TestTheLossMustHaveBeenMeasured(Base):
    """A number is not a result; a run that reports nothing is flagged."""

    def test_a_run_that_reports_nothing_is_flagged(self):
        m = adapter.run_training(self.config(), self.out,
                                 _run=lambda a, c=None: None)
        self.assertGreaterEqual(m.get("metrics_unavailable", 0), 1)
        self.assertNotIn("final_loss", m)

    def test_a_run_that_reports_a_loss_is_not_flagged(self):
        m = adapter.run_training(
            self.config(), self.out,
            _run=lambda a, c=None: {"epochs": 1, "final_loss": 3.5})
        self.assertEqual(m["final_loss"], 3.5)
        self.assertNotIn("metrics_unavailable", m)

    def test_a_non_numeric_loss_is_counted_not_written(self):
        m = adapter.run_training(
            self.config(), self.out,
            _run=lambda a, c=None: {"epochs": 1, "final_loss": "low"})
        self.assertNotIn("final_loss", m)
        self.assertGreaterEqual(m.get("metrics_unavailable", 0), 1)


class TestTheDeviceIsResolved(Base):
    """The device is decided from the config, not sniffed from the hardware.
    The upstream forward hard-coded `.cuda()`; the fork and resolve_device make
    `device: cpu` real. See docs/GPU.md section 4. (Referenced by the device
    mutation spec.)"""

    def trainer(self):
        return load("mar_trainer", METHOD / "train_step1_mar.py")

    @needs_deps
    def test_asking_for_cuda_without_one_is_refused(self):
        from unittest import mock
        t = self.trainer()
        with mock.patch.object(t.torch.cuda, "is_available",
                               return_value=False):
            with self.assertRaises(RuntimeError) as e:
                t.resolve_device("cuda", 0)
            self.assertIn("cuda", str(e.exception).lower())
            self.assertEqual(t.resolve_device("cpu", 0).type, "cpu")
            self.assertEqual(t.resolve_device("auto", 0).type, "cpu")

    @needs_deps
    def test_cpu_is_honoured_even_where_cuda_exists(self):
        from unittest import mock
        t = self.trainer()
        with mock.patch.object(t.torch.cuda, "is_available",
                               return_value=True):
            self.assertEqual(t.resolve_device("cpu", 0).type, "cpu")
            self.assertEqual(t.resolve_device("auto", 0).type, "cuda")
            self.assertEqual(t.resolve_device("cuda", 0).type, "cuda")

    def test_run_resolves_the_device_rather_than_sniffing_it(self):
        """Structural, needs no torch: run() must go through resolve_device."""
        import ast
        src = (METHOD / "train_step1_mar.py").read_text()
        run_fn = next(n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == "run")
        called = {n.func.id for n in ast.walk(run_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("resolve_device", called)


class TestASmokeRun(Base):
    def run_adapter(self, **over):
        tiny_cached_latents(self.tmp / "cached")
        cfg = self.tmp / "resolved.json"
        cfg.write_text(json.dumps(self.config(**over)), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        return cfg, subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(cfg),
             "--out", str(self.out)],
            cwd=METHOD, env=env, capture_output=True, text=True)

    @needs_deps
    def test_it_completes_and_satisfies_the_contract(self):
        cfg, r = self.run_adapter()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    @needs_deps
    def test_the_manifest_records_the_pinned_upstream(self):
        """A submodule method must say which upstream produced the result."""
        self.run_adapter()
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["upstream"], adapter.UPSTREAM)

    @unittest.skipUnless(HAVE_DEPS and torch.cuda.is_available(),
                         "no CUDA device; the GPU path cannot be exercised here")
    def test_a_real_run_on_cuda_produces_a_loadable_encoder(self):
        """The GPU path on real hardware: the case the patched `.cuda()` call
        sites and the device resolution exist for. A device-placement mistake
        raises inside training, so a run that finishes and writes a non-empty
        encoder is the GPU path working end to end."""
        cfg, r = self.run_adapter(device="cuda")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn("cuda", r.stdout.lower())
        saved = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(saved, "encoder.pt is empty after a CUDA run")

    @needs_deps
    def test_the_encoder_is_the_encoder_side_of_the_model(self):
        self.run_adapter()
        state = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(state)
        self.assertFalse([k for k in state if k.startswith("decoder")],
                         "the decoder is in encoder.pt")
        self.assertFalse([k for k in state if k.startswith("diffloss")],
                         "the diffusion loss head is in encoder.pt")

    @needs_deps
    def test_nothing_is_written_outside_out(self):
        before = {p for p in self.tmp.rglob("*")}
        self.run_adapter()
        stray = [p for p in self.tmp.rglob("*")
                 if p not in before and not p.is_relative_to(self.out)
                 and not p.is_relative_to(self.tmp / "cached")
                 and p != self.tmp / "resolved.json"]
        self.assertEqual(stray, [], f"written outside --out: {stray}")

    @needs_deps
    def test_the_loss_is_recorded_as_a_pretext_number(self):
        self.run_adapter()
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        self.assertIn("final_pretext_loss", m, "the run reported no loss")
        self.assertNotIn("metrics_unavailable", m)
        for k, v in m.items():
            self.assertIsInstance(v, (int, float))
            self.assertNotIsInstance(v, bool)

    @needs_deps
    def test_the_same_config_twice_gives_the_same_encoder(self):
        digests = []
        for name in ("a", "b"):
            self.out = self.tmp / name
            self.run_adapter()
            digests.append(hashlib.sha256(
                (self.out / "encoder.pt").read_bytes()).hexdigest())
        self.assertEqual(digests[0], digests[1])

    @needs_deps
    def test_a_missing_dataset_is_reported_as_a_failure(self):
        cfg = self.tmp / "resolved.json"
        cfg.write_text(json.dumps(self.config(
            data_root=str(self.tmp / "absent"))), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(cfg),
             "--out", str(self.out)],
            cwd=METHOD, env=env, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)

    @needs_deps
    def test_the_encoder_pt_it_wrote_loads_back(self):
        """The round trip: writing an encoder.pt and never reading one back is
        how a file that loads nothing goes unnoticed. Weights are compared, not
        just the absence of an exception."""
        self.run_adapter()
        saved = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(saved, "encoder.pt is empty")
        model = adapter.load_encoder(saved, self.config())
        loaded = model.state_dict()
        pairs = 0
        for key, want in saved.items():
            got = loaded.get(key)
            if got is None:
                continue
            pairs += 1
            self.assertTrue(torch.equal(got, want), f"{key} came back changed")
        self.assertGreater(pairs, 0, "no saved weight reached the model")


class TestTheOriginalIsReferencedNotCopied(unittest.TestCase):
    """The model is a submodule, so there are no copied science files to hash;
    what must hold is that the port owns only the thin loop and calls the
    upstream."""

    def test_the_body_lives_in_run_and_main_only_parses(self):
        import ast
        src = (METHOD / "train_step1_mar.py").read_text()
        top = {n.name for n in ast.parse(src).body
               if isinstance(n, ast.FunctionDef)}
        for fn in ("run", "main", "build_parser"):
            self.assertIn(fn, top)
        main_src = src[src.index("def main("):]
        self.assertLess(len(main_src.splitlines()), 12,
                        "main() should parse and delegate, nothing more")

    def test_it_asks_for_deterministic_seeding(self):
        import ast
        src = (METHOD / "train_step1_mar.py").read_text()
        run_fn = next(n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == "run")
        called = {n.func.id for n in ast.walk(run_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("make_deterministic", called)

    def test_the_engine_is_not_imported(self):
        """engine_mar imports torch_fidelity and cv2 and calls
        torch.cuda.synchronize(); importing it would drag those in and break a
        CPU run. The port owns the loop instead.

        Checked against the actual imports, not the source text: the docstring
        names engine_mar to explain why it is avoided, and a substring search
        would match that -- the too-wide-scope mistake this project keeps a list
        of."""
        import ast
        src = (METHOD / "train_step1_mar.py").read_text()
        imported = set()
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Import):
                imported |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                imported.add(n.module.split(".")[0])
        self.assertNotIn("engine_mar", imported)


if __name__ == "__main__":
    unittest.main()
