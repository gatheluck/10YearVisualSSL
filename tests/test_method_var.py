#!/usr/bin/env python3
"""Specification for var (Tian et al., 2024; arXiv:2404.02905).

The second port on the `third_party/` submodule mechanism, and the first
`submodule+adapter` one: the model is the pinned upstream under `third_party/var`,
imported not copied, and pinned **directly** (no fork) because it runs on a CPU
or a GPU unmodified.

The shape follows mar: the config is translated and refused by name, the device
is resolved, the output stays under `--out`, and a hermetic smoke runs a real
training step -- here over a **tiny random VQVAE** and a few fabricated images,
so no VQVAE is downloaded. `encoder.pt` is the VAR representation side; the round
trip is proved here.
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
METHOD = ROOT / "methods" / "var"
BIN = ROOT / "bin"
UPSTREAM = ROOT / "third_party" / "var"

try:
    import torch                                       # noqa: F401
    import numpy                                       # noqa: F401
    import torchvision                                 # noqa: F401
    import huggingface_hub                             # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

needs_deps = unittest.skipUnless(
    HAVE_DEPS, "var needs torch, numpy, torchvision and huggingface_hub")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("var_adapter", METHOD / "adapter" / "__init__.py")

# A model small enough to train a step on a CPU in a moment. patch_nums[-1]=3
# over the VQVAE's 16x downsample fixes the image size at 48; the encoder
# dimensions come from depth/ch and are the real architecture, just small.
TRAIN = {"patch_nums": [1, 2, 3], "vocab_size": 16, "Cvae": 32, "ch": 32,
         "num_classes": 4, "depth": 2, "shared_aln": False, "attn_l2_norm": True,
         "epochs": 1, "batch_size": 2, "num_workers": 0, "lr": 1.0e-4,
         "weight_decay": 0.05, "grad_clip": 1.0, "vqvae_ckpt": ""}

# The architecture keys `build_vae_var` needs, shared by both stages: the
# linear_eval probe reads the VQVAE, which is built the same way.
MODEL = {"patch_nums": [1, 2, 3], "vocab_size": 16, "Cvae": 32, "ch": 32,
         "num_classes": 4, "depth": 2, "shared_aln": False, "attn_l2_norm": True}

# A linear_eval config small enough to run a probe on a CPU in a moment. It
# carries the model arch (to build the VQVAE), the tokeniser checkpoint
# (`vqvae_ckpt`, empty here -> a random VQVAE, as in the smoke), and the probe
# hyperparameters. img_size 32 over the VQVAE's 16x downsample -> a 2x2 map.
EVAL_TRAIN = {**MODEL, "epochs": 2, "batch_size": 2, "num_workers": 0,
              "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0, "img_size": 32,
              "vqvae_ckpt": ""}


def tiny_imagefolder(root: Path, n: int = 4) -> Path:
    """A minimal ImageFolder of fabricated images under train/ -- no download,
    no VQVAE."""
    import numpy as np
    from PIL import Image
    cls = root / "train" / "class0"
    cls.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(n):
        Image.fromarray(rng.randint(0, 256, (64, 64, 3), dtype="uint8")).save(
            cls / f"{i}.png")
    return root


def tiny_split(root: Path, per: int = 3) -> Path:
    """A labelled ImageFolder with train/ and val/, two classes each, so a
    linear probe has something separable to fit. No download, no VQVAE."""
    import numpy as np
    from PIL import Image
    rng = np.random.RandomState(0)
    for split in ("train", "val"):
        for label, cls in enumerate(("c0", "c1")):
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(per):
                # A per-class colour bias, so the probe can do better than chance
                # and best/final are not identically zero.
                base = np.full((64, 64, 3), label * 120, dtype="uint8")
                noise = rng.randint(0, 64, (64, 64, 3), dtype="uint8")
                Image.fromarray((base + noise).astype("uint8")).save(
                    d / f"{i}.png")
    return root


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="var-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def config(self, **over) -> dict:
        cfg = {"stage": "pretrain", "seed": 0,
               "data_root": str(self.tmp / "data"),
               "device": "cpu", "train": dict(TRAIN)}
        for k, v in over.items():
            if k == "train" and v is not None:
                cfg["train"] = {**cfg["train"], **v}
            else:
                cfg[k] = v
        return cfg

    def eval_config(self, **over) -> dict:
        cfg = {"stage": "linear_eval", "seed": 0,
               "data_root": str(self.tmp / "data"),
               "device": "cpu", "train": dict(EVAL_TRAIN)}
        for k, v in over.items():
            if k == "train" and v is not None:
                cfg["train"] = {**cfg["train"], **v}
            else:
                cfg[k] = v
        return cfg


class TestThePinnedUpstream(unittest.TestCase):
    @needs_checkout
    def test_the_adapter_records_the_checked_out_commit(self):
        """The gitlink in the working tree must be the commit the adapter names
        in every manifest. Gated on a checkout: no git, no pin to read."""
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=UPSTREAM,
                           capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("the submodule is not checked out here")
        self.assertEqual(r.stdout.strip(), adapter.UPSTREAM["commit"])

    def test_provenance_agrees_with_the_adapter(self):
        prov = json.loads((METHOD / "provenance.json").read_text())
        self.assertEqual(prov["upstream"]["commit"], adapter.UPSTREAM["commit"])
        self.assertEqual(prov["upstream"]["repo"], adapter.UPSTREAM["repo"])

    def test_the_upstream_is_pinned_directly_not_a_fork(self):
        """Unlike mar, VAR needs no patch, so it pins the upstream directly:
        the URL is FoundationVision/VAR and there is no fork indirection."""
        prov = json.loads((METHOD / "provenance.json").read_text())
        self.assertIn("FoundationVision/VAR", adapter.UPSTREAM["repo"])
        self.assertNotIn("fork_of", prov["upstream"])

    @needs_deps
    def test_the_pinned_upstream_is_imported_not_another_methods_models(self):
        """`models` is a package name several methods define; the upstream
        defines it too. Caching another method's `models` first must not decide
        which `build_vae_var` the adapter gets."""
        two = ROOT / "methods" / "02_vae"
        load_from(two, "models", two / "models" / "__init__.py")  # poison cache
        trainer = load("var_trainer", METHOD / "train_pretrain_var.py")
        trainer._load_upstream()
        self.assertTrue(
            sys.modules["models"].__file__.startswith(str(UPSTREAM)),
            "the upstream `models` was not imported from third_party/var")


class TestConfigTranslation(Base):
    def test_every_setting_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["training"]["epochs"], 1)
        self.assertEqual(built["model"]["depth"], 2)
        self.assertEqual(built["model"]["patch_nums"], [1, 2, 3])
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
        # step2 is a real captured stage that this port does not bring across;
        # linear_eval is now a known stage and is exercised below.
        with self.assertRaises(adapter.ConfigError):
            adapter.to_run_config(self.config(stage="step2"),
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
    def test_only_the_representation_side_comes_out(self):
        got = adapter.extract_encoder({
            "word_embed.weight": 1, "blocks.0.attn.mat_qkv.weight": 2,
            "class_emb.weight": 3, "pos_start": 4, "lvl_embed.weight": 5,
            "head.weight": 6, "head_nm.ada_lin.1.weight": 7})
        self.assertEqual(
            set(got), {"word_embed.weight", "blocks.0.attn.mat_qkv.weight",
                       "class_emb.weight", "pos_start", "lvl_embed.weight"})

    def test_the_generative_head_is_left_out(self):
        got = adapter.extract_encoder({"blocks.0.w": 1, "head.weight": 2,
                                       "head_nm.ada_lin.1.weight": 3})
        self.assertNotIn("head.weight", got)
        self.assertNotIn("head_nm.ada_lin.1.weight", got)

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"head.weight": 1})
        self.assertIn("encoder", str(e.exception).lower())

    def test_an_empty_checkpoint_is_refused(self):
        with self.assertRaises(RuntimeError):
            adapter.extract_encoder({})


class TestBuildingTheVQVAEDoesNotLeakGlobalInit(unittest.TestCase):
    """Building the tokeniser must not corrupt the other methods' models.

    The pinned upstream's `build_vae_var` (`third_party/var/models/__init__.py`),
    for its own build speed, does `setattr(clz, 'reset_parameters', lambda self:
    None)` over eight `torch.nn` classes -- globally and permanently. Every module
    built *afterwards* then skips initialisation and keeps its `torch.empty` memory
    (NaN / denormal garbage). The whole test suite and the downstream harness hold
    every method in one process, and VAR sorts before the ViT methods, so a leaked
    patch turned a later method's freshly built model NaN (`videomae`'s
    checkpoint-loads-and-probes went not-finite once it was added right after VAR).
    `build_vqvae` is the one place that calls `build_vae_var`, so it must snapshot
    and restore that global init exactly -- for every path (pretraining as well as
    linear_eval), not only the downstream provider that used to wrap it.
    """

    # The exact eight classes the upstream patches (third_party/var/models
    # __init__.build_vae_var). Named here so a class it stops restoring is caught.
    PATCHED = (
        "Linear", "LayerNorm", "BatchNorm2d", "SyncBatchNorm",
        "Conv1d", "Conv2d", "ConvTranspose1d", "ConvTranspose2d",
    )

    @needs_deps
    def test_it_restores_torch_default_init_for_the_next_method(self):
        import torch
        import torch.nn as nn

        trainer = load("var_trainer", METHOD / "train_pretrain_var.py")

        absent = object()
        before = {name: getattr(nn, name).__dict__.get("reset_parameters", absent)
                  for name in self.PATCHED}

        vae, _var = trainer.build_vqvae(dict(MODEL), "", torch.device("cpu"))

        # The function object each class carried (or its absence) is put back, so
        # nothing downstream sees the no-op patch.
        for name in self.PATCHED:
            after = getattr(nn, name).__dict__.get("reset_parameters", absent)
            self.assertIs(
                after, before[name],
                f"nn.{name}.reset_parameters was left patched by build_vqvae")

        # The property that actually bit videomae: a module built *after* the
        # tokeniser is initialised, not left as `torch.empty` garbage. A no-op
        # reset_parameters leaves the weight uninitialised (NaN / not varying).
        probe = nn.Linear(64, 64)
        self.assertTrue(torch.isfinite(probe.weight).all(),
                        "a Linear built after build_vqvae is not finite")
        self.assertGreater(float(probe.weight.std()), 0.0,
                           "a Linear built after build_vqvae was not initialised")


class TestTheLossMustHaveBeenMeasured(Base):
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
    """Referenced by the device mutation spec."""

    def trainer(self):
        return load("var_trainer", METHOD / "train_pretrain_var.py")

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
        import ast
        src = (METHOD / "train_pretrain_var.py").read_text()
        run_fn = next(n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == "run")
        called = {n.func.id for n in ast.walk(run_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("resolve_device", called)


class TestASmokeRun(Base):
    def run_adapter(self, **over):
        tiny_imagefolder(self.tmp / "data")
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
        self.run_adapter()
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["upstream"], adapter.UPSTREAM)

    @unittest.skipUnless(HAVE_DEPS and torch.cuda.is_available(),
                         "no CUDA device; the GPU path cannot be exercised here")
    def test_a_real_run_on_cuda_produces_a_loadable_encoder(self):
        """The GPU path on real hardware -- the case device resolution exists
        for. A device-placement mistake raises inside training, so a run that
        finishes and writes a non-empty encoder is the GPU path working."""
        cfg, r = self.run_adapter(device="cuda")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn("cuda", r.stdout.lower())
        saved = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(saved, "encoder.pt is empty after a CUDA run")

    @needs_deps
    def test_the_encoder_is_the_representation_side(self):
        self.run_adapter()
        state = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(state)
        self.assertFalse([k for k in state if k.startswith("head")],
                         "the generative head is in encoder.pt")

    @needs_deps
    def test_nothing_is_written_outside_out(self):
        before = {p for p in self.tmp.rglob("*")}
        self.run_adapter()
        stray = [p for p in self.tmp.rglob("*")
                 if p not in before and not p.is_relative_to(self.out)
                 and not p.is_relative_to(self.tmp / "data")
                 and p != self.tmp / "resolved.json"]
        self.assertEqual(stray, [], f"written outside --out: {stray}")

    @needs_deps
    def test_the_loss_is_recorded_as_a_pretext_number(self):
        self.run_adapter()
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        self.assertIn("final_pretext_loss", m)
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
        """The round trip: weights are compared, not just the absence of an
        exception."""
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
    def test_the_body_lives_in_run_and_main_only_parses(self):
        import ast
        src = (METHOD / "train_pretrain_var.py").read_text()
        top = {n.name for n in ast.parse(src).body
               if isinstance(n, ast.FunctionDef)}
        for fn in ("run", "main", "build_parser"):
            self.assertIn(fn, top)
        main_src = src[src.index("def main("):]
        self.assertLess(len(main_src.splitlines()), 12,
                        "main() should parse and delegate, nothing more")

    def test_it_asks_for_deterministic_seeding(self):
        import ast
        src = (METHOD / "train_pretrain_var.py").read_text()
        run_fn = next(n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == "run")
        called = {n.func.id for n in ast.walk(run_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("make_deterministic", called)

    def test_the_ddp_trainer_is_not_imported(self):
        """The upstream trainer.py/dist.py assume DDP and a cuda AMP context;
        the port owns its loop instead. Checked against actual imports, not the
        source text (the docstring names them to say they are avoided)."""
        import ast
        src = (METHOD / "train_pretrain_var.py").read_text()
        imported = set()
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Import):
                imported |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                imported.add(n.module.split(".")[0])
        self.assertNotIn("trainer", imported)
        self.assertNotIn("dist", imported)


class TestLinearEvalConfig(Base):
    """The second stage: a linear probe on the VQVAE tokeniser. It reads its
    own key set and, unlike step 1, produces no encoder."""

    def test_linear_eval_is_an_accepted_stage(self):
        # Must not raise: linear_eval is a stage this port implements.
        adapter.to_run_config(self.eval_config(), out=self.out)

    def test_a_missing_eval_setting_is_refused_by_name(self):
        for key in EVAL_TRAIN:
            with self.subTest(key=key):
                cfg = self.eval_config()
                cfg["train"] = {k: v for k, v in EVAL_TRAIN.items() if k != key}
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(cfg, out=self.out)
                self.assertIn(key, str(e.exception))

    def test_an_unknown_eval_setting_is_refused_by_name(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.eval_config(train={"grad_clip": 1.0}),
                                  out=self.out)
        self.assertIn("grad_clip", str(e.exception))

    def test_a_step1_only_key_is_not_silently_accepted_in_eval(self):
        # grad_clip belongs to step 1; the probe never reads it.
        self.assertNotIn("grad_clip", EVAL_TRAIN)

    def test_an_eval_config_that_sets_output_is_refused(self):
        cfg = self.eval_config()
        cfg["output"] = {"checkpoint_dir": "/anywhere"}
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("--out", str(e.exception))

    def test_an_unknown_device_is_refused_in_eval(self):
        with self.assertRaises(adapter.ConfigError):
            adapter.to_run_config(self.eval_config(device="tpu"), out=self.out)


class TestTheEvalProducesNoEncoder(Base):
    """CONTRACT section 3: a stage that produces no encoder must say so, and a
    stage that does must not claim otherwise."""

    def _reason_for(self, cfg: dict) -> "str | None":
        path = self.tmp / "resolved.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        return adapter._absent_reason(path)

    def test_linear_eval_declares_it_produces_no_encoder(self):
        reason = self._reason_for(self.eval_config())
        self.assertIsInstance(reason, str)
        self.assertTrue(reason)

    def test_step1_produces_an_encoder_so_gives_no_reason(self):
        self.assertIsNone(self._reason_for(self.config()))


class TestTheEvalMetricsAreComparable(unittest.TestCase):
    """The probe's numbers go in the comparable column, and the mapping is
    checked against the one vocabulary (adapterlib), not a private copy."""

    def _vocab(self):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import adapterlib
        return adapterlib

    def test_every_mapped_eval_name_is_in_the_vocabulary(self):
        al = self._vocab()
        for target in adapter.LINEAR_EVAL_METRIC_NAMES.values():
            if target is None:
                continue
            self.assertIn(target, al.METRIC_VOCABULARY)

    def test_the_four_probe_accuracies_are_linear_probe_names(self):
        al = self._vocab()
        mapped = set(adapter.LINEAR_EVAL_METRIC_NAMES.values())
        for name in ("best_linear_probe_top1_accuracy",
                     "final_linear_probe_top1_accuracy",
                     "best_linear_probe_top5_accuracy",
                     "final_linear_probe_top5_accuracy"):
            self.assertIn(name, mapped)
            self.assertEqual(al.METRIC_VOCABULARY[name], al.COMPARABLE)


class TestTheProbeReadsTheVqvaeNotTheTransformer(Base):
    """The representation is the VQVAE encoder's continuous features, global
    average-pooled to Cvae dims -- faithful to the lab's ARSSL protocol
    (extract_var_features) and NOT the VAR transformer this port trains. Its
    dimensionality distinguishes the two: the transformer's width is not Cvae."""

    def modules(self):
        trainer = load("var_trainer", METHOD / "train_pretrain_var.py")
        ev = load("var_eval", METHOD / "evaluate_linear_var.py")
        return trainer, ev

    @needs_deps
    def test_features_are_the_vqvae_encoder_pooled_to_cvae_dims(self):
        import torch
        trainer, ev = self.modules()
        trainer.make_deterministic(0)
        vae, _var = trainer.build_vqvae(
            dict(EVAL_TRAIN), EVAL_TRAIN["vqvae_ckpt"], torch.device("cpu"))
        imgs = torch.zeros(2, 3, EVAL_TRAIN["img_size"], EVAL_TRAIN["img_size"])
        feats = ev.encode(vae, imgs)
        self.assertEqual(tuple(feats.shape), (2, EVAL_TRAIN["Cvae"]))


class TestALinearEvalSmoke(Base):
    def run_adapter(self, **over):
        tiny_split(self.tmp / "data")
        cfg = self.tmp / "resolved.json"
        cfg.write_text(json.dumps(self.eval_config(**over)), encoding="utf-8")
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
    def test_it_reports_the_comparable_probe_numbers(self):
        self.run_adapter()
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        for name in ("best_linear_probe_top1_accuracy",
                     "final_linear_probe_top1_accuracy",
                     "best_linear_probe_top5_accuracy",
                     "final_linear_probe_top5_accuracy"):
            self.assertIn(name, m)
        self.assertNotIn("metrics_unavailable", m)

    @needs_deps
    def test_it_produces_no_encoder_and_the_manifest_says_so(self):
        self.run_adapter()
        self.assertFalse((self.out / "encoder.pt").exists(),
                         "linear_eval must not write an encoder")
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["stage"], "linear_eval")
        self.assertEqual(man["status"], "ok", man.get("error", ""))
        self.assertIn("encoder_absent_reason", man)

    @needs_deps
    def test_a_missing_dataset_is_reported_as_a_failure(self):
        cfg = self.tmp / "resolved.json"
        cfg.write_text(json.dumps(self.eval_config(
            data_root=str(self.tmp / "absent"))), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(cfg),
             "--out", str(self.out)],
            cwd=METHOD, env=env, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)

    @unittest.skipUnless(HAVE_DEPS and torch.cuda.is_available(),
                         "no CUDA device; the GPU path cannot be exercised here")
    def test_the_probe_runs_on_cuda(self):
        """The GPU path on real hardware -- the device invariant (docs/GPU.md
        section 4) holds for linear_eval too. A device-placement mistake raises
        inside feature extraction, so a run that finishes on cuda with the
        comparable numbers written is the GPU path working."""
        cfg, r = self.run_adapter(device="cuda")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn("cuda", r.stdout.lower())
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        self.assertIn("final_linear_probe_top1_accuracy", m)


if __name__ == "__main__":
    unittest.main()
