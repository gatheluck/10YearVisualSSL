#!/usr/bin/env python3
"""Specification for 24_beit (Bao et al., 2021; arXiv:2106.08254).

BEiT: masked image modeling with discrete targets. A dVAE tokenizer turns each
image into a grid of discrete visual tokens; a random block of image patches is
replaced by a shared learned mask token in the ViT input; the ViT predicts the
visual tokens at the masked positions -- a cross-entropy over the dVAE vocabulary.
The ViT is the lab's own (LayerScale blocks, no timm); the tokenizer is the frozen
OpenAI DALL-E dVAE for a real run (a hash-pinned download, imported lazily), and a
random torch-only tokenizer for the hermetic smoke, so CI downloads nothing.

`encoder.pt` is the BEiT backbone trunk (patch_embed, cls_token, pos_embed, blocks,
norm); the shared mask token and the MIM head are training machinery and are
excluded. `linear_eval` probes the backbone's mean-pooled patch tokens (embed_dim,
CLS excluded). The captured step 2 (ViT fine-tuning) is excluded, as in every port.
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

ROOT = Path(__file__).resolve().parent.parent
METHOD = ROOT / "methods" / "24_beit"
BIN = ROOT / "bin"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import adapterlib                            # noqa: E402

try:
    import torch                                       # noqa: F401
    import numpy                                       # noqa: F401
    import torchvision                                 # noqa: F401
    from PIL import Image                              # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

needs_deps = unittest.skipUnless(
    HAVE_DEPS, "24_beit needs torch, numpy, torchvision, Pillow")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("beit_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to run a step on a CPU: a tiny ViT at 32px (patch16 -> a 2x2 = 4
# patch grid, embed_dim 32, 2 blocks), a random tokenizer over a 16px token view
# (stride 8 -> a 2x2 = 4 token grid, matching the patch grid), a 16-word codebook,
# 2 of 4 patches masked. The paper's ViT-Base/16 / 224px / 8192 tokens / 800
# epochs live in the shipped config.
IMG = 32
PATCH = 16
VOCAB = 16
EMBED = 32
DEPTH = 2
HEADS = 4
MLP = 2.0
DPR = 0.0
INIT = 0.1
TOKEN_SIZE = 16
GRID = IMG // PATCH          # 2 -> 4 patches
NUM_MASK = 2
MIN_MASK = 1

MODEL = {"img_size": IMG, "patch_size": PATCH, "vocab_size": VOCAB,
         "embed_dim": EMBED, "depth": DEPTH, "num_heads": HEADS,
         "mlp_ratio": MLP, "drop_path_rate": DPR, "init_values": INIT}
TOKENIZER = {"ckpt": "", "token_size": TOKEN_SIZE, "input_is_mapped": True}
STEP1_ONLY = {"num_workers": 0, "num_masking_patches": NUM_MASK,
              "min_num_patches": MIN_MASK, "epochs": 1, "batch_size": 2,
              "lr": 1.0e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8,
              "weight_decay": 0.05, "warmup_epochs": 0, "clip_grad": 3.0}
TRAIN = {**MODEL, **TOKENIZER, **STEP1_ONLY}
EVAL_TRAIN = {**MODEL, "epochs": 2, "batch_size": 2, "num_workers": 0,
              "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}


def tiny_imagefolder(root: Path, n: int = 6) -> Path:
    import numpy as np
    from PIL import Image
    cls = root / "class0"
    cls.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(n):
        Image.fromarray(rng.randint(0, 256, (48, 48, 3), dtype="uint8")).save(
            cls / f"{i}.png")
    return root


def tiny_split(root: Path, per: int = 3) -> Path:
    import numpy as np
    from PIL import Image
    rng = np.random.RandomState(0)
    for split in ("train", "val"):
        for label, cls in enumerate(("c0", "c1")):
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(per):
                base = np.full((48, 48, 3), label * 120, dtype="uint8")
                noise = rng.randint(0, 64, (48, 48, 3), dtype="uint8")
                Image.fromarray((base + noise).astype("uint8")).save(
                    d / f"{i}.png")
    return root


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="beit-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def config(self, **over) -> dict:
        cfg = {"stage": "pretrain", "seed": 0, "data_root": str(self.tmp / "data"),
               "device": "cpu", "train": dict(TRAIN)}
        for k, v in over.items():
            if k == "train" and v:
                cfg["train"] = {**cfg["train"], **v}
            elif k != "train":
                cfg[k] = v
        return cfg

    def eval_config(self, **over) -> dict:
        cfg = {"stage": "linear_eval", "seed": 0,
               "data_root": str(self.tmp / "data"), "device": "cpu",
               "encoder": str(self.tmp / "encoder.pt"),
               "train": dict(EVAL_TRAIN)}
        for k, v in over.items():
            if k == "train" and v:
                cfg["train"] = {**cfg["train"], **v}
            elif k != "train":
                cfg[k] = v
        return cfg


class TestTheModel(unittest.TestCase):
    def models(self):
        return load("beit_models", METHOD / "models" / "__init__.py")

    def _model(self, m):
        return m.build_beit(img_size=IMG, patch_size=PATCH, vocab_size=VOCAB,
                            embed_dim=EMBED, depth=DEPTH, num_heads=HEADS,
                            mlp_ratio=MLP, drop_path_rate=DPR, init_values=INIT)

    def _mask(self, torch, b=2, k=NUM_MASK):
        mask = torch.zeros(b, GRID * GRID, dtype=torch.bool)
        mask[:, :k] = True
        return mask

    @needs_deps
    def test_forward_returns_vocab_logits_at_the_masked_positions(self):
        import torch
        model = self._model(self.models()).eval()
        x = torch.randn(2, 3, IMG, IMG)
        with torch.no_grad():
            logits = model(x, self._mask(torch))
        # 2 images x 2 masked patches = 4 rows, VOCAB columns.
        self.assertEqual(tuple(logits.shape), (2 * NUM_MASK, VOCAB))
        self.assertTrue(torch.isfinite(logits).all())

    @needs_deps
    def test_the_number_of_rows_tracks_the_mask(self):
        import torch
        model = self._model(self.models()).eval()
        x = torch.randn(2, 3, IMG, IMG)
        with torch.no_grad():
            none = model(x, torch.zeros(2, GRID * GRID, dtype=torch.bool))
            allm = model(x, torch.ones(2, GRID * GRID, dtype=torch.bool))
        self.assertEqual(none.shape[0], 0)
        self.assertEqual(allm.shape[0], 2 * GRID * GRID)

    @needs_deps
    def test_the_mask_token_is_applied_at_masked_positions(self):
        # With every patch masked, the input carries only the mask token (plus
        # position), so changing the mask token changes the logits. If the mask
        # token were not applied, the logits would be independent of it.
        import torch
        model = self._model(self.models()).eval()
        x = torch.randn(2, 3, IMG, IMG)
        full = torch.ones(2, GRID * GRID, dtype=torch.bool)
        with torch.no_grad():
            before = model(x, full)
            model.mask_token.zero_().add_(5.0)
            after = model(x, full)
        self.assertFalse(torch.allclose(before, after),
                         "changing the mask token did not change the "
                         "fully-masked logits -- the mask token is not applied")

    @needs_deps
    def test_get_encoder_mean_pools_the_patch_tokens(self):
        import torch
        model = self._model(self.models())
        enc = model.get_encoder()
        feats = enc(torch.randn(2, 3, IMG, IMG))
        self.assertEqual(tuple(feats.shape), (2, EMBED))

    @needs_deps
    def test_the_encoder_carries_no_mim_head(self):
        model = self._model(self.models())
        enc = model.get_encoder()
        self.assertFalse(any(k.startswith("head.") or k == "mask_token"
                             for k in enc.state_dict()),
                         "the encoder must not carry the MIM head or mask token")


class TestTheTokenizer(unittest.TestCase):
    def models(self):
        return load("beit_models", METHOD / "models" / "__init__.py")

    @needs_deps
    def test_random_tokenizer_returns_int_token_grids_in_range(self):
        import torch
        m = self.models()
        tok = m.build_tokenizer(vocab_size=VOCAB, ckpt="", stride=8)
        out = tok(torch.rand(2, 3, TOKEN_SIZE, TOKEN_SIZE))
        self.assertEqual(tuple(out.shape), (2, (TOKEN_SIZE // 8) ** 2))
        self.assertEqual(out.dtype, torch.int64)
        self.assertGreaterEqual(int(out.min()), 0)
        self.assertLess(int(out.max()), VOCAB)

    @needs_deps
    def test_empty_ckpt_gives_the_random_tokenizer(self):
        m = self.models()
        tok = m.build_tokenizer(vocab_size=VOCAB, ckpt="")
        self.assertIsInstance(tok, m.RandomDVAETokenizer)

    @needs_deps
    def test_random_tokenizer_is_deterministic_under_a_seed(self):
        import torch
        m = self.models()
        x = torch.rand(2, 3, TOKEN_SIZE, TOKEN_SIZE)
        torch.manual_seed(0)
        a = m.build_tokenizer(vocab_size=VOCAB, ckpt="", stride=8)(x)
        torch.manual_seed(0)
        b = m.build_tokenizer(vocab_size=VOCAB, ckpt="", stride=8)(x)
        self.assertTrue(torch.equal(a, b))

    @needs_deps
    def test_the_dalle_path_points_at_the_pinned_submodule(self):
        tok = load("beit_dvae", METHOD / "models" / "dvae_tokenizer.py")
        self.assertEqual(tok._DALLE_SUBMODULE.name, "dall_e")
        self.assertEqual(tok._DALLE_SUBMODULE.parent.name, "third_party")

    @needs_deps
    def test_a_real_ckpt_needs_dall_e_and_says_so(self):
        # With the submodule unreachable and dall_e not importable, a real
        # checkpoint raises a helpful ImportError rather than a bare one. The
        # submodule path is pointed at a bogus location so the guard fires
        # deterministically, whether or not dall_e's own deps happen to be around.
        import sys as _sys
        from pathlib import Path as _Path
        from unittest import mock
        tok = load("beit_dvae2", METHOD / "models" / "dvae_tokenizer.py")
        purged = {k: _sys.modules.pop(k) for k in list(_sys.modules)
                  if k == "dall_e" or k.startswith("dall_e.")}
        try:
            with mock.patch.object(tok, "_DALLE_SUBMODULE",
                                   _Path("/no/such/dall_e_submodule")):
                with self.assertRaises(ImportError) as e:
                    tok.build_tokenizer(vocab_size=VOCAB,
                                        ckpt="/no/such/encoder.pkl")
            self.assertIn("DALL-E", str(e.exception))
        finally:
            _sys.modules.update(purged)


class TestTheMaskGenerator(unittest.TestCase):
    def dataset_mod(self):
        return load("beit_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_the_flat_mask_has_exactly_the_requested_count(self):
        import numpy as np
        np.random.seed(0)
        import random
        random.seed(0)
        gen = self.dataset_mod().BEiTMaskingGenerator(
            input_size=(GRID, GRID), num_masking_patches=NUM_MASK,
            min_num_patches=MIN_MASK)
        mask = gen()
        self.assertEqual(mask.shape, (GRID * GRID,))
        self.assertTrue(set(np.unique(mask)).issubset({False, True}))
        self.assertEqual(int(mask.sum()), NUM_MASK)


class TestTheDataset(Base):
    def dataset_mod(self):
        return load("beit_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_an_item_is_patch_token_mask_and_label(self):
        tiny_imagefolder(self.tmp / "data" / "train")
        ds = self.dataset_mod().BEiTPretrainDataset(
            str(self.tmp / "data"), img_size=IMG, patch_size=PATCH,
            token_size=TOKEN_SIZE, num_masking_patches=NUM_MASK,
            min_masking_patches=MIN_MASK)
        patch, token, mask, label = ds[0]
        self.assertEqual(tuple(patch.shape), (3, IMG, IMG))
        self.assertEqual(tuple(token.shape), (3, TOKEN_SIZE, TOKEN_SIZE))
        self.assertEqual(tuple(mask.shape), (GRID * GRID,))
        self.assertEqual(int(mask.sum()), NUM_MASK)


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_backbone_trunk_comes_out(self):
        got = adapter.extract_encoder({
            "patch_embed.proj.weight": 1,
            "blocks.0.norm1.weight": 2,
            "cls_token": 3, "pos_embed": 4, "norm.weight": 5,
            "mask_token": 6, "head.weight": 7, "head.bias": 8})
        self.assertEqual(set(got), {"patch_embed.proj.weight",
                                    "blocks.0.norm1.weight", "cls_token",
                                    "pos_embed", "norm.weight"})

    def test_the_mask_token_and_head_are_left_out(self):
        got = adapter.extract_encoder({"norm.weight": 1, "mask_token": 2,
                                       "head.weight": 3})
        self.assertEqual(set(got), {"norm.weight"})

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"head.weight": 1, "mask_token": 2})
        self.assertIn("empty", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["model"]["embed_dim"], EMBED)
        self.assertEqual(built["tokenizer"]["token_size"], TOKEN_SIZE)
        self.assertEqual(built["masking"]["num_masking_patches"], NUM_MASK)
        self.assertEqual(built["training"]["epochs"], 1)

    def test_a_missing_step1_setting_is_refused_by_name(self):
        for key in TRAIN:
            with self.subTest(key=key):
                cfg = self.config()
                cfg["train"] = {k: v for k, v in TRAIN.items() if k != key}
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(cfg, out=self.out)
                self.assertIn(key, str(e.exception))

    def test_an_unknown_step1_setting_is_refused(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(train={"nonsense": 1}),
                                  out=self.out)
        self.assertIn("nonsense", str(e.exception))

    def test_an_unknown_stage_is_refused(self):
        with self.assertRaises(adapter.ConfigError):
            adapter.to_run_config(self.config(stage="step2"), out=self.out)

    def test_output_is_refused(self):
        cfg = self.config()
        cfg["output"] = {"save_dir": "/anywhere"}
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("--out", str(e.exception))


class TestLinearEvalConfig(Base):
    def test_linear_eval_is_accepted(self):
        adapter.to_run_config(self.eval_config(), out=self.out)

    def test_the_encoder_must_be_named(self):
        cfg = self.eval_config()
        del cfg["encoder"]
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("encoder", str(e.exception))

    def test_step1_only_settings_are_not_part_of_the_probe(self):
        cfg = self.eval_config(train={"num_masking_patches": NUM_MASK})
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("num_masking_patches", str(e.exception))


class TestTheEvalProducesNoEncoder(Base):
    def _reason(self, cfg):
        p = self.tmp / "resolved.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        return adapter._absent_reason(p)

    def test_linear_eval_declares_no_encoder(self):
        self.assertTrue(self._reason(self.eval_config()))

    def test_step1_gives_no_reason(self):
        self.assertIsNone(self._reason(self.config()))


class TestTheMetricsAreInTheVocabulary(unittest.TestCase):
    def test_step1_maps_a_pretext_loss(self):
        self.assertEqual(adapter.STEP1_METRIC_NAMES["final_loss"],
                         "final_pretext_loss")
        for target in adapter.STEP1_METRIC_NAMES.values():
            if target is not None:
                self.assertIn(target, adapterlib.METRIC_VOCABULARY)

    def test_eval_maps_the_comparable_probe_numbers(self):
        mapped = set(adapter.LINEAR_EVAL_METRIC_NAMES.values())
        for name in ("best_linear_probe_top1_accuracy",
                     "final_linear_probe_top1_accuracy",
                     "best_linear_probe_top5_accuracy",
                     "final_linear_probe_top5_accuracy"):
            self.assertIn(name, mapped)
            self.assertEqual(adapterlib.METRIC_VOCABULARY[name],
                             adapterlib.COMPARABLE)


class TestTheDeviceIsResolved(Base):
    def trainer(self):
        return load("beit_trainer", METHOD / "train_step1_beit.py")

    @needs_deps
    def test_asking_for_cuda_without_one_is_refused(self):
        from unittest import mock
        t = self.trainer()
        with mock.patch.object(t.torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                t.resolve_device("cuda", 0)
            self.assertEqual(t.resolve_device("cpu", 0).type, "cpu")
            self.assertEqual(t.resolve_device("auto", 0).type, "cpu")

    def test_run_resolves_the_device(self):
        import ast
        src = (METHOD / "train_step1_beit.py").read_text()
        run_fn = next(n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == "run")
        called = {n.func.id for n in ast.walk(run_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("resolve_device", called)
        self.assertIn("make_deterministic", called)


class TestAStep1Smoke(Base):
    def run_adapter(self, **over):
        tiny_imagefolder(self.tmp / "data" / "train")
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
    def test_it_writes_an_encoder_and_a_pretext_loss(self):
        self.run_adapter()
        self.assertTrue((self.out / "encoder.pt").is_file())
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        self.assertIn("final_pretext_loss", m)

    @needs_deps
    def test_the_manifest_records_the_pinned_upstream(self):
        """A real run imports the pinned DALL-E tokenizer submodule, so the
        manifest must name it (CONTRACT: upstream is recorded for every
        submodule-using method). The value is fixed, so the smoke -- which uses
        a random tokenizer -- still records it."""
        self.run_adapter()
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["upstream"], adapter.UPSTREAM)

    @needs_deps
    def test_the_encoder_pt_it_wrote_loads_back(self):
        self.run_adapter()
        import torch
        saved = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(saved)
        load("this_methods_models", METHOD / "models" / "__init__.py")
        model = adapter.load_encoder(saved, self.eval_config())
        loaded = model.state_dict()
        pairs = 0
        for key, want in saved.items():
            got = loaded.get(key)
            if got is None:
                continue
            pairs += 1
            self.assertTrue(torch.equal(got, want), f"{key} came back changed")
        self.assertGreater(pairs, 0, "no saved weight reached the model")

    @needs_deps
    def test_the_same_config_twice_gives_the_same_encoder(self):
        digests = []
        for name in ("a", "b"):
            self.out = self.tmp / name
            self.run_adapter()
            digests.append(hashlib.sha256(
                (self.out / "encoder.pt").read_bytes()).hexdigest())
        self.assertEqual(digests[0], digests[1])

    @unittest.skipUnless(HAVE_DEPS and torch.cuda.is_available(),
                         "no CUDA device; the GPU path cannot be exercised here")
    def test_a_real_run_on_cuda_produces_a_loadable_encoder(self):
        cfg, r = self.run_adapter(device="cuda")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn("cuda", r.stdout.lower())


class TestALinearEvalSmoke(Base):
    def _step1(self):
        tiny_split(self.tmp / "data")
        s1data = self.tmp / "s1data"
        tiny_imagefolder(s1data / "train")
        s1cfg = {"stage": "pretrain", "seed": 0, "data_root": str(s1data),
                 "device": "cpu", "train": dict(TRAIN)}
        p = self.tmp / "s1.json"
        p.write_text(json.dumps(s1cfg), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        s1out = self.tmp / "s1out"
        r = subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(p),
             "--out", str(s1out)],
            cwd=METHOD, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        return s1out

    def run_eval(self, **over):
        s1out = self._step1()
        cfg = self.eval_config(encoder=str(s1out / "encoder.pt"), **over)
        p = self.tmp / "resolved.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(p),
             "--out", str(self.out)],
            cwd=METHOD, env=env, capture_output=True, text=True)
        return p, r

    @needs_deps
    def test_it_completes_and_satisfies_the_contract(self):
        cfg, r = self.run_eval()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    @needs_deps
    def test_it_reports_the_comparable_probe_numbers(self):
        self.run_eval()
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        for name in ("best_linear_probe_top1_accuracy",
                     "final_linear_probe_top1_accuracy"):
            self.assertIn(name, m)

    @needs_deps
    def test_it_produces_no_encoder_and_says_so(self):
        self.run_eval()
        self.assertFalse((self.out / "encoder.pt").exists())
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["stage"], "linear_eval")
        self.assertEqual(man["status"], "ok", man.get("error", ""))
        self.assertIn("encoder_absent_reason", man)


class TestTheOriginalIsReferencedNotCopied(unittest.TestCase):
    def test_no_distributed_or_tensorboard_machinery_is_used(self):
        import ast
        tree = ast.parse((METHOD / "train_step1_beit.py").read_text())
        used = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                used.add(n.attr)
            elif isinstance(n, ast.Name):
                used.add(n.id)
        self.assertNotIn("DistributedDataParallel", used)
        self.assertNotIn("SummaryWriter", used)
        self.assertNotIn("autocast", used)


if __name__ == "__main__":
    unittest.main()
