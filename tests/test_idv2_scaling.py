"""Scaling input membership, optimizer-clock updates and continuation."""
import copy
import gzip
import hashlib
import io
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
import test_method_10_inst_disc as base


def prepared(root):
    from PIL import Image
    root.mkdir()
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), (23, 45, 67)).save(stream, format="PNG")
    payload = stream.getvalue()
    compressed = gzip.compress(payload)
    (root / "samples.tar").write_bytes(payload + compressed)
    rows = [{"sample_id": str(i), "synset": "n0001", "member_name": name,
                 "tar_relpath": "samples.tar", "offset": offset, "size": size}
            for i, (name, offset, size) in enumerate([
                ("keep.PNG", 0, len(payload)), ("other.PNG", len(payload), len(compressed)),
                ("ILSVRC2012_val_00001.JPEG", 0, len(payload)),
                ("collision.PNG", 0, len(payload))])]
    (root / "MANIFEST.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "SUCCESS.json").write_text('{"status":"ok"}')
    return rows


def tiny_model(arch):
    import torch as t
    from idv2_scaling import ScalingModel
    from timm.models.vision_transformer import VisionTransformer
    with mock.patch("timm.create_model", side_effect=lambda *a, **kw: VisionTransformer(
            img_size=224, patch_size=16, embed_dim=8, depth=1, num_heads=2, num_classes=0,
            global_pool="", dynamic_img_size=kw["dynamic_img_size"],
            norm_layer=lambda dim, **kw: t.nn.LayerNorm(dim, eps=1e-6, **kw))):
        return ScalingModel(arch)


def tiny_config(root):
    return {"device": "cpu", "seed": 0, "model": "vit_base_patch16_224",
            "precision": "float32", "prepared_root": str(root / "source"),
            "validation_root": str(root / "val"), "epochs": 2, "microbatch": 1,
            "accum": 2, "num_workers": 0, "resume": None}


@base.needs_deps
class InputTests(unittest.TestCase):
    def setUp(self):
        self.mod = base.load("indexed_tar", base.METHOD / "data/indexed_tar.py")

    def test_membership_exclusion_order_gzip_and_no_source_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "prepared"
            rows = prepared(source)
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            val = root / "val"; val.mkdir(); (val / "collision.JPEG").touch()
            out = root / "frozen"
            meta = self.mod.freeze(source, val, out)
            self.assertEqual((meta["n_train"], meta["n_excluded"]), (2, 2))
            manifest = out / "TRAIN_MANIFEST.jsonl"
            kept = [json.loads(s) for s in manifest.read_text().splitlines()]
            self.assertEqual(kept, [dict(row, instance_id=i) for i, row in enumerate(rows[:2])])
            self.assertEqual(meta["manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
            dataset = self.mod.IndexedTarDataset(manifest, source, lambda im: im.getpixel((0, 0)))
            self.assertEqual(dataset[0], ((23, 45, 67), 0, 0))
            self.assertEqual(dataset[1], ((23, 45, 67), 1, 0))
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
            with self.assertRaises(FileExistsError):
                self.mod.freeze(source, val, out)

    def test_missing_val_and_empty_population_are_not_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared(root / "source")
            with self.assertRaises(FileNotFoundError):
                self.mod.freeze(root / "source", root / "missing", root / "out")
            val = root / "val"; val.mkdir()
            for name in ("keep.PNG", "other.PNG", "collision.PNG"):
                (val / name).touch()
            with self.assertRaisesRegex(ValueError, "empty"):
                self.mod.freeze(root / "source", val, root / "empty")

    def test_invalid_rows_escape_and_corrupt_images_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rows = prepared(root / "source")
            manifest = root / "manifest.jsonl"
            for update in ({"instance_id": 4}, {"offset": -1}, {"size": 0},
                           {"size": True}, {"offset": .5}, {"sample_id": ""}, {"synset": None},
                           {"tar_relpath": "../samples.tar"}, {"tar_relpath": "/tmp/samples.tar"}):
                manifest.write_text(json.dumps({**rows[0], "instance_id": 0, **update}) + "\n")
                with self.subTest(update=update), self.assertRaises(ValueError):
                    self.mod.IndexedTarDataset(manifest, root / "source", lambda im: im)
            manifest.write_text(json.dumps(dict(rows[0], instance_id=0, offset=99999)) + "\n")
            data = self.mod.IndexedTarDataset(manifest, root / "source", lambda im: im)
            with self.assertRaisesRegex(ValueError, "sample"):
                data[0]
            outside = root / "outside.tar"; outside.write_bytes(b"not an image")
            (root / "source/escape.tar").symlink_to(outside)
            manifest.write_text(json.dumps(dict(rows[0], tar_relpath="escape.tar", instance_id=0)) + "\n")
            data = self.mod.IndexedTarDataset(manifest, root / "source", lambda im: im)
            with self.assertRaisesRegex(ValueError, "sample"):
                data[0]
            manifest.write_text("")
            with self.assertRaisesRegex(ValueError, "empty"):
                self.mod.IndexedTarDataset(manifest, root / "source", lambda im: im)


@base.needs_deps
@base.needs_timm
class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.mod = base.load("idv2_scaling", base.METHOD / "idv2_scaling.py")
        self.parts = base.load("idv2_components", base.METHOD / "idv2_components.py")

    def test_optimizer_clock_has_ten_thousand_warmup_updates(self):
        for step, expected in ((0, 6e-8), (5000, .00030006), (9999, .0006), (10000, .0006),
                               (50000, .0003), (90000, 0.0), (100000, 0.0)):
            self.assertAlmostEqual(self.mod.schedule(step, 90000), expected)

    def test_reference_templates_and_real_preset_dimensions(self):
        import torch as t
        for key, arch, width, depth, heads in (("vitb", "vit_base_patch16_224", 768, 12, 12),
                                              ("vitl", "vit_large_patch16_224", 1024, 24, 16)):
            cfg = json.loads((base.METHOD / f"configs/idv2_scaling_{key}.json").read_text())
            self.mod.validate(cfg)
            self.assertEqual((cfg["model"], cfg["microbatch"], cfg["accum"], cfg["epochs"], cfg["seed"], cfg["precision"]),
                             (arch, 8, 4, 90, 0, "bfloat16"))
            with t.device("meta"):
                model = self.mod.build_model(arch)
            self.assertEqual(model.encoder.embed_dim, width)
            self.assertEqual(len(model.encoder.blocks), depth)
            self.assertEqual(model.encoder.blocks[0].attn.num_heads, heads)
            self.assertEqual(model.encoder.blocks[0].norm1.eps, 1e-6)
            self.assertEqual(model.fc.out_features, 128)
            self.assertTrue(model.encoder.grad_checkpointing)
        model = tiny_model("vit_large_patch16_224").eval()
        for size in (224, 96):
            image = t.randn(2, 3, size, size)
            expected = t.nn.functional.normalize(model.fc(model.encoder.forward_features(image)[:, 0]), dim=1)
            actual = model(image)
            t.testing.assert_close(actual, expected)
            t.testing.assert_close(actual.norm(dim=1), t.ones(2))
        with mock.patch("timm.create_model", return_value=t.nn.Linear(2, 3)), self.assertRaises(TypeError):
            self.mod.build_model("vit_base_patch16_224")

    def test_invalid_accumulated_inputs_are_refused(self):
        import torch as t
        model = t.nn.Linear(3, 4)
        optimizer = t.optim.AdamW(model.parameters())
        bank = self.mod.ScalingBank(3, 4, .2, .5, 2)
        views = [t.ones(2, 3) for _ in range(6)]
        batch = (views, t.tensor([0, 1]), None)
        for batches, precision in (([], "float32"), ([batch], "auto"), ([batch], "bfloat16"),
                                   ([(views[:5], batch[1], None)], "float32")):
            with self.subTest(precision=precision), self.assertRaises(ValueError):
                self.mod.train_step(model, bank, optimizer, batches, precision)
        self.assertFalse(optimizer.state)

    def test_setup_preserves_fixed_recipe_and_rejects_zero_updates(self):
        import torch as t
        from data.indexed_tar import freeze

        from downstream.extended_distributed import session
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared(root / "source"); (root / "val").mkdir()
            out = root / "out"; out.mkdir()
            freeze(root / "source", root / "val", out / "membership")
            cfg = dict(tiny_config(root), seed=17)
            with session("cpu") as context, mock.patch.object(self.mod, "build_model", side_effect=tiny_model):
                model, bank, optimizer, loader, _, generator, steps = self.mod._setup(cfg, out, context)
                self.assertEqual((bank.temperature, bank.momentum, bank.num_negatives, bank.feature_dim), (.07, .5, 4096, 128))
                self.assertEqual((optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["weight_decay"]), (.0006, .05))
                self.assertEqual((steps, len(loader), generator.initial_seed()), (1, 3, 17))
                t.manual_seed(17); expected = tiny_model(cfg["model"])
                for p, q in zip(model.parameters(), expected.parameters()):
                    t.testing.assert_close(p, q, rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "population"):
                    self.mod._setup(dict(cfg, accum=4), out, context)

    def test_incompatible_resume_and_epoch30_export(self):
        import torch as t
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared(root / "source"); (root / "val").mkdir()
            cfg = tiny_config(root); out = root / "partial"; out.mkdir()
            with mock.patch.object(self.mod, "build_model", side_effect=tiny_model):
                self.mod.run(dict(cfg, epochs=1), out)
                original = t.load(out / "checkpoint_latest.pth", weights_only=False)
                for i, update in enumerate(({"scaling_version": 2}, {"contract": {}}, {"epoch": 2, "global_step": 3},
                        {"epoch": -1, "global_step": 0},
                        {"global_step": 99}, {"rank_rng": []}, {"optimizer": {}})):
                    path = root / f"bad{i}.pth"; t.save(dict(original, **update), path)
                    dest = root / f"bad{i}"; dest.mkdir()
                    with self.subTest(update=update), self.assertRaisesRegex(ValueError, "incompatible"):
                        self.mod.run(dict(cfg, resume=str(path)), dest)
                dest = root / "extended"; dest.mkdir()
                self.mod.run(dict(cfg, epochs=30, resume=str(out / "checkpoint_latest.pth")), dest)
                state = t.load(dest / "checkpoint_epoch_30.pth", weights_only=False)
                self.assertEqual((state["epoch"], state["global_step"]), (29, 30))
                self.assertTrue((dest / "encoder_epoch30.pt").is_file())
                self.assertEqual(json.loads((dest / "results.json").read_text())["dropped_microbatches_per_rank"], 1)

    def test_cli_delivery_and_existing_output_protection(self):
        from downstream.contract import verify
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared(root / "source"); (root / "val").mkdir()
            cfg = dict(tiny_config(root), epochs=1)
            path = root / "config.json"; path.write_text(json.dumps(cfg))
            out = root / "out"
            with mock.patch.object(self.mod, "build_model", side_effect=tiny_model):
                self.assertEqual(self.mod.main(["--config", str(path), "--out", str(out)]), 0)
            self.assertEqual(verify(out, path, 0), (True, []))
            manifest = json.loads((out / "run_manifest.json").read_text())
            self.assertEqual(manifest["method_ref"], "idv2_scaling")
            before = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}
            self.assertEqual(self.mod.main(["--config", str(path), "--out", str(out)]), 1)
            self.assertEqual(before, {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()})
            unsafe = root / "source/new-output"
            with self.assertRaisesRegex(ValueError, "overlap"), mock.patch.object(self.mod, "build_model", side_effect=tiny_model):
                self.mod.main(["--config", str(path), "--out", str(unsafe)])
            self.assertFalse(unsafe.exists())

    def test_accumulated_update_matches_independent_six_view_objective(self):
        import torch as t
        import torch.nn.functional as F
        t.manual_seed(9)
        model = t.nn.Sequential(t.nn.Linear(3, 4), self.parts.Normalize())
        bank = self.parts.make_bank(7, 4, .2, .5, 2, "multicrop")
        expected = copy.deepcopy(model); eb = copy.deepcopy(bank)
        optimizer = t.optim.AdamW(model.parameters(), lr=.001)
        eo = t.optim.AdamW(expected.parameters(), lr=.001)
        batches = [([t.randn(2, 3) for _ in range(6)], t.tensor(ids), None)
                   for ids in ((0, 1), (1, 2))]
        rng = t.get_rng_state(); losses = []; observations = []; ids = []
        eo.zero_grad()
        for views, idx, _ in batches:
            neg = t.randint(0, 7, (2, 2)); rows = eb.memory[t.cat((idx[:, None], neg), 1)].detach()
            embeddings = list(expected(t.cat(views[:2])).chunk(2)) + list(expected(t.cat(views[2:])).chunk(4))
            terms = []
            for z in embeddings:
                exp = t.exp(t.bmm(rows, z.unsqueeze(2)).squeeze(2) / .2)
                new_z = exp.detach().float().mean() * 7
                eb.Z.copy_(new_z if eb.Z.item() < 0 else .5 * eb.Z + .5 * new_z)
                c = 2 * eb.Z.item() / 7
                terms.append((-t.log(exp[:, 0] / (exp[:, 0] + c) + 1e-7)
                              -t.log(c / (exp[:, 1:] + c) + 1e-7).sum(1)).mean())
            loss = t.stack(terms).mean(); (loss / 2).backward(); losses.append(loss.detach())
            observations.append(F.normalize(.5 * (embeddings[0].detach() + embeddings[1].detach()), dim=1)); ids.append(idx)
        eo.step(); eb.update_memory(t.cat(observations), t.cat(ids))
        t.set_rng_state(rng)
        actual = self.mod.train_step(model, bank, optimizer, batches, "float32")
        self.assertAlmostEqual(actual, t.stack(losses).mean().item(), places=6)
        for p, q in zip(model.parameters(), expected.parameters()):
            t.testing.assert_close(p, q, rtol=0, atol=0)
            t.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        t.testing.assert_close(bank.memory, eb.memory, rtol=0, atol=0)
        t.testing.assert_close(bank.Z, eb.Z)

    def test_nonfinite_does_not_publish_an_optimizer_update(self):
        import torch as t
        model = t.nn.Sequential(t.nn.Linear(3, 4), self.parts.Normalize())
        bank = self.parts.make_bank(3, 4, .2, .5, 2, "multicrop")
        optimizer = t.optim.AdamW(model.parameters())
        before = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.mod.train_step(model, bank, optimizer,
                [([t.full((2, 3), float("nan")) for _ in range(6)], t.tensor([0, 1]), None)], "float32")
        self.assertFalse(optimizer.state)
        for key, value in before.items():
            t.testing.assert_close(value, model.state_dict()[key])

    def test_partition_caps_new_observation_before_ema_and_recovers_invalid_state(self):
        import torch as t
        bank = self.mod.ScalingBank(3, 4, 1., .5, 2)
        bank.Z.fill_(2)
        bank.update_partition(t.full((2, 3), float("inf")))
        self.assertAlmostEqual(bank.Z.item(), 1 + .5 * 3 * math.e, places=5)
        bank.Z.fill_(float("nan"))
        bank.update_partition(t.ones(2, 3))
        self.assertEqual(bank.Z.item(), 3)

    def test_real_training_exact_epoch_resume_and_delivery(self):
        import torch as t
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared(root / "source")
            val = root / "val"; val.mkdir(); (val / "collision.PNG").touch()
            cfg = tiny_config(root)
            with mock.patch.object(self.mod, "build_model", side_effect=tiny_model), mock.patch.object(self.mod, "schedule", wraps=self.mod.schedule) as clock:
                for name, epochs, resume in (("full", 2, None), ("partial", 1, None),
                                             ("resumed", 2, str(root / "partial/checkpoint_latest.pth"))):
                    out = root / name; out.mkdir()
                    self.mod.run(dict(cfg, epochs=epochs, resume=resume), out)
                self.assertEqual([call.args for call in clock.call_args_list], [(0, 90), (1, 90), (0, 90), (1, 90)])
                full = t.load(root / "full/checkpoint_latest.pth", weights_only=False)
                resumed = t.load(root / "resumed/checkpoint_latest.pth", weights_only=False)
                for field in ("model", "bank"):
                    for key, value in full[field].items():
                        t.testing.assert_close(value, resumed[field][key], rtol=0, atol=0)
                self.assertEqual(full["global_step"], 2)
                self.assertEqual(full["epoch"], 1)
                self.assertEqual(full["contract"]["schedule_epochs"], 90)
                export = t.load(root / "full/encoder.pt", weights_only=True)
                self.assertEqual(export["model"], cfg["model"])
                self.assertNotIn("fc.weight", export["state_dict"])
                with self.assertRaisesRegex(RuntimeError, "keys"):
                    base.adapter.load_encoder(export, {"train": base.VIT_TRAIN_TINY})
                encoder = self.mod.load_encoder(root / "full/encoder.pt").eval()
                expected = tiny_model(cfg["model"]).eval()
                expected.load_state_dict(full["model"])
                images = t.randn(2, 3, 224, 224)
                t.testing.assert_close(encoder(images), expected.encoder.forward_features(images)[:, 0], rtol=0, atol=0)
                for update in ({"model": "unknown"}, {"scaling_encoder_version": 0}, {"readout": "mean"}):
                    invalid = root / "invalid-encoder.pt"; t.save(dict(export, **update), invalid)
                    with self.assertRaisesRegex(ValueError, "encoder"):
                        self.mod.load_encoder(invalid)
                t.save(dict(export, state_dict={}), invalid)
                with self.assertRaises(RuntimeError):
                    self.mod.load_encoder(invalid)
                result = json.loads((root / "full/results.json").read_text())
                self.assertFalse(result["canonical_eligible"])
                self.assertEqual(result["global_batch_size"], 2)
                self.assertEqual(result["optimizer_steps_per_epoch"], 1)
                changed = root / "changed"; changed.mkdir()
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    self.mod.run(dict(cfg, seed=1, resume=str(root / "partial/checkpoint_latest.pth")), changed)

    def test_configuration_errors_and_checkpoint_health(self):
        import torch as t
        cfg = tiny_config(Path("unused"))
        for key, value in (("epochs", 91), ("epochs", 0), ("seed", True), ("accum", 0),
                           ("microbatch", 0), ("num_workers", -1), ("model", "unknown"),
                           ("precision", "bfloat16"), ("precision", "auto"), ("device", "auto"), ("extra", 1)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.mod.validate(dict(cfg, **{key: value}))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared(root / "source"); (root / "val").mkdir()
            cfg = tiny_config(root); cfg["epochs"] = 1
            out = root / "out"; out.mkdir()
            step = t.optim.AdamW.step
            def corrupt(optimizer, *args, **kwargs):
                result = step(optimizer, *args, **kwargs)
                with t.no_grad():
                    optimizer.param_groups[0]["params"][0].fill_(float("nan"))
                return result
            with mock.patch.object(self.mod, "build_model", side_effect=tiny_model), mock.patch.object(t.optim.AdamW, "step", corrupt), self.assertRaisesRegex(ValueError, "nonfinite"):
                self.mod.run(cfg, out)
            self.assertFalse((out / "checkpoint_latest.pth").exists())


def distributed_worker(root, in1k=False):
    from datetime import timedelta

    import torch as t

    from downstream.extended_distributed import session
    mod = base.load("idv2_scaling", base.METHOD / "idv2_scaling.py")
    root = Path(root); rank = int(os.environ["RANK"])
    t.distributed.init_process_group("gloo", timeout=timedelta(seconds=20))
    cfg = tiny_config(root)
    if in1k:
        cfg.update(profile="in1k300_vitl", model="vit_large_patch16_224")
    with session("cpu") as context, mock.patch.object(mod, "build_model", side_effect=tiny_model):
        # Compare DDP's accumulated gradient to explicit per-parameter averaging.
        from idv2_components import Normalize
        class MeanAdamW(t.optim.AdamW):
            def step(self, closure=None):
                for group in self.param_groups:
                    for parameter in group["params"]:
                        t.distributed.all_reduce(parameter.grad)
                        parameter.grad.div_(2)
                return super().step(closure)
        t.manual_seed(21)
        raw = t.nn.Sequential(t.nn.Linear(3, 4), Normalize())
        expected = copy.deepcopy(raw)
        model = t.nn.parallel.DistributedDataParallel(raw)
        calls = []
        def reduce(state, bucket):
            calls.append(1)
            value = bucket.buffer()
            t.distributed.all_reduce(value); value.div_(2)
            future = t.futures.Future(); future.set_result(value)
            return future
        model.register_comm_hook(None, reduce)
        bank = mod.ScalingBank(7, 4, .2, .5, 2); bank.distributed = True
        bank.synchronize_state(); oracle = copy.deepcopy(bank)
        optimizer = t.optim.AdamW(model.parameters(), lr=.001)
        eo = MeanAdamW(expected.parameters(), lr=.001)
        for step in range(3):
            t.manual_seed(20 + rank + step * 2)
            batches = [([t.randn(2, 3) for _ in range(6)], t.tensor([rank, rank + 1]), None) for _ in range(4)]
            rng = t.get_rng_state(); calls.clear()
            actual = mod.train_step(model, bank, optimizer, batches, "float32")
            assert len(calls) == 1, calls
            t.set_rng_state(rng)
            reference = mod.train_step(expected, oracle, eo, batches, "float32")
            assert abs(actual - reference) < 1e-6
            for p, q in zip(raw.parameters(), expected.parameters()):
                t.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)
                t.testing.assert_close(p.grad, q.grad, rtol=1e-6, atol=1e-7)
            t.testing.assert_close(bank.memory, oracle.memory)
            t.testing.assert_close(bank.Z, oracle.Z)
        for name, epochs, resume in (("full", 2, None), ("partial", 1, None), ("resumed", 2, str(root / "partial/checkpoint_latest.pth"))):
            out = root / name
            context.call(lambda out=out: out.mkdir(), leader=True)
            mod.run(dict(cfg, epochs=epochs, resume=resume), out)
        full = t.load(root / "full/checkpoint_latest.pth", weights_only=False)
        resumed = t.load(root / "resumed/checkpoint_latest.pth", weights_only=False)
        for field in ("model", "bank"):
            for key, value in full[field].items():
                t.testing.assert_close(value, resumed[field][key], rtol=0, atol=0)
        assert len(full["rank_rng"]) == 2
        # Source read errors must reach every rank before gradient collectives.
        out = root / "read-failure"; context.call(lambda out=out: out.mkdir(), leader=True)
        import data.indexed_tar as indexed
        from data.imagefolder_scaling import FrozenImageFolder
        reader = FrozenImageFolder if in1k else indexed.IndexedTarDataset
        original = reader.__getitem__
        def fail(self, index):
            if rank == 1:
                raise ValueError("rank-local unreadable sample")
            return original(self, index)
        with mock.patch.object(reader, "__getitem__", fail):
            try:
                mod.run(cfg, out)
            except RuntimeError as exc:
                assert "rank-local unreadable sample" in str(exc)
            else:
                raise AssertionError("rank read failure was ignored")
        assert not (out / "checkpoint_latest.pth").exists()
    t.distributed.destroy_process_group()


@base.needs_deps
@base.needs_timm
class DistributedTests(unittest.TestCase):
    def test_two_rank_continuation_and_failure_propagation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rows = prepared(root / "source"); (root / "val").mkdir()
            # Five kept images: sampler/tail dropping must not pad the population.
            rows = [dict(rows[0], sample_id=str(i), member_name=f"keep{i}.PNG") for i in range(5)]
            (root / "source/MANIFEST.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
            env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=str(base.ROOT))
            proc = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--master-addr=127.0.0.1",
                f"--master-port={port}", "--nproc-per-node=2", str(Path(__file__).resolve()), str(root)],
                env=env, capture_output=True, text=True, timeout=100, check=False)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            result = json.loads((root / "full/results.json").read_text())
            self.assertEqual(result["global_batch_size"], 4)
            self.assertEqual(result["optimizer_steps_per_epoch"], 1)
            self.assertEqual(result["dropped_microbatches_per_rank"], 0)


if __name__ == "__main__":
    distributed_worker(sys.argv[1], in1k=len(sys.argv) > 2)
