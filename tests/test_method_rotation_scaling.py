"""Source-supported rotation scaling updates and portable continuation."""

import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
    from timm.models.vision_transformer import VisionTransformer
    from torch import nn

    HAVE = True
except ImportError:
    HAVE = False

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "methods/06_rotation_prediction/rotation_scaling.py"
    assert path.is_file(), "native rotation scaling training entry is missing"
    spec = importlib.util.spec_from_file_location("rotation_scaling", path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@unittest.skipUnless(HAVE, "torch and timm required")
class TestScaling(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.m = module()
        self.patch = mock.patch.dict(
            self.m.PRESETS, {"vit_base_patch16_224": (16, 1, 2)}
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_native_initialization_and_forward(self):
        torch.manual_seed(42)
        actual = self.m.build_model("vit_base_patch16_224")
        torch.manual_seed(42)
        expected = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=16,
            depth=1,
            num_heads=2,
            num_classes=4,
            mlp_ratio=4.0,
            drop_rate=0.1,
            attn_drop_rate=0.0,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
        )
        self.assertEqual(set(actual.state_dict()), set(expected.state_dict()))
        for k, v in actual.state_dict().items():
            torch.testing.assert_close(v, expected.state_dict()[k], rtol=0, atol=0)
        x = torch.rand(2, 3, 224, 224)
        torch.manual_seed(23)
        a = actual(x)
        torch.manual_seed(23)
        b = expected(x)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_rotation_order_normalization_and_transform(self):
        x = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4) / 48
        images, labels = self.m.rotations(x)
        self.assertEqual(labels.tolist(), [0, 1, 2, 3])
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        for k in range(4):
            torch.testing.assert_close(
                images[k], (torch.rot90(x, k, (1, 2)) - mean) / std
            )
        transform = self.m.train_transform()
        self.assertEqual(
            [type(t).__name__ for t in transform.transforms],
            ["Resize", "RandomCrop", "RandomHorizontalFlip", "ToTensor"],
        )
        self.assertEqual(transform.transforms[0].size, 256)
        self.assertEqual(transform.transforms[1].size, (224, 224))

    def test_schedule_updates_clipping_and_tail(self):
        for step in (0, 1, 9999, 10000, 14000, 18000, 99999):
            expected = (
                max(0.0006 * (step + 1) / 10000, 1e-8)
                if step < 10000
                else 0.000001
                + 0.5
                * (0.0006 - 0.000001)
                * (1 + math.cos(math.pi * min(max((step - 10000) / 8000, 0), 1)))
            )
            self.assertAlmostEqual(self.m.schedule(step, 18000), expected, places=14)
        a = nn.Sequential(nn.Flatten(), nn.Linear(12, 4))
        b = copy.deepcopy(a)
        oa = torch.optim.AdamW(a.parameters(), lr=0.0006, weight_decay=0.05)
        ob = torch.optim.AdamW(b.parameters(), lr=0.0006, weight_decay=0.05)
        batches = [(torch.randn(4, 3, 2, 2) * 20, torch.arange(4)) for _ in range(2)]
        for _ in range(2):
            ob.zero_grad(set_to_none=True)
            losses = []
            for x, y in batches:
                loss = nn.functional.cross_entropy(b(x), y)
                losses.append(loss.detach())
                (loss / 2).backward()
            nn.utils.clip_grad_norm_(b.parameters(), 1.0)
            ob.step()
            loss = self.m.train_step(a, oa, batches, "float32", 2)
            self.assertAlmostEqual(loss, torch.stack(losses).mean().item(), places=5)
            for p, q in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        before = copy.deepcopy(a.state_dict())
        count = oa.state_dict()["state"][0]["step"].clone()
        self.m.train_step(a, oa, batches[:1], "float32", 2)
        for k, v in a.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        torch.testing.assert_close(oa.state_dict()["state"][0]["step"], count)
        with self.assertRaises(ValueError):
            self.m.train_step(a, oa, batches, "bfloat16", 2)

    def test_invalid_batches_and_nonfinite_loss_never_update(self):
        model = nn.Sequential(nn.Flatten(), nn.Linear(12, 4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006)
        images = torch.full((4, 3, 2, 2), float("nan"))
        labels = torch.arange(4)
        before = copy.deepcopy(model.state_dict())
        for batches, precision, count in [
            ([], "float32", 1),
            ([(images, labels)], "unknown", 1),
            ([(images, labels)], "float32", 1),
        ]:
            with self.assertRaises(ValueError):
                self.m.train_step(model, optimizer, batches, precision, count)
        self.assertFalse(optimizer.state)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_nonfinite_gradients_never_reach_optimizer(self):
        model = nn.Linear(3, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006)
        before = copy.deepcopy(model.state_dict())
        handle = model.weight.register_hook(
            lambda grad: torch.full_like(grad, float("inf"))
        )
        self.addCleanup(handle.remove)
        with self.assertRaises(ValueError):
            self.m.train_step(
                model, optimizer, [(torch.ones(4, 3), torch.arange(4))], "float32", 1
            )
        self.assertFalse(optimizer.state)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_both_presets_and_large_constructor(self):
        self.assertEqual(
            module().PRESETS,
            {
                "vit_base_patch16_224": (768, 12, 12),
                "vit_large_patch16_224": (1024, 24, 16),
            },
        )
        with mock.patch.dict(self.m.PRESETS, {"vit_large_patch16_224": (24, 2, 3)}):
            torch.manual_seed(42)
            model = self.m.build_model("vit_large_patch16_224")
        torch.manual_seed(42)
        expected = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=24,
            depth=2,
            num_heads=3,
            num_classes=4,
            mlp_ratio=4.0,
            drop_rate=0.1,
            attn_drop_rate=0.0,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
        )
        for key, value in model.state_dict().items():
            torch.testing.assert_close(
                value, expected.state_dict()[key], rtol=0, atol=0
            )

    def test_resolved_configuration_and_example(self):
        from tests._checkout import is_checkout

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.fixture(Path(tmp))
            self.m.validate(cfg)
            for key, value in [
                ("device", "auto"),
                ("precision", "half"),
                ("prepared_root", ""),
                ("validation_root", None),
                ("resume", False),
                ("seed", True),
                ("num_workers", -1),
                ("microbatch", 0),
                ("epochs", 0),
                ("extra", 1),
            ]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.m.validate(dict(cfg, **{key: value}))
        if is_checkout():
            self.m.validate(
                json.loads((ROOT / "docs/examples/rotation_scaling.json").read_text())
            )

    def fixture(self, root):
        from tests.test_idv2_scaling import prepared

        rows = prepared(root / "source")
        (root / "val").mkdir()
        rows = [
            dict(rows[0], sample_id=str(i), member_name=f"keep{i}.PNG")
            for i in range(5)
        ]
        (root / "source/MANIFEST.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        return {
            "device": "cpu",
            "seed": 42,
            "model": "vit_base_patch16_224",
            "precision": "float32",
            "prepared_root": str(root / "source"),
            "validation_root": str(root / "val"),
            "epochs": 2,
            "microbatch": 1,
            "accum": 2,
            "num_workers": 0,
            "resume": None,
        }

    def test_retained_epochs_export_matching_head_free_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            out = root / "out"
            out.mkdir()
            self.m.run(dict(cfg, epochs=90), out)
            for epoch in (30, 60, 90):
                state = torch.load(
                    out / f"checkpoint_epoch_{epoch}.pth", weights_only=True
                )
                export = torch.load(out / f"encoder_epoch{epoch}.pt", weights_only=True)
                self.assertEqual(state["epoch"], epoch)
                self.assertEqual(state["global_step"], epoch * 2)
                expected = {
                    k: v for k, v in state["model"].items() if not k.startswith("head.")
                }
                self.assertEqual(set(export["state_dict"]), set(expected))
                for key, value in expected.items():
                    torch.testing.assert_close(
                        export["state_dict"][key], value, rtol=0, atol=0
                    )

    def test_epoch_loss_weights_partial_accumulation_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            out = root / "out"
            out.mkdir()
            actual = []
            original = self.m.train_step

            def record(*args):
                value = original(*args)
                actual.append((value, len(args[2])))
                return value

            with mock.patch.object(self.m, "train_step", side_effect=record):
                self.m.run(dict(cfg, epochs=1), out)
            expected = sum(value * count for value, count in actual) / sum(
                count for _, count in actual
            )
            result = json.loads((out / "results.json").read_text())
            self.assertAlmostEqual(result["final_loss"], expected, places=12)

    def test_shared_indexed_reader_is_the_existing_implementation(self):
        from downstream.indexed_tar import IndexedTarDataset

        spec = importlib.util.spec_from_file_location(
            "reader_compat", ROOT / "methods/10_inst_disc/data/indexed_tar.py"
        )
        assert spec is not None and spec.loader is not None
        compat = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compat)
        self.assertIs(compat.IndexedTarDataset, IndexedTarDataset)

    def test_cli_continuation_export_and_fail_closed_inputs(self):
        from downstream import contract
        from downstream import spatial_backbones as sb

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            before = {p.name: p.read_bytes() for p in (root / "source").iterdir()}
            for name, epochs, resume in [
                ("full", 2, None),
                ("part", 1, None),
                ("resumed", 2, str(root / "part/checkpoint_latest.pth")),
            ]:
                config = dict(cfg, epochs=epochs, resume=resume)
                path = root / (name + ".json")
                path.write_text(json.dumps(config))
                out = root / name
                self.assertEqual(
                    self.m.main(["--config", str(path), "--out", str(out)]), 0
                )
                self.assertTrue(contract.verify(out, path, 0)[0])
            full = torch.load(root / "full/checkpoint_latest.pth", weights_only=True)
            resumed = torch.load(
                root / "resumed/checkpoint_latest.pth", weights_only=True
            )
            for k, v in full["model"].items():
                torch.testing.assert_close(v, resumed["model"][k], rtol=0, atol=0)
            for key, state in full["optimizer"]["state"].items():
                for k, v in state.items():
                    torch.testing.assert_close(
                        v, resumed["optimizer"]["state"][key][k], rtol=0, atol=0
                    )
            self.assertEqual(full["global_step"], 4)
            result = json.loads((root / "full/results.json").read_text())
            self.assertEqual(result["dropped_microbatches_per_rank"], 1)
            self.assertEqual(result["global_source_batch_size"], 2)
            self.assertEqual(result["global_rotation_batch_size"], 8)
            self.assertFalse(result["canonical_eligible"])
            provider = sb._load_provider(sb.discover_providers()["rotation_scaling"])
            with mock.patch.dict(
                provider.PRESETS, {"vit_base_patch16_224": (16, 1, 2)}
            ):
                encoder = provider.build(
                    {
                        "kind": "rotation_scaling",
                        "encoder": str(root / "full/encoder.pt"),
                        "arch": "vit_base_patch16_224",
                        "img_size": 224,
                        "patch_size": 16,
                    }
                )
                self.assertEqual(
                    encoder.classification_features(
                        torch.rand(2, 3, 224, 224), adaptation="frozen"
                    ).shape,
                    (2, 16),
                )
            self.assertEqual(
                before, {p.name: p.read_bytes() for p in (root / "source").iterdir()}
            )
            for key, value in [
                ("model", "unknown"),
                ("epochs", 91),
                ("accum", True),
                ("precision", "bfloat16"),
            ]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.m.validate(dict(cfg, **{key: value}))
            cases = {
                "weight_decay": lambda b: b["optimizer"]["param_groups"][0].update(
                    weight_decay=0
                ),
                "identity": lambda b: b.update(identity="wrong"),
                "step": lambda b: b.update(global_step=999),
                "epoch": lambda b: b.update(epoch=True),
                "format": lambda b: b.update(format="native"),
                "rng": lambda b: b.update(rng=b["rng"] * 2),
                "missing_slot": lambda b: b["optimizer"]["state"].pop(0),
                "rate": lambda b: b["optimizer"]["param_groups"][0].update(lr=1.0),
                "moment": lambda b: b["optimizer"]["state"][0]["exp_avg"].fill_(
                    float("nan")
                ),
                "model": lambda b: next(iter(b["model"].values())).fill_(float("nan")),
            }
            for name, corrupt in cases.items():
                bad = copy.deepcopy(full)
                corrupt(bad)
                torch.save(bad, root / "bad.pt")
                path = root / "bad.json"
                path.write_text(
                    json.dumps(dict(cfg, epochs=3, resume=str(root / "bad.pt")))
                )
                out = root / ("bad-" + name)
                with self.subTest(corruption=name):
                    with mock.patch.object(
                        self.m,
                        "train_step",
                        side_effect=AssertionError(
                            "invalid checkpoint reached training"
                        ),
                    ) as update:
                        self.assertNotEqual(
                            self.m.main(["--config", str(path), "--out", str(out)]), 0
                        )
                        update.assert_not_called()
                    self.assertFalse((out / "encoder.pt").exists())
            manifest = (root / "full" / contract.MANIFEST).read_bytes()
            self.assertNotEqual(
                self.m.main(
                    ["--config", str(root / "full.json"), "--out", str(root / "full")]
                ),
                0,
            )
            self.assertEqual(manifest, (root / "full" / contract.MANIFEST).read_bytes())


class TestCI(unittest.TestCase):
    def test_downstream_job_runs_scaling_contracts(self):
        from tests._checkout import is_checkout
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import HAVE_YAML, parsed

        if not HAVE_YAML or not is_checkout():
            self.skipTest("checkout and PyYAML required")
        command = next(
            s["run"]
            for s in parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
            if s.get("name")
            == "Run Basic5 component contracts with downstream dependencies"
        )
        self.assertTrue(
            _runs_finetune_tests(command, module="tests.test_method_rotation_scaling")
        )


@unittest.skipUnless(HAVE, "torch and timm required")
class TestDistributed(unittest.TestCase):
    def test_two_rank_updates_continuation_and_read_failure(self):
        import os
        import socket
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = TestScaling().fixture(root)
            (root / "config.json").write_text(json.dumps(cfg))
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = dict(
                os.environ,
                OMP_NUM_THREADS="1",
                PYTHONPATH=str(ROOT),
                GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--master-addr=127.0.0.1",
                    f"--master-port={port}",
                    "--nproc-per-node=2",
                    str(Path(__file__).resolve()),
                    str(root),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = json.loads((root / "full/results.json").read_text())
            self.assertEqual(result["global_source_batch_size"], 4)
            self.assertEqual(result["global_step"], 2)


def distributed_worker(root):
    from downstream.extended_distributed import session
    from downstream.indexed_tar import IndexedTarDataset

    m = module()
    m.PRESETS["vit_base_patch16_224"] = (16, 1, 2)
    cfg = json.loads((root / "config.json").read_text())
    with session("cpu") as context:
        torch.manual_seed(42)
        raw = nn.Sequential(nn.Flatten(), nn.Linear(12, 4))
        oracle = copy.deepcopy(raw)
        ddp = torch.nn.parallel.DistributedDataParallel(raw)
        optimizer = torch.optim.AdamW(raw.parameters(), lr=0.0006, weight_decay=0.05)
        reference = torch.optim.AdamW(oracle.parameters(), lr=0.0006, weight_decay=0.05)
        rank_batches = [
            [
                (
                    torch.randn(
                        4,
                        3,
                        2,
                        2,
                        generator=torch.Generator().manual_seed(100 + 10 * rank + i),
                    ),
                    torch.arange(4),
                )
                for i in range(2)
            ]
            for rank in range(2)
        ]
        for _ in range(2):
            reference.zero_grad(set_to_none=True)
            for i in range(2):
                images = torch.cat([rank_batches[r][i][0] for r in range(2)])
                labels = torch.cat([rank_batches[r][i][1] for r in range(2)])
                (nn.functional.cross_entropy(oracle(images), labels) / 2).backward()
            nn.utils.clip_grad_norm_(oracle.parameters(), 1.0)
            reference.step()
            m.train_step(ddp, optimizer, rank_batches[context.rank], "float32", 2)
            for actual, expected in zip(raw.parameters(), oracle.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(
                    actual.grad, expected.grad, rtol=1e-6, atol=1e-7
                )
        for name, epochs, resume in [
            ("full", 2, None),
            ("part", 1, None),
            ("resumed", 2, str(root / "part/checkpoint_latest.pth")),
        ]:
            out = root / name
            context.call(lambda out=out: out.mkdir(), leader=True)
            built = []
            factory = m.build_model

            def build(arch, factory=factory, built=built):
                value = factory(arch)
                built.append(value)
                return value

            with mock.patch.object(m, "build_model", side_effect=build):
                m.run(dict(cfg, epochs=epochs, resume=resume), out)
            all_states = [None] * context.world
            torch.distributed.all_gather_object(all_states, built[0].state_dict())
            for state in all_states[1:]:
                for key, value in all_states[0].items():
                    torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        full = torch.load(root / "full/checkpoint_latest.pth", weights_only=True)
        resumed = torch.load(root / "resumed/checkpoint_latest.pth", weights_only=True)
        for key, value in full["model"].items():
            torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0)
        for key, state in full["optimizer"]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(
                    value, resumed["optimizer"]["state"][key][name], rtol=0, atol=0
                )
        assert len(full["rng"]) == 2
        out = root / "failed"
        context.call(lambda: out.mkdir(), leader=True)
        original = IndexedTarDataset.__getitem__

        def fail(self, index):
            if context.rank == 1:
                raise ValueError("rank-local input failure")
            return original(self, index)

        with mock.patch.object(IndexedTarDataset, "__getitem__", fail):
            try:
                m.run(cfg, out)
            except RuntimeError as exc:
                assert "rank-local input failure" in str(exc)
            else:
                raise AssertionError("rank-local input error was ignored")
        assert not (out / "encoder.pt").exists()


if __name__ == "__main__":
    import sys

    distributed_worker(Path(sys.argv[1]))
