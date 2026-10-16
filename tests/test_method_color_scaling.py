"""Indexed color scaling recipe and portable training contracts."""

import importlib
import unittest

try:
    import torch

    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "torch required")
class TestColorScaling(unittest.TestCase):
    def entry(self):
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[1]
            / "methods/03_colorization/color_scaling.py"
        )
        self.assertTrue(path.is_file(), "indexed color scaling training is missing")
        return importlib.import_module("methods.03_colorization.color_scaling")

    def test_native_epoch_schedule(self):
        m = self.entry()
        for batch in (32, 1024):
            peak, floor = 0.00015 * batch / 256, 1e-6 * batch / 256
            for epoch in (0, 1, 9, 10, 30, 60, 89, 90):
                expected = floor + (peak - floor) * (
                    epoch / 10
                    if epoch < 10
                    else 0.5
                    * (
                        1
                        + float(torch.cos(torch.tensor((epoch - 10) / 80 * 3.14159265)))
                    )
                )
                self.assertEqual(m.schedule(epoch, batch), expected)

    def test_full_preset_geometry_and_documented_config(self):
        import json
        from pathlib import Path

        from tests._checkout import is_checkout

        m = self.entry()
        # Counts measured from the original constructor on the meta device.
        expected = {
            "vit_base_patch16_224": (768, 12, 12, 91742969),
            "vit_large_patch16_224": (1024, 24, 16, 310294521),
        }
        self.assertEqual(set(m.PRESETS), set(expected))
        for arch, (width, depth, heads, count) in expected.items():
            with torch.device("meta"):
                model = m.build_model(arch)
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            self.assertEqual(model.embed_dim, width)
            self.assertEqual(len(model.encoder.blocks), depth)
            self.assertEqual(model.encoder.blocks[0].attn.num_heads, heads)
            self.assertEqual(tuple(model.encoder.pos_embed.shape), (1, 197, width))
            self.assertEqual(model.decoder[-1].out_channels, 313)
        if is_checkout():
            m.validate(
                json.loads(
                    (
                        Path(__file__).resolve().parents[1]
                        / "docs/examples/colorization_scaling.json"
                    ).read_text()
                )
            )

    def test_weighted_update_matches_oracle(self):
        import copy

        m = self.entry()
        torch.manual_seed(3)
        model = torch.nn.Conv2d(1, 3, 1)
        other = copy.deepcopy(model)
        a = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=0.05)
        b = torch.optim.AdamW(other.parameters(), lr=0.0006, weight_decay=0.05)
        x, y, weights = (
            torch.rand(2, 1, 4, 4) * 40,
            torch.randint(3, (2, 4, 4)),
            torch.tensor([0.1, 2.0, 0.7]),
        )
        expected = torch.nn.functional.cross_entropy(other(x), y, weight=weights)
        expected.backward()
        self.assertGreater(
            torch.nn.utils.clip_grad_norm_(other.parameters(), 1.0).item(), 1.0
        )
        b.step()
        loss = m.train_step(model, a, x, y, weights, None)
        self.assertEqual(loss, expected.item())
        for v, w in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(v, w, rtol=0, atol=0)
            torch.testing.assert_close(v.grad, w.grad, rtol=0, atol=0)

    def test_nonfinite_refuses_before_backward(self):
        m = self.entry()
        model = torch.nn.Conv2d(1, 3, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            m.train_step(
                model,
                optimizer,
                torch.full((2, 1, 4, 4), float("nan")),
                torch.zeros(2, 4, 4, dtype=torch.long),
                torch.ones(3),
                None,
            )
        self.assertFalse(optimizer.state)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def fixture(self, root):
        import json

        from tests.test_idv2_scaling import prepared

        rows = prepared(root / "source")
        (root / "val").mkdir()
        rows = [
            dict(rows[0], sample_id=str(i), member_name=f"keep{i}.PNG")
            for i in range(4)
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
            "microbatch": 2,
            "num_workers": 0,
        }

    def test_actual_training_exports_and_preserves_inputs(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest import mock

        m = self.entry()
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            config = root / "resolved.json"
            config.write_text(json.dumps(cfg))
            original = (root / "source/MANIFEST.jsonl").read_bytes()
            with (
                mock.patch.dict(m.PRESETS, {cfg["model"]: (16, 1, 2)}),
                mock.patch.object(m, "CROP", 16),
                mock.patch.object(m, "RESIZE", 20),
            ):
                self.assertEqual(
                    m.main(["--config", str(config), "--out", str(root / "out")]), 0
                )
            from downstream.contract import MANIFEST

            self.assertEqual(
                json.loads((root / "out" / MANIFEST).read_text())["status"], "ok"
            )
            result = json.loads((root / "out/results.json").read_text())
            self.assertFalse(result["canonical_eligible"])
            self.assertEqual(result["global_step"], 4)
            self.assertEqual((root / "source/MANIFEST.jsonl").read_bytes(), original)
            export = torch.load(root / "out/encoder.pt", weights_only=True)
            self.assertEqual(export["kind"], "colorization_scaling")
            self.assertFalse(
                any(k.startswith("decoder.") for k in export["state_dict"])
            )
            ckpt = torch.load(root / "out/checkpoint_latest.pth", weights_only=True)
            self.assertEqual(ckpt["epoch"], 2)
            self.assertEqual(ckpt["global_step"], 4)
            self.assertEqual(ckpt["class_weights"].shape, (313,))
            before = (root / "out/encoder.pt").read_bytes()
            self.assertNotEqual(
                m.main(["--config", str(config), "--out", str(root / "out")]), 0
            )
            self.assertEqual((root / "out/encoder.pt").read_bytes(), before)

    def test_invalid_configs_and_nonfinite_gradient(self):
        import tempfile
        from pathlib import Path

        m = self.entry()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.fixture(Path(tmp))
            m.validate(cfg)
            for k, v in [
                ("epochs", 0),
                ("epochs", 91),
                ("epochs", True),
                ("microbatch", 0),
                ("seed", -1),
                ("num_workers", -1),
                ("model", "other"),
                ("precision", "bfloat16"),
                ("precision", "float16"),
                ("device", "auto"),
                ("resume", "guess.pth"),
                ("prepared_root", ""),
            ]:
                with self.subTest(key=k, value=v), self.assertRaises(ValueError):
                    m.validate(dict(cfg, **{k: v}))
        model = torch.nn.Conv2d(1, 3, 1)
        handle = model.weight.register_hook(lambda g: g * float("inf"))
        opt = torch.optim.AdamW(model.parameters())
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            m.train_step(
                model,
                opt,
                torch.rand(2, 1, 4, 4),
                torch.zeros(2, 4, 4, dtype=torch.long),
                torch.ones(3),
                None,
            )
        handle.remove()
        self.assertFalse(opt.state)

    def test_native_initialization_order(self):
        from unittest import mock

        m = self.entry()
        old = importlib.import_module("methods.03_colorization.models.vit_colorization")
        nn = torch.nn
        torch.manual_seed(42)
        ref = nn.Module()
        ref.patch_embed = old.PatchEmbed(16, 16, 1, 16)
        ref.cls_token = nn.Parameter(torch.zeros(1, 1, 16))
        ref.pos_embed = nn.Parameter(torch.zeros(1, 2, 16))
        ref.pos_drop = nn.Dropout(0.1)
        ref.blocks = nn.ModuleList([old.Block(16, 2, drop=0.1, attn_drop=0.1)])
        ref.norm = nn.LayerNorm(16)
        ref.decoder = old.ColorizationViT._build_decoder(16, 313)
        nn.init.trunc_normal_(ref.cls_token, std=0.02)
        nn.init.trunc_normal_(ref.pos_embed, std=0.02)
        ref.apply(old.ColorizationViT._init_weights)
        torch.manual_seed(42)
        with (
            mock.patch.dict(m.PRESETS, {"vit_base_patch16_224": (16, 1, 2)}),
            mock.patch.object(m, "CROP", 16),
        ):
            actual = m.build_model("vit_base_patch16_224")
        expected = ref.state_dict()
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(
                value, expected[key.removeprefix("encoder.")], atol=0, rtol=0
            )

    def test_class_weights_native_sample_and_formula(self):
        import numpy as np

        m = self.entry()
        visited = []

        class Data:
            def __len__(self):
                return 10003

            def __getitem__(self, index):
                visited.append(index)
                return (
                    (
                        None,
                        torch.tensor([[index % 313, index % 313, (index + 3) % 313]]),
                    ),
                    index,
                    0,
                )

        weights = m.class_weights(Data())
        expected_indices = np.random.RandomState(42).choice(10003, 10000, replace=False)
        self.assertEqual(visited, expected_indices.tolist())
        counts = np.bincount(
            np.concatenate(
                [
                    expected_indices % 313,
                    expected_indices % 313,
                    (expected_indices + 3) % 313,
                ]
            ),
            minlength=313,
        ).astype(np.float64)
        expected = 1 / (0.5 * counts / (counts.sum() + 1e-8) + 0.5 / 313 + 1e-8)
        torch.testing.assert_close(
            weights,
            torch.tensor((expected / expected.mean()).astype(np.float32)),
            rtol=0,
            atol=0,
        )

    def test_input_recipe_and_nonfinite_refusal(self):
        from unittest import mock

        import numpy as np
        from PIL import Image

        m = self.entry()
        image = Image.fromarray(
            np.random.RandomState(8).randint(0, 256, (260, 270, 3)).astype(np.uint8)
        )
        from torchvision import transforms

        for training in (True, False):
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomCrop(224)
                    if training
                    else transforms.CenterCrop(224),
                ]
                + ([transforms.RandomHorizontalFlip()] if training else [])
            )
            torch.manual_seed(10)
            rgb = transform(image)
            light, ab = m._data._lab_components(rgb, native_eval=True)
            expected = torch.tensor((light / 100).astype(np.float32)).unsqueeze(0)
            target = torch.tensor(m._quant.quantize_ab_fast(ab))
            torch.manual_seed(10)
            actual, labels = m.Inputs(training)(image)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(labels, target, rtol=0, atol=0)
        with (
            mock.patch.object(
                m._data,
                "_lab_components",
                return_value=(np.array([float("nan")]), np.zeros(2)),
            ),
            self.assertRaisesRegex(ValueError, "nonfinite"),
        ):
            m.Inputs()(image)

    def test_retained_epochs_and_invalid_reads(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest import mock

        m = self.entry()
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            out = root / "out"
            out.mkdir()
            with (
                mock.patch.object(m, "build_model", side_effect=lambda arch: tiny()),
                mock.patch.object(m, "CROP", 4),
                mock.patch.object(m, "RESIZE", 8),
            ):
                m.run(dict(cfg, epochs=90), out)
            for epoch in (30, 60, 90):
                checkpoint = torch.load(
                    out / f"checkpoint_epoch_{epoch}.pth", weights_only=True
                )
                export = torch.load(out / f"encoder_epoch{epoch}.pt", weights_only=True)
                self.assertEqual(checkpoint["epoch"], epoch)
                self.assertEqual(checkpoint["global_step"], epoch * 2)
                for k, v in export["state_dict"].items():
                    torch.testing.assert_close(
                        v, checkpoint["model"]["encoder." + k], rtol=0, atol=0
                    )
            (root / "source/samples.tar").write_bytes(b"broken")
            config = root / "config.json"
            config.write_text(json.dumps(cfg))
            with mock.patch.object(m, "build_model", side_effect=lambda arch: tiny()):
                self.assertEqual(
                    m.main(["--config", str(config), "--out", str(root / "bad")]), 1
                )
            self.assertFalse((root / "bad/encoder.pt").exists())
            from downstream.contract import MANIFEST

            self.assertEqual(
                json.loads((root / "bad" / MANIFEST).read_text())["status"], "failed"
            )

    def test_ci_registration(self):
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
            _runs_finetune_tests(command, module="tests.test_method_color_scaling")
        )

    def test_frozen_export_loads_and_matches_trained_trunk(self):
        import importlib.util

        if importlib.util.find_spec("timm") is None:
            self.skipTest("frozen evaluation requires timm")
        import tempfile
        from pathlib import Path
        from unittest import mock

        from downstream import spatial_backbones as sb

        m = self.entry()
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            out = root / "out"
            out.mkdir()
            with (
                mock.patch.dict(m.PRESETS, {cfg["model"]: (16, 1, 2)}),
                mock.patch.object(m, "CROP", 16),
                mock.patch.object(m, "RESIZE", 20),
            ):
                m.run(dict(cfg, epochs=1), out)
            provider = sb._load_provider(
                sb.discover_providers()["colorization_scaling"]
            )
            factory = lambda arch: m._model._ViTTrunk(
                img_size=16, embed_dim=16, depth=1, num_heads=2
            )
            with mock.patch.object(provider, "build_encoder", side_effect=factory):
                model = provider.build(
                    {
                        "kind": "colorization_scaling",
                        "encoder": str(out / "encoder.pt"),
                        "arch": cfg["model"],
                        "img_size": 224,
                        "patch_size": 16,
                    }
                )
            images = torch.rand(2, 3, 16, 16)
            actual = model.classification_features(images, adaptation="frozen")
            state = torch.load(out / "checkpoint_latest.pth", weights_only=True)
            reference = factory(cfg["model"]).eval()
            reference.load_state_dict(
                {
                    k.removeprefix("encoder."): v
                    for k, v in state["model"].items()
                    if k.startswith("encoder.")
                }
            )
            from downstream.frozen_vit import lightness_input

            rgb = (
                images * images.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
                + images.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
            )
            torch.testing.assert_close(
                actual,
                torch.nn.functional.normalize(
                    reference(lightness_input(rgb))[:, 0], dim=-1
                ),
                rtol=0,
                atol=0,
            )

    def test_amp_scaler_order(self):
        from contextlib import nullcontext
        from unittest import mock

        m = self.entry()
        calls = []

        class Scaler:
            def scale(self, loss):
                calls.append("scale")
                return loss

            def unscale_(self, optimizer):
                calls.append("unscale")

            def step(self, optimizer):
                calls.append("step")
                optimizer.step()

            def update(self):
                calls.append("update")

        model = torch.nn.Conv2d(1, 3, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        original = torch.nn.utils.clip_grad_norm_

        def clip(*a, **kw):
            calls.append("clip")
            return original(*a, **kw)

        with (
            mock.patch.object(
                torch, "autocast", return_value=nullcontext()
            ) as autocast,
            mock.patch.object(torch.nn.utils, "clip_grad_norm_", side_effect=clip),
        ):
            m.train_step(
                model,
                optimizer,
                torch.rand(2, 1, 4, 4),
                torch.zeros(2, 4, 4, dtype=torch.long),
                torch.ones(3),
                Scaler(),
            )
        autocast.assert_called_once_with("cuda", dtype=torch.float16)
        self.assertEqual(calls, ["scale", "unscale", "clip", "step", "update"])


def tiny():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Conv2d(1, 4, 1)
            self.decoder = torch.nn.Conv2d(4, 313, 1)

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return Model()


@unittest.skipUnless(HAVE, "torch required")
class TestDistributed(unittest.TestCase):
    def test_two_rank_training_and_read_error(self):
        import json
        import os
        import socket
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = TestColorScaling().fixture(root)
            (root / "config.json").write_text(json.dumps(cfg))
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = dict(
                os.environ,
                OMP_NUM_THREADS="1",
                PYTHONPATH=str(Path(__file__).resolve().parents[1]),
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
            self.assertEqual(result["global_batch_size"], 4)
            self.assertEqual(result["global_step"], 2)


def worker(root):
    import copy
    import json
    from unittest import mock

    from downstream.extended_distributed import session
    from downstream.indexed_tar import IndexedTarDataset

    m = importlib.import_module("methods.03_colorization.color_scaling")
    m.PRESETS["vit_base_patch16_224"] = (16, 1, 2)
    m.__dict__.update(CROP=16, RESIZE=20)
    cfg = json.loads((root / "config.json").read_text())
    with session("cpu") as context:
        torch.manual_seed(4)
        raw = torch.nn.Conv2d(1, 3, 1)
        reference = copy.deepcopy(raw)
        ddp = torch.nn.parallel.DistributedDataParallel(raw)
        opt = torch.optim.AdamW(raw.parameters(), lr=0.0006, weight_decay=0.05)
        other = torch.optim.AdamW(reference.parameters(), lr=0.0006, weight_decay=0.05)
        g = torch.Generator().manual_seed(8)
        images = [torch.rand(2, 1, 4, 4, generator=g) for _ in range(2)]
        labels = [torch.randint(3, (2, 4, 4), generator=g) for _ in range(2)]
        weights = torch.tensor([0.1, 2.0, 0.7])
        for _ in range(2):
            other.zero_grad(set_to_none=True)
            loss = torch.stack(
                [
                    torch.nn.functional.cross_entropy(reference(x), y, weight=weights)
                    for x, y in zip(images, labels)
                ]
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            other.step()
            m.train_step(
                ddp, opt, images[context.rank], labels[context.rank], weights, None
            )
            for p, q in zip(raw.parameters(), reference.parameters()):
                torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-6, atol=1e-7)
        out = root / "full"
        context.call(lambda: out.mkdir(), leader=True)
        built = []
        factory = m.build_model

        def build(arch):
            value = factory(arch)
            built.append(value)
            return value

        with mock.patch.object(m, "build_model", side_effect=build):
            m.run(cfg, out)
        states = [None] * context.world
        torch.distributed.all_gather_object(states, dict(built[0].named_parameters()))
        for k, v in states[0].items():
            torch.testing.assert_close(v, states[1][k], rtol=0, atol=0)
        saved = torch.load(out / "checkpoint_latest.pth", weights_only=True)
        assert len(saved["rng"]) == 2
        out = root / "bad"
        context.call(lambda: out.mkdir(), leader=True)
        original = IndexedTarDataset.__getitem__

        def fail(self, index):
            if context.rank == 1:
                raise ValueError("rank-local read failure")
            return original(self, index)

        with mock.patch.object(IndexedTarDataset, "__getitem__", fail):
            try:
                m.run(cfg, out)
            except RuntimeError as exc:
                assert "rank-local read failure" in str(exc)
            else:
                raise AssertionError("read failure was ignored")
        assert not (out / "encoder.pt").exists()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    worker(Path(sys.argv[1]))
