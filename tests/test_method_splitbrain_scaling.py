"""Behavioral contracts for indexed cross-channel scaling training."""

import importlib
import unittest
from pathlib import Path

try:
    import timm
    import torch

    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "torch and timm required")
class TestSplitScaling(unittest.TestCase):
    def entry(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "methods/08_split_brain/splitbrain_scaling.py"
        )
        self.assertTrue(path.is_file(), "indexed SplitBrain training is missing")
        return importlib.import_module("methods.08_split_brain.splitbrain_scaling")

    def test_documented_config(self):
        import json

        from tests._checkout import is_checkout

        if not is_checkout():
            self.skipTest("checkout required")
        path = (
            Path(__file__).resolve().parents[1]
            / "docs/examples/splitbrain_scaling.json"
        )
        self.assertTrue(path.is_file(), "resolved example missing")
        self.entry().validate(json.loads(path.read_text()))

    def test_presets_and_native_norm(self):
        m = self.entry()
        for arch, count in [
            ("vit_base_patch16_224", 47278315),
            ("vit_large_patch16_224", 157203947),
        ]:
            with torch.device("meta"):
                model = m.build_model(arch)
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            self.assertEqual(model.net1.encoder.norm.eps, 1e-6)
            self.assertEqual(model.net2.encoder.patch_embed.proj.in_channels, 2)

    def test_quantization_and_resize(self):
        m = self.entry()
        torch.manual_seed(7)
        light = torch.rand(2, 1, 5, 7) * 2 - 1
        ab = torch.rand(2, 2, 5, 7) * 2 - 1
        l, a = m.targets(light, ab)
        code = torch.tensor(m._data.load_ab_codebook(), dtype=torch.float32)
        expected = (
            (((ab.permute(0, 2, 3, 1) * 128).unsqueeze(-2) - code) ** 2)
            .sum(-1)
            .argmin(-1)
        )
        torch.testing.assert_close(a, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            l, ((light[:, 0] * 50 + 50) / 2).floor().clamp(0, 49).long(), rtol=0, atol=0
        )
        self.assertEqual(
            m.resize_target(a, (2, 3)).tolist(), a[:, [0, 2]][:, :, [0, 2, 4]].tolist()
        )
        torch.manual_seed(19)
        dense = torch.rand(2, 2, 32, 32) * 2 - 1
        values = dense.permute(0, 2, 3, 1)
        expected = ((values.unsqueeze(-2) * 128 - code) ** 2).sum(-1).argmin(-1)
        alternative = ((values.unsqueeze(-2) * 127 - code) ** 2).sum(-1).argmin(-1)
        self.assertTrue((expected != alternative).any())
        torch.testing.assert_close(
            m.targets(torch.zeros(2, 1, 32, 32), dense)[1], expected, rtol=0, atol=0
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            m.targets(light * float("nan"), ab)

    def test_two_loss_update_and_discarded_tail(self):
        import copy

        m = self.entry()
        torch.manual_seed(11)
        torch.set_num_threads(1)

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Conv2d(1, 313, 1)
                self.l = torch.nn.Conv2d(2, 50, 1)
                self.calls = 0

            def forward(self, l, a):
                self.calls += 1
                return self.a(l), self.l(a)

        model = Tiny()
        other = copy.deepcopy(model)
        opt = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=0.05)
        oracle = torch.optim.AdamW(other.parameters(), lr=0.0006, weight_decay=0.05)
        batches = [
            (torch.randn(2, 1, 4, 4) * 10, torch.randn(2, 2, 4, 4) * 10)
            for _ in range(2)
        ]
        losses = []
        for l, a in batches:
            lt, at = m.targets(l, a)
            ap, lp = other(l, a)
            loss = torch.nn.functional.cross_entropy(
                ap, at
            ) + torch.nn.functional.cross_entropy(lp, lt)
            losses.append(loss.item())
            (loss / 2).backward()
        self.assertGreater(
            sum(p.grad.square().sum().item() for p in other.parameters()), 1
        )
        oracle.step()
        self.assertAlmostEqual(
            m.train_step(model, opt, batches, "float32", 2), sum(losses) / 2
        )
        for p, q in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        saved = copy.deepcopy(model.state_dict())
        m.train_step(model, opt, batches[:1], "float32", 2)
        self.assertEqual(model.calls, 3)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, saved[k], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            m.train_step(
                model,
                opt,
                [(batches[0][0] * float("nan"), batches[0][1])],
                "float32",
                1,
            )

    def test_indexed_run_and_export(self):
        import json
        import tempfile
        from unittest import mock

        from tests.test_method_color_scaling import TestColorScaling

        m = self.entry()
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = dict(
                TestColorScaling().fixture(root), accum=1, lab_backend="fallback"
            )
            m.validate(cfg)
            for key, value in [
                ("epochs", 0),
                ("epochs", 91),
                ("accum", True),
                ("microbatch", 0),
                ("lab_backend", "auto"),
                ("precision", "float16"),
                ("resume", ""),
                ("unexpected", 1),
            ]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    m.validate(dict(cfg, **{key: value}))
            config = root / "config.json"
            config.write_text(json.dumps(cfg))
            with (
                mock.patch.dict(m.PRESETS, {cfg["model"]: (16, 1, 2)}),
                mock.patch.object(m, "CROP", 16),
            ):
                self.assertEqual(
                    m.main(["--config", str(config), "--out", str(root / "out")]), 0
                )
            result = json.loads((root / "out/results.json").read_text())
            self.assertEqual(result["global_step"], 4)
            self.assertFalse(result["canonical_eligible"])
            state = torch.load(root / "out/checkpoint_latest.pth", weights_only=True)
            self.assertEqual(state["epoch"], 2)
            export = torch.load(root / "out/encoder.pt", weights_only=True)
            with (
                mock.patch.dict(m.PRESETS, {cfg["model"]: (16, 1, 2)}),
                mock.patch.object(m, "CROP", 16),
            ):
                trained = m.build_model(cfg["model"])
                trained.load_state_dict(state["model"])
            from downstream.frozen_vit import validate_state

            encoders = torch.nn.ModuleList([trained.net1.encoder, trained.net2.encoder])
            validate_state(encoders, export["state_dict"])
            self.assertFalse(any("decoder" in k for k in export["state_dict"]))
            self.assertEqual(
                m.main(["--config", str(config), "--out", str(root / "out")]), 1
            )

    def test_native_factory_initialization_and_contiguous_forward(self):
        from unittest import mock

        m = self.entry()
        torch.manual_seed(23)
        with (
            mock.patch.dict(m.PRESETS, {"vit_base_patch16_224": (16, 1, 2)}),
            mock.patch.object(m, "CROP", 32),
        ):
            model = m.build_model("vit_base_patch16_224")
        torch.manual_seed(23)
        branches = []
        for channels, classes in ((1, 313), (2, 50)):
            encoder = timm.create_model(
                "vit_base_patch16_224",
                pretrained=False,
                num_classes=0,
                img_size=32,
                in_chans=channels,
                embed_dim=16,
                depth=1,
                num_heads=2,
                mlp_ratio=4.0,
            )
            decoder = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(16, 256, 4, 2, 1),
                torch.nn.ReLU(inplace=True),
                torch.nn.ConvTranspose2d(256, 128, 4, 2, 1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(128, classes, 1),
            )
            branches.append((encoder, decoder))
        for branch, (encoder, decoder), channels in zip(
            (model.net1, model.net2), branches, (1, 2)
        ):
            for key, value in encoder.state_dict().items():
                torch.testing.assert_close(
                    value, branch.encoder.state_dict()[key], rtol=0, atol=0
                )
            x = torch.randn(2, channels, 32, 32)
            expected = decoder(
                encoder.forward_features(x)[:, 1:]
                .transpose(1, 2)
                .reshape(2, 16, 2, 2)
                .contiguous()
            ).contiguous()
            actual = branch(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(actual.is_contiguous())

    def test_inputs_and_pickling(self):
        import pickle
        from unittest import mock

        import numpy as np
        from PIL import Image
        from torchvision import transforms

        m = self.entry()
        image = Image.fromarray(
            np.random.RandomState(5).randint(256, size=(256, 260, 3)).astype(np.uint8)
        )
        torch.manual_seed(7)
        rgb = transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
        )(image)
        lab = m._data.rgb2lab(np.array(rgb), native_eval=True)
        torch.manual_seed(7)
        light, ab = pickle.loads(pickle.dumps(m.Inputs("fallback")))(image)
        torch.testing.assert_close(
            light,
            torch.from_numpy((lab[..., 0] - 50) / 50).unsqueeze(0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            ab, torch.from_numpy(lab[..., 1:] / 128).permute(2, 0, 1), rtol=0, atol=0
        )
        with (
            mock.patch.object(
                m._data, "rgb2lab", return_value=np.full((224, 224, 3), np.nan)
            ),
            self.assertRaises(ValueError),
        ):
            m.Inputs("fallback")(image)
        with self.assertRaises(ValueError):
            m.Inputs("auto")

    def test_invalid_steps_and_gradient_guard(self):
        from unittest import mock

        m = self.entry()
        with (
            mock.patch.dict(m.PRESETS, {"vit_base_patch16_224": (16, 1, 2)}),
            mock.patch.object(m, "CROP", 16),
        ):
            model = m.build_model("vit_base_patch16_224")
        optimizer = torch.optim.AdamW(model.parameters())
        batch = [(torch.randn(2, 1, 16, 16), torch.randn(2, 2, 16, 16))]
        for batches, precision, accum in (
            ([], "float32", 1),
            (batch, "other", 1),
            (batch, "bfloat16", 1),
            (batch, "float32", 0),
        ):
            with self.assertRaises(ValueError):
                m.train_step(model, optimizer, batches, precision, accum)
        handle = next(model.parameters()).register_hook(
            lambda grad: torch.full_like(grad, float("inf"))
        )
        try:
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                m.train_step(model, optimizer, batch, "float32", 1)
        finally:
            handle.remove()
        self.assertFalse(optimizer.state)

    def test_retention_tail_clock_and_schedule(self):
        import json
        import math
        import tempfile
        from unittest import mock

        from tests.test_method_color_scaling import TestColorScaling

        m = self.entry()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = dict(
                TestColorScaling().fixture(root),
                epochs=90,
                microbatch=1,
                accum=3,
                lab_backend="fallback",
            )
            out = root / "out"
            out.mkdir()
            rates = []

            def step(model, opt, batches, precision, accum):
                rates.append(opt.param_groups[0]["lr"])
                return 2.0 if len(batches) == 3 else 10.0

            with (
                mock.patch.dict(m.PRESETS, {cfg["model"]: (16, 1, 2)}),
                mock.patch.object(m, "CROP", 16),
                mock.patch.object(m, "train_step", side_effect=step),
            ):
                m.run(cfg, out)
            result = json.loads((out / "results.json").read_text())
            self.assertEqual(result["global_step"], 90)
            self.assertEqual(result["final_loss"], 4)
            self.assertEqual(rates[0], m.schedule(0, 90))
            self.assertEqual(rates[2], m.schedule(1, 90))
            for epoch in (30, 60, 90):
                self.assertEqual(
                    torch.load(
                        out / f"checkpoint_epoch_{epoch}.pth", weights_only=True
                    )["epoch"],
                    epoch,
                )
                self.assertTrue((out / f"encoder_epoch{epoch}.pt").is_file())
        for step in (0, 1, 9999, 10000, 15000, 90000):
            expected = (
                max(0.0006 * (step + 1) / 10000, 1e-8)
                if step < 10000
                else 1e-6
                + (0.0006 - 1e-6)
                * 0.5
                * (1 + math.cos(math.pi * min(max((step - 10000) / 80000, 0), 1)))
            )
            self.assertAlmostEqual(m.schedule(step, 90000), expected, places=15)


def assert_exact(left, right):
    """Compare complete nested training state, including scalar RNG metadata."""
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_exact(a, b)
    else:
        assert left == right, (left, right)


@unittest.skipUnless(HAVE, "torch and timm required")
class TestContinuation(unittest.TestCase):
    def test_identity_binds_runtime_and_membership(self):
        import tempfile
        from types import SimpleNamespace
        from unittest import mock

        from tests.test_method_color_scaling import TestColorScaling

        m = importlib.import_module("methods.08_split_brain.splitbrain_scaling")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(
                TestColorScaling().fixture(Path(tmp)), accum=1, lab_backend="fallback"
            )
            context = SimpleNamespace(world=1)
            membership = {"manifest_sha256": "original"}
            expected = m.continuation_identity(cfg, membership, [0, 1], context)
            self.assertEqual(
                expected,
                m.continuation_identity(
                    dict(cfg, resume="file", epochs=90), membership, [0, 1], context
                ),
            )
            for meta, loader, ctx in (
                ({"manifest_sha256": "other"}, [0, 1], context),
                (membership, [0], context),
                (membership, [0, 1], SimpleNamespace(world=2)),
            ):
                self.assertNotEqual(
                    expected, m.continuation_identity(cfg, meta, loader, ctx)
                )
            with mock.patch.object(m, "CROP", 16):
                self.assertNotEqual(
                    expected, m.continuation_identity(cfg, membership, [0, 1], context)
                )
            with mock.patch.dict(m.PRESETS, {cfg["model"]: (16, 1, 2)}):
                self.assertNotEqual(
                    expected, m.continuation_identity(cfg, membership, [0, 1], context)
                )

    def test_epoch_continuation_and_rejections(self):
        import copy
        import hashlib
        import tempfile
        from unittest import mock

        from tests.test_method_color_scaling import TestColorScaling

        m = importlib.import_module("methods.08_split_brain.splitbrain_scaling")
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = dict(
                TestColorScaling().fixture(root),
                accum=3,
                microbatch=1,
                lab_backend="fallback",
                epochs=3,
            )
            with (
                mock.patch.dict(m.PRESETS, {cfg["model"]: (16, 1, 2)}),
                mock.patch.object(m, "CROP", 16),
            ):
                m.run(cfg, root / "whole")
                m.run(dict(cfg, epochs=1), root / "first")
                path = root / "first/checkpoint_latest.pth"
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                resumed = dict(cfg, resume=str(path))
                # RED: an explicit portable checkpoint must be accepted.
                m.validate(resumed)
                m.run(resumed, root / "resumed")
                whole = torch.load(
                    root / "whole/checkpoint_latest.pth", weights_only=True
                )
                other = torch.load(
                    root / "resumed/checkpoint_latest.pth", weights_only=True
                )
                for key in (
                    "model",
                    "optimizer",
                    "rng",
                    "global_step",
                    "epoch",
                    "identity",
                ):
                    assert_exact(whole[key], other[key])
                self.assertEqual(other["global_step"], 3)
                assert_exact(
                    torch.load(root / "whole/encoder.pt", weights_only=True),
                    torch.load(root / "resumed/encoder.pt", weights_only=True),
                )
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
                saved = torch.load(path, weights_only=True)
                cases = []

                def altered(label, change):
                    value = copy.deepcopy(saved)
                    change(value)
                    cases.append((label, value))

                altered(
                    "legacy",
                    lambda s: s.update(format="splitbrain_scaling_training_v1"),
                )
                altered("identity", lambda s: s.update(identity="wrong"))
                altered("extra-key", lambda s: s.update(unexpected=True))
                altered("epoch", lambda s: s.update(epoch=3))

                def at_target(s):
                    s.update(epoch=3, global_step=3)
                    s["optimizer"]["param_groups"][0]["lr"] = m.schedule(3, 90)
                    for slot in s["optimizer"]["state"].values():
                        slot["step"].fill_(3)

                altered("epoch-at-target", at_target)
                altered("epoch-type", lambda s: s.update(epoch=True))
                altered("counter", lambda s: s.update(global_step=0))

                # Every optimizer field is coherent with two updates, but one
                # epoch of this loader can only have completed one update.
                def excess_updates(s):
                    s["global_step"] = 2
                    s["optimizer"]["param_groups"][0]["lr"] = m.schedule(2, 90)
                    for slot in s["optimizer"]["state"].values():
                        slot["step"].fill_(2)

                altered("counter-coherent-optimizer", excess_updates)
                altered("counter-type", lambda s: s.update(global_step=True))
                altered("recorded-config", lambda s: s["config"].update(seed=99))
                altered(
                    "membership",
                    lambda s: s["membership"].update(manifest_sha256="wrong"),
                )
                altered("rank-count", lambda s: s.update(rng=[]))
                altered("rng", lambda s: s["rng"][0].update(torch=torch.zeros(3)))
                altered("optimizer-slots", lambda s: s["optimizer"]["state"].clear())
                altered("optimizer-schema", lambda s: s["optimizer"].update(extra=0))
                altered(
                    "optimizer-step",
                    lambda s: next(iter(s["optimizer"]["state"].values()))[
                        "step"
                    ].fill_(0),
                )
                altered(
                    "optimizer-policy",
                    lambda s: s["optimizer"]["param_groups"][0].update(
                        weight_decay=0.1
                    ),
                )
                altered(
                    "rate", lambda s: s["optimizer"]["param_groups"][0].update(lr=1.0)
                )
                altered(
                    "model",
                    lambda s: next(iter(s["model"].values())).fill_(float("nan")),
                )
                altered(
                    "moment",
                    lambda s: next(iter(s["optimizer"]["state"].values()))[
                        "exp_avg"
                    ].fill_(float("nan")),
                )
                for label, value in cases:
                    bad = root / (label + ".pth")
                    torch.save(value, bad)
                    raw = m.build_model(cfg["model"])
                    original = copy.deepcopy(raw.state_dict())
                    opt = torch.optim.AdamW(
                        raw.parameters(), lr=m.schedule(0, 90), weight_decay=0.05
                    )
                    loader = torch.utils.data.DataLoader(
                        torch.arange(4),
                        batch_size=1,
                        generator=torch.Generator().manual_seed(123),
                    )
                    context = m.Session(torch.device("cpu"))
                    rng = m._rng(loader, context.device)
                    with (
                        self.subTest(preflight=label),
                        self.assertRaises((ValueError, RuntimeError)),
                    ):
                        m.restore(
                            str(bad), saved["identity"], raw, opt, loader, context, 1, 3
                        )
                    assert_exact(raw.state_dict(), original)
                    self.assertFalse(opt.state)
                    assert_exact(m._rng(loader, context.device), rng)
                    with (
                        self.subTest(label=label),
                        mock.patch.object(m, "train_step") as step,
                    ):
                        with self.assertRaises((ValueError, RuntimeError)):
                            m.run(dict(resumed, resume=str(bad)), root / label)
                        step.assert_not_called()
                for key, value in (("seed", 7), ("accum", 2), ("microbatch", 2)):
                    with (
                        self.subTest(recipe=key),
                        self.assertRaises((ValueError, RuntimeError)),
                    ):
                        m.run(dict(resumed, **{key: value}), root / ("changed-" + key))

    def test_worker_continuation_cli(self):
        import json
        import tempfile
        from unittest import mock

        from tests.test_method_color_scaling import TestColorScaling

        m = importlib.import_module("methods.08_split_brain.splitbrain_scaling")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = dict(
                TestColorScaling().fixture(root),
                accum=1,
                lab_backend="fallback",
                num_workers=1,
            )
            with mock.patch.dict(m.PRESETS, {cfg["model"]: (16, 1, 2)}):
                # Keep crop unpatched: spawn workers import the original module.
                def run(name, **changes):
                    config = root / (name + ".json")
                    config.write_text(json.dumps(dict(cfg, **changes)))
                    self.assertEqual(
                        m.main(["--config", str(config), "--out", str(root / name)]), 0
                    )
                    return torch.load(
                        root / name / "checkpoint_latest.pth", weights_only=True
                    )

                whole = run("whole")
                run("first", epochs=1)
                resumed = run(
                    "resumed", resume=str(root / "first/checkpoint_latest.pth")
                )
                for key in ("model", "optimizer", "rng", "global_step", "identity"):
                    assert_exact(whole[key], resumed[key])


class TestCI(unittest.TestCase):
    def test_registered(self):
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
            _runs_finetune_tests(command, module="tests.test_method_splitbrain_scaling")
        )


@unittest.skipUnless(HAVE, "torch and timm required")
class TestDistributed(unittest.TestCase):
    def test_two_rank_update_and_read_failure(self):
        import json
        import os
        import socket
        import subprocess
        import sys
        import tempfile

        from tests.test_method_color_scaling import TestColorScaling

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = dict(
                TestColorScaling().fixture(root), accum=1, lab_backend="fallback"
            )
            (root / "config.json").write_text(json.dumps(cfg))
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
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
                env=dict(
                    os.environ,
                    OMP_NUM_THREADS="1",
                    PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                    GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo",
                ),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = json.loads((root / "out/results.json").read_text())
            self.assertEqual(result["global_step"], 2)
            self.assertEqual(result["global_batch_size"], 4)


def worker(root):
    import copy
    import json
    from unittest import mock

    from downstream.extended_distributed import session
    from downstream.indexed_tar import IndexedTarDataset

    m = importlib.import_module("methods.08_split_brain.splitbrain_scaling")
    m.PRESETS["vit_base_patch16_224"] = (16, 1, 2)
    m.CROP = 16
    cfg = json.loads((root / "config.json").read_text())
    with session("cpu") as context:
        torch.manual_seed(5)
        raw = m.build_model(cfg["model"])
        other = copy.deepcopy(raw)
        ddp = torch.nn.parallel.DistributedDataParallel(raw)
        a = torch.optim.AdamW(raw.parameters(), lr=0.0006, weight_decay=0.05)
        b = torch.optim.AdamW(other.parameters(), lr=0.0006, weight_decay=0.05)
        x, y = torch.randn(4, 1, 16, 16), torch.randn(4, 2, 16, 16)
        m.train_step(other, b, [(x, y)], "float32", 1)
        m.train_step(
            ddp, a, [(x.chunk(2)[context.rank], y.chunk(2)[context.rank])], "float32", 1
        )
        for p, q in zip(raw.parameters(), other.parameters()):
            torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6)
        built = []
        factory = m.build_model

        def build(arch):
            value = factory(arch)
            built.append(value)
            return value

        with mock.patch.object(m, "build_model", side_effect=build):
            m.run(cfg, root / "out")
        states = [None] * context.world
        torch.distributed.all_gather_object(states, built[0].state_dict())
        for key, value in states[0].items():
            torch.testing.assert_close(value, states[1][key], rtol=0, atol=0)
        saved = torch.load(root / "out/checkpoint_latest.pth", weights_only=True)
        assert len(saved["rng"]) == 2
        m.run(dict(cfg, epochs=1), root / "first")
        m.run(
            dict(cfg, resume=str(root / "first/checkpoint_latest.pth")),
            root / "resumed",
        )
        resumed = torch.load(root / "resumed/checkpoint_latest.pth", weights_only=True)
        for key in ("model", "optimizer", "rng", "global_step", "epoch", "identity"):
            assert_exact(saved[key], resumed[key])
        get = IndexedTarDataset.__getitem__

        def broken(self, index):
            if context.rank == 1:
                raise OSError("injected read failure")
            return get(self, index)

        with mock.patch.object(IndexedTarDataset, "__getitem__", broken):
            try:
                m.run(cfg, root / "broken")
            except RuntimeError as exc:
                assert "injected read failure" in str(exc)
            else:
                raise AssertionError("rank failure was swallowed")


if __name__ == "__main__":
    import sys

    worker(Path(sys.argv[1]))
