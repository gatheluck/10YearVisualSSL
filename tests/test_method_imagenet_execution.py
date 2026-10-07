"""Explicit ImageNet execution must preserve updates and epoch continuation."""

import copy
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HAVE = all(
    importlib.util.find_spec(n) is not None
    for n in ("torch", "torchvision", "numpy", "PIL")
)
if HAVE:
    import torch

    from downstream import imagenet, optimization
    from tests import test_basic5_imagenet as images
    from tests import test_method_imagenet_finetune as ft


@unittest.skipUnless(HAVE, "image dependencies required")
class Execution(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def config(self, root=Path("/unused"), adaptation="frozen", kind="vjepa2_1"):
        cfg = images.TestImageNet().config(root)
        cfg["adaptation"] = adaptation
        cfg["backbone"]["kind"] = kind
        if adaptation == "finetune":
            cfg.update(
                optimizer_profile="basic5_finetune_v1",
                finetune_recipe=ft.PROFILES[kind],
            )
        else:
            cfg["preprocessing_profile"] = "captured_provider_v1"
        if adaptation == "attentive":
            from downstream.spatial_backbones import attentive_profile

            if attentive_profile(kind):
                cfg["reader_profile"] = attentive_profile(kind)
        cfg["execution"] = {
            "profile": "captured_imagenet_v1",
            "accumulation_steps": 2,
            "precision": "fp32",
            "tail_policy": "discard",
        }
        cfg["probe"].update(batch_size=1, max_train_samples=5, max_val_samples=0)
        return cfg

    def test_twenty_four_routes_and_effective_batch_are_explicit(self):
        for kind in ft.PROFILES:
            for adaptation in ("frozen", "attentive", "finetune"):
                with self.subTest(kind=kind, adaptation=adaptation):
                    cfg = self.config(adaptation=adaptation, kind=kind)
                    imagenet.validate_config(cfg)
                    report = optimization.resolve_optimization(cfg, imagenet.TASK)
                    self.assertEqual(report["effective_batch"], 2)
                    self.assertEqual(report["accumulation_steps"], 2)
                    from downstream.imagenet_execution import resolve

                    expected_stride = (
                        17
                        if kind in ("dinov3_hf", "raev2_k7", "vjepa2_1")
                        else 1
                        if kind == "vggt_omega"
                        else 1000
                    )
                    self.assertEqual(resolve(cfg)["rank_seed_stride"], expected_stride)
                    self.assertAlmostEqual(
                        report["lr"], report["base_lr"] * 2 / report["reference_batch"]
                    )
                    cfg["scheduler_profile"] = optimization.REFERENCE_SCHEDULE
                    cfg["probe"]["batch_size"] = report["reference_batch"] // 2
                    report = optimization.resolve_optimization(cfg, imagenet.TASK)
                    self.assertEqual(report["epochs"], 100)
                    self.assertEqual(
                        report["effective_batch"], report["reference_batch"]
                    )

    def test_invalid_execution_and_ambiguous_distributed_source_are_refused(self):
        cfg = self.config()
        for change in (
            {"profile": "typo"},
            {"accumulation_steps": True},
            {"accumulation_steps": 0},
            {"accumulation_steps": 1.5},
            {"tail_policy": "flush_scaled"},
            {"precision": "fp16"},
            {"unknown": 1},
        ):
            bad = copy.deepcopy(cfg)
            bad["execution"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                imagenet.validate_config(bad)
        for change in ({"execution": None}, {"resume": None}, {"task": "ssv2_video"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                imagenet.validate_config(cfg | change)
        bad = copy.deepcopy(cfg)
        bad["probe"]["max_steps_per_epoch"] = 1
        with self.assertRaises(ValueError):
            imagenet.validate_config(bad)
        bad = copy.deepcopy(cfg)
        bad["probe"]["epochs"] = 101
        with self.assertRaises(ValueError):
            imagenet.validate_config(bad)
        from downstream.imagenet_execution import resolve

        bad = copy.deepcopy(cfg)
        bad["backbone"]["kind"] = "unknown"
        with self.assertRaises(ValueError):
            resolve(bad)
        env = {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}
        with mock.patch.dict(os.environ, env):
            report = optimization.resolve_optimization(cfg, imagenet.TASK)
            self.assertEqual(report["effective_batch"], 4)
            self.assertEqual(report["world_size"], 2)
            with self.assertRaises(ValueError):
                imagenet.validate_config(self.config(kind="vggt_omega"))
            legacy = copy.deepcopy(cfg)
            legacy.pop("execution")
            with self.assertRaises(ValueError):
                imagenet.validate_config(legacy)

    def test_finetune_accumulation_matches_source_updates_and_discards_tail(self):
        from downstream import extended_execution as execution

        for adaptation in ("frozen", "attentive", "finetune"):
            for algorithm in (torch.optim.SGD, torch.optim.AdamW):
                with self.subTest(adaptation=adaptation, algorithm=algorithm):
                    torch.manual_seed(3)
                    actual = torch.nn.Linear(3, 2)
                    expected = copy.deepcopy(actual)
                    batches = [torch.randn(2, 3) * 10 for _ in range(5)]
                    opt = algorithm(actual.parameters(), lr=0.02)
                    ref = algorithm(expected.parameters(), lr=0.02)
                    schedule = torch.optim.lr_scheduler.LambdaLR(
                        opt, lambda t: 1 / (t + 1)
                    )
                    rates = []
                    ref.zero_grad(set_to_none=True)
                    for i, x in enumerate(batches):
                        (expected(x).square().mean() / 2).backward()
                        if (i + 1) % 2 == 0:
                            if adaptation != "frozen":
                                torch.nn.utils.clip_grad_norm_(
                                    expected.parameters(), 1.0
                                )
                            ref.step()
                            ref.zero_grad(set_to_none=True)
                            rates.append(0.02 / (len(rates) + 2))
                            for group in ref.param_groups:
                                group["lr"] = rates[-1]
                    stats = execution.train_epoch(
                        actual,
                        batches,
                        lambda x, m=actual: m(x).square().mean(),
                        opt,
                        schedule,
                        accumulation_steps=2,
                        tail_policy="discard",
                        adaptation=adaptation,
                        trainable_backbone=adaptation == "finetune",
                    )
                    self.assertEqual(
                        stats,
                        {
                            "microbatches": 5,
                            "updates": 2,
                            "tail_updates": 0,
                            "discarded_microbatches": 1,
                        },
                    )
                    self.assertEqual(schedule.last_epoch, 2)
                    for a, b in zip(actual.parameters(), expected.parameters()):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                        self.assertIsNone(a.grad)

    def fixture(self, root):
        images.TestImageNet().fixture(root)
        for split in ("train", "val"):
            for folder in (root / split).iterdir():
                for i in (1, 2):
                    (folder / f"{i}.png").write_bytes((folder / "0.png").read_bytes())

    def equal_tree(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a:
                self.equal_tree(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.equal_tree(x, y)
        else:
            self.assertEqual(a, b)

    def test_lp_ap_ft_resume_matches_uninterrupted_model_optimizer_and_rng(self):
        if not images.HAVE:
            self.skipTest("real encoder dependencies required")
        for adaptation in ("frozen", "attentive", "finetune"):
            with (
                self.subTest(adaptation=adaptation),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                data = root / "data"
                self.fixture(data)
                cfg = self.config(data, adaptation)
                cfg["probe"]["epochs"] = 3
                for name, epochs, resume in (
                    ("full", 3, None),
                    ("first", 1, None),
                    ("resumed", 3, root / "first/resume.pt"),
                ):
                    current = copy.deepcopy(cfg)
                    current["probe"]["epochs"] = epochs
                    if resume:
                        current["resume"] = str(resume)
                    path = root / f"{name}.json"
                    path.write_text(json.dumps(current))
                    self.assertEqual(
                        imagenet.main(
                            ["--config", str(path), "--out", str(root / name)]
                        ),
                        0,
                    )
                    self.assertTrue((root / name / "resume.pt").is_file())
                    from downstream import contract

                    self.assertTrue(contract.verify(root / name, path, 0)[0])
                full = torch.load(root / "full/resume.pt", weights_only=True)
                resumed = torch.load(root / "resumed/resume.pt", weights_only=True)
                self.equal_tree(full, resumed)
                if adaptation == "finetune":
                    exported = torch.load(
                        root / "full/finetune_model.pt", weights_only=True
                    )["model"]
                else:
                    exported = torch.load(root / "full/probe.pt", weights_only=True)
                self.equal_tree(full["model"], exported)
                self.assertEqual(full["runtime"]["updates"], 6)
                self.assertEqual(full["runtime"]["discarded_microbatches"], 3)
                self.assertEqual(
                    any(k.startswith("backbone.") for k in full["model"]),
                    adaptation == "finetune",
                )
                report = json.loads((root / "resumed/results.json").read_text())
                self.assertEqual(report["continuation"]["start_epoch"], 1)
                self.assertFalse(report["continuation"]["legacy_native_compatible"])
                self.assertEqual(report["final"]["images"], 6)
                self.assertFalse(report["canonical_eligible"])
                self.assertFalse(report["record_value"])

    def test_all_real_model_routes_keep_expected_training_scope(self):
        from tests import test_method_hf_basic5 as hf
        from tests import test_method_patch_basic5 as patch

        if not (images.HAVE and hf.HAVE and patch.HAVE):
            self.skipTest("full encoder dependencies required")
        from tests.test_method_cradiov4_h import TestRadio
        from tests.test_method_raev2 import TestK7Backbone
        from tests.test_method_vggt_omega import TestOmega

        for kind in ft.PROFILES:
            owner = (
                hf.TestVisionFamilies()
                if kind in ("clip_hf", "siglip2_g", "dinov3_hf")
                else patch.TestPatchFamilies()
                if kind == "cosmos3_super_vm"
                else TestRadio()
                if kind == "cradiov4_h"
                else TestK7Backbone()
                if kind == "raev2_k7"
                else TestOmega()
                if kind == "vggt_omega"
                else None
            )
            if owner:
                owner.setUp()
            try:
                for adaptation in ("frozen", "attentive", "finetune"):
                    with (
                        self.subTest(kind=kind, adaptation=adaptation),
                        tempfile.TemporaryDirectory() as tmp,
                    ):
                        root = Path(tmp)
                        data = root / "data"
                        self.fixture(data)
                        cfg = self.config(data, adaptation, kind)
                        if owner:
                            cfg["backbone"], _ = (
                                owner.fixture(kind)
                                if kind
                                in (
                                    "clip_hf",
                                    "siglip2_g",
                                    "dinov3_hf",
                                    "cosmos3_super_vm",
                                )
                                else owner.fixture()
                            )
                        original = imagenet.ImageClassifier
                        models = []
                        states = []

                        def build(
                            *args,
                            original=original,
                            models=models,
                            states=states,
                            **kwargs,
                        ):
                            model = original(*args, **kwargs)
                            models.append(model)
                            states.append(copy.deepcopy(model.state_dict()))
                            return model

                        with mock.patch.object(imagenet, "ImageClassifier", build):
                            imagenet.run(cfg, root / "out")
                        model = models[0]
                        initial = states[0]
                        changed = any(
                            not torch.equal(v, initial["backbone." + k])
                            for k, v in model.backbone.state_dict().items()
                        )
                        self.assertEqual(changed, adaptation == "finetune")
                        self.assertFalse(
                            torch.equal(
                                model.classifier.weight, initial["classifier.weight"]
                            )
                        )
                        self.assertTrue(all(p.grad is None for p in model.parameters()))
                        report = json.loads((root / "out/results.json").read_text())
                        self.assertEqual(report["execution"]["updates"], 2)
                        self.assertEqual(
                            report["execution"]["discarded_microbatches"], 1
                        )
            finally:
                if owner:
                    owner.doCleanups()

    def test_identity_corruption_bf16_and_existing_output_fail_closed(self):
        if not images.HAVE:
            self.skipTest("real encoder dependencies required")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            self.fixture(data)
            cfg = self.config(data, "finetune")
            imagenet.run(cfg, root / "first")
            source = root / "first/resume.pt"
            original = source.read_bytes()
            cfg["resume"] = str(source)
            cfg["probe"]["epochs"] = 2
            mutations = []
            for section, key, value in (
                ("execution", "precision", "bf16"),
                ("execution", "accumulation_steps", 1),
                ("probe", "batch_size", 2),
                ("probe", "image_size", 48),
            ):
                bad = copy.deepcopy(cfg)
                bad[section][key] = value
                mutations.append(bad)
            mutations.append(cfg | {"seed": 1})
            for i, bad in enumerate(mutations):
                path = root / f"bad{i}.json"
                path.write_text(json.dumps(bad))
                out = root / f"bad{i}"
                self.assertEqual(
                    imagenet.main(["--config", str(path), "--out", str(out)]), 1
                )
                self.assertEqual(
                    json.loads((out / "run_manifest.json").read_text())["status"],
                    "failed",
                )
                self.assertFalse((out / "resume.pt").exists())
            valid = torch.load(source, weights_only=True)
            for field in ("model", "optimizer", "scheduler", "rng"):
                corrupt = copy.deepcopy(valid)
                corrupt.pop(field)
                badpath = root / f"corrupt-{field}.pt"
                torch.save(corrupt, badpath)
                with self.assertRaises(ValueError):
                    imagenet.run(
                        cfg | {"resume": str(badpath)}, root / f"refused-{field}"
                    )
            image = data / "train/n00000001/0.png"
            image.write_bytes(image.read_bytes() + b"changed-input")
            with self.assertRaisesRegex(ValueError, "identity"):
                imagenet.run(cfg, root / "changed-data")
            self.assertEqual(source.read_bytes(), original)
            path = root / "existing.json"
            path.write_text(json.dumps(cfg))
            before = sorted((root / "first").iterdir())
            self.assertEqual(
                imagenet.main(["--config", str(path), "--out", str(root / "first")]), 1
            )
            self.assertEqual(before, sorted((root / "first").iterdir()))
            self.assertEqual(source.read_bytes(), original)

    def test_device_override_and_autocast_cover_train_and_evaluation(self):
        if not images.HAVE:
            self.skipTest("real encoder dependencies required")
        from downstream import extended_execution as execution

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root / "data")
            cfg = self.config(root / "data")
            cfg["device"] = "cuda"
            path = root / "config.json"
            path.write_text(json.dumps(cfg))
            self.assertEqual(
                imagenet.main(
                    [
                        "--config",
                        str(path),
                        "--out",
                        str(root / "override"),
                        "--device",
                        "cpu",
                    ]
                ),
                0,
            )
            cfg["device"] = "cpu"
            cfg["execution"]["precision"] = "bf16"
            active = []
            losses = []
            original_forward = imagenet.ImageClassifier.forward
            original_loss = torch.nn.functional.cross_entropy

            def forward(model, x):
                active.append(torch.is_autocast_enabled("cpu"))
                return original_forward(model, x)

            def loss(logits, y):
                losses.append(logits.dtype)
                return original_loss(logits, y)

            def autocast(device, precision):
                self.assertEqual(device.type, "cpu")
                self.assertEqual(precision, "bf16")
                return torch.autocast("cpu", dtype=torch.bfloat16)

            with (
                mock.patch.object(execution, "autocast_context", side_effect=autocast),
                mock.patch.object(imagenet.ImageClassifier, "forward", forward),
                mock.patch.object(
                    torch.nn.functional, "cross_entropy", side_effect=loss
                ),
            ):
                imagenet.run(cfg, root / "autocast")
            self.assertEqual(len(active), 11)
            self.assertTrue(all(active))
            self.assertEqual(losses, [torch.float32] * 5)

    def test_execution_ft_uses_soft_targets_and_rejects_invalid_membership(self):
        if not images.HAVE:
            self.skipTest("real encoder dependencies required")
        from downstream import extended_execution as execution
        from downstream.imagenet_execution import _data

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root / "data")
            cfg = self.config(root / "data", "finetune")
            cfg["probe"]["epochs"] = 2
            logits = []
            targets = []
            checked = []
            original_forward = imagenet.ImageClassifier.forward
            original_epoch = execution.train_epoch

            def forward(model, x):
                value = original_forward(model, x)
                logits.append(value.detach().float())
                return value

            def mix(x, y, classes, recipe):
                target = 0.6 * torch.nn.functional.one_hot(
                    y, classes
                ) + 0.4 * torch.nn.functional.one_hot((y + 1) % classes, classes)
                targets.append(target.float())
                return x, target.float()

            def epoch(model, loader, fn, *args, **kwargs):
                def wrapped(batch):
                    result = fn(batch)
                    expected = (
                        -(targets[-1] * torch.nn.functional.log_softmax(logits[-1], -1))
                        .sum(-1)
                        .mean()
                    )
                    torch.testing.assert_close(
                        result.detach(), expected, rtol=0, atol=0
                    )
                    checked.append(float(result.detach()))
                    return result

                return original_epoch(model, loader, wrapped, *args, **kwargs)

            with (
                mock.patch.object(imagenet.ImageClassifier, "forward", forward),
                mock.patch.object(imagenet.imagenet_finetune, "mix", side_effect=mix),
                mock.patch.object(execution, "train_epoch", side_effect=epoch),
            ):
                imagenet.run(cfg, root / "soft")
            self.assertEqual(len(checked), 10)
            self.assertEqual(len(targets), 10)
            source = imagenet.ImageNetImages

            def mismatch(*args, **kwargs):
                data = source(*args, **kwargs)
                if args[1] == "val":
                    data.class_to_idx = {"wrong": 0}
                return data

            with (
                mock.patch.object(imagenet, "ImageNetImages", side_effect=mismatch),
                self.assertRaises(ValueError),
            ):
                _data(cfg, imagenet)

            def too_many(*args, **kwargs):
                data = source(*args, **kwargs)
                data.classes = list(range(1001))
                return data

            with (
                mock.patch.object(imagenet, "ImageNetImages", side_effect=too_many),
                self.assertRaises(ValueError),
            ):
                _data(cfg, imagenet)

    def test_two_cpu_ranks_lp_ap_ft_resume_and_full_evaluation(self):
        if not images.HAVE:
            self.skipTest("real encoder dependencies required")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root / "data")
            for adaptation in ("frozen", "attentive", "finetune"):
                cfg = self.config(root / "data", adaptation)
                cfg["probe"].update(epochs=3, max_train_samples=0, max_val_samples=5)
                (root / f"{adaptation}.json").write_text(json.dumps(cfg))
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--master-addr=127.0.0.1",
                    f"--master-port={port}",
                    "--nproc-per-node=2",
                    "--module",
                    "tests." + Path(__file__).stem,
                    "--worker",
                    tmp,
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                output, _ = proc.communicate(timeout=150)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                output, _ = proc.communicate(timeout=10)
                self.fail("distributed worker timed out: " + output[-12000:])
            self.assertEqual(proc.returncode, 0, output[-12000:])
            for adaptation in ("frozen", "attentive", "finetune"):
                full = torch.load(
                    root / f"{adaptation}-full/resume.pt", weights_only=True
                )
                resumed = torch.load(
                    root / f"{adaptation}-resumed/resume.pt", weights_only=True
                )
                self.equal_tree(full, resumed)
                self.assertEqual(len(full["rng"]), 2)
                self.assertEqual(full["runtime"]["updates"], 3)
                self.assertEqual(full["runtime"]["effective_batch"], 4)
                report = json.loads(
                    (root / f"{adaptation}-resumed/results.json").read_text()
                )
                self.assertEqual(report["final"]["images"], 5)
                self.assertEqual(report["execution"]["world_size"], 2)

    def test_direct_execution_preserves_existing_output_bytes(self):
        if not images.HAVE:
            self.skipTest("real encoder dependencies required")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root / "data")
            out = root / "existing"
            out.mkdir()
            (out / "results.json").write_bytes(b"previous result")
            with self.assertRaises(FileExistsError):
                imagenet.run(self.config(root / "data"), out)
            self.assertEqual((out / "results.json").read_bytes(), b"previous result")
            self.assertEqual([p.name for p in out.iterdir()], ["results.json"])


class Delivery(unittest.TestCase):
    def test_ci_runs_execution(self):
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import HAVE_YAML, parsed

        if not HAVE_YAML:
            self.skipTest("CI YAML parser required")
        command = next(
            s["run"]
            for s in parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
            if s.get("name")
            == "Run Basic5 component contracts with downstream dependencies"
        )
        self.assertTrue(
            _runs_finetune_tests(command, module="tests.test_method_imagenet_execution")
        )

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "docs").is_dir(), "docs unavailable"
    )
    def test_documented_example_selects_profile(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "docs/examples/imagenet_execution.json"
        )
        self.assertTrue(path.is_file())
        cfg = json.loads(path.read_text())
        self.assertEqual(cfg["execution"]["profile"], "captured_imagenet_v1")
        if HAVE:
            imagenet.validate_config(cfg)


def worker(root):
    root = Path(root)
    torch.set_num_threads(1)
    from downstream import extended_distributed as distributed

    # Reuse one owned process group for the nine independent runner invocations.
    # Rapid destroy/reinitialize cycles can race on the same torchrun store.
    with distributed.session("cpu"):
        for adaptation in ("frozen", "attentive", "finetune"):
            cfg = json.loads((root / f"{adaptation}.json").read_text())
            for name, epochs, resume in (
                ("full", 3, None),
                ("first", 1, None),
                ("resumed", 3, root / f"{adaptation}-first/resume.pt"),
            ):
                current = copy.deepcopy(cfg)
                current["probe"]["epochs"] = epochs
                if resume:
                    current["resume"] = str(resume)
                rc = distributed.cli(
                    json.dumps(current).encode(),
                    root / f"{adaptation}-{name}",
                    imagenet.run,
                    imagenet.TASK,
                )
                if rc:
                    raise RuntimeError(f"{adaptation}/{name} failed")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        unittest.main()
