"""Online image classification components and their fail-closed boundary."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    import timm
    import einops
    from torchvision import transforms as T
    from PIL import Image
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "Image classification dependencies required")
class TestImageNet(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("downstream.imagenet"),
                             "online ImageNet component runner is missing")
        return importlib.import_module("downstream.imagenet")

    def config(self, root):
        from tests.test_method_vjepa2_1_training import TestFinetuneOptimization
        return dict(task="imagenet_classification", seed=0, device="cpu", data_root=str(root),
                    profile="capture_basic5_components", adaptation="frozen",
                    optimizer_profile="basic5_frozen_v1",
                    backbone=TestFinetuneOptimization().config()["backbone"],
                    probe=dict(epochs=1, batch_size=2, lr="protocol", num_workers=0,
                               image_size=32, max_train_samples=2, max_val_samples=2,
                               max_steps_per_epoch=0))

    def test_dataset_geometry_and_normalization_match_reference_operations(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            for train in (False, True):
                ds = m.ImageNetImages(root, "train" if train else "val", 32)
                image = Image.open(ds.samples[0][0]).convert("RGB")
                torch.manual_seed(7)
                if train:
                    params = T.RandomResizedCrop.get_params(image, (.08, 1.), (.75, 4./3.))
                    image = T.functional.resized_crop(image, *params, [32, 32], T.InterpolationMode.BILINEAR)
                    if torch.rand(1).item() < .5:
                        image = T.functional.hflip(image)
                else:
                    image = T.functional.center_crop(T.functional.resize(image, 256,
                            T.InterpolationMode.BILINEAR), [32, 32])
                expected = T.functional.normalize(T.functional.to_tensor(image),
                                                  [.485, .456, .406], [.229, .224, .225])
                torch.manual_seed(7)
                actual, label = ds[0]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(label, 0)

    def fixture(self, root):
        import numpy as np
        image = np.random.RandomState(9).randint(0, 256, (39, 57, 3), dtype="uint8")
        for split in ("train", "val"):
            for name in ("n00000001", "n00000002"):
                folder = root / split / name
                folder.mkdir(parents=True)
                Image.fromarray(image).save(folder / "0.png")

    def test_optimizer_and_schedule_for_both_supported_tracks(self):
        from downstream import optimization as opt
        for adaptation in ("frozen", "attentive"):
            cfg = self.config(Path("/unused"))
            cfg.update(adaptation=adaptation, scheduler_profile=opt.REFERENCE_SCHEDULE)
            cfg["probe"].update(epochs=100, batch_size=256)
            report = opt.resolve_optimization(cfg, cfg["task"])
            self.assertEqual(report["optimizer"], "SGD" if adaptation == "frozen" else "AdamW")
            self.assertEqual(report["weight_decay"], 0. if adaptation == "frozen" else .05)
            optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=report["lr"])
            schedule = opt.build_task_scheduler(optimizer, report, 2)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], .1 if adaptation == "frozen" else 1e-6)
            for _ in range(200):
                optimizer.step()
                schedule.step()
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0. if adaptation == "frozen" else 1e-6)

    def test_real_frozen_and_attentive_clis_preserve_contract_and_backbone(self):
        from downstream import contract
        from unittest import mock
        m = self.module()
        for adaptation in ("frozen", "attentive"):
            with self.subTest(adaptation=adaptation), tempfile.TemporaryDirectory() as d:
                root, out, path = Path(d)/"data", Path(d)/"out", Path(d)/"config.json"
                self.fixture(root)
                cfg = self.config(root)
                cfg["adaptation"] = adaptation
                path.write_text(json.dumps(cfg))
                models, states = [], []
                original = m.ImageClassifier
                def create(*args, **kwargs):
                    model = original(*args, **kwargs)
                    models.append(model)
                    states.append(copy.deepcopy(model.state_dict()))
                    return model
                with mock.patch.object(m, "ImageClassifier", create):
                    self.assertEqual(m.main(["--config", str(path), "--out", str(out)]), 0)
                model = models[0]
                for name, value in model.backbone.state_dict().items():
                    torch.testing.assert_close(value, states[0]["backbone."+name], rtol=0, atol=0)
                self.assertFalse(torch.equal(model.classifier.weight, states[0]["classifier.weight"]))
                result = json.loads((out/"results.json").read_text())
                self.assertFalse(result["canonical_eligible"])
                self.assertFalse(result["record_value"])
                self.assertEqual(result["num_classes"], 1000)
                self.assertTrue(contract.verify(out, path, 0)[0])

    def test_verified_readout_and_gradients_for_all_three_model_components(self):
        from downstream import spatial_backbones as sb
        m = self.module()
        for adaptation in ("frozen", "attentive", "finetune"):
            builder = sb.build_trainable_backbone if adaptation == "finetune" else sb.build_frozen_backbone
            backbone = builder(self.config(Path("/unused"))["backbone"], torch.device("cpu"))
            model = m.ImageClassifier(backbone, adaptation)
            x = torch.randn(2, 3, 32, 32)
            tokens = backbone.forward_features(x).flatten(2).transpose(1, 2)
            expected = model.reader(tokens) if adaptation == "attentive" else torch.nn.functional.normalize(tokens.mean(1), dim=-1)
            with torch.no_grad():
                model.classifier.weight.normal_()
            torch.testing.assert_close(model(x), model.classifier(expected), rtol=0, atol=0)
            model(x).square().mean().backward()
            self.assertEqual(any(p.grad is not None for p in backbone.parameters()), adaptation == "finetune")

    def test_refuses_unknown_recipe_provider_and_class_mapping(self):
        m = self.module()
        cfg = self.config(Path("/absent"))
        for change in ({"adaptation": "finetune"}, {"profile": "legacy"},
                       {"backbone": {"kind": "vit"}}, {"mystery": True},
                       {"device": "typo"}, {"scheduler_profile": "typo"},
                       {"task": "wrong"}):
            with self.assertRaises(ValueError):
                m.validate_config(cfg | change)
        for key, value in (("image_size", 0), ("num_workers", -1), ("epochs", True),
                           ("batch_size", 1.5), ("lr", .1), ("unknown", 1)):
            invalid = copy.deepcopy(cfg)
            invalid["probe"][key] = value
            with self.assertRaises(ValueError):
                m.validate_config(invalid)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/"data"
            self.fixture(root)
            (root/"val/n00000002").rename(root/"val/n00000003")
            cfg = self.config(root)
            with self.assertRaisesRegex(ValueError, "class mapping"):
                m.run(cfg, Path(d)/"out")

    def test_nonfinite_training_fails_without_update_and_records_failure(self):
        from unittest import mock
        m = self.module()
        with tempfile.TemporaryDirectory() as d:
            root, out, path = Path(d)/"data", Path(d)/"out", Path(d)/"config.json"
            self.fixture(root)
            path.write_text(json.dumps(self.config(root)))
            with mock.patch.object(m.F, "cross_entropy", return_value=torch.tensor(float("nan"))), \
                 mock.patch.object(torch.optim.SGD, "step") as step:
                self.assertEqual(m.main(["--config", str(path), "--out", str(out)]), 1)
                step.assert_not_called()
            manifest = json.loads((out/"run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("non-finite", manifest["error"])
            self.assertFalse((out/"results.json").exists())

    def test_real_cli_reference_schedule_advances_after_updates(self):
        from unittest import mock
        m = self.module()
        with tempfile.TemporaryDirectory() as d:
            root, out, path = Path(d)/"data", Path(d)/"out", Path(d)/"config.json"
            self.fixture(root)
            for directory in (root/"train").iterdir():
                for i in range(1, 128):
                    (directory/f"{i}.png").write_bytes((directory/"0.png").read_bytes())
            cfg = self.config(root)
            cfg["probe"].update(batch_size=256, epochs=100, max_train_samples=0)
            cfg["scheduler_profile"] = "basic5_reference_schedule_v1"
            path.write_text(json.dumps(cfg))
            rates, original = [], torch.optim.SGD.step
            def step(optimizer, *args, **kwargs):
                rates.append(optimizer.param_groups[0]["lr"])
                return original(optimizer, *args, **kwargs)
            with mock.patch.object(torch.optim.SGD, "step", step):
                self.assertEqual(m.main(["--config", str(path), "--out", str(out)]), 0)
            self.assertEqual(len(rates), 100)
            self.assertEqual(rates[0], .1)
            self.assertGreater(rates[0], rates[-1])
            report = json.loads((out/"results.json").read_text())["optimization"]["schedule"]
            self.assertEqual(report["updates_completed"], 100)
            self.assertEqual(report["next_update_lr"], [0.])
