"""SAM trunk and final-merger Basic5 boundaries on real local tiny models."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
from tests._checkout import needs_checkout

try:
    import torch
    from torch.nn import functional as F
    from transformers import (
        Sam3ViTConfig,
        Sam3ViTModel,
        Qwen3VLVisionConfig,
        Qwen3VLVisionModel,
    )

    HAVE = True
except ImportError:
    HAVE = False

KINDS = ("sam3_trunk", "cosmos3_super_vm")


@unittest.skipUnless(HAVE, "torch and vision model dependencies required")
class TestPatchFamilies(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(47)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self, kind):
        if kind == "sam3_trunk":
            cfg = Sam3ViTConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                patch_size=4,
                image_size=32,
                pretrain_image_size=32,
                window_size=2,
                global_attn_indexes=[1],
            )
            model = Sam3ViTModel(cfg)
        else:
            cfg = Qwen3VLVisionConfig(
                hidden_size=32,
                intermediate_size=64,
                depth=2,
                num_heads=4,
                patch_size=4,
                temporal_patch_size=2,
                spatial_merge_size=2,
                out_hidden_size=24,
                num_position_embeddings=64,
                deepstack_visual_indexes=[0],
            )
            model = Qwen3VLVisionModel(cfg)
        model.eval()
        folder = self.root / kind
        model.save_pretrained(folder)
        return (
            dict(
                kind=kind,
                arch="fixture",
                encoder=str(folder),
                img_size=32,
                patch_size=4,
            ),
            model,
        )

    def build(self, spec, trainable=False):
        from downstream import spatial_backbones as sb

        self.assertIn(spec["kind"], sb.KINDS, "Basic5 provider missing")
        return (sb.build_trainable_backbone if trainable else sb.build_frozen_backbone)(
            spec, torch.device("cpu")
        )

    def expected(self, kind, model, x):
        if kind == "sam3_trunk":
            from transformers.models.sam3.modeling_sam3 import Sam3ViTRotaryEmbedding

            pixel = F.pad(x, (0, -x.shape[-1] % 4, 0, -x.shape[-2] % 4))
            h, w = pixel.shape[-2] // 4, pixel.shape[-1] // 4
            for layer in model.layers:
                if layer.window_size == 0:
                    layer.rotary_emb = Sam3ViTRotaryEmbedding(
                        model.config, w, h, scale=2 / h
                    )
            tokens = model(pixel_values=pixel).last_hidden_state
        else:
            raw = (
                x * x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
                + x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            )
            pixel = (
                F.pad(raw, (0, -x.shape[-1] % 8, 0, -x.shape[-2] % 8), value=0.5) - 0.5
            ) / 0.5
            # Independent explicit merge-window/patch/channel/time oracle.
            h, w = pixel.shape[-2] // 8, pixel.shape[-1] // 8
            patches = torch.stack(
                [
                    pixel[b, :, yy : yy + 4, xx : xx + 4]
                    .unsqueeze(1)
                    .repeat(1, 2, 1, 1)
                    .flatten()
                    for b in range(x.shape[0])
                    for by in range(h)
                    for bx in range(w)
                    for yy in (by * 8, by * 8 + 4)
                    for xx in (bx * 8, bx * 8 + 4)
                ]
            )
            grid = torch.tensor([[1, h * 2, w * 2]] * x.shape[0])
            out = model(hidden_states=patches, grid_thw=grid)
            tokens = out.pooler_output.reshape(x.shape[0], h * w, 24)
        return tokens.float().transpose(1, 2).reshape(x.shape[0], -1, h, w)

    def test_spatial_pooling_video_gradients_and_updates(self):
        for kind in KINDS:
            spec, reference = self.fixture(kind)
            bb = self.build(spec, True)
            opt = torch.optim.SGD(bb.parameters(), lr=0.01)
            ref_opt = torch.optim.SGD(reference.parameters(), lr=0.01)
            for step in range(3):
                raw = torch.rand(2, 3, 19 + step, 29 + step)
                x = (
                    (raw - raw.new_tensor([0.485, 0.456, 0.406])[None, :, None, None])
                    / raw.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
                ).requires_grad_()
                y = x.detach().clone().requires_grad_()
                actual = bb.forward_features(x)
                expected = self.expected(kind, reference, y)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
                weights = torch.randn_like(actual)
                opt.zero_grad()
                ref_opt.zero_grad()
                (actual * weights).sum().backward()
                (expected * weights).sum().backward()
                torch.testing.assert_close(x.grad, y.grad, atol=4e-5, rtol=4e-5)
                opt.step()
                ref_opt.step()
            for a, b in zip(bb.model.parameters(), reference.parameters()):
                torch.testing.assert_close(a, b, atol=4e-5, rtol=4e-5)
            x = x.detach()
            pooled = self.expected(kind, reference, x).mean((2, 3))
            torch.testing.assert_close(
                bb.classification_features(x, adaptation="frozen"),
                F.normalize(pooled, dim=-1),
                atol=4e-5,
                rtol=4e-5,
            )
            torch.testing.assert_close(
                bb.classification_features(x, adaptation="finetune"),
                pooled,
                atol=4e-5,
                rtol=4e-5,
            )
            video = x.unsqueeze(0)
            torch.testing.assert_close(
                bb.classification_features(video, adaptation="frozen", video=True),
                pooled.mean(0, keepdim=True),
                atol=4e-5,
                rtol=4e-5,
            )
            with torch.no_grad():
                self.assertFalse(bb.forward_features(x).requires_grad)

    def test_frozen_state_and_parameter_policy(self):
        from downstream.spatial_backbones import supports_capture_pyramid

        for kind in KINDS:
            spec, _ = self.fixture(kind)
            frozen = self.build(spec)
            frozen.train()
            self.assertFalse(frozen.training)
            self.assertFalse(any(p.requires_grad for p in frozen.parameters()))
            image = torch.rand(1, 3, 19, 29, requires_grad=True)
            self.assertFalse(frozen.forward_features(image).requires_grad)
            self.assertFalse(
                frozen.classification_features(image, adaptation="frozen").requires_grad
            )
            self.assertFalse(supports_capture_pyramid(kind))
            bb = self.build(spec, True)
            endpoint, policy = bb.finetune_group_policy()
            self.assertEqual(endpoint, 2)
            self.assertEqual(set(policy), set(dict(bb.named_parameters())))
            for name, (layer, nd) in policy.items():
                if ".layers.0." in name or ".blocks.0." in name:
                    self.assertEqual(layer, 1)
                if ".layers.1." in name or ".blocks.1." in name:
                    self.assertEqual(layer, 2)
                if "merger." in name and "deepstack" not in name:
                    self.assertEqual(layer, 2)
                if "deepstack_merger_list.0." in name:
                    self.assertEqual(layer, 1)
                if name.endswith("bias") or "norm" in name:
                    self.assertTrue(nd)
                if "position_embeddings" in name or ".pos_embed." in name:
                    self.assertEqual((layer, nd), (0, True))
                if name.endswith(".weight") and any(
                    token in name for token in ("attention.q_proj", "attn.qkv")
                ):
                    self.assertFalse(nd)
            with self.assertRaisesRegex(ValueError, "reader|attentive"):
                bb.classification_features(
                    torch.rand(1, 3, 32, 32), adaptation="attentive"
                )

    def test_invalid_inputs_outputs_and_policy_are_refused(self):
        for kind in KINDS:
            spec, _ = self.fixture(kind)
            bb = self.build(spec, True)
            for x in (torch.rand(1, 4, 32, 32), torch.rand(1, 3, 0, 32)):
                with self.assertRaisesRegex(ValueError, "images"):
                    bb.forward_features(x)
            with mock.patch.object(
                bb.model,
                "forward",
                return_value=SimpleNamespace(
                    last_hidden_state=torch.zeros(1, 0, 32), pooler_output=None
                ),
            ):
                with self.assertRaisesRegex(ValueError, "grid"):
                    bb.forward_features(torch.rand(1, 3, 32, 32))
                with self.assertRaisesRegex(ValueError, "grid"):
                    bb.classification_features(
                        torch.rand(1, 3, 32, 32), adaptation="frozen"
                    )
            bb.depth += 1
            with self.assertRaisesRegex(ValueError, "blocks"):
                bb.finetune_group_policy()

    def test_global_rope_updates_buffers_without_replacing_modules(self):
        spec, _ = self.fixture("sam3_trunk")
        bb = self.build(spec)
        original = [layer.rotary_emb for layer in bb.model.layers]
        window = original[0].rope_embeddings_cos.clone()
        for hw in ((19, 29), (32, 16), (19, 29)):
            bb.forward_features(torch.rand(1, 3, *hw))
            self.assertTrue(
                all(a is b.rotary_emb for a, b in zip(original, bb.model.layers))
            )
            torch.testing.assert_close(
                original[0].rope_embeddings_cos, window, atol=0, rtol=0
            )
            hp, wp = (hw[0] + 3) // 4, (hw[1] + 3) // 4
            self.assertEqual(
                (original[1].end_x, original[1].end_y, original[1].scale),
                (wp, hp, 2 / hp),
            )

    def test_strict_local_loading_and_architecture(self):
        from safetensors.torch import load_file, save_file

        for kind in KINDS:
            spec, _ = self.fixture(kind)
            if kind == "sam3_trunk":
                with self.assertRaisesRegex(ValueError, "image_size"):
                    self.build(dict(spec, img_size=224))
            for update in (
                dict(encoder=""),
                dict(encoder="remote/model"),
                dict(arch="released"),
                dict(arch="typo"),
                dict(patch_size=8),
            ):
                with self.subTest(kind=kind, update=update), self.assertRaises(
                    (ValueError, RuntimeError)
                ):
                    self.build(dict(spec, **update))
            path = Path(spec["encoder"]) / "model.safetensors"
            state = load_file(path)
            original = copy.deepcopy(state)
            state.pop(next(iter(state)))
            save_file(state, path)
            with self.assertRaisesRegex(
                (ValueError, RuntimeError), "missing|incomplete"
            ):
                self.build(spec)
            original["unexpected_tower.weight"] = torch.ones(2)
            save_file(original, path)
            with self.assertRaisesRegex(ValueError, "unexpected|incomplete"):
                self.build(spec)
            raw = json.loads((Path(spec["encoder"]) / "config.json").read_text())
            raw["model_type"] = "wrong_family"
            (Path(spec["encoder"]) / "config.json").write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "family"):
                self.build(spec)

    def test_official_trunk_file_loads_without_changing_extraction(self):
        from tests.test_method_sam3 import TestTheTrunkConverter

        raw = TestTheTrunkConverter()._official_trunk(H=64, depth=2, mlp=128, grid=2)
        path = self.root / "trunk.pt"
        torch.save(raw, path)
        spec = dict(
            kind="sam3_trunk",
            encoder=str(path),
            arch="fixture",
            img_size=28,
            patch_size=14,
        )
        bb = self.build(spec, True)
        from methods.sam3.sam3_trunk import load_official_trunk

        reference = load_official_trunk(str(path), img_size=28)
        x = torch.rand(2, 3, 28, 28)
        expected = (
            reference(pixel_values=x)
            .last_hidden_state.transpose(1, 2)
            .reshape(2, 64, 2, 2)
        )
        torch.testing.assert_close(bb.forward_features(x), expected)
        with self.assertRaises(ValueError):
            self.build(dict(spec, arch="released"))
        with self.assertRaises(ValueError):
            self.build(dict(spec, patch_size=16))
        with self.assertRaisesRegex(ValueError, "arch"):
            self.build(dict(spec, arch="typo"))

    def test_documented_examples(self):
        from downstream.imagenet import validate_config
        from tests.test_basic5_imagenet import TestImageNet

        path = Path(__file__).resolve().parents[1] / "docs/BASIC5_PATCH_PROVIDERS.md"
        self.assertTrue(path.is_file(), "provider documentation missing")
        specs = json.loads(path.read_text().split("```json\n")[1].split("```")[0])
        self.assertEqual({s["kind"] for s in specs}, set(KINDS))
        for spec in specs:
            cfg = TestImageNet().config(Path("/data/imagenet"))
            cfg["backbone"] = spec
            validate_config(cfg)

    @needs_checkout
    def test_ci_coverage(self):
        from tests.test_ci import parsed, HAVE_YAML

        if not HAVE_YAML:
            self.skipTest("PyYAML required")
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests

        commands = [
            s["run"]
            for s in parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
            if s.get("name")
            == "Run Basic5 component contracts with downstream dependencies"
        ]
        self.assertEqual(len(commands), 1)
        self.assertTrue(
            _runs_finetune_tests(commands[0], module="tests.test_method_patch_basic5")
        )

    def test_task_entrypoints(self):
        from tests import test_basic5_optimization as helper, test_basic5_imagenet as ih

        if not (helper.HAVE and ih.HAVE):
            self.skipTest("Full downstream dependencies required")
        from downstream import imagenet, contract
        from scipy.io import savemat

        for kind in KINDS:
            spec, _ = self.fixture(kind)
            cases = list(helper.TestOptimization().cases()) + [
                (imagenet, ih.TestImageNet().fixture, ih.TestImageNet().config)
            ]
            for module, fixture, factory in cases:
                if module is helper.coco:
                    continue
                for adaptation in ("frozen", "finetune"):
                    if module is imagenet and adaptation == "finetune":
                        continue
                    with self.subTest(
                        kind=kind, task=module.TASK, adaptation=adaptation
                    ):
                        folder = self.root / kind / module.TASK / adaptation
                        root, path, out = (
                            folder / "data",
                            folder / "cfg.json",
                            folder / "out",
                        )
                        fixture(root)
                        if module is helper.nyuv2:
                            savemat(
                                root / "labeled/splits.mat",
                                {
                                    "trainNdxs": [[1], [2], [3]],
                                    "testNdxs": [[4], [5], [6]],
                                },
                            )
                        cfg = (
                            factory(root)
                            if module is imagenet
                            else helper.TestOptimization().config(
                                module, factory, root, adaptation
                            )
                        )
                        cfg["backbone"] = spec
                        if adaptation == "finetune":
                            cfg["optimizer_profile"] = "basic5_finetune_v1"
                        cfg["probe"].update(epochs=1, max_val_samples=2)
                        path.write_text(json.dumps(cfg))
                        self.assertEqual(
                            module.main(["--config", str(path), "--out", str(out)]), 0
                        )
                        self.assertTrue(contract.verify(out, path, 0)[0])
                        self.assertFalse(
                            json.loads((out / "results.json").read_text())[
                                "canonical_eligible"
                            ]
                        )
