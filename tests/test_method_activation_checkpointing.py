"""Recomputation must preserve observable outputs, gradients and state keys."""

import unittest

try:
    import torch

    from downstream import spatial_backbones as sb
    from tests import test_method_hf_basic5 as hf
    from tests import test_method_patch_basic5 as patch
    from tests import test_method_raev2 as rae
    from tests import test_method_vjepa2_1 as video
    from tests.test_basic5_readers import SPEC

    HAVE = hf.HAVE and patch.HAVE and rae.HAVE and video.HAVE
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "Torch and vision dependencies required")
class TestActivationCheckpointing(unittest.TestCase):
    def fixtures(self):
        for owner, kinds in (
            (hf.TestVisionFamilies(), hf.KINDS),
            (patch.TestPatchFamilies(), ("cosmos3_super_vm",)),
        ):
            owner.setUp()
            self.addCleanup(owner.doCleanups)
            for kind in kinds:
                yield owner.fixture(kind)[0]
        owner = rae.TestK7Backbone()
        owner.setUp()
        self.addCleanup(owner.doCleanups)
        yield owner.fixture()[0]
        yield video.TestBackbone().spec()

    def test_recomputation_preserves_outputs_gradients_updates_and_state_keys(self):
        torch.set_num_threads(1)
        for spec in self.fixtures():
            with self.subTest(kind=spec["kind"]):
                torch.manual_seed(27)
                plain = sb.build_trainable_backbone(spec, torch.device("cpu"))
                torch.manual_seed(27)
                checked = sb.build_trainable_backbone(
                    dict(spec, activation_checkpointing=True), torch.device("cpu")
                )
                self.assertEqual(list(plain.state_dict()), list(checked.state_dict()))
                checked.load_state_dict(plain.state_dict(), strict=True)
                counts = [0]

                def count(module, args, counts=counts):
                    counts[0] += 1

                hooks = [
                    m.register_forward_pre_hook(count)
                    for m in checked.modules()
                    if isinstance(m, torch.nn.Linear)
                ]
                x = torch.randn(2, 3, 32, 32)
                plain_forward = plain.forward_features
                checked_forward = checked.forward_features
                assert callable(plain_forward) and callable(checked_forward)
                expected = plain_forward(x)
                actual = checked_forward(x)
                forward_calls = counts[0]
                assert isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor)
                target = torch.randn_like(expected)
                ((expected - target) ** 2).mean().backward()
                ((actual - target) ** 2).mean().backward()
                self.assertGreater(
                    counts[0], forward_calls, "backward must actually recompute"
                )
                torch.testing.assert_close(actual, expected)
                for name, parameter in plain.named_parameters():
                    torch.testing.assert_close(
                        parameter.grad, dict(checked.named_parameters())[name].grad
                    )
                for model in (plain, checked):
                    torch.optim.SGD(model.parameters(), lr=0.01).step()
                for name, state in plain.state_dict().items():
                    torch.testing.assert_close(state, checked.state_dict()[name])
                for hook in hooks:
                    hook.remove()
                plain.eval()
                checked.eval()
                with torch.no_grad():
                    torch.testing.assert_close(plain_forward(x), checked_forward(x))

    def test_invalid_selection_is_rejected_before_loading_weights(self):
        for value in (1, "true", None, [], {}):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "activation_checkpointing"),
            ):
                sb.build_trainable_backbone(
                    dict(SPEC, activation_checkpointing=value), torch.device("cpu")
                )
        with self.assertRaisesRegex(ValueError, "activation_checkpointing"):
            sb.build_trainable_backbone(
                dict(SPEC, activation_checkpointing=True), torch.device("cpu")
            )
        with self.assertRaisesRegex(ValueError, "activation_checkpointing"):
            sb.build_frozen_backbone(
                {"kind": "siglip2_g", "activation_checkpointing": True},
                torch.device("cpu"),
            )

    def test_task_validation_refuses_checkpointing_for_frozen_or_attentive(self):
        from downstream.attention import validate_adaptation

        for adaptation in ("frozen", "attentive"):
            cfg = dict(
                profile="capture_basic5_components",
                adaptation=adaptation,
                **(
                    {"reader_profile": "captured_cross_self_v1"}
                    if adaptation == "attentive"
                    else {}
                ),
                backbone={"kind": "siglip2_g", "activation_checkpointing": True},
            )
            with (
                self.subTest(adaptation=adaptation),
                self.assertRaisesRegex(ValueError, "activation_checkpointing"),
            ):
                validate_adaptation(cfg)

    def test_disabled_setting_preserves_default_and_enable_is_idempotent(self):
        from downstream.activation_checkpointing import enable, validate

        torch.manual_seed(8)
        baseline = sb.build_trainable_backbone(SPEC, torch.device("cpu"))
        torch.manual_seed(8)
        disabled = sb.build_trainable_backbone(
            dict(SPEC, activation_checkpointing=False), torch.device("cpu")
        )
        for key, tensor in baseline.state_dict().items():
            torch.testing.assert_close(
                tensor, disabled.state_dict()[key], rtol=0, atol=0
            )
        self.assertIsNone(validate({"kind": "unknown"}, trainable=False))
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.blocks = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.5))]
        )
        enable(model, ("blocks", "model"))
        forward = model.model.blocks[0].forward
        enable(model, ("blocks", "model"))
        self.assertIs(forward, model.model.blocks[0].forward)
        with self.assertRaisesRegex(ValueError, "native flag"):
            enable(model, ("native_flag", "model"))
        with self.assertRaisesRegex(ValueError, "unknown"):
            enable(model, ("typo", "model"))
        model.model.blocks = torch.nn.ModuleList()
        with self.assertRaisesRegex(ValueError, "nonempty"):
            enable(model, ("blocks", "model"))

    def test_all_dense_and_video_configs_accept_the_explicit_option(self):
        from pathlib import Path

        from downstream import ade20k, coco, nyuv2, ssv2
        from tests.test_downstream_ade20k import smoke_config as ade_config
        from tests.test_downstream_coco import smoke_config as coco_config
        from tests.test_downstream_nyuv2 import smoke_config as nyu_config
        from tests.test_downstream_ssv2 import smoke_config as video_config

        for module, factory in (
            (ade20k, ade_config),
            (nyuv2, nyu_config),
            (coco, coco_config),
            (ssv2, video_config),
        ):
            cfg = factory(Path("/unused"))
            cfg.update(profile="capture_basic5_components", adaptation="finetune")
            cfg["backbone"].update(kind="vjepa2_1", activation_checkpointing=True)
            with self.subTest(task=module.__name__):
                module.validate_config(cfg)

    def test_block_recomputation_preserves_dropout_rng_and_gradients(self):
        import copy

        from downstream.activation_checkpointing import enable

        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(4, 4), torch.nn.Dropout(0.5), torch.nn.Linear(4, 4)
                )
            ]
        )
        plain = copy.deepcopy(model)
        enable(model, ("blocks", "model"))
        x = torch.randn(3, 4)
        torch.manual_seed(42)
        expected = plain.get_submodule("model.blocks.0")(x)
        expected.square().sum().backward()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        actual = model.model.blocks[0](x)
        actual.square().sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for left, right in zip(plain.parameters(), model.parameters(), strict=True):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
