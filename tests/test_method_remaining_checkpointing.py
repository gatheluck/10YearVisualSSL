"""Native ViT and whole-aggregator recomputation on image and video paths."""

import copy
import unittest

try:
    import torch
    from timm.models.vision_transformer import VisionTransformer
    from torch import nn

    from downstream import spatial_backbones as sb
    from tests import test_method_cradiov4_h as radio_tests
    from tests import test_method_vggt_omega as omega_tests

    HAVE = True
except ImportError:
    HAVE = False


if HAVE:

    class NativeRadio(nn.Module):
        """Interface fixture with real timm blocks; official CPE is checked privately."""

        def __init__(self):
            super().__init__()
            self.model = VisionTransformer(
                img_size=32,
                patch_size=16,
                embed_dim=32,
                depth=2,
                num_heads=4,
                reg_tokens=9,
                num_classes=0,
            )
            self.input_conditioner = nn.Identity()
            self.adaptors = nn.ModuleDict()
            self.num_summary_tokens = 10
            self.summary_dim = 64
            self.register_buffer("summary_idxs", torch.tensor([0, 1]))

        def forward(self, images):
            tokens = self.model.forward_features(images)
            return tokens[:, :2].flatten(1), tokens[:, 10:]


@unittest.skipUnless(HAVE, "Torch and timm required")
class TestRemainingCheckpointing(unittest.TestCase):
    def fixtures(self):
        radio = radio_tests.TestRadio()
        radio.setUp()
        self.addCleanup(radio.doCleanups)
        spec, reference = radio.fixture()
        reference.radio_model = NativeRadio()
        yield dict(spec, embed_dim=32, depth=2)
        omega = omega_tests.TestOmega()
        omega.setUp()
        self.addCleanup(omega.doCleanups)
        yield omega.fixture()[0]

    def test_native_image_video_and_detection_recompute_with_equal_updates(self):
        torch.set_num_threads(1)
        for spec in self.fixtures():
            for route in ("image", "video", "detection", "summary", "video_summary"):
                with self.subTest(kind=spec["kind"], route=route):
                    plain = sb.build_trainable_backbone(spec, torch.device("cpu"))
                    try:
                        checked = sb.build_trainable_backbone(
                            dict(spec, activation_checkpointing=True),
                            torch.device("cpu"),
                        )
                    except ValueError as exc:
                        self.fail(
                            f"source-supported recomputation is unavailable: {exc}"
                        )
                    checked.load_state_dict(plain.state_dict(), strict=True)
                    self.assertEqual(
                        list(plain.state_dict()), list(checked.state_dict())
                    )
                    x = (
                        torch.randn(2, 2, 3, 32, 32)
                        if route.startswith("video")
                        else torch.randn(2, 3, 32, 32)
                    )
                    if route == "detection":
                        x = x.sigmoid()

                    def forward(model, route=route, x=x):
                        if route in ("summary", "video_summary"):
                            return model.classification_features(
                                x, adaptation="finetune", video=route == "video_summary"
                            )
                        if route == "video":
                            return model.video_tokens(x)
                        if route == "detection":
                            return model.forward_detection_features(x)
                        return model.forward_features(x)

                    calls = [0]

                    def count(module, args, calls=calls):
                        calls[0] += 1

                    hooks = [
                        m.register_forward_pre_hook(count)
                        for m in checked.modules()
                        if isinstance(m, nn.Linear)
                    ]
                    before = copy.deepcopy(plain.state_dict())
                    torch.manual_seed(83)
                    expected = forward(plain)
                    torch.manual_seed(83)
                    actual = forward(checked)
                    forward_calls = calls[0]
                    target = torch.randn_like(expected)
                    (expected - target).square().mean().backward()
                    (actual - target).square().mean().backward()
                    self.assertGreater(calls[0], forward_calls)
                    torch.testing.assert_close(actual, expected)
                    for name, param in plain.named_parameters():
                        torch.testing.assert_close(
                            param.grad, dict(checked.named_parameters())[name].grad
                        )
                    for model in (plain, checked):
                        torch.optim.SGD(model.parameters(), lr=0.03).step()
                    self.assertTrue(
                        any(
                            not torch.equal(before[name], tensor)
                            for name, tensor in plain.state_dict().items()
                        )
                    )
                    for name, tensor in plain.state_dict().items():
                        torch.testing.assert_close(tensor, checked.state_dict()[name])
                    for hook in hooks:
                        hook.remove()
                    plain.eval()
                    checked.eval()
                    with torch.no_grad():
                        torch.testing.assert_close(forward(plain), forward(checked))

    def test_whole_module_preserves_registration_and_eval_bypasses_checkpoint(self):
        from unittest import mock

        from torch.utils.checkpoint import checkpoint

        from downstream.activation_checkpointing import enable

        owner = omega_tests.TestOmega()
        owner.setUp()
        self.addCleanup(owner.doCleanups)
        spec = owner.fixture()[0]
        model = sb.build_trainable_backbone(spec, torch.device("cpu"))
        keys = list(model.state_dict())
        forward = model.forward_features
        assert callable(forward)
        with mock.patch("torch.utils.checkpoint.checkpoint", wraps=checkpoint) as run:
            output = forward(torch.randn(2, 3, 32, 32))
            assert isinstance(output, torch.Tensor)
            output.square().mean().backward()
            self.assertEqual(run.call_count, 0)
            model.zero_grad()
            enable(model, ("native_flag", ""))
            enable(model, ("native_flag", ""))
            self.assertEqual(keys, list(model.state_dict()))
            model.eval()
            output = forward(torch.randn(2, 3, 32, 32))
            assert isinstance(output, torch.Tensor)
            output.square().mean().backward()
            self.assertEqual(run.call_count, 0)
            model.train()
            model.zero_grad()
            output = forward(torch.randn(2, 3, 32, 32))
            assert isinstance(output, torch.Tensor)
            output.square().mean().backward()
            self.assertEqual(run.call_count, 1)
            self.assertFalse(run.call_args.kwargs["use_reentrant"])

    def test_copied_aggregator_recomputes_its_own_parameters(self):
        owner = omega_tests.TestOmega()
        owner.setUp()
        self.addCleanup(owner.doCleanups)
        spec = owner.fixture()[0]
        original = sb.build_trainable_backbone(
            dict(spec, activation_checkpointing=True), torch.device("cpu")
        )
        cloned = copy.deepcopy(original)
        forward = cloned.forward_features
        assert callable(forward)
        output = forward(torch.randn(2, 3, 32, 32))
        assert isinstance(output, torch.Tensor)
        loss = output.square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is None for p in original.parameters()))
        self.assertTrue(
            any(
                p.grad is not None and bool(p.grad.abs().sum() > 0)
                for p in cloned.parameters()
            )
        )
